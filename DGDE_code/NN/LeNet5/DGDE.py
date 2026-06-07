import copy
import random
import time
import torch
import numpy as np
import torch.nn as nn
from ACC_LOSS import *
from datasets import *
from models import *
import joblib


class DGDE():
    def __init__(self, trainloader, testloader, valloader, criterion, lambda_reg,
                 population_size, niche_size, T_per, num_epoch, F, K_base, K_0,
                 pho_high, pho_low, fitness_p):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.criterion = criterion

        self.population_size = int(population_size)
        self.niche_size = niche_size
        self.T_per = T_per  # 周期性更新探索集的步数
        self.num_epoch = num_epoch  # 对应 budget
        self.lambda_reg = lambda_reg

        # 算法特有参数
        self.F = F
        self.K_base = K_base
        self.K_0 = K_0
        self.pho_high = pho_high
        self.pho_low = pho_low
        self.fitness_p = fitness_p
        self.t = 0

    # --- 基础评估函数 (与 NGDE 相同) ---
    def calculate_loss(self, model, inputs, targets):
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def evaluate_model(self, model, dataloader):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, outputs = self.calculate_loss(model, inputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return total_loss / len(dataloader), 100.0 * correct / total

    def prob(self, cost, Fitness):
        eps = 1e-8
        p = (cost - min(Fitness)) / (max(Fitness) - min(Fitness) + eps)
        return p

    # --- 核心更新策略 (对应 DGDE 的 update1, 2, 3) ---
    def DGDE_update1(self, model, optimizer):
        """标准梯度下降"""
        model.train()
        for inputs, targets in self.trainloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            loss, _ = self.calculate_loss(model, inputs, targets)
            loss.backward()
            optimizer.step()
        return model

    def DGDE_update2(self, model, best_model, optimizer):
        """梯度下降 + 基础引导变异"""
        model = self.DGDE_update1(model, optimizer)
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_model.parameters()):
                param.data.add_(self.K_base * (best_param.data - param.data))
        return model

    def DGDE_update3(self, population, model, best_model, optimizer):
        """梯度下降 + 动态引导 + 差分变异 (Explore 模式)"""
        model = self.DGDE_update1(model, optimizer)
        with torch.no_grad():
            j, k = random.sample(range(self.population_size), 2)
            xj, xk = population[j].parameters(), population[k].parameters()
            dynamic_K = self.K_0 + (self.t / self.num_epoch) ** 0.5
            for param, best_param, pj, pk in zip(model.parameters(), best_model.parameters(), xj, xk):
                diff_best = best_param.data - param.data
                diff_pop = pj.data - pk.data
                param.data.add_(dynamic_K * diff_best + self.F * diff_pop)
        return model

    # --- 辅助工具函数 (保持与 NGDE 命名一致) ---
    def model_distance(self, model1, model2):
        distance = 0.0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            distance += torch.sum((p1 - p2) ** 2).item()
        return distance ** 0.5

    def distance_sort(self, individual, population):
        Distance = [self.model_distance(individual, p) for p in population]
        # 返回 (模型, 距离, 全局索引) 以便后续处理
        return sorted(zip(population, Distance, range(len(population))), key=lambda x: x[1])

    def fitness_sort(self, population):
        Fitness = [get_loss(m, self.valloader, self.device, self.criterion) for m in population]
        Fitness_sort = sorted(zip(population, Fitness), key=lambda x: x[1])
        return Fitness_sort[0][0], Fitness, Fitness_sort

    def get_niche(self, population):
        """返回 Niche 的索引集合，确保与 DGDE 逻辑一致"""
        Niche_Groups = []
        remaining_indices = list(range(len(population)))
        while len(remaining_indices) >= self.niche_size:
            # 选出表现最好的
            sub_pop = [population[i] for i in remaining_indices]
            best_m, _, _ = self.fitness_sort(sub_pop)
            # 计算到其他人的距离
            d_sort = self.distance_sort(best_m, sub_pop)
            # 取最近的 N 个全局索引
            niche_indices = [remaining_indices[d_sort[k][2]] for k in range(self.niche_size)]
            Niche_Groups.append(niche_indices)
            remaining_indices = [i for i in remaining_indices if i not in niche_indices]
        if remaining_indices:
            Niche_Groups.append(remaining_indices)
        return Niche_Groups

    def build_explore_set(self, niche_indices, population, fitness_sub):
        E_i = []
        p_values = [self.prob(f, fitness_sub) for f in fitness_sub]
        high_pos = [i for i, p in enumerate(p_values) if p <= self.fitness_p]
        low_pos = [i for i, p in enumerate(p_values) if p > self.fitness_p]

        if high_pos:
            num = max(1, int(len(high_pos) * self.pho_high))
            E_i.extend([niche_indices[i] for i in random.sample(high_pos, min(num, len(high_pos)))])
        if low_pos:
            num = max(1, int(len(low_pos) * (1.0 - self.pho_low)))
            E_i.extend([niche_indices[i] for i in random.sample(low_pos, min(num, len(low_pos)))])
        return list(set(E_i))

    def weight_l1(self, model):
        l1_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return l1_reg

    def weight_l0(self, model, threshold):
        l0_norm, total_elements = 0, 0
        for param in model.parameters():
            total_elements += param.nelement()
            l0_norm += (param.data.abs() >= threshold).sum().item()
        return l0_norm / total_elements if total_elements > 0 else 0

    def DGDE(self, optimizers, population, filename):
        thresholds = [0.01, 0.001, 0.0001]
        results = {
            "best_test_accs": [], "best_test_losses": [],
            "best_valid_accs": [], "best_valid_losses": [],
            "sparsity_epochs": {t: [] for t in thresholds},
            "weight_epochs": [], "num_niches": []
        }
        explore_set = []

        while self.t < self.num_epoch:
            print(f"Epoch [{self.t + 1}/{self.num_epoch}]")
            start_time = time.time()

            # 1. 划分 Niche 索引
            Niche_Indices_List = self.get_niche(population)
            results["num_niches"].append(len(Niche_Indices_List))

            # 2. 周期性更新 Explore Set
            if self.t % self.T_per == 0:
                explore_set = []
                for niche_idxs in Niche_Indices_List:
                    f_sub = [get_loss(population[i], self.valloader, self.device, self.criterion) for i in niche_idxs]
                    explore_set.extend(self.build_explore_set(niche_idxs, population, f_sub))

            # 3. 种群更新
            for niche_idxs in Niche_Indices_List:
                sub_pop = [population[i] for i in niche_idxs]
                best_niche_m, f_sub, _ = self.fitness_sort(sub_pop)

                for idx in niche_idxs:
                    model, opt = population[idx], optimizers[idx]
                    f_model = get_loss(model, self.valloader, self.device, self.criterion)

                    if self.model_distance(model, best_niche_m) < 1e-9:
                        population[idx] = self.DGDE_update1(model, opt)
                    elif idx in explore_set:
                        population[idx] = self.DGDE_update3(population, model, best_niche_m, opt)
                    else:
                        population[idx] = self.DGDE_update2(model, best_niche_m, opt)

            # 4. 评估与记录 (与 NGDE 相同)
            best_ind, _, _ = self.fitness_sort(population)
            v_loss, v_acc = acc_loss(best_ind, self.valloader, self.device, self.criterion)
            t_loss, t_acc = acc_loss(best_ind, self.testloader, self.device, self.criterion)

            results["best_valid_accs"].append(v_acc)
            results["best_valid_losses"].append(v_loss)
            results["best_test_accs"].append(t_acc)
            results["best_test_losses"].append(t_loss)
            results["weight_epochs"].append(self.weight_l1(best_ind).item())

            for th in thresholds:
                results["sparsity_epochs"][th].append(self.weight_l0(best_ind, th))

            print(f"Best Valid Acc: {v_acc:.2f}%, Sparsity(0.001): {results['sparsity_epochs'][0.001][-1]:.4f}")
            self.t += 1

        with open(filename, "wb") as f:
            joblib.dump(results, f)