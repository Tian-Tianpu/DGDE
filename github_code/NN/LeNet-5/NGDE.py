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

class NGDE():

    def __init__(self, trainloader, testloader, valloader, criterion, lambda_reg,
                 population_size, F, fitness_p, niche_size, num_epoch):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.device = device
        self.criterion = criterion
        self.niche_size = niche_size
        self.lambda_reg = lambda_reg
        self.F = F
        self.fitness_p = fitness_p
        self.population_size = int(population_size)
        self.num_epoch = num_epoch
        self.t = 0

    # 定义计算损失函数
    def calculate_loss(self, model, inputs, targets):
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    # 定义评估模型函数
    def evaluate_model(self, model, dataloader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, outputs = self.calculate_loss(model, inputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def prob(self, cost, Fitness):
        p = (cost - min(Fitness)) / (max(Fitness) - min(Fitness) + 10e-8)
        return p

    # 定义训练函数
    def NGDE_update1(self, model, optimizer):
        model.train()
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 参数更新
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time

        return model, epoch_time

    def NGDE_update2(self, model, best_model, optimizer):
        model.train()
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_model.parameters()):
                param.data = param.data + self.F * (best_param.data - param.data)

        end_time = time.time()
        epoch_time = end_time - start_time

        return model, epoch_time

    def NGDE_update3(self, population, model, optimizer, best_model):

        model.train()

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            j, k = random.sample(range(self.population_size), 2)
            xj, xk = population[j].parameters(), population[k].parameters()
            for param, best_param, param_j, param_k in zip(model.parameters(), best_model.parameters(), xj, xk):
                param.data = param.data + self.F * (best_param.data - param.data) + self.F * (param_j.data - param_k.data)

        end_time = time.time()
        epoch_time = end_time - start_time

        return model, epoch_time

    def update(self, model, best_model, population, fitness_model, Fitness, optimizer):
        # 对种群个体进行排序，按照Fitness值从小到大排序
        if fitness_model == np.min(Fitness):
            model_update, epoch_time = self.NGDE_update1(model, optimizer)
            return model_update, epoch_time
        else:
            p = self.prob(fitness_model, Fitness)
            if p > self.fitness_p:
                model_update, epoch_time = self.NGDE_update3(population, model, optimizer, best_model)
                return model_update, epoch_time
            else:
                model_update, epoch_time = self.NGDE_update2(model, best_model, optimizer)
                return model_update, epoch_time

    def model_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """计算两个模型之间的欧氏距离（L2范数）。"""
        distance = 0.0
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            distance += torch.sum((param1 - param2) ** 2).item()
        return distance ** 0.5

    def distance_sort(self, individual, population):
        Distance = np.zeros(len(population))
        for i in range(len(population)):
            Distance[i] = self.model_distance(individual, population[i])

        Distance_sort = sorted(zip(population, Distance), key=lambda x: x[1], reverse=False)
        return Distance_sort

    def get_niche(self, population):
        Niche = []
        for i in range(int(self.population_size / self.niche_size)):
            best_individual, Fitness, Fitness_sort = self.fitness_sort(population)
            Distance_sort = self.distance_sort(best_individual, population)
            population1, niche = [], []
            for j in range(len(Distance_sort)):
                population1.append(Distance_sort[j][0])
            # print("niche size:", self.niche_size)
            for k in range(0, self.niche_size):
                niche.append(Distance_sort[k][0])
            Niche.append(niche)
            del population1[:self.niche_size]
            population = population1
        if len(population) != 0:
            Niche.append(population)
        return Niche

    def fitness(self, population):
        ## fitness加入参数的L2范数作为正则项
        Fitness = [get_loss(model, self.valloader, self.device, self.criterion) for model in population]
        return Fitness

    def fitness_sort(self, population):
        Fitness = self.fitness(population)
        Fitness_sort = sorted(zip(population, Fitness), key=lambda x: x[1], reverse=False)
        # print(Fitness_sort)
        best_individual = Fitness_sort[0][0]
        return best_individual, Fitness, Fitness_sort

    def Near_select(self, individual1, cost1, population):#, offspring_population):
        d_sort = self.distance_sort(individual1, population)
        individual_nearest = d_sort[1][0]
        cost_nearest = get_loss(individual_nearest,self.valloader, self.device, self.criterion)
        if cost1 <= cost_nearest:
            return individual1
        else:
            return individual_nearest


    def calculate_sparsity(self, individual):
        total_weights = 0
        zero_weights = 0
        for param in individual.parameters():
            total_weights += param.numel()
            zero_weights += torch.sum(param == 0).item()
        return zero_weights / total_weights

    def weight_l1(self, model):
        # 计算L1正则化项
        l1_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return l1_reg

    def weight_l0(self, model, threshold):
        l0_norm = 0
        total_elements = 0
        for param in model.parameters():
            total_elements += param.nelement()
            temp_param = param.data.clone()
            temp_param[temp_param.abs() < threshold] = 0
            l0_norm += (temp_param != 0).sum().item()
        return l0_norm / total_elements if total_elements > 0 else 0

    def NGDE(self, optimizers, population, filename):  # 添加 filename 参数

        thresholds = [0.01, 0.001, 0.0001]# 使用初始化时的 prune_threshold

        results = {
            "train_losses": [],
            "test_losses": [],
            "valid_losses": [],
            "train_accs": [],
            "test_accs": [],
            "valid_accs": [],
            "epoch_times": [],
            "weight_individuals": [],
            "sparsity_weights": {threshold: [] for threshold in thresholds},
            "best_train_accs": [],
            "best_train_losses": [],
            "best_test_accs": [],
            "best_test_losses": [],
            "best_valid_accs": [],
            "best_valid_losses": [],
            "epoch_times_all": [],
            "weight_epochs": [],
            "num_niches": [],
            "sparsity_epochs": {threshold: [] for threshold in thresholds}
        }

        best_individual, Fitness, Fitness_sort = self.fitness_sort(population)
        print("Fitness:", Fitness)

        while self.t < self.num_epoch:
            print(f"Epoch [{self.t + 1}/{self.num_epoch}]")
            train_losses, test_losses, valid_losses = [], [], []
            train_accs, test_accs, valid_accs = [], [], []
            epoch_times, weight_individuals, sparsity_individuals = [], [], []

            start_time = time.time()
            niches = self.get_niche(population)
            results["num_niches"].append(len(niches))
            print("Number of niches:", len(niches))

            for niche in niches:
                best_individual, fitness, _ = self.fitness_sort(niche)
                for model, optimizer in zip(niche, optimizers):
                    fitness_model = get_loss(model, self.valloader, self.device, self.criterion)
                    model, epoch_time = self.update(model, best_individual, population, fitness_model, fitness,
                                                    optimizer)

                    train_loss, train_acc = self.evaluate_model(model, self.trainloader)
                    test_loss, test_acc = self.evaluate_model(model, self.testloader)
                    valid_loss, valid_acc = self.evaluate_model(model, self.valloader)
                    weight = self.weight_l1(model)
                    weight_individuals.append(weight.item())
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    valid_losses.append(valid_loss)
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    valid_accs.append(valid_acc)
                    epoch_times.append(time.time() - start_time)

                    sparsity = {threshold: self.weight_l0(model, threshold) for threshold in thresholds}
                    for threshold, value in sparsity.items():
                        results["sparsity_weights"][threshold].append(value)  # 修改处

                    print(
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, "
                        f"Test Acc: {test_acc:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, "
                        f"Weight: {weight:.4f}, Sparsity: {sparsity}"
                    )

            population = [self.Near_select(model, get_loss(model, self.valloader, self.device, self.criterion), population) for model in population]
            end_time = time.time()
            results["epoch_times_all"].append(end_time - start_time)

            best_individual, _, _ = self.fitness_sort(population)
            best_train_loss, best_train_acc = acc_loss(best_individual, self.trainloader, self.device, self.criterion)
            best_test_loss, best_test_acc = acc_loss(best_individual, self.testloader, self.device, self.criterion)
            best_valid_loss, best_valid_acc = acc_loss(best_individual, self.valloader, self.device, self.criterion)
            weight_epoch = self.weight_l1(best_individual).item()

            results["train_losses"].append(train_losses)
            results["test_losses"].append(test_losses)
            results["valid_losses"].append(valid_losses)
            results["train_accs"].append(train_accs)
            results["test_accs"].append(test_accs)
            results["valid_accs"].append(valid_accs)
            results["epoch_times"].append(epoch_times)
            results["weight_individuals"].append(weight_individuals)
            results["best_train_losses"].append(best_train_loss)
            results["best_train_accs"].append(best_train_acc)
            results["best_test_losses"].append(best_test_loss)
            results["best_test_accs"].append(best_test_acc)
            results["best_valid_losses"].append(best_valid_loss)
            results["best_valid_accs"].append(best_valid_acc)
            results["weight_epochs"].append(weight_epoch)

            sparsity_epoch = {threshold: self.weight_l0(best_individual, threshold) for threshold in thresholds}
            for threshold, value in sparsity_epoch.items():
                results["sparsity_epochs"][threshold].append(value)

            print(
                f"Best Individual - Train Loss: {best_train_loss:.4f}, Train Acc: {best_train_acc:.2f}%, "
                f"Test Loss: {best_test_loss:.4f}, Test Acc: {best_test_acc:.2f}%, "
                f"Valid Loss: {best_valid_loss:.4f}, Valid Acc: {best_valid_acc:.2f}%, "
                f"Weight: {weight_epoch:.4f}, Sparsity: {sparsity_epoch}"
            )
            self.t += 1

        with open(filename, "wb") as f:
            joblib.dump(results, f)
        print(f"Results saved to {filename}")
        #


def load_and_print_results(filename):
    """加载 .pkl 文件并打印所有结果。

    Args:
        filename (str): .pkl 文件名。
    """
    try:
        results = joblib.load(filename)
        for key, value in results.items():
            print(f"{key}:")
            if isinstance(value, list):
                for item in value:
                    print(item)
            else:
                print(value)
            print("-" * 20)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def print_sparsity(filename):
    """加载 .pkl 文件并打印不同 threshold 的 sparsity。

    Args:
        filename (str): .pkl 文件名。
    """
    try:
        results = joblib.load(filename)
        sparsity_epochs = results["sparsity_epochs"]

        print("Sparsity for different thresholds:")
        for threshold, sparsity_list in sparsity_epochs.items():
            print(f"Threshold: {threshold}")
            print(f"Sparsity: {sparsity_list}")
            print("-" * 20)  # 分隔不同 threshold 的 sparsity
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
