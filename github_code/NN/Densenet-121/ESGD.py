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

class ESGD():

    def __init__(self, trainloader, testloader, valloader, criterion, gd_epochs, evo_epochs, lambda_reg,
                 population_size, F, num_epoch):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.device = device
        self.criterion = criterion
        self.lambda_reg = lambda_reg
        self.gd_epochs = gd_epochs
        self.evo_epochs = evo_epochs
        self.F = F
        self.population_size = int(population_size)  # 转换为整数
        self.num_epoch = num_epoch
        self.t = 0
        self.t_gd = 0
        self.t_evo = 0

    # 定义计算损失函数
    def calculate_loss(self, model, inputs, targets):
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        loss = loss + (self.lambda_reg / 2) * l2_reg
        return loss, outputs

    def weight_l1(self, model):
        # 计算L1正则化项
        l1_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return l1_reg


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
                l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
                for param in model.parameters():
                    l2_reg = l2_reg + torch.norm(param, 2)
                loss = loss + (self.lambda_reg / 2) * l2_reg
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    # 定义训练函数
    def update_GD(self, model, optimizer):
        model.train()
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 参数更新
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            # 计算L2正则化项
            l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)
            # 将L2正则化项加入到损失中
            loss = loss + (self.lambda_reg / 2) * l2_reg

            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time


        return model, epoch_time

    def update_ES(self, population, model, best_model):

        model.train()
        start_time = time.time()

        with torch.no_grad():
            j, k = random.sample(range(self.population_size), 2)
            xj, xk = population[j].parameters(), population[k].parameters()
            for param, best_param, param_j, param_k in zip(model.parameters(), best_model.parameters(), xj, xk):
                param.data = param.data + self.F * (best_param.data - param.data) + self.F * (param_j.data - param_k.data)

        end_time = time.time()
        epoch_time = end_time - start_time

        return model, epoch_time
    #



    def fitness(self, population):
        Fitness = [get_loss(model, self.valloader, self.device, self.criterion) for model in population]
        return Fitness

    def fitness_sort(self, population):
        Fitness = self.fitness(population)
        Fitness_sort = sorted(zip(population, Fitness), key=lambda x: x[1], reverse=False)
        best_individual = Fitness_sort[0][0]
        return best_individual, Fitness, Fitness_sort

    def select(self,  Fitness_sort):
        next_generation = []
        new_fitness = []
        num_best_selected = self.population_size // 2
        num_random_selected = self.population_size - num_best_selected

        # 选择前50%最优个体
        next_generation.extend([individual for individual, fitness in Fitness_sort[0:num_best_selected]])
        new_fitness.extend([fitness for individual, fitness in Fitness_sort[0:num_best_selected]])

        # 随机选取50%个体
        remaining_individuals = [individual for individual, fitness in Fitness_sort[num_best_selected:]]
        remaining_fitness = [fitness for individual, fitness in Fitness_sort[num_best_selected:]]
        random_indices = np.random.choice(len(remaining_individuals), num_random_selected, replace=False)
        next_generation.extend([remaining_individuals[i] for i in random_indices])
        new_fitness.extend([remaining_fitness[i] for i in random_indices])

        # 更新种群
        population = next_generation
        return population, new_fitness

    def weight_l0(self, model, threshold):
        l0_norm = 0
        total_elements = 0

        for param in model.parameters():
            total_elements += param.nelement()
            temp_param = param.data.clone()
            temp_param[temp_param.abs() < threshold] = 0
            l0_norm += (temp_param != 0).sum().item()
        l0_norm = l0_norm / total_elements if total_elements > 0 else 0

        return l0_norm


    def calculate_sparsity(self, individual):
        total_weights = 0
        zero_weights = 0
        for param in individual.parameters():
            total_weights += param.numel()
            zero_weights += torch.sum(param == 0).item()

        sparsity = zero_weights / total_weights
        return sparsity

    def ESGD(self, optimizers, population, filename):  # 添加 filename 参数

        thresholds = [0.01, 0.001, 0.0001]

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
            "sparsity_epochs": {threshold: [] for threshold in thresholds}
        }

        best_individual, Fitness, Fitness_sort = self.fitness_sort(population)
        print("Fitness:", Fitness)

        while self.t < self.num_epoch:
            print(f"Epoch [{self.t + 1}/{self.num_epoch}]; GD Epoch [{self.t_gd + 1}/{self.gd_epochs}]")
            while self.t_gd < self.gd_epochs and self.t < self.num_epoch:
                loss = []
                updated_population = []
                train_losses, test_losses = [], []
                train_accs, test_accs = [], []
                valid_losses, valid_accs = [], []
                epoch_times, weight_individuals, sparsity_individuals = [], [], []

                begin_time_gd = time.time()
                for i, (model, optimizer) in enumerate(zip(population, optimizers)):
                    fitness_model = get_loss(model, self.valloader, self.device, self.criterion)
                    update_model, epoch_time = self.update_GD(model, optimizer)

                    updated_population.append(update_model)

                    train_loss, train_acc = self.evaluate_model(update_model, self.trainloader)
                    loss.append(train_loss)
                    test_loss, test_acc = self.evaluate_model(update_model, self.testloader)
                    valid_loss, valid_acc = self.evaluate_model(update_model, self.valloader)
                    weight = self.weight_l1(model)
                    weight_individuals.append(weight.item())
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    valid_losses.append(valid_loss)
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    valid_accs.append(valid_acc)
                    epoch_times.append(time.time() - begin_time_gd)

                    sparsity = {threshold: self.weight_l0(model, threshold) for threshold in thresholds}
                    for threshold, value in sparsity.items():
                        results["sparsity_weights"][threshold].append(value)

                    print(
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, "
                        f"Test Acc: {test_acc:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, "
                        f"Weight: {weight:.4f}, Sparsity: {sparsity}"
                    )

                results["train_losses"].append(list(train_losses))
                results["test_losses"].append(list(test_losses))
                results["valid_losses"].append(list(valid_losses))
                results["train_accs"].append(list(train_accs))
                results["test_accs"].append(list(test_accs))
                results["valid_accs"].append(list(valid_accs))
                results["epoch_times"].append(list(epoch_times))
                results["weight_individuals"].append(weight_individuals)

                best_individual, Fitness, Fitness_sort = self.fitness_sort(updated_population + population)
                population, Fitness = self.select(Fitness_sort)

                end_time_gd = time.time()
                results["epoch_times_all"].append(end_time_gd - begin_time_gd)

                # 选择最优个体
                print("min fitness in validation data: {} ".format(np.min(Fitness)))
                test_loss_best, test_acc_best = self.evaluate_model(best_individual, self.testloader)
                train_loss_best, train_acc_best = self.evaluate_model(best_individual, self.trainloader)
                valid_loss_best, valid_acc_best = self.evaluate_model(best_individual, self.valloader)
                Weight = self.weight_l1(best_individual)

                sparsity_epoch = {threshold: self.weight_l0(best_individual, threshold) for threshold in thresholds}
                for threshold, value in sparsity_epoch.items():
                    results["sparsity_epochs"][threshold].append(value)

                print(
                    f"Best Individual: Train Loss: {train_loss_best:.4f}, Train Acc: {train_acc_best:.2f}%, "
                    f"Test Loss: {test_loss_best:.4f}, Test Acc: {test_acc_best:.2f}%, "
                    f"Valid Loss: {valid_loss_best:.4f}, Valid Acc: {valid_acc_best:.2f}%, "
                    f"Weight: {Weight:.4f},Weight sparsity: {sparsity_epoch}"
                    f",time_all:{results['epoch_times_all'][-1]:.2f}s")

                results["best_train_losses"].append(train_loss_best)
                results["best_test_losses"].append(test_loss_best)
                results["best_valid_losses"].append(valid_loss_best)
                results["best_train_accs"].append(train_acc_best)
                results["best_test_accs"].append(test_acc_best)
                results["best_valid_accs"].append(valid_acc_best)
                results["weight_epochs"].append(Weight.item())

                self.t += 1
                self.t_gd += 1

            self.t_gd = 0

        print(f"Epoch [{self.t + 1}/{self.num_epoch}]; ES Epoch [{self.t_evo + 1}/{self.evo_epochs}]")
        while self.t_evo < self.evo_epochs and self.t < self.num_epoch:
            loss = []
            updated_population = []
            train_losses, test_losses = [], []
            train_accs, test_accs = [], []
            valid_losses, valid_accs = [], []
            epoch_times, Weight_individual, sparsity_individuals = [], [], []

            begin_time_evo = time.time()
            best_individual, Fitness, Fitness_sort = self.fitness_sort(population)
            for i, (model, optimizer) in enumerate(zip(population, optimizers)):
                update_model, epoch_time = self.update_ES(population, model, best_individual)

                updated_population.append(update_model)

                train_loss, train_acc = self.evaluate_model(update_model, self.trainloader)
                loss.append(train_loss)
                test_loss, test_acc = self.evaluate_model(update_model, self.testloader)
                valid_loss, valid_acc = self.evaluate_model(update_model, self.valloader)
                weight = self.weight_l1(update_model)

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                valid_losses.append(valid_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                valid_accs.append(valid_acc)
                epoch_times.append(epoch_time)
                Weight_individual.append(weight.item())
                sparsity = {threshold: self.weight_l0(model, threshold) for threshold in thresholds}
                for threshold, value in sparsity.items():
                    results["sparsity_weights"][threshold].append(value)
                print(
                    f"Individual {i + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
                    f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, "
                    f"Weight: {weight:.4f}, Time: {epoch_time:.2f}s")

                results["train_losses"].append(list(train_losses))
                results["test_losses"].append(list(test_losses))
                results["valid_losses"].append(list(valid_losses))
                results["train_accs"].append(list(train_accs))
                results["test_accs"].append(list(test_accs))
                results["valid_accs"].append(list(valid_accs))
                results["epoch_times"].append(list(epoch_times))
                results["weight_individuals"].append(Weight_individual)

                best_individual, Fitness, Fitness_sort = self.fitness_sort(updated_population + population)
                population, Fitness = self.select(Fitness_sort)

                end_time_evo = time.time()
                results["epoch_times_all"].append(end_time_evo - begin_time_evo)

                # 选择最优个体
                print("min fitness in validation data: {} ".format(np.min(Fitness)))
                test_loss_best, test_acc_best = self.evaluate_model(best_individual, self.testloader)
                train_loss_best, train_acc_best = self.evaluate_model(best_individual, self.trainloader)
                valid_loss_best, valid_acc_best = self.evaluate_model(best_individual, self.valloader)
                Weight = self.weight_l1(best_individual)

                sparsity_epoch = {threshold: self.weight_l0(best_individual, threshold) for threshold in thresholds}
                for threshold, value in sparsity_epoch.items():
                    results["sparsity_epochs"][threshold].append(value)

                print(
                    f"Best Individual: Train Loss: {train_loss_best:.4f}, Train Acc: {train_acc_best:.2f}%, "
                    f"Test Loss: {test_loss_best:.4f}, Test Acc: {test_acc_best:.2f}%, "
                    f"Valid Loss: {valid_loss_best:.4f}, Valid Acc: {valid_acc_best:.2f}%, "
                    f"Weight: {Weight:.4f},Weight sparsity: {sparsity_epoch}"
                    f",time_all:{results['epoch_times_all'][-1]:.2f}s")

                results["best_train_losses"].append(train_loss_best)
                results["best_test_losses"].append(test_loss_best)
                results["best_valid_losses"].append(valid_loss_best)
                results["best_train_accs"].append(train_acc_best)
                results["best_test_accs"].append(test_acc_best)
                results["best_valid_accs"].append(valid_acc_best)
                results["weight_epochs"].append(Weight.item())

                self.t += 1
                self.t_evo += 1
            self.t_evo = 0

        with open(filename, "wb") as f:
            joblib.dump(results, f)
        print(f"Results saved to {filename}")

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
#
#
# if __name__ == "__main__":
#
#     batch_size = 128
#     num_epoch = 4
#     population_size = 2
#     lambda_reg = 0.001
#     learning_rate = 0.001
#     F = 0.5
#     gd_epochs = 2
#     evo_epochs = 2
#
#     import torch.optim as optim
#     import torch.nn as nn
#
#
#     (train_dataset, val_dataset, test_dataset,
#      trainloader, valloader, testloader, n) = choose_dataset_subset("MNIST", batch_size, 100)
#
#     print('n:', n)
#
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
#     print(f"Number of classes: {n}")
#
#
#     criterion = nn.CrossEntropyLoss()
#     optimizers = [optim.SGD(LeNet5(n).parameters(), lr=learning_rate) for _ in range(population_size)]
#     # optimizers = [optim.Adam(LeNet5(n).parameters(), lr=learning_rate) for _ in range(population_size)]
#     # population = [create_grayscale_model("resnet18", n).cuda() for _ in range(population_size)]
#     population = [create_grayscale_model("lenet5", n) for _ in range(population_size)]
#     # population = [create_grayscale_model("resnet18", n) for _ in range(population_size)]
#
#     # n_class = num_classes
#
#     filename = "./ESGD_SGD_results.pkl"
#
#
#     esgd = ESGD(trainloader, testloader, valloader, criterion, gd_epochs, evo_epochs, lambda_reg,
#                  population_size, F, num_epoch)
#     esgd.ESGD(optimizers, population, filename)
#
#     results = joblib.load(filename)
#     print("Results loaded from:", filename)
#     print("Best Test Accuracies:", results["best_test_accs"])
#
#     load_and_print_results(filename)
#     print_sparsity(filename)

    # def ESGD(self, optimizers, population, prune_threshold):
    #
    #     # 训练模型
    #     train_losses_all, test_losses_all = [], []
    #     train_accs_all, test_accs_all = [], []
    #     valid_losses_all, valid_accs_all = [], []
    #     epoch_times_all, Weight_individual_all, Sparsity_weight_all = [], [], []
    #
    #     Train_Accuracy_list, Train_cost_list, Test_Accuracy_list, Test_cost_list, Valid_cost_list, Valid_Accuracy_list = [], [], [], [], [], []
    #     times, Weight_epoch, Sparsity_weight = [], [], []
    #
    #     best_individual, Fitness, Fitness_sort = self.fitness_sort(population)
    #     print("Fitness:", Fitness)
    #
    #     while self.t < self.num_epoch:
    #         print(f"Epoch [{self.t + 1}/{self.num_epoch}]; GD Epoch [{self.t_gd + 1}/{self.gd_epochs}]")
    #         while self.t_gd < self.gd_epochs and self.t < self.num_epoch:
    #             loss = []
    #             updated_population = []
    #             train_losses, test_losses = [], []
    #             train_accs, test_accs = [], []
    #             valid_losses, valid_accs = [], []
    #             epoch_times, Weight_individual = [], []
    #             Sparsity_weight_individual = []
    #
    #             begin_time_gd = time.time()
    #             for i, (model, optimizer) in enumerate(zip(population, optimizers)):
    #                 fitness_model = get_loss(model, self.valloader, self.device, self.criterion)
    #                 update_model, epoch_time = self.update_GD(model, optimizer)
    #
    #                 updated_population.append(update_model)
    #
    #                 train_loss, train_acc = acc_loss(update_model, self.trainloader, self.device, self.criterion)
    #                 loss.append(train_loss)
    #                 test_loss, test_acc = acc_loss(update_model, self.testloader, self.device, self.criterion)
    #                 valid_loss, valid_acc = acc_loss(update_model, self.valloader, self.device, self.criterion)
    #                 weight = self.weight_l1(update_model)
    #
    #                 train_losses.append(train_loss)
    #                 test_losses.append(test_loss)
    #                 valid_losses.append(valid_loss)
    #                 train_accs.append(train_acc)
    #                 test_accs.append(test_acc)
    #                 valid_accs.append(valid_acc)
    #                 epoch_times.append(epoch_time)
    #                 sparsity = self.weight_l0(update_model, prune_threshold)
    #                 Sparsity_weight_individual.append(sparsity)
    #                 Weight_individual.append(int(weight))
    #
    #                 print(
    #                     f"Individual {i + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
    #                     f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
    #                     f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, "
    #                     f"Weight: {weight:.4f}, Time: {epoch_time:.2f}s")
    #
    #             train_losses_all.append(list(train_losses))
    #             test_losses_all.append(list(test_losses))
    #             valid_losses_all.append(list(valid_losses))
    #             train_accs_all.append(list(train_accs))
    #             test_accs_all.append(list(test_accs))
    #             valid_accs_all.append(list(valid_accs))
    #             epoch_times_all.append(list(epoch_times))
    #             Weight_individual_all.append(list(Weight_individual))
    #             Sparsity_weight_all.append(list(Sparsity_weight))
    #
    #             best_individual, Fitness, Fitness_sort = self.fitness_sort(updated_population + population)
    #             population, Fitness = self.select(Fitness_sort)
    #
    #             end_time_gd = time.time()
    #             epoch_time_all = end_time_gd - begin_time_gd
    #
    #             # 选择最优个体
    #             print("min fitness in validation data: {} ".format(np.min(Fitness)))
    #             test_loss_best, test_acc_best = acc_loss(best_individual, self.testloader, self.device, self.criterion)
    #             train_loss_best, train_acc_best = acc_loss(best_individual, self.trainloader, self.device,
    #                                                        self.criterion)
    #             valid_loss_best, valid_acc_best = acc_loss(best_individual, self.valloader, self.device, self.criterion)
    #             Weight = self.weight_l1(best_individual)
    #
    #             # 存储每一代中最优个体的精度和损失
    #
    #             sparsity_best = self.weight_l0(best_individual, prune_threshold)
    #
    #             print(
    #                 f"Best Individual: Train Loss: {train_loss_best:.4f}, Train Acc: {train_acc_best:.2f}%, "
    #                 f"Test Loss: {test_loss_best:.4f}, Test Acc: {test_acc_best:.2f}%, "
    #                 f"Valid Loss: {valid_loss_best:.4f}, Valid Acc: {valid_acc_best:.2f}%, "
    #                 f"Weight: {Weight:.4f},Weight sparsity: {sparsity_best:.4f},time_all:{epoch_time_all:.2f}s")
    #
    #             # 存储每一代中最优个体的精度和损失
    #
    #             Train_cost_list.append(train_loss_best)
    #             Test_cost_list.append(test_loss_best)
    #             Valid_cost_list.append(valid_loss_best)
    #             Train_Accuracy_list.append(train_acc_best)
    #             Test_Accuracy_list.append(test_acc_best)
    #             Valid_Accuracy_list.append(valid_acc_best)
    #             times.append(epoch_time_all)
    #             Weight_epoch.append(int(Weight))
    #             Sparsity_weight.append(sparsity_best)
    #
    #             self.t += 1
    #             self.t_gd += 1
    #
    #         self.t_gd = 0
    #
    #         while self.t_evo < self.evo_epochs and self.t < self.num_epoch:
    #             loss = []
    #             updated_population = []
    #             train_losses, test_losses = [], []
    #             train_accs, test_accs = [], []
    #             valid_losses, valid_accs = [], []
    #             epoch_times, Weight_individual = [], []
    #             Sparsity_weight_individual = []
    #
    #             begin_time_evo = time.time()
    #             best_individual, Fitness, Fitness_sort = self.fitness_sort(population)
    #             for i, (model, optimizer) in enumerate(zip(population, optimizers)):
    #                 # fitness_model = get_loss(model, self.valloader, self.device, self.criterion)
    #                 update_model, epoch_time = self.update_ES(population, model, best_individual)
    #
    #                 updated_population.append(update_model)
    #
    #                 train_loss, train_acc = acc_loss(update_model, self.trainloader, self.device, self.criterion)
    #                 loss.append(train_loss)
    #                 test_loss, test_acc = acc_loss(update_model, self.testloader, self.device, self.criterion)
    #                 valid_loss, valid_acc = acc_loss(update_model, self.valloader, self.device, self.criterion)
    #                 weight = self.weight_l1(update_model)
    #
    #                 train_losses.append(train_loss)
    #                 test_losses.append(test_loss)
    #                 valid_losses.append(valid_loss)
    #                 train_accs.append(train_acc)
    #                 test_accs.append(test_acc)
    #                 valid_accs.append(valid_acc)
    #                 epoch_times.append(epoch_time)
    #                 Weight_individual.append(int(weight))
    #                 sparsity = self.weight_l0(update_model, prune_threshold)
    #                 Sparsity_weight_individual.append(sparsity)
    #
    #
    #                 print(
    #                     f"Individual {i + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
    #                     f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
    #                     f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, "
    #                     f"Weight: {weight:.4f}, Time: {epoch_time:.2f}s")
    #
    #             train_losses_all.append(list(train_losses))
    #             test_losses_all.append(list(test_losses))
    #             valid_losses_all.append(list(valid_losses))
    #             train_accs_all.append(list(train_accs))
    #             test_accs_all.append(list(test_accs))
    #             valid_accs_all.append(list(valid_accs))
    #             epoch_times_all.append(list(epoch_times))
    #             Weight_individual_all.append(list(Weight_individual))
    #             Sparsity_weight_all.append(list(Sparsity_weight))
    #
    #             best_individual, Fitness, Fitness_sort = self.fitness_sort(updated_population + population)
    #             population, Fitness = self.select(Fitness_sort)
    #
    #             end_time_evo = time.time()
    #             epoch_time_all = end_time_evo - begin_time_evo
    #
    #             # 选择最优个体
    #             print("min fitness in validation data: {} ".format(np.min(Fitness)))
    #             test_loss_best, test_acc_best = acc_loss(best_individual, self.testloader, self.device, self.criterion)
    #             train_loss_best, train_acc_best = acc_loss(best_individual, self.trainloader, self.device,
    #                                                        self.criterion)
    #             valid_loss_best, valid_acc_best = acc_loss(best_individual, self.valloader, self.device, self.criterion)
    #             Weight = self.weight_l1(best_individual)
    #
    #             sparsity_best = self.weight_l0(best_individual, prune_threshold)
    #
    #             print(
    #                 f"Best Individual: Train Loss: {train_loss_best:.4f}, Train Acc: {train_acc_best:.2f}%, "
    #                 f"Test Loss: {test_loss_best:.4f}, Test Acc: {test_acc_best:.2f}%, "
    #                 f"Valid Loss: {valid_loss_best:.4f}, Valid Acc: {valid_acc_best:.2f}%, "
    #                 f"Weight: {Weight:.4f},Weight sparsity: {sparsity_best:.4f},time_all:{epoch_time_all:.2f}s")
    #
    #             # 存储每一代中最优个体的精度和损失
    #
    #             Train_cost_list.append(train_loss_best)
    #             Test_cost_list.append(test_loss_best)
    #             Valid_cost_list.append(valid_loss_best)
    #             Train_Accuracy_list.append(train_acc_best)
    #             Test_Accuracy_list.append(test_acc_best)
    #             Valid_Accuracy_list.append(valid_acc_best)
    #             times.append(epoch_time_all)
    #             Weight_epoch.append(int(Weight))
    #             Sparsity_weight.append(sparsity_best)
    #
    #             self.t += 1
    #             self.t_evo += 1
    #
    #
    #     return train_losses_all, test_losses_all, valid_losses_all, train_accs_all, test_accs_all, valid_accs_all, \
    #            epoch_times_all, Weight_individual_all, Sparsity_weight_all, \
    #            Train_Accuracy_list, Train_cost_list, Test_Accuracy_list, Test_cost_list, Valid_cost_list, \
    #            Valid_Accuracy_list, times, Weight_epoch, Sparsity_weight
    #
    #
