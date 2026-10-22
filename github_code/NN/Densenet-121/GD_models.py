import torch.optim as optim
import torch
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import LeNet5
import numpy as np
import joblib
from ACC_LOSS import *

class GDs:
    def __init__(self, num_epoch, criterion, train_loader, test_loader, valid_loader, population_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.population_size = population_size
        self.best_test_accuracies = []
        self.best_test_losses = []
        self.population_test_accuracies = []
        self.population_test_losses = []
        self.population_train_accuracies = []
        self.population_train_losses = []
        self.population_times = []
        self.population_val_accuracies = []
        self.population_val_losses = []
        self.best_val_accuracies = []
        self.best_val_losses = []

    def train(self, model, optimizer, data_loader):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        accuracy = 100. * correct / total
        avg_loss = train_loss / len(data_loader)
        return avg_loss, accuracy

    def evaluate(self, model, data_loader):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(data_loader)
        return avg_loss, accuracy

    def GD(self, optimizers, population, filename):
        for epoch in range(self.num_epoch):
            print(f"Epoch [{epoch + 1}/{self.num_epoch}]")
            epoch_test_accuracies = []
            epoch_test_losses = []
            epoch_train_accuracies = []
            epoch_train_losses = []
            epoch_val_accuracies = []
            epoch_val_losses = []
            epoch_times = []

            for i, model in enumerate(population):
                optimizer = optimizers[i]
                start_time = time.time()
                gd_train_loss, gd_train_acc = self.train(model, optimizer, self.train_loader)

                gd_test_loss, gd_test_acc = acc_loss(model, self.test_loader, self.device, self.criterion)
                gd_val_loss, gd_val_acc = acc_loss(model, self.valid_loader, self.device, self.criterion)
                end_time = time.time()
                elapsed_time = end_time - start_time

                epoch_train_accuracies.append(gd_train_acc)
                epoch_train_losses.append(gd_train_loss)
                epoch_test_accuracies.append(gd_test_acc)
                epoch_test_losses.append(gd_test_loss)
                epoch_val_accuracies.append(gd_val_acc)
                epoch_val_losses.append(gd_val_loss)
                epoch_times.append(elapsed_time)

                print(
                    f"Individual {i + 1}: Train Loss: {gd_train_loss:.4f} | Train Acc: {gd_train_acc:.2f}% | Test Loss: {gd_test_loss:.4f} | Test Acc: {gd_test_acc:.2f}% | Val Loss: {gd_val_loss:.4f} | Val Acc: {gd_val_acc:.2f}% | Time: {elapsed_time:.2f}s")

            self.population_train_accuracies.append(epoch_train_accuracies)
            self.population_train_losses.append(epoch_train_losses)
            self.population_test_accuracies.append(epoch_test_accuracies)
            self.population_test_losses.append(epoch_test_losses)
            self.population_val_accuracies.append(epoch_val_accuracies)
            self.population_val_losses.append(epoch_val_losses)
            self.population_times.append(epoch_times)

            best_individual_index = np.argmax(self.population_test_accuracies[epoch])
            best_individual = population[best_individual_index]
            best_test_loss = self.population_test_losses[epoch][best_individual_index]
            best_test_acc = self.population_test_accuracies[epoch][best_individual_index]
            best_train_loss = self.population_train_losses[epoch][best_individual_index]
            best_train_acc = self.population_train_accuracies[epoch][best_individual_index]
            best_val_loss = self.population_val_losses[epoch][best_individual_index]
            best_val_acc = self.population_val_accuracies[epoch][best_individual_index]

            self.best_test_accuracies.append(best_test_acc)
            self.best_test_losses.append(best_test_loss)
            self.best_val_accuracies.append(best_val_acc)
            self.best_val_losses.append(best_val_loss)

            print(
                f"Epoch {epoch + 1} Best Individual {best_individual_index + 1}: "
                f"Train Loss: {best_train_loss:.4f} | Train Acc: {best_train_acc:.2f}% | "
                f"Test Loss: {best_test_loss:.4f} | Test Acc: {best_test_acc:.2f}% | "
                f"Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_acc:.2f}%")

        results = {
            "best_test_accuracies": self.best_test_accuracies,
            "best_test_losses": self.best_test_losses,
            "population_test_accuracies": self.population_test_accuracies,
            "population_test_losses": self.population_test_losses,
            "population_train_accuracies": self.population_train_accuracies,
            "population_train_losses": self.population_train_losses,
            "population_times": self.population_times,
            "population_val_accuracies": self.population_val_accuracies,
            "population_val_losses": self.population_val_losses,
            "best_val_accuracies": self.best_val_accuracies,
            "best_val_losses": self.best_val_losses
        }
        with open(filename, "wb") as f:
            joblib.dump(results, f)
        print(f"Results saved to {filename}")





# 1. 数据集加载器
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
# val_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
#
# # 2. 损失函数
# criterion = torch.nn.CrossEntropyLoss()
#
# # 3. 模型种群
# population_size = 5
# population = [LeNet5(10) for _ in range(population_size)]
#
# # 4. 优化器列表
# optimizers = [optim.SGD(model.parameters(), lr=0.001) for model in population]
#
# # 5. 超参数
# num_epoch = 50
# learning_rate = 0.001 #优化器列表的lr应该与此处相同
#
# msgd = GD(num_epoch, criterion, train_loader, test_loader, learning_rate, population_size)
#
# msgd.MSGD(optimizers, population)
#
#
# class MSGD:
#     def __init__(self, num_epoch,criterion, train_loader, test_loader,
#                  learning_rate, population_size):
#         # self.model = model
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = device
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.criterion = criterion
#         self.num_epoch = num_epoch
#         self.learning_rate = learning_rate
#         self.population_size = population_size
#
#     # def train(self, data_loader):
#     #     self.model.train()
#     #     train_loss = 0
#     #     correct = 0
#     #     total = 0
#     #     for data, target in data_loader:
#     #         data, target = data.to(self.device), target.to(self.device)
#     #         self.optimizer.zero_grad()
#     #         output = self.model(data)
#     #         loss = self.criterion(output, target)
#     #         loss.backward()
#     #         self.optimizer.step()
#     #         train_loss += loss.item()
#     #         _, predicted = output.max(1)
#     #         total += target.size(0)
#     #         correct += predicted.eq(target).sum().item()
#     #     accuracy = 100. * correct / total
#     #     avg_loss = train_loss / len(data_loader)
#     #     return avg_loss, accuracy
#
#     def train(self, model, optimizer, data_loader):
#         model.train()
#         train_loss = 0
#         correct = 0
#         total = 0
#         for data, target in data_loader:
#             data, target = data.to(self.device), target.to(self.device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             # optimizer.step()
#             train_loss += loss.item()
#             _, predicted = output.max(1)
#             total += target.size(0)
#             correct += predicted.eq(target).sum().item()
#         accuracy = 100. * correct / total
#         avg_loss = train_loss / len(data_loader)
#         return avg_loss, accuracy
#
#     def evaluate(self, data_loader):
#         self.model.eval()
#         test_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for data, target in data_loader:
#                 data, target = data.to(self.device), target.to(self.device)
#                 output = self.model(data)
#                 loss = self.criterion(output, target)
#                 test_loss += loss.item()
#                 _, predicted = output.max(1)
#                 total += target.size(0)
#                 correct += predicted.eq(target).sum().item()
#         accuracy = 100. * correct / total
#         avg_loss = test_loss / len(data_loader)
#         return avg_loss, accuracy
#
#     def MSGD(self, optimizer, population):
#
#         train_losses = [[] for _ in range(self.population_size + 3)]
#         train_accuracies = [[] for _ in range(self.population_size + 3)]
#         test_losses = [[] for _ in range(self.population_size + 3)]
#         test_accuracies = [[] for _ in range(self.population_size + 3)]
#
#         times = {f"Population {i + 1}": [] for i in range(self.population_size)}
#
#         for epoch in range(self.num_epoch):
#             print(f"Epoch [{epoch + 1}/{self.num_epoch}]")
#             losses = []
#             accuracies = []
#             for i, model in enumerate(population):
#                 start_time = time.time()
#                 sgd_train_loss, sgd_train_acc = self.train(model, optimizer, self.train_loader)
#                 sgd_test_loss, sgd_test_acc = test(model, self.test_loader, self.criterion, self.device)
#                 end_time = time.time()
#                 elapsed_time = end_time - start_time
#
#                 times[f"Population {i + 1}"].append(elapsed_time)
#                 losses.append(sgd_train_loss)
#                 accuracies.append(sgd_test_acc)
#                 train_losses[i].append(sgd_train_loss)
#                 train_accuracies[i].append(sgd_train_acc)
#                 test_losses[i].append(sgd_test_loss)
#                 test_accuracies[i].append(sgd_test_acc)
#                 print(
#                     f"Population {i + 1}: Train Loss: {sgd_train_loss:.4f} | Train Acc: {sgd_train_acc:.2f}% | Test Loss: {sgd_test_loss:.4f} | Test Acc: {sgd_test_acc:.2f}%| Time: {elapsed_time:.2f}s")
#
#             best_individual_index = np.argmin(test_accuracies[:self.population_size], axis=0)[epoch]  # 找到每一代中测试集损失最小的个体
#             best_individual = population[best_individual_index]
#             best_test_loss = test_losses[best_individual_index][epoch]
#             best_test_acc = test_accuracies[best_individual_index][epoch]
#             best_train_loss = train_losses[best_individual_index][epoch]
#             best_train_acc = train_accuracies[best_individual_index][epoch]
#
#             print(
#                 f"Epoch {epoch + 1} Best Individual {best_individual_index + 1}: "
#                 f"Train Loss: {best_train_loss:.4f} | Train Acc: {best_train_acc:.2f}% | "
#                 f"Test Loss: {best_test_loss:.4f} | Test Acc: {best_test_acc:.2f}%")
#
#
#
#
# # 1. 数据集加载器
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
# val_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
#
# # 2. 损失函数
# criterion = torch.nn.CrossEntropyLoss()
#
# # 3. 模型种群
# population_size = 5
# # population = [LeNet5(10).cuda() for _ in range(population_size)]
# population = [LeNet5(10) for _ in range(population_size)]
#
# # 4. 优化器列表
# optimizers = [optim.SGD(model.parameters(), lr=0.001) for model in population]
#
# # 5. 超参数
# better_ratio = 0.8
# num_epoch = 10
# learning_rate = 0.5
#
#
# msgd = MSGD(num_epoch, criterion, train_loader, test_loader, learning_rate, population_size)
#
# msgd.MSGD(optimizers, population)