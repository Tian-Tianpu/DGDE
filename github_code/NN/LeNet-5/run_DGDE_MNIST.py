import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import time
import os
from ACC_LOSS import *
from datasets import *
from models import *
from DGDE import DGDE

# ================= 1. 实验超参数配置 =================
batch_size = 128
num_epoch = 100  # 对应论文中的 K [cite: 155]
population_size = 5  # 种群大小 μ [cite: 129]
lambda_reg = 0  # L1 正则化系数 λ [cite: 316, 501]
niche_size = 2  # 小生境大小 [cite: 163]
T_per = 1  # 探索集更新周期

# DGDE 算法特有参数 [cite: 308, 309]
F = 0.5  # 变异因子
K_base = 0.5  # 基础引导因子 (update2)
K_0 = 1.1  # 动态引导初始值 (update3)
pho_high = 0.8  # 高适应度采样比例 [cite: 481]
pho_low = 0.8  # 低适应度采样比例
# FP = [0.1, 0.5]  # 适应度阈值 pf [cite: 252]

# SGD 专用参数：通常 LR 设为 0.01 或 0.1 效果较好
# LR =  [0.1]
#[0.0001, 0.001, 0.01]

LRs=[ .001, .0001, .01]
FP=[.1, .3, .5]
K_B=[.3, .5, .7]
K_0s=[1.01, 1.03, 1.05, 1.07]


model = "lenet5"
Data = ["MNIST"]
Sub_Size = [5000]
# K_0s = [1.3, 1.5, 1.7]
# KB = [0.001, 0.01, 0.1, 0.3, 0.5]
# Fs = [0.001, 0.01, 0.1, 0.3, 0.5]

# ================= 2. 实验主循环 =================
results_list = []

for learning_rate  in LRs:
    for fitness_p in FP:
        for dataname in Data:
            for subset_size in Sub_Size:
                for K_0 in K_0s:
                    for K_base in K_B:
                        # 数据加载
                        (train_dataset, val_dataset, test_dataset,
                         trainloader, valloader, testloader, n) = choose_dataset_subset(dataname, batch_size,
                                                                                        subset_size)

                        criterion = nn.CrossEntropyLoss()
                        figure_save_path = f"./new_Results_DGDE/{dataname}_{subset_size}/"
                        if not os.path.exists(figure_save_path):
                            os.makedirs(figure_save_path)

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        print(f"Using device: {device}")

                        # ======================DGDE=========================
                        print("=======Running DGDE=======")

                        print(f"\n[Running] DGDE  | LR: {learning_rate} | Dataset: {dataname} | K_0: {K_0}")

                        # 初始化种群
                        population = [create_grayscale_model(model, n).to(device) for _ in range(population_size)]

                        # 关键改进：使用带动量的 SGD 优化器组 [cite: 80]
                        # optimizers = [optim.SGD(model.to(device).parameters(), lr=learning_rate, weight_decay=1e-4) for
                        #               model in population]
                        # optimizers = [optim.Adam(model.to(device).parameters(), lr=learning_rate, weight_decay=1e-4) for
                        #               model in
                        #               population]

                        optimizers = [
                            torch.optim.Adam(
                                model.parameters(),
                                lr=learning_rate,
                                betas=(0.9, 0.999),
                                eps=1e-8,
                                weight_decay=0,
                                amsgrad=False,
                                maximize=False,
                            )
                            for model in population
                        ]
                        # filename = save_path + f"results_LR{learning_rate}_DGDE_SGD.pkl"
                        filename = figure_save_path + f"results_lr{learning_rate}_K0{K_0}_f_p{fitness_p}_KB_{K_base}_DGDE.pkl"

                        # 实例化 DGDE
                        dgde = DGDE(trainloader, testloader, valloader, criterion, lambda_reg,
                                    population_size, niche_size, T_per, num_epoch,
                                    F, K_base, K_0, pho_high, pho_low, fitness_p)

                        # 执行算法
                        start_time = time.time()
                        dgde.DGDE(optimizers, population, filename)
                        end_time = time.time()

                        # 记录结果
                        res = joblib.load(filename)
                        results_list.append({
                            "algorithm": "DGDE",
                            "learning_rate": learning_rate,
                            "F": F,
                            "fitness_p": fitness_p,
                            "K_base": K_base,
                            "K_0": K_0,
                            "best_test_acc": res["best_test_accs"][-1],
                            "best_valid_loss": res["best_valid_losses"][-1],
                            "best_valid_accs": res["best_valid_accs"][-1],
                            "best_valid_losses": res["best_valid_losses"][-1],
                            # "experiment_time": experiment_time  # 监测稀疏化 [cite: 497, 499]
                            "time": end_time - start_time
                        })







# 保存汇总数据
# pd.DataFrame(results_list).to_csv("./Results_DGDE_SGD/summary_sgd.csv", index=False)

df = pd.DataFrame(results_list)
csv_filename = "./new_Results_DGDE/subset_parameter_results.csv"
df.to_csv(csv_filename, index=False)
print(f"Experiment results saved to {csv_filename}")