import numpy as np
import joblib
import os
import PGD
import time


def run_gd_algorithms(objective_function, gradient_function, x_min, x_max, figure_save_path, dimension, learning_rate, budget,
                       population_size, mutate_thresh, mutate_strength, gradient_thresh):
    """
    运行 MGD 和 MPGD 算法并保存结果。

    Args:
        function_name (str): 目标函数名称。
        figure_save_path (str): 保存结果的路径。
        dimension (int): 问题维度。
        learning_rate (float): 学习率。
        budget (int): 预算。
        population_size (int): 种群大小。
        mutate_thresh (int): 变异阈值。
        mutate_strength (float): 变异强度。
        gradient_thresh (float): 梯度阈值。
    """

    # f_shift = np.random.uniform(-100, 100, dimension)
    # f_matrix = np.random.rand(dimension, dimension)
    # q, r = np.linalg.qr(f_matrix)
    # f_matrix = q
    # f_bias = 300.0
    #
    # objective_function, gradient_function, x_min, x_max = select_function(function_name)

    # ------------------
    print("=====Running MGD_model======")
    MGD_model = PGD.MGD(learning_rate=learning_rate, budget=budget, population_size=population_size, dimension=dimension)
    start_time = time.time()
    MGD_model.MGD(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    MGD_model.run_time = end_time - start_time # 添加运行时间

    with open(figure_save_path + 'MGD_model.pkl', 'wb') as f:
        joblib.dump(MGD_model, f)

    print(f"Best Fitness: {MGD_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in MGD_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    # ------------------
    print("=====Running MPGD_model======")
    MPGD_model = PGD.MPGD(learning_rate=learning_rate, budget=budget, population_size=population_size, dimension=dimension,
                         mutate_thresh=mutate_thresh, mutate_strength=mutate_strength, gradient_thresh=gradient_thresh)
    start_time = time.time()
    MPGD_model.MPGD(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    MPGD_model.run_time = end_time - start_time # 添加运行时间

    with open(figure_save_path + 'MPGD_model.pkl', 'wb') as f:
        joblib.dump(MPGD_model, f)

    print(f"Best Fitness: {MPGD_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in MPGD_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")