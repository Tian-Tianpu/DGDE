import numpy as np
import joblib
import os
from mealpy import FloatVar
import DE, SHADE, GA
import time


def run_evolutionary_algorithms(objective_function, x_min, x_max, figure_save_path, dimension, epoch, pop_size,
                                wf, cr, strategy, jade_pt, jade_ap):
    """
    Args:
        function_name (str): 目标函数名称。
        dimension (int): 问题维度。
        epoch (int): 迭代次数。
        pop_size (int): 种群大小。
        wf (float): 权重因子。
        cr (float): 交叉概率。
        strategy (int): DE 策略。
        jade_pt (float): JADE pt 参数。
        jade_ap (float): JADE ap 参数。
    """

    # f_shift = np.random.uniform(-100, 100, dimension)
    # f_matrix = np.random.rand(dimension, dimension)
    # q, r = np.linalg.qr(f_matrix)
    # f_matrix = q
    # f_bias = 300.0
    #
    # objective_function, gradient_function, x_min, x_max = select_function(function_name)
    #


    problem_dict = {
        "bounds": FloatVar(lb=(int(x_min),) * int(dimension), ub=(int(x_max),) * int(dimension), name="delta"),
        "minmax": "min",
        "obj_func": objective_function
    }

    # ------------------
    print("=====Running OriginalDE_model======")

    OriginalDE_model = DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, strategy=strategy)
    start_time = time.time()
    OriginalDE_model.solve(problem_dict)
    end_time = time.time()
    OriginalDE_model.run_time = end_time - start_time

    with open(figure_save_path + '/DE_model.pkl', 'wb') as f:
        joblib.dump(OriginalDE_model, f)

    print(f"Best Fitness: {OriginalDE_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in OriginalDE_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    # ------------------
    print("=====Running JADE_model======")
    JADE_model = DE.JADE(epoch=epoch, pop_size=pop_size, miu_f=wf, miu_cr=cr, pt=jade_pt, ap=jade_ap)
    start_time = time.time()
    JADE_model.solve(problem_dict)
    end_time = time.time()
    JADE_model.run_time = end_time - start_time

    with open(figure_save_path + '/JADE_model.pkl', 'wb') as f:
        joblib.dump(JADE_model, f)

    print(f"Best Fitness: {JADE_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in JADE_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    # ------------------
    print("=====Running SADE_model======")
    SADE_model = DE.SADE(epoch=epoch, pop_size=pop_size)
    start_time = time.time()
    SADE_model.solve(problem_dict)
    end_time = time.time()
    SADE_model.run_time = end_time - start_time

    with open(figure_save_path + '/SADE_model.pkl', 'wb') as f:
        joblib.dump(SADE_model, f)

    print(f"Best Fitness: {SADE_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in SADE_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    # ------------------
    print("=====Running SAP_DE_model======")
    SAP_DE_model = DE.SAP_DE(epoch=epoch, pop_size=pop_size)
    start_time = time.time()
    SAP_DE_model.solve(problem_dict)
    end_time = time.time()
    SAP_DE_model.run_time = end_time - start_time

    with open(figure_save_path + '/SAP_DE_model.pkl', 'wb') as f:
        joblib.dump(SAP_DE_model, f)

    print(f"Best Fitness: {SAP_DE_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in SAP_DE_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    # ------------------
    print("=====Running OriginalSHADE_model======")
    OriginalSHADE_model = SHADE.OriginalSHADE(epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, strategy=strategy)
    start_time = time.time()
    OriginalSHADE_model.solve(problem_dict)
    end_time = time.time()
    OriginalSHADE_model.run_time = end_time - start_time

    with open(figure_save_path + '/SHADE_model.pkl', 'wb') as f:
        joblib.dump(OriginalSHADE_model, f)

    print(f"Best Fitness: {OriginalSHADE_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in OriginalSHADE_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    # ------------------
    print("=====Running L_SHADE_model======")
    L_SHADE_model = SHADE.L_SHADE(epoch=epoch, pop_size=pop_size, miu_f=wf, miu_cr=cr)
    start_time = time.time()
    L_SHADE_model.solve(problem_dict)
    end_time = time.time()
    L_SHADE_model.run_time = end_time - start_time

    with open(figure_save_path + '/L_SHADE_model.pkl', 'wb') as f:
        joblib.dump(L_SHADE_model, f)

    print(f"Best Fitness: {L_SHADE_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in L_SHADE_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    # ------------------
    print("=====Running BaseGA_model======")
    BaseGA_model = GA.BaseGA(epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, strategy=strategy)
    start_time = time.time()
    BaseGA_model.solve(problem_dict)
    end_time = time.time()
    BaseGA_model.run_time = end_time - start_time

    with open(figure_save_path + '/GA_model.pkl', 'wb') as f:
        joblib.dump(BaseGA_model, f)

    print(f"Best Fitness: {BaseGA_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in BaseGA_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")