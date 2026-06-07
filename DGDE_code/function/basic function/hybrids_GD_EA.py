import numpy as np
import joblib
import os
import EGD, ESGD, NGDE, DGDE
import PGD
# import EAPSO
import time

"""
运行 MPIS, NGDE, EGD, ESGD, MGD, MPGD, EAPSO 算法并保存结果. 
"""


def run_hybrid_algorithms(args):
    run_id, params = args
    """运行单次实验的函数，用于并行处理"""
    # 解包参数
    objective_function = params['objective_function']
    gradient_function = params['gradient_function']
    x_min = params['x_min']
    x_max = params['x_max']
    figure_save_path = params['figure_save_path']
    dimension = params['dimension']
    learning_rate = params['learning_rate']
    budget = params['budget']
    population_size = params['population_size']
    F = params['F']
    fitness_p = params['fitness_p']
    niche_size = params['niche_size']
    pop_strength = params['pop_strength']
    mutate_strength = params['mutate_strength']
    if_escape_fitness = params['if_escape_fitness']
    gradient_thresh = params['gradient_thresh']
    mutate_thresh = params['mutate_thresh']
    mutate_turns = params['mutate_turns']
    gd_budget = params['gd_budget']
    evo_budget = params['evo_budget']
    # F_D = params['F_D']
    # w = params['w']
    mutate_thresh = params['mutate_thresh']
    mutate_strength = params['mutate_strength']
    gradient_thresh = params['gradient_thresh']
    K_0 = params['K_0']
    K_base = params['K_base']
    T_per = params['T_per']
    pho_high = params['pho_high']
    pho_low = params['pho_low']
    # p_m = params['p_m']





    # 设置随机种子（确保可重复性）
    np.random.seed(run_id)
    print(f"运行 #{run_id + 1:02d}")

    # ------------------
    print("=====Running DGDE_model======")

    DGDE_model = DGDE.DGDE(population_size=population_size, dimension=dimension, niche_size=niche_size,
                           budget=budget, F=F, K_0=K_0, K_base=K_base, T_per=T_per, pho_high=pho_high,
                           pho_low=pho_low, fitness_p=fitness_p, learning_rate=learning_rate)
    start_time = time.time()
    DGDE_model.DGDE(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    DGDE_model.run_time = end_time - start_time

    results_DGDE = {
        'run_id': run_id,
        'seed': run_id,  # 随机种子
        'run_time': DGDE_model.run_time,  # 运行时间
        # 每一代的最优适应值
        'best_fitness': DGDE_model.best_history,
        'best_solution': DGDE_model.g_best.target.fitness,
        # 参数信息
        'parameters': {
            'population_size': population_size,
            'dimension': dimension,
            'budget': budget,
            'learning_rate': learning_rate,
            # 'F_D': F_D,
            # 'w': w,
            'x_min': x_min,
            'x_max': x_max
        }
    }
    dgde_filename = os.path.join(figure_save_path, f'DGDE_model_run_{run_id}.pkl')
    with open(dgde_filename, 'wb') as f:
        joblib.dump(results_DGDE, f)

    print(f"Best Fitness: {DGDE_model.g_best.target.fitness}")
    fitness_values = [agent.target.fitness for agent in DGDE_model.pop]
    print(f"Mean Fitness: {np.mean(fitness_values)}")
    print(f"Variance of Fitness: {np.var(fitness_values)}")

    ## ------------------
    print("=====Running NGDE_model======")

    NGDE_model = NGDE.NGDE(population_size=population_size, dimension=dimension, niche_size=niche_size,
                           budget=budget, F=F, fitness_p=fitness_p, learning_rate=learning_rate)
    start_time = time.time()
    NGDE_model.NGDE(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    NGDE_model.run_time = end_time - start_time

    results_NGDE = {
        'run_id': run_id,
        'seed': run_id,  # 随机种子
        'run_time': NGDE_model.run_time,  # 运行时间
        # 每一代的最优适应值
        'best_fitness': NGDE_model.best_history,
        'best_solution': NGDE_model.g_best.target.fitness,
        # 参数信息
        'parameters': {
            'population_size': population_size,
            'dimension': dimension,
            'budget': budget,
            'learning_rate': learning_rate,
            # 'F_D': F_D,
            # 'w': w,
            'x_min': x_min,
            'x_max': x_max
        }
    }
    ngde_filename = os.path.join(figure_save_path, f'NGDE_model_run_{run_id}.pkl')
    with open(ngde_filename, 'wb') as f:
        joblib.dump(results_NGDE, f)

    print(f"Best Fitness: {NGDE_model.g_best.target.fitness}")

    ### ------------------
    print("=====Running EGD_model======")

    EGD_model = EGD.EGD(population_size=population_size, dimension=dimension, pop_strength=pop_strength,
                        mutate_strength=mutate_strength, if_escape_fitness=if_escape_fitness,
                        budget=budget, gradient_thresh=gradient_thresh, mutate_thresh=mutate_thresh,
                        learning_rate=learning_rate, mutate_turns=mutate_turns)
    start_time = time.time()
    EGD_model.EGD(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    EGD_model.run_time = end_time - start_time

    results_EGD = {
        'run_id': run_id,
        'seed': run_id,  # 随机种子
        'run_time': EGD_model.run_time,  # 运行时间
        # 每一代的最优适应值
        'best_fitness': EGD_model.best_history,
        'best_solution': EGD_model.g_best.target.fitness,
        # 参数信息
        'parameters': {
            'population_size': population_size,
            'dimension': dimension,
            'budget': budget,
            'learning_rate': learning_rate,
            # 'F_D': F_D,
            # 'w': w,
            'x_min': x_min,
            'x_max': x_max
        }
    }
    egd_filename = os.path.join(figure_save_path, f'EGD_model_run_{run_id}.pkl')
    with open(egd_filename, 'wb') as f:
        joblib.dump(results_EGD, f)

    print(f"Best Fitness: {EGD_model.g_best.target.fitness}")

    # # ------------------
    print("=====Running ESGD_model======")

    ESGD_model = ESGD.ESGD(population_size=population_size, mutate_strength=mutate_strength,
                           learning_rate=learning_rate, budget=budget,
                           gd_budget=gd_budget, evo_budget=evo_budget, dimension=dimension)
    start_time = time.time()
    ESGD_model.ESGD(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    ESGD_model.run_time = end_time - start_time

    results_ESGD = {
        'run_id': run_id,
        'seed': run_id,  # 随机种子
        'run_time': ESGD_model.run_time,  # 运行时间
        # 每一代的最优适应值
        'best_fitness': ESGD_model.best_history,
        'best_solution': ESGD_model.g_best.target.fitness,
        # 参数信息
        'parameters': {
            'population_size': population_size,
            'dimension': dimension,
            'budget': budget,
            'learning_rate': learning_rate,
            # 'F_D': F_D,
            # 'w': w,
            'x_min': x_min,
            'x_max': x_max
        }
    }

    esgd_filename = os.path.join(figure_save_path, f'ESGD_model_run_{run_id}.pkl')
    with open(esgd_filename, 'wb') as f:
        joblib.dump(results_ESGD, f)

    print(f"Best Fitness: {ESGD_model.g_best.target.fitness}")

    # ------------------------------- gradient_algorithms 2个
    # MGD,MPGD
    # # ------------------
    print("=====Running MGD_model======")
    MGD_model = PGD.MGD(learning_rate=learning_rate, budget=budget, population_size=population_size,
                        dimension=dimension)
    start_time = time.time()
    MGD_model.MGD(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    MGD_model.run_time = end_time - start_time
    # 保存结果
    results_MGD_model = {
        'run_id': run_id,
        'seed': run_id,  # 随机种子
        'run_time': MGD_model.run_time,  # 运行时间
        # 每一代的最优适应值
        'best_fitness': MGD_model.best_history,  # 100轮的每一轮
        'best_solution': MGD_model.g_best.target.fitness,  # 第100轮
        # 参数信息
        'parameters': {
            'population_size': population_size,
            'dimension': dimension,
            'budget': budget,
            'learning_rate': learning_rate,
            # 'F_D': F_D,
            # 'w': w,
            'x_min': x_min,
            'x_max': x_max
        }
    }
    MGD_model_filename = os.path.join(figure_save_path, f'MGD_model_run_{run_id}.pkl')
    with open(MGD_model_filename, 'wb') as f:
        joblib.dump(results_MGD_model, f)

    print(f"Best Fitness: {MGD_model.g_best.target.fitness}")

    # ##------------------
    print("=====Running MPGD_model======")
    MPGD_model = PGD.MPGD(learning_rate=learning_rate, budget=budget, population_size=population_size,
                          dimension=dimension,
                          mutate_thresh=mutate_thresh, mutate_strength=mutate_strength, gradient_thresh=gradient_thresh)
    start_time = time.time()
    MPGD_model.MPGD(x_min, x_max, objective_function, gradient_function)
    end_time = time.time()
    MPGD_model.run_time = end_time - start_time  # 添加运行时间

    # 保存结果
    results_MPGD_model = {
        'run_id': run_id,
        'seed': run_id,  # 随机种子
        'run_time': MPGD_model.run_time,  # 运行时间
        # 每一代的最优适应值
        'best_fitness': MPGD_model.best_history,  # 100轮的每一轮
        'best_solution': MPGD_model.g_best.target.fitness,  # 第100轮
        # 参数信息
        'parameters': {
            'population_size': population_size,
            'dimension': dimension,
            'budget': budget,
            'learning_rate': learning_rate,
            # 'F_D': F_D,
            # 'w': w,
            'x_min': x_min,
            'x_max': x_max
        }
    }
    MPGD_model_filename = os.path.join(figure_save_path, f'MPGD_model_run_{run_id}.pkl')
    with open(MPGD_model_filename, 'wb') as f:
        joblib.dump(results_MPGD_model, f)

    print(f"Best Fitness: {MPGD_model.g_best.target.fitness}")
