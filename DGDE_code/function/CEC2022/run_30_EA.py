from others_algorithms import run_others_algorithms
import os
from all_function_new import select_function
from multiprocessing import Pool

budget = 100
population_size = 20
# EAs
wf = 0.5
cr = 0.5
strategy_DE = 0
# JADE
jade_pt, jade_ap = 0.1, 0.1
# PGD
mutate_thresh = 10
mutate_strength = 1.0
gradient_thresh = 0.1
# EGD
pop_strength = 0.5
if_escape_fitness = 0.1
mutate_turns = 5
# ESGD
gd_budget = 10
evo_budget = 10
# NGDE
F = 0.5
niche_size = 5
fitness_p = 0.5
# PSOS
c1 = 0.5
c2 = 0.5
#ES
strategy_ES = 1
verbose=True
# MPIS
# F_D = 0.5
Dimension = [1000]
learning_rate = 0.1
F = 0.5
w = 0.5

# FUNS = [3]
def main_parallel():
    params = {
        'budget': budget,
        'population_size': population_size,
        # EAs
        'wf': wf,
        'cr': cr,
        'strategy_DE': strategy_DE,
        'strategy_ES': strategy_ES,
        # JADE
        'jade_pt': jade_pt,
        'jade_ap': jade_ap,
        # PGD
        'mutate_thresh': mutate_thresh,
        'mutate_strength': mutate_strength,
        'gradient_thresh': gradient_thresh,
        # EGD
        'pop_strength': pop_strength,
        'if_escape_fitness': if_escape_fitness,
        'mutate_turns': mutate_turns,
        # ESGD
        'gd_budget': gd_budget,
        'evo_budget': evo_budget,
        # NGDE
        'F': F,
        'niche_size': niche_size,
        'fitness_p': fitness_p,
        # PSOS
        'c1': c1,
        'c2': c2,
        # MPIS
        # 'F_D': F_D,
        'dimension': dimension,
        'learning_rate': learning_rate,
        # 'w': w,
        'objective_function': objective_function,
        'gradient_function': gradient_function,
        'x_min': x_min,
        'x_max': x_max,
        'figure_save_path': figure_save_path,
        'verbose': verbose
    }

    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)

    # 准备参数列表
    num_runs = 1
    tasks = [(i, params) for i in range(num_runs)]

    print(f"{num_runs} run")

    # 使用多进程并行运行
    num_processes = min(4, os.cpu_count() - 1)  # 使用4个进程或CPU核心数-1
    with Pool(processes=num_processes) as pool:
       pool.map(run_others_algorithms, tasks)

#=================================================================
# # for dimension in Dimension:
#     for function_name in FUNS:
#         print("-------------Running-" + str(dimension) + "D-Func" + str(function_name) + "-------------")
#         figure_save_path = ("./results_CEC2022_repeats/"+ "/0lr_0w/"+ "Func_" + str(function_name) + +str(dimension) + "D/"
#                                     )
#         objective_function, gradient_function, x_min, x_max = select_function(function_name, dimension)

#         if __name__ == "__main__":
#             results = main_parallel()
#=================================================================
for dimension in Dimension:
    for function_name in range(1,13):
        print("-------------Running-" + str(dimension) + "D-Func" + str(function_name) + "-------------")
        figure_save_path = ("./results_CEC2022_one/"+ "0lr_0F0/" +  "Func_"+ str(function_name) + "_" +str(dimension) + "D/"
                                    )
        objective_function, gradient_function, x_min, x_max = select_function(function_name, dimension)

        if __name__ == "__main__":
            results = main_parallel()







