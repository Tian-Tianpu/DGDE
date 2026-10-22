import numpy as np
import joblib
import os
import DE, SHADE, GA
import ES
from PSOS import *
# import EAPSO
import time
from mealpy import FloatVar

"""
OriginalDE_model, JADE_model, SADE_model, SAP_DE_model, OriginalSHADE_model, L_SHADE_model, BaseGA_model

"""

def run_others_algorithms(args):
    run_id, params = args
    #运行单次实验的函数，用于并行处理
    """
      Args:
          objective_function (callable): The objective function to minimize.
          x_min (float or array-like): Lower bound of the search space.
          x_max (float or array-like): Upper bound of the search space.
          figure_save_path (str): Directory to save results.
          dimension (int): Dimensionality of the problem.
          epoch (int): Number of iterations.
          pop_size (int): Population size.
    # DE/ES/GA
          wf (float): Weight factor (for DE/ES/GA).
          cr (float): Crossover probability (for DE/ES/GA).
          strategy (int): DE/ES/GA strategy identifier.
          jade_pt (float): JADE-specific pt parameter.
          jade_ap (float): JADE-specific ap parameter.
          verbose (bool): Whether to print progress and results.
    # MGD/MPGD
          mutate_thresh (int): 变异阈值
          mutate_strength (float): 变异强度
          gradient_thresh (float): 梯度阈值
    """
    #参数
    objective_function = params['objective_function']
    x_min = params['x_min']
    x_max = params['x_max']
    figure_save_path = params['figure_save_path']
    dimension = params['dimension']
    learning_rate = params['learning_rate']
    budget = params['budget']
    population_size  = params['population_size']
    # evolutionary_algorithms
    wf = params['wf']
    cr = params['cr']
    strategy_DE = params['strategy_DE']
    jade_pt = params['jade_pt']
    jade_ap = params['jade_ap']
    # evolutionary_strategy_algorithms
    strategy_ES = params['strategy_ES']
    verbose = params['verbose']
    c1 = params['c1']
    c2 = params['c2']

    # Seed
    np.random.seed(run_id)
    print(f"运行 #{run_id + 1:02d}")

    os.makedirs(figure_save_path, exist_ok=True)
    # Define problem
    problem_dict = {
        "bounds": FloatVar(lb=(int(x_min),) * int(dimension), ub=(int(x_max),) * int(dimension), name="delta"),
        "minmax": "min",
        "obj_func": objective_function
    }

    # Helper function to run a model
    def run_and_save_model(model_class, model_name, **kwargs):
        if verbose:
            print(f"=====Running {model_name}======")
        model = model_class(**kwargs)
        start_time = time.time()
        model.solve(problem_dict)
        end_time = time.time()
        model.run_time = end_time - start_time
        # Save model
        results = {
            'run_id': run_id,
            'seed': run_id,  # 随机种子
            'run_time': model.run_time,  # 运行时间
            # 每一代的最优适应值
            'best_fitness': model.best_history,  # 100轮的每一轮
            'best_solution': model.g_best.target.fitness,  # 第100轮
            # 参数信息
            'parameters': {
                'population_size': population_size,
                'dimension': dimension,
                'budget': budget,
                'wf': wf,
                'cr': cr,
                'jade_pt': jade_pt,
                'jade_ap': jade_ap
            }
        }
        file_path = os.path.join(figure_save_path, f"{model_name}_model_run_{run_id}.pkl")
        with open(file_path, 'wb') as f:
            joblib.dump(results, f)
        print(f"Best Fitness: {model.g_best.target.fitness}")
        # return model

    # Dictionary to store all models
    models = {}

    #------------------------------evolutionary_algorithms 7个
    #OriginalDE_model, JADE_model, SADE_model, SAP_DE_model, OriginalSHADE_model, L_SHADE_model, BaseGA_model
    # ----------------------------------------------------------
    models["OriginalDE"] = run_and_save_model(DE.OriginalDE, "DE", epoch=budget, pop_size=population_size,
                                              wf=wf, cr=cr, strategy=strategy_DE)
    models["JADE"] = run_and_save_model(DE.JADE, "JADE", epoch=budget, pop_size=population_size,
                                              miu_f=wf, miu_cr=cr, pt=jade_pt, ap=jade_ap)
    models["SADE"] = run_and_save_model(DE.SADE, "SADE", epoch=budget,
                                        pop_size=population_size)
    models["SAP_DE"] = run_and_save_model(DE.SAP_DE, "SAP_DE", epoch=budget,
                                          pop_size=population_size)
    models["OriginalSHADE"] = run_and_save_model(SHADE.OriginalSHADE, "SHADE", epoch=budget, pop_size=population_size,
                                              wf=wf, cr=cr, strategy=strategy_DE)
    models["L_SHADE"] = run_and_save_model(SHADE.L_SHADE, "L_SHADE", epoch=budget, pop_size=population_size,
                                             miu_f=wf, miu_cr=cr)
    models["BaseGA"] = run_and_save_model(GA.BaseGA, "GA", epoch=budget, pop_size=population_size,
                                              wf=wf, cr=cr, strategy=strategy_DE)

    #------------------------------- evolutionary_strategy_algorithms 5个
    # OriginalES, LevyES, CMA_ES, MAP_CMA_ES,LRA_CMA_ES
    # --------------------------------------------
    # models["OriginalES"] = run_and_save_model(ES.OriginalES, "OriginalES",epoch=budget, pop_size=population_size,
    #                                           wf=wf, cr=cr, strategy=strategy_ES)
    # models["LevyES"] = run_and_save_model(ES.LevyES, "LevyES", epoch=budget, pop_size=population_size,
    #                                       wf=wf, cr=cr, strategy=strategy_ES)
    # models["CMA_ES"] = run_and_save_model(ES.CMA_ES, "CMA_ES",epoch=budget, pop_size=population_size,
    #                                       wf=wf, cr=cr, strategy=strategy_ES)
    # models["MAP_CMA_ES"] = run_and_save_model(ES.MAP_CMA_ES, "MAP_CMA_ES",epoch=budget, pop_size=population_size,
    #                                           wf=wf, cr=cr, strategy=strategy_ES)
    # models["LRA_CMA_ES"] = run_and_save_model(ES.LRA_CMA_ES, "LRA_CMA_ES", epoch=budget, pop_size=population_size,
    #                                           wf=wf, cr=cr, strategy=strategy_ES)

    # --------------------------------------PSO_variant_algorithms 6个
    # OriginalPSO_model, LDW_PSO_model, P_PSO_model, HPSO_TVAC_PSO_model, C_PSO_model, CL_PSO_model, EAPSO_model
    # --------------------------------------------
    models["OriginalPSO"] = run_and_save_model(OriginalPSO, "PSO", epoch=budget,
                                               pop_size=population_size, c1=c1, c2=c2, w=0.4)
    models["LDW_PSO"] = run_and_save_model(LDW_PSO, "LDW_PSO", epoch=budget,
                                               pop_size=population_size, c1=c1, c2=c2, w_min=0.4, w_max=0.9)
    models["P_PSO"] = run_and_save_model(P_PSO, "P_PSO", epoch=budget,
                                               pop_size=population_size)
    models["HPSO_TVAC_PSO"] = run_and_save_model(HPSO_TVAC, "HPSO_TVAC_PSO", epoch=budget,
                                               pop_size=population_size, ci=0.5, cf=0.1)
    models["C_PSO"] = run_and_save_model(C_PSO, "C_PSO", epoch=budget,
                                               pop_size=population_size, c_local=c1, w_min=0.4, w_max=0.9, max_flag=7)
    models["CL_PSO"] = run_and_save_model(CL_PSO, "CL_PSO", epoch=budget,
                                               pop_size=population_size, c1=c1, c2=c2, w=0.4)




