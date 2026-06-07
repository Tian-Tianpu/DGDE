'''
NDGDE
'''

import numpy as np
import random
import joblib
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from mealpy.utils.target import Target
from all_function import select_function

def get_fitness(x, f):
    fitness = f(x)
    return fitness

def get_gradient(x, f_gradient):
    grad = f_gradient(x)
    return grad

class DGDE(Optimizer):
    def __init__(self, population_size, dimension, niche_size, T_per, budget, F, K_base, K_0, pho_high, pho_low, fitness_p,
                 learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.niche_size = niche_size
        self.dimension = dimension
        self.budget = budget  # 最大迭代次数
        self.F = F
        self.K_base = K_base
        self.K_0 = K_0
        self.T_per = T_per
        self.pho_high = pho_high
        self.pho_low = pho_low
        # self.p_m = p_m
        self.M = int(self.population_size / 2)  # 未特别使用
        self.fitness_p = fitness_p  # 适应度概率阈值（决定使用哪种更新策略）
        self.learning_rate = learning_rate  # 学习率
        self.t = 0
        self.population_history = []
        self.sort_flag = False
        self.population_all = []
        self.best_history = []


    def initial(self, x_min, x_max):
        population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension]) #初始化种群（在定义域内均匀分布）
        return population

    def Fitness(self, population, f):
        fitness = [get_fitness(population[p], f) for p in range(len(population))] #使用get_fitness批量计算种群的适应度
        return fitness #返回适应度列表

    def Gradient(self, x, f_gradient): #执行一次梯度下降操作；并返回更新后的x
        grad = get_gradient(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def is_domin(self, x, x_min, x_max): #边界修复函数，把越界元素拉回边界最近的一侧
        for i in range(len(x)):
            if x[i] < x_min or x[i] > x_max:
                # 注意：这里把标量与向量比较，要确保 x_min / x_max 是与 x 元素同尺度（你的原实现亦如此）
                if self.p_distance(x[i], x_min) < self.p_distance(x[i], x_max):
                    x[i] = x_min
                else:
                    x[i] = x_max
        return x

    def fitness_sort(self, population, f): #按适应度排序种群，返回：最佳个体；所有适应度；排序后的（个体、适应度）列表
        Fitness = self.Fitness(population, f)
        sort = sorted(zip(population, Fitness), key=lambda x: x[1], reverse=False)
        best_p = sort[0][0]
        return best_p, Fitness, sort

    def p_distance(self, x, y):#计算两个向量的平方和距离（L2范数的平方）；返回：距离值；
        dist = (np.square(np.sum((np.array(x) - np.array(y))))) ** 0.5
        return dist

    def distance_sort(self, x, population):
        Distance = np.zeros(len(population))
        for i in range(len(population)):
            Distance[i] = self.p_distance(x, population[i])
        sort = sorted(zip(range(len(population)), Distance), key=lambda x: x[1])
        return sort  # [(index_in_list, distance), ...]

    def get_niche(self, population, f):
        """
        返回 Niche：每个 niche 是原始 population 的索引列表（全局索引）。
        逻辑：在 remaining_indices 中反复选取当前最优个体（适应度最小），并把与其最近的 self.niche_size 个体
        组成一个小生境；然后从 remaining_indices 中删除这些索引，继续划分，直到无法再划分完整的 niche。
        剩余不足一个完整 niche 的索引也作为最后一个 niche 返回。
        """
        Niche = []
        # population 可能是 numpy.ndarray 或 list，统一按索引访问
        remaining_indices = list(range(len(population)))

        while len(remaining_indices) >= self.niche_size:
            # 在剩余个体中找到适应度最优的索引（全局索引）
            best_idx = min(remaining_indices, key=lambda idx: get_fitness(population[idx], f))
            best_p = population[best_idx]

            # 构造剩余个体的局部列表（与 remaining_indices 一一对应）
            local_pop = [population[idx] for idx in remaining_indices]
            # 计算 best_p 到 local_pop 的距离排序（返回 local 索引）
            sort = self.distance_sort(best_p, local_pop)  # [(local_index, dist), ...]

            # 取最近的 self.niche_size 个（local 索引），转换回全局索引
            niche_local_idxs = [sort[k][0] for k in range(self.niche_size)]
            niche_global = [remaining_indices[li] for li in niche_local_idxs]

            Niche.append(niche_global)

            # 从 remaining_indices 中删除这些全局索引
            remaining_indices = [idx for idx in remaining_indices if idx not in niche_global]

        # 若剩余不足一个完整小生境，也把它作为最后一个 niche（全局索引列表）
        if remaining_indices:
            Niche.append(remaining_indices)

        return Niche

    def update1(self, x, f_gradient):
        grad = self.Gradient(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def update2(self, x, best_x, f_gradient):
        N = len(x)
        x[0:N:1] = x[0:N:1] + self.K_base * (best_x[0:N:1]-x[0:N:1])
        grad = self.Gradient(x, f_gradient)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def update3(self, x, best_x, population, f_gradient):
        # population 是整个种群（list of arrays）
        k = np.random.randint(0, len(population), 2)
        N = len(x)
        x[0:N:1] = x[0:N:1] + (self.K_0 + (self.t / self.budget) ** 0.5 ) * (best_x[0:N:1]-x[0:N:1]) + self.F * (population[k[0]][0:N:1] - population[k[1]][0:N:1])
        grad = self.Gradient(x, f_gradient)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def build_explore_set(self, niche, population, Fitness, f):
        """
        输入:
            niche: 当前小生境的**全局索引列表**（例如 [3, 7, 9, ...]）
            population: 全局种群（list 或 ndarray）
            Fitness: 对应 sub_population 的适应度列表（顺序与 niche 一致）
            f: 目标函数
        返回:
            E_i: 选中的全局索引列表（用于后续 t 到 t+T_per-1 的两类更新策略分配）
        选择规则（按你要求的改动）:
            - 在 p > fitness_p 的候选中，随机选取 pho_high%（向下取整，若结果为0且候选非空则至少选1）
            - 在 p <= fitness_p 的候选中，随机选取 (1 - pho_low)%（同样保证不越界）
        """
        E_i = []

        if len(niche) == 0:
            return E_i

        # niche_individuals 保证与 Fitness 顺序一致
        niche_individuals = [population[idx] for idx in niche]

        # 计算 p 值（归一化适应度）——这里使用传入的 Fitness（对应 sub_population）
        p_values = [self.prob(ind, Fitness, f) for ind in niche_individuals]

        # 高适应度候选（p > fitness_p）：在这些中随机抽取 pho_high 比例
        high_positions = [i for i, p in enumerate(p_values) if p >= self.fitness_p]
        if len(high_positions) > 0:
            k_high = int(len(high_positions) * (1 - self.pho_high))
            # if k_high <= 0:
            #     k_high = 1  # 保证至少选择 1 个（当 high_positions 非空时）
            # if k_high > len(high_positions):
            #     k_high = len(high_positions)
            selected_high_pos = list(np.random.choice(high_positions, size=k_high, replace=False))
            E_i.extend([niche[pos] for pos in selected_high_pos])

        # 低适应度候选（p <= fitness_p）：在这些中随机抽取 (1 - pho_low) 比例
        low_positions = [i for i, p in enumerate(p_values) if p < self.fitness_p]
        if len(low_positions) > 0:
            k_low = int(len(low_positions) * self.pho_low)
            if k_low <= 0:
                k_low = 1
            if k_low > len(low_positions):
                k_low = len(low_positions)
            selected_low_pos = list(np.random.choice(low_positions, size=k_low, replace=False))
            E_i.extend([niche[pos] for pos in selected_low_pos])

        # 去重并保持为列表
        E_i = list(dict.fromkeys(E_i))

        return E_i

    def prob(self, x, Fitness, f):
        """
        计算个体 x 在该子种群（对应 Fitness）内的归一化适应度概率
        p = (f(x) - min(Fitness)) / (max(Fitness)-min(Fitness)+eps)
        注意：Fitness 应该是与 x 所在子种群顺序一致的 fitness 列表
        """
        eps = 1e-8
        p = (f(x) - min(Fitness)) / (max(Fitness) - min(Fitness) + eps)
        return p

    def update(self, x_idx, x, best_x, population, f_gradient, x_min, x_max, explore_set):
        # 兼容 x_idx 为全局索引（int）
        if self.p_distance(x, best_x) == 0:
            x = self.is_domin(self.update1(x, f_gradient), x_min, x_max)
        elif x_idx in explore_set:
            x = self.is_domin(self.update3(x, best_x, population, f_gradient), x_min, x_max)
        else:
            x = self.is_domin(self.update2(x, best_x,  f_gradient), x_min, x_max)
        return x

    def generate_target(self, fitness):
        return Target(fitness)

    def DGDE(self, x_min, x_max, f, f_gradient):
        # 初始化
        population = self.initial(x_min, x_max)
        fitness = self.Fitness(population, f)
        Fitness_all = [fitness]
        Population_all = [population]
        history = [np.min(fitness)]
        best_history = [np.min(fitness)]
        population = list(population)  # 转为 list 以便后续索引/替换
        niche_history = []
        niche_info_history = []

        # 初始划分 niche：返回每个 niche 的全局索引列表
        Niche = self.get_niche(population, f)

        while self.t < self.budget:
            offspring_population_L = []
            niche_info = []

            if self.t % self.T_per == 0:
                explore_set = []
                # 对每个 niche 计算 sub_population / sub_fitness，然后从该 niche 中抽取 explore 集合索引
                for niche in Niche:
                    # niche 是全局索引列表
                    sub_population = [population[i] for i in niche]
                    best_x, Fitness_sub, Fitness_sort = self.fitness_sort(sub_population, f)
                    # build_explore_set 要求 (niche_indices, population, Fitness_sub, f)
                    E_i = self.build_explore_set(niche, population, Fitness_sub, f)
                    explore_set.extend(E_i)

            # 对所有个体做一次更新（使用 explore_set 判定是否用 update2）
            for niche in Niche:
                sub_population = [population[i] for i in niche]
                best_x, Fitness_sub, Fitness_sort = self.fitness_sort(sub_population, f)
                for x_idx in niche:  # x_idx 是全局索引
                    x = population[x_idx]
                    x1 = self.update(x_idx, x, best_x, population, f_gradient, x_min, x_max, explore_set)
                    offspring_population_L.append(np.array(x1))

            # 更新 population
            population = offspring_population_L
            self.t += 1

            # 记录历史信息
            fitness = self.Fitness(population, f)
            Fitness_all.append(fitness)
            Population_all.append(population)
            history.append(np.min(fitness))
            self.best_history.append(min(np.min(fitness), best_history[-1]))
            # 保证 population 长度等于 population_size（假设算法不改变大小）
            self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])
            self.population_all.append(population)

            niche_history.append(len(Niche))
            niche_info_history.append(niche_info)

        # 构造 Agent 列表
        self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in range(self.population_size)]
        self.g_best = self.get_sorted_population(self.pop, "min")[0]
        self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])

        self.niche_history = niche_history
        self.niche_info_history = niche_info_history
