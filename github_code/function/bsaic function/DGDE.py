'''
DGDE：对应论文 Algorithm 1、Algorithm 2 和公式 (4)--(9)。
保留原构造参数及 DGDE() 调用方式；T_per 仅为兼容旧脚本保留。
budget 表示代数 T，不是函数评价次数；每代重新独立抽取更新策略。
'''

import numpy as np
from optimizer_support import Optimizer
from optimizer_support import Agent
from optimizer_support import Target


def get_fitness(x, f):
    fitness = f(x)
    return fitness


def get_gradient(x, f_gradient):
    grad = f_gradient(x)
    return grad


class DGDE(Optimizer):
    def __init__(self, population_size, dimension, niche_size, T_per, budget, F, K_base, K_0, pho_high, pho_low, fitness_p,
                 learning_rate, boundary_handling='none', max_nfe=None, max_nge=None, **kwargs):
        super().__init__(**kwargs)
        for name, value, lower in [('population_size', population_size, 1), ('dimension', dimension, 1),
                                   ('niche_size', niche_size, 1), ('budget', budget, 0)]:
            if not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(name + ' must be an integer >= ' + str(lower))
        if not np.isfinite(F) or F <= 0 or not 0 < K_base <= 1 or not np.isfinite(K_0) or K_0 <= 1:
            raise ValueError('Require F > 0, 0 < K_base <= 1 and K_0 > 1')
        if not all(0 <= p <= 1 for p in (pho_high, pho_low, fitness_p)):
            raise ValueError('pho_high, pho_low and fitness_p must be in [0, 1]')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if boundary_handling not in ('none', 'clip'):
            raise ValueError("boundary_handling must be 'none' or 'clip'")
        for name, value in [('max_nfe', max_nfe), ('max_nge', max_nge)]:
            if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(name + ' must be a nonnegative integer or None')
        if max_nfe is not None and max_nfe < population_size:
            raise ValueError('max_nfe must cover initialization of the whole population')
        self.boundary_handling = boundary_handling
        self.max_nfe = max_nfe  # 含初始化的直接目标函数调用上限
        self.max_nge = max_nge  # 梯度回调调用上限，非梯度内部的函数调用数
        self.nfe = 0
        self.nge = 0
        self.population_size = population_size
        self.dimension = dimension
        self.niche_size = niche_size
        self.T_per = T_per  # 论文没有周期性固定策略，故此参数不参与更新
        self.budget = budget
        self.F = F
        self.K_base = K_base
        self.K_0 = K_0
        self.pho_high = pho_high  # rho_high：优质个体采用式 (6) 的概率
        self.pho_low = pho_low  # rho_low：较差个体采用式 (7) 的概率
        self.fitness_p = fitness_p
        self.learning_rate = learning_rate
        self.M = (population_size + niche_size - 1) // niche_size
        self.sort_flag = False
        self.t = 0
        self.population_history = []
        self.population_all = []
        self.best_history = []

    def initial(self, x_min, x_max):
        return np.random.uniform(x_min, x_max, [self.population_size, self.dimension])

    def Fitness(self, population, f):
        return [self._fitness(x, f) for x in population]

    def _fitness(self, x, f):
        if self.max_nfe is not None and self.nfe >= self.max_nfe:
            raise RuntimeError('Objective evaluation budget exhausted')
        self.nfe += 1
        value = float(get_fitness(x, f))
        if not np.isfinite(value):
            raise ValueError('Objective returned a non-finite value')
        return value

    def Gradient(self, x, f_gradient):
        # 仅执行一次梯度校正，不修改传入的当代个体。
        x = np.array(x, dtype=float, copy=True)
        if self.max_nge is not None and self.nge >= self.max_nge:
            raise RuntimeError('Gradient evaluation budget exhausted')
        self.nge += 1
        grad = np.asarray(get_gradient(x, f_gradient), dtype=float)
        if grad.shape != x.shape or not np.all(np.isfinite(grad)):
            raise ValueError('Gradient must be finite and have the same shape as x')
        return x - self.learning_rate * grad

    def is_domin(self, x, x_min, x_max):
        # 显式可选的盒约束扩展，不属于论文式 (5)、(9)。
        return np.clip(x, x_min, x_max)

    def fitness_sort(self, population, f):
        Fitness = self.Fitness(population, f)
        sort = sorted(zip(population, Fitness), key=lambda x: x[1])
        return sort[0][0], Fitness, sort

    def p_distance(self, x, y):
        return np.linalg.norm(np.asarray(x) - np.asarray(y))

    def distance_sort(self, x, population):
        Distance = [self.p_distance(x, y) for y in population]
        return sorted(zip(range(len(population)), Distance), key=lambda x: x[1])

    def get_niche(self, population, f, Fitness=None):
        # Algorithm 1：只在初始化时划分，中心为未分配个体中的最优者。
        if Fitness is None:
            Fitness = self.Fitness(population, f)
        Niche = []
        remaining_indices = list(range(len(population)))
        while remaining_indices:
            best_idx = min(remaining_indices, key=lambda idx: Fitness[idx])
            neighbors = [idx for idx in remaining_indices if idx != best_idx]
            neighbors.sort(key=lambda idx: self.p_distance(population[best_idx], population[idx]))
            niche = [best_idx] + neighbors[:self.niche_size - 1]
            Niche.append(niche)
            assigned = set(niche)
            remaining_indices = [idx for idx in remaining_indices if idx not in assigned]
        return Niche

    def update1(self, x, f_gradient):
        return self.Gradient(x, f_gradient)  # 式 (5)

    def update2(self, x, best_x, f_gradient):
        y = x + self.K_base * (best_x - x)  # 式 (6)
        return self.Gradient(y, f_gradient)  # 式 (9)，在 y 处求梯度

    def update3(self, x, best_x, population, f_gradient):
        # 全种群 P^t 中抽取两个不同索引；论文只要求 alpha_1 != alpha_2。
        k = np.random.choice(len(population), size=2, replace=False)
        K_explore = self.K_0 + (self.t / self.budget) ** 0.5  # 式 (8)，t 从 0 开始
        y = x + K_explore * (best_x - x) + self.F * (population[k[0]] - population[k[1]])
        return self.Gradient(y, f_gradient)  # 式 (7)、(9)

    def prob(self, x, Fitness, f):
        spread = max(Fitness) - min(Fitness)
        # 等适应度时式 (4) 未定义，约定 p=0，避免除零。
        return 0.0 if spread == 0 else (self._fitness(x, f) - min(Fitness)) / spread

    def build_explore_set(self, niche, population, Fitness, f):
        # Algorithm 2：每个非最优个体独立进行 Bernoulli 抽样，无固定人数。
        E_i = []
        best_pos = int(np.argmin(Fitness))
        spread = max(Fitness) - min(Fitness)
        for pos, x_idx in enumerate(niche):
            if pos == best_pos:
                continue
            p = 0.0 if spread == 0 else (Fitness[pos] - min(Fitness)) / spread
            if p < self.fitness_p:
                explore = np.random.random() >= self.pho_high
            else:
                explore = np.random.random() < self.pho_low
            if explore:
                E_i.append(x_idx)
        return E_i

    def update(self, x_idx, x, best_x, population, f_gradient, x_min, x_max, explore_set, best_idx=None):
        is_best = x_idx == best_idx if best_idx is not None else np.array_equal(x, best_x)
        if is_best:
            x1 = self.update1(x, f_gradient)
        elif x_idx in explore_set:
            x1 = self.update3(x, best_x, population, f_gradient)
        else:
            x1 = self.update2(x, best_x, f_gradient)
        # 默认严格执行论文未投影的迭代；x_min/x_max 默认仅控制初始化。
        if self.boundary_handling == 'clip':
            return self.is_domin(x1, x_min, x_max)
        return x1

    def generate_target(self, fitness):
        return Target(fitness)

    def DGDE(self, x_min, x_max, f, f_gradient):
        # 重复调用同一实例时重新开始一次完整实验。
        self.t = 0
        self.nfe = 0
        self.nge = 0
        self.stop_reason = 'generations'
        self.population_history = []
        self.population_all = []
        self.best_history = []
        self.niche_history = []
        self.niche_info_history = []
        population = np.array(self.initial(x_min, x_max), dtype=float, copy=True)
        fitness = self.Fitness(population, f)
        self.initial_population = population.copy()
        self.initial_fitness = list(fitness)
        self.history = [min(fitness)]  # 当代最优（包含初始代）
        best_fitness = min(fitness)
        best_solution = population[int(np.argmin(fitness))].copy()
        Niche = self.get_niche(population, f, fitness)
        self.Niche = [list(niche) for niche in Niche]

        while self.t < self.budget:
            # 只执行完整的同步代；额度不足时保留上一完整代，不产生半代结果。
            if self.max_nfe is not None and self.nfe + self.population_size > self.max_nfe:
                self.stop_reason = 'max_nfe'
                break
            if self.max_nge is not None and self.nge + self.population_size > self.max_nge:
                self.stop_reason = 'max_nge'
                break
            # 所有引导点和差分向量均来自 P^t，写入独立的 P^(t+1)。
            offspring_population_L = np.empty_like(population)
            niche_info = []
            for niche in Niche:
                Fitness_sub = [fitness[idx] for idx in niche]
                best_idx = niche[int(np.argmin(Fitness_sub))]
                best_x = population[best_idx]
                explore_set = set(self.build_explore_set(niche, population, Fitness_sub, f))
                for x_idx in niche:
                    offspring_population_L[x_idx] = self.update(
                        x_idx, population[x_idx], best_x, population, f_gradient,
                        x_min, x_max, explore_set, best_idx=best_idx)
                niche_info.append({'indices': list(niche), 'best_idx': best_idx,
                                   'explore_set': sorted(explore_set)})

            population = offspring_population_L
            self.t += 1
            fitness = self.Fitness(population, f)
            current_best = int(np.argmin(fitness))
            self.history.append(fitness[current_best])
            if fitness[current_best] < best_fitness:
                best_fitness = fitness[current_best]
                best_solution = population[current_best].copy()
            self.best_history.append(best_fitness)  # 累计最优，每代一项
            self.population_history.append([(x.tolist(), fit) for x, fit in zip(population, fitness)])
            self.population_all.append(population.copy())
            self.niche_history.append(len(Niche))
            self.niche_info_history.append(niche_info)

        self.pop = [Agent(solution=x.copy(), target=self.generate_target(fit))
                    for x, fit in zip(population, fitness)]
        self.g_best = self.get_sorted_population(self.pop, 'min')[0]  # 论文要求的最终种群最优
        self.best_so_far = Agent(solution=best_solution, target=self.generate_target(best_fitness))
        return self.pop, self.g_best
