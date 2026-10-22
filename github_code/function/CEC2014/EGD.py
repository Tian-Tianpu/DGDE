
import numpy as np
import copy
import joblib
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from mealpy.utils.target import Target


def get_fitness(x, f):
    fitness = f(x)
    return fitness

def get_gradient(x, f_gradient):
    grad = f_gradient(x)
    return grad



class EGD(Optimizer):
    def __init__(self, population_size, dimension, pop_strength, mutate_strength, if_escape_fitness,
                 budget, gradient_thresh, mutate_thresh, learning_rate, mutate_turns, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.dimension = dimension
        self.pop_strength = pop_strength
        self.mutate_strength = mutate_strength
        self.if_escape_fitness = if_escape_fitness
        self.budget = budget
        self.gradient_thresh = gradient_thresh
        self.mutate_thresh = mutate_thresh
        self.learning_rate = learning_rate
        self.mutate_turns = mutate_turns
        self.population_history = []
        self.sort_flag = False
        self.population_all = []
        self.best_history = []

    def initial(self, x_min, x_max):
        population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension])
        return population

    def Fitness(self, population, f):
        fitness = [get_fitness(population[p], f) for p in range(len(population))]
        return fitness

    def gd_gradient(self, x, f_gradient):
        grad = get_gradient(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def mutate(self, x):
        return x + np.random.uniform(-self.mutate_strength, self.mutate_strength, [self.dimension])

    def set_mutate_strength(self, start, end, num_population):
        return np.linspace(start, end, num=num_population).tolist()

    def p_distance(self, x, y):
        dist = (np.square(np.sum((np.array(x) - np.array(y))))) ** 0.5
        return dist

    def is_domin(self, x, x_min, x_max):
        for i in range(len(x)):
            if x[i] < x_min or x[i] > x_max:
                if self.p_distance(x[i], x_min) < self.p_distance(x[i], x_max):
                    x[i] = x_min
                else:
                    x[i] = x_max
        return x

    def generate_target(self, fitness):
        return Target(fitness)

    def EGD(self, x_min, x_max, f, f_gradient):
        INF = 10000
        mutate_times = 0
        population = self.initial(x_min, x_max)
        population_fitness = [INF for _ in range(self.population_size)]
        population_index = [i for i in range(self.population_size)]
        update_index = [1 for _ in range(self.population_size)]
        is_escape_index = [1 for _ in range(self.population_size)]
        last_mutate_turn = [0 for _ in range(self.population_size)]
        population_backup = np.zeros([self.population_size, self.dimension])
        fitness_backup = [0 for _ in range(self.population_size)]
        if self.pop_strength != 0:
            population_mutate_strength = self.set_mutate_strength(self.mutate_strength,
                                                             self.pop_strength * self.mutate_strength,
                                                             self.population_size)
        else:
            population_mutate_strength = [self.mutate_strength for _ in range(self.population_size)]

        fitness = self.Fitness(population, f)
        history = [np.min(fitness)]
        best_history = [np.min(fitness)]

        t = 0
        while t < self.budget:
            for p in range(self.population_size):
                if update_index[p]:
                    x_gradient = get_gradient(population[p], f_gradient)
                    population_fitness[p] = get_fitness(population[p], f)
                    if np.linalg.norm(x_gradient, 2) < self.gradient_thresh and \
                            t - last_mutate_turn[p] > self.mutate_thresh:
                        population_backup[p] = population[p]
                        fitness_backup[p] = population_fitness[p]
                        update_index[p] = 0
                    else:
                        population[p] = self.is_domin(self.gd_gradient(population[p], f_gradient), x_min, x_max)
            fitness = self.Fitness(population, f)
            history.append(np.min(fitness))
            best_history.append(min(np.min(fitness), best_history[-1]))

            if np.sum(update_index) == 0:
                m_t = 0
                while m_t < self.mutate_turns:
                    for p in range(self.population_size):
                        if update_index[p] == 0:
                            if m_t == 0:
                                last_mutate_turn[p] = t
                                population[p] = self.is_domin(self.mutate(population[p]), x_min, x_max)
                                mutate_times += 1
                            else:
                                population[p] = self.is_domin(self.gd_gradient(population[p], f_gradient), x_min, x_max)
                            population_fitness[p] = get_fitness(population[p], f)
                    t += 1
                    m_t += 1
                    fitness = self.Fitness(population, f)
                    history.append(np.min(fitness))
                    # self.best_history.append(min(np.min(fitness), best_history[-1]))

                for p in range(self.population_size):
                    if update_index[p] == 0:
                        if fitness_backup[p] - population_fitness[p] >= self.if_escape_fitness:
                            is_escape_index[p] = 1
                            population_backup[p] = population[p]
                            fitness_backup[p] = population_fitness[p]
                        else:
                            is_escape_index[p] = 0
                            population[p] = population_backup[p]
                            population_fitness[p] = fitness_backup[p]
                fitness = self.Fitness(population, f)
                fitness_average = np.mean(fitness)
                min_index = int(np.argmin(fitness))
                for p in range(self.population_size):
                    if update_index[p] == 0:
                        if p == min_index:
                            if population_fitness[p] >= fitness_backup[p]:
                                population[p] = population_backup[p]
                                population_fitness[p] = fitness_backup[p]
                        if is_escape_index[p] == 0 and population_fitness[p] >= fitness_average:
                            population[p] = population[min_index]
                            population_fitness[p] = population_fitness[min_index]
                            population_index[p] = population_index[min_index]
                        update_index[p] = 1
            else:
                pass
            t += 1
            self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])
            self.population_all.append(population.tolist())
            self.best_history.append(min(np.min(fitness), best_history[-1]))

        self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in range(self.population_size)]
        self.g_best = self.get_sorted_population(self.pop, "min")[0]

        self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])
        t = 0

    #     return self
    #
    # def save_population_history(self, filename):
    #     with open(filename, 'wb') as f:
    #         joblib.dump(self, f)



#
# def objective_function(x):
#     return np.sum(x**2)
#
# def gradient_function(x):
#     return 2 * x
#
#






#
# def get_fitness(x, f):
#     fitness = f(x)
#     return fitness
#
# def get_gradient(x, f_gradient):
#     grad = f_gradient(x)
#     return grad
#
# def set_mutate_strength(start, end, num_population):
#     return np.linspace(start, end, num=num_population).tolist()
#
#
# class EGD():
#     def __init__(self, population_size, dimension, pop_strength, mutate_strength, if_escape_fitness,
#                  budget, gradient_thresh, mutate_thresh, learning_rate, mutate_turns):
#         self.population_size = population_size
#         self.dimension = dimension
#         self.pop_strength = pop_strength
#         self.mutate_strength = mutate_strength
#         self.if_escape_fitness = if_escape_fitness
#         self.budget = budget
#         self.gradient_thresh = gradient_thresh
#         self.mutate_thresh = mutate_thresh
#         self.learning_rate = learning_rate
#         self.mutate_turns = mutate_turns
#
#
#     def initial(self, x_min, x_max):
#         population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension])
#         return population
#
#     def Fitness(self, population, f):
#         fitness = [get_fitness(population[p], f) for p in range(len(population))]
#         return fitness
#
#     def gd_gradient(self, x, f_gradient):
#         grad = get_gradient(x, f_gradient)
#         N = len(x)
#         x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
#         return x
#
#     def mutate(self, x):
#         return x + np.random.uniform(-self.mutate_strength, self.mutate_strength, [self.dimension])
#
#     def p_distance(self, x, y):
#         dist = (np.square(np.sum((np.array(x) - np.array(y))))) ** 0.5
#         return dist
#
#     def is_domin(self, x, x_min, x_max):
#         for i in range(len(x)):
#             if x[i] < x_min or x[i] > x_max:
#                 if self.p_distance(x[i], x_min) < self.p_distance(x[i], x_max):
#                     x[i] = x_min
#                 else:
#                     x[i] = x_max
#         return x
#
#
#     def EGD(self, x_min, x_max, f, f_gradient):
#         INF = 10000
#         mutate_times = 0
#         population = self.initial(x_min, x_max)
#         population_fitness = [INF for _ in range(self.population_size)]
#         population_index = [i for i in range(self.population_size)]
#         update_index = [1 for _ in range(self.population_size)]
#         is_escape_index = [1 for _ in range(self.population_size)]
#         last_mutate_turn = [0 for _ in range(self.population_size)]
#         population_backup = np.zeros([self.population_size, self.dimension])
#         fitness_backup = [0 for _ in range(self.population_size)]
#         if self.pop_strength != 0:
#             population_mutate_strength = set_mutate_strength(self.mutate_strength,
#                                                              self.pop_strength * self.mutate_strength,
#                                                              self.population_size)
#         else:
#             population_mutate_strength = [self.mutate_strength for _ in range(self.population_size)]
#
#         fitness = self.Fitness(population, f)
#         Fitness_all = [fitness]
#         Population_all = [population]
#         history = [np.min(fitness)]
#         best_history = [np.min(fitness)]
#
#         t = 0
#         while t < self.budget:
#             for p in range(self.population_size):
#                 if update_index[p]:
#                     x_gradient = get_gradient(population[p], f_gradient)
#                     population_fitness[p] = get_fitness(population[p], f)
#                     if np.linalg.norm(x_gradient, 2) < self.gradient_thresh and \
#                             t - last_mutate_turn[p] > self.mutate_thresh:
#                         population_backup[p] = population[p]
#                         fitness_backup[p] = population_fitness[p]
#                         update_index[p] = 0
#                     else:
#                         population[p] = self.is_domin(self.gd_gradient(population[p], f_gradient), x_min, x_max)
#                         # population[p] = self.is_domin((population[p] - self.learning_rate * x_gradient), x_min, x_max)
#             fitness = self.Fitness(population, f)
#             Fitness_all.append(fitness)
#             Population_all.append(population)
#             history.append(np.min(fitness))
#             best_history.append(min(np.min(fitness), best_history[-1]))
#
#             if np.sum(update_index) == 0:
#                 m_t = 0
#                 while m_t < self.mutate_turns:
#                     for p in range(self.population_size):
#                         if update_index[p] == 0:
#                             if m_t == 0:
#                                 last_mutate_turn[p] = t
#                                 population[p] = self.is_domin(self.mutate(population[p]), x_min, x_max)
#                                 mutate_times += 1
#                             else:
#                                 population[p] = self.is_domin(self.gd_gradient(population[p], f_gradient), x_min, x_max)
#                             population_fitness[p] = get_fitness(population[p], f)
#                     t += 1
#                     m_t += 1
#                     fitness = self.Fitness(population, f)
#                     Fitness_all.append(fitness)
#                     Population_all.append(population)
#                     history.append(np.min(fitness))
#                     best_history.append(min(np.min(fitness), best_history[-1]))
#
#                 for p in range(self.population_size):
#                     if update_index[p] == 0:
#                         if fitness_backup[p] - population_fitness[p] >= self.if_escape_fitness:
#                             is_escape_index[p] = 1
#                             population_backup[p] = population[p]
#                             fitness_backup[p] = population_fitness[p]
#                         else:
#                             is_escape_index[p] = 0
#                             population[p] = population_backup[p]
#                             population_fitness[p] = fitness_backup[p]
#                 fitness = self.Fitness(population, f)
#                 Fitness_all.append(fitness)
#                 Population_all.append(population)
#                 fitness_average = np.mean(fitness)
#                 min_index = int(np.argmin(fitness))
#                 for p in range(self.population_size):
#                     if update_index[p] == 0:
#                         if p == min_index:
#                             if population_fitness[p] >= fitness_backup[p]:
#                                 population[p] = population_backup[p]
#                                 population_fitness[p] = fitness_backup[p]
#                         if is_escape_index[p] == 0 and population_fitness[p] >= fitness_average:
#                             population[p] = population[min_index]
#                             population_fitness[p] = population_fitness[min_index]
#                             population_index[p] = population_index[min_index]
#                         update_index[p] = 1
#             else:
#                 pass
#             t += 1
#         return history, best_history, Fitness_all, Population_all
