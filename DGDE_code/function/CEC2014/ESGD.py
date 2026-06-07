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

class ESGD(Optimizer):
    def __init__(self, population_size, mutate_strength, learning_rate,
                 budget, gd_budget, evo_budget, dimension, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.mutate_strength = mutate_strength
        self.learning_rate = learning_rate
        self.M = int(self.population_size / 2)
        self.budget = budget
        self.gd_budget = gd_budget
        self.evo_budget = evo_budget
        self.dimension = dimension
        self.t = 0
        self.t_gd = 0
        self.t_evo = 0
        self.optimizer = None
        self.population_history = []
        self.sort_flag = False
        self.population_all = []
        self.best_history = []

    def initial(self, x_min, x_max):
        population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension])
        return population

    def Fitness(self, population, f):
        fitness = [self.get_fitness_esgd(population[p], f) for p in range(len(population))]
        return fitness

    def get_fitness_esgd(self, x, f):
        return get_fitness(x, f)

    def get_gradient_esgd(self, x, f_gradient):
        return get_gradient(x, f_gradient)

    def esgd_gradient(self, x, f_gradient):
        grad = self.get_gradient_esgd(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def esgd_mutate(self, x):
        return x + np.random.uniform(-self.mutate_strength, self.mutate_strength, [self.dimension])

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

    def esgd_generate_offspring(self, population, f, x_min, x_max):
        fitness = self.Fitness(population, f)
        best_idx = np.argmin(fitness)
        offspring = np.zeros([self.population_size, self.dimension])
        for i in range(self.population_size):
            offspring[i] = self.is_domin(self.esgd_mutate(population[best_idx]), x_min, x_max)
        return offspring

    def selection(self, population, offspring, f):
        new_x = np.zeros([self.population_size, self.dimension])
        fitness = self.Fitness(population, f)
        x_idx = np.argsort(fitness)
        offspring_fitness = self.Fitness(offspring, f)
        off_idx = np.argsort(offspring_fitness)
        x_ite = 0
        off_ite = 0
        for i in range(self.M):
            if offspring_fitness[off_idx[off_ite]] < fitness[x_idx[x_ite]]:
                new_x[i] = copy.deepcopy(offspring[off_idx[off_ite]])
                off_ite += 1
            else:
                new_x[i] = copy.deepcopy(population[x_idx[x_ite]])
                x_ite += 1
        for j in range(self.population_size - self.M):
            if np.random.rand() < 0.5:
                idx = np.random.randint(x_ite + 1, self.population_size)
                new_x[self.M + j] = copy.deepcopy(population[x_idx[idx]])
            else:
                idx = np.random.randint(off_ite + 1, self.population_size)
                new_x[self.M + j] = copy.deepcopy(offspring[off_idx[idx]])
        population = copy.deepcopy(new_x)
        population_backup = copy.deepcopy(population)
        return population, population_backup

    def generate_target(self, fitness):
        return Target(fitness)

    def ESGD(self, x_min, x_max, f, f_gradient):
        population = self.initial(x_min, x_max)
        population_backup = copy.deepcopy(population)

        fitness = self.Fitness(population, f)
        # history = [np.min(fitness)]
        best_history = [np.min(fitness)]

        while self.t < self.budget:
            while self.t_gd < self.gd_budget and self.t < self.budget:
                fitness_backup = copy.deepcopy(fitness)
                for p in range(self.population_size):
                    population[p] = self.is_domin(self.esgd_gradient(population[p], f_gradient), x_min, x_max)
                    fitness[p] = self.get_fitness_esgd(population[p], f)
                    if fitness[p] > fitness_backup[p]:
                        population[p] = copy.deepcopy(population_backup[p])
                        fitness[p] = copy.deepcopy(fitness_backup[p])
                    else:
                        population_backup[p] = copy.deepcopy(population[p])
                        fitness_backup[p] = copy.deepcopy(fitness[p])
                fitness = self.Fitness(population, f)
                self.best_history.append(np.min(fitness))


                self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in
                            range(self.population_size)]
                self.g_best = self.get_sorted_population(self.pop, "min")[0]
                self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])

                self.t += 1
                self.t_gd += 1
            self.t_gd = 0

            while self.t_evo < self.evo_budget and self.t < self.budget:
                offspring = self.esgd_generate_offspring(population, f, x_min, x_max)
                population, population_backup = self.selection(population, offspring, f)
                fitness = self.Fitness(population, f)
                self.best_history.append(np.min(fitness))


                self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in
                            range(self.population_size)]
                self.g_best = self.get_sorted_population(self.pop, "min")[0]
                self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])




                self.t += 1
                self.t_evo += 1
            self.t_evo = 0
            self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])
            self.population_all.append(population.tolist())



        # return self
# class ESGD(Optimizer):
#     def __init__(self, population_size, mutate_strength, learning_rate,
#                  budget, gd_budget, evo_budget, dimension, **kwargs):
#         super().__init__(**kwargs)
#         self.population_size = population_size
#         self.mutate_strength = mutate_strength
#         self.learning_rate = learning_rate
#         self.seed = 123
#         self.M = int(self.population_size / 2)
#         np.random.seed(self.seed)
#         self.budget = budget
#         self.gd_budget = gd_budget
#         self.evo_budget = evo_budget
#         self.dimension = dimension
#         self.t = 0
#         self.t_gd = 0
#         self.t_evo = 0
#         self.optimizer = None
#         self.population_history = []
#         self.sort_flag = False
#         self.population_all = []
#
#     def initial(self, x_min, x_max):
#         population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension])
#         return population
#
#     def Fitness(self, population, f):
#         fitness = [self.get_fitness_esgd(population[p], f) for p in range(len(population))]
#         return fitness
#
#     def get_fitness_esgd(self, x, f):
#         return get_fitness(x, f)
#
#     def get_gradient_esgd(self, x, f_gradient):
#         return get_gradient(x, f_gradient)
#
#     def esgd_gradient(self, x, f_gradient):
#         grad = self.get_gradient_esgd(x, f_gradient)
#         N = len(x)
#         x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
#         return x
#
#     def esgd_mutate(self, x):
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
#     def esgd_generate_offspring(self, population, f, x_min, x_max):
#         fitness = self.Fitness(population, f)
#         best_idx = np.argmin(fitness)
#         offspring = np.zeros([self.population_size, self.dimension])
#         for i in range(self.population_size):
#             offspring[i] = self.is_domin(self.esgd_mutate(population[best_idx]), x_min, x_max)
#         return offspring
#
#     def selection(self, population, offspring, f):
#         new_x = np.zeros([self.population_size, self.dimension])
#         fitness = self.Fitness(population, f)
#         x_idx = np.argsort(fitness)
#         offspring_fitness = self.Fitness(offspring, f)
#         off_idx = np.argsort(offspring_fitness)
#         x_ite = 0
#         off_ite = 0
#         for i in range(self.M):
#             if offspring_fitness[off_idx[off_ite]] < fitness[x_idx[x_ite]]:
#                 new_x[i] = copy.deepcopy(offspring[off_idx[off_ite]])
#                 off_ite += 1
#             else:
#                 new_x[i] = copy.deepcopy(population[x_idx[x_ite]])
#                 x_ite += 1
#         for j in range(self.population_size - self.M):
#             if np.random.rand() < 0.5:
#                 idx = np.random.randint(x_ite + 1, self.population_size)
#                 new_x[self.M + j] = copy.deepcopy(population[x_idx[idx]])
#             else:
#                 idx = np.random.randint(off_ite + 1, self.population_size)
#                 new_x[self.M + j] = copy.deepcopy(offspring[off_idx[idx]])
#         population = copy.deepcopy(new_x)
#         population_backup = copy.deepcopy(population)
#         return population, population_backup
#
#     def generate_target(self, fitness):
#         return Target(fitness)
#
#     def ESGD(self, x_min, x_max, f, f_gradient):
#         population = self.initial(x_min, x_max)
#         population_backup = copy.deepcopy(population)
#
#         fitness = self.Fitness(population, f)
#         history = [np.min(fitness)]
#         best_history = [np.min(fitness)]
#
#         while self.t < self.budget:
#             while self.t_gd < self.gd_budget and self.t < self.budget:
#                 fitness_backup = copy.deepcopy(fitness)
#                 for p in range(self.population_size):
#                     population[p] = self.is_domin(self.esgd_gradient(population[p], f_gradient), x_min, x_max)
#                     fitness[p] = self.get_fitness_esgd(population[p], f)
#                     if fitness[p] > fitness_backup[p]:
#                         population[p] = copy.deepcopy(population_backup[p])
#                         fitness[p] = copy.deepcopy(fitness_backup[p])
#                     else:
#                         population_backup[p] = copy.deepcopy(population[p])
#                         fitness_backup[p] = copy.deepcopy(fitness[p])
#                 fitness = self.Fitness(population, f)
#                 history.append(np.min(fitness))
#                 best_history.append(min(np.min(fitness), best_history[-1]))
#                 self.t += 1
#                 self.t_gd += 1
#             self.t_gd = 0
#             while self.t_evo < self.evo_budget and self.t < self.budget:
#                 offspring = self.esgd_generate_offspring(population, f, x_min, x_max)
#                 population, population_backup = self.selection(population, offspring, f)
#                 self.t += 1
#                 self.t_evo += 1
#                 fitness = self.Fitness(population, f)
#                 history.append(np.min(fitness))
#                 best_history.append(min(np.min(fitness), best_history[-1]))
#             self.t_evo = 0
#             self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])
#             self.population_all.append(population.tolist())
#
#         self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in
#                     range(self.population_size)]
#         self.g_best = self.get_sorted_population(self.pop, "min")[0]
#         self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])
#
#         # return self
#     # def ESGD(self, x_min, x_max, f, f_gradient):
#     #     population = self.initial(x_min, x_max)
#     #     population_backup = copy.deepcopy(population)
#     #
#     #
#     #     fitness = self.Fitness(population, f)
#     #     history = [np.min(fitness)]
#     #     best_history = [np.min(fitness)]
#     #
#     #     while self.t < self.budget:
#     #         while self.t_gd < self.gd_budget and self.t < self.budget:
#     #             for p in range(self.population_size):
#     #                 population[p] = self.is_domin(self.esgd_gradient(population[p], f_gradient), x_min, x_max)
#     #                 fitness[p] = self.get_fitness_esgd(population[p], f)
#     #                 if fitness[p] > copy.deepcopy(fitness_backup[p]):
#     #                     population[p] = copy.deepcopy(population_backup[p])
#     #                     fitness[p] = copy.deepcopy(fitness_backup[p])
#     #                 else:
#     #                     population_backup[p] = copy.deepcopy(population[p])
#     #                     fitness_backup[p] = copy.deepcopy(fitness[p])
#     #             fitness = self.Fitness(population, f)
#     #             history.append(np.min(fitness))
#     #             best_history.append(min(np.min(fitness), best_history[-1]))
#     #             self.t += 1
#     #             self.t_gd += 1
#     #         self.t_gd = 0
#     #         while self.t_evo < self.evo_budget and self.t < self.budget:
#     #             offspring = self.esgd_generate_offspring(population, f, x_min, x_max)
#     #             population, population_backup = self.selection(population, offspring, f)
#     #             self.t += 1
#     #             self.t_evo += 1
#     #             fitness = self.Fitness(population, f)
#     #             history.append(np.min(fitness))
#     #             best_history.append(min(np.min(fitness), best_history[-1]))
#     #         self.t_evo = 0
#     #         self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])
#     #         self.population_all.append(population.tolist())
#     #
#     #     self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in range(self.population_size)]
#     #     self.g_best = self.get_sorted_population(self.pop, "min")[0]
#     #     self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])
#
#     #     return self
#     #
#     # def save_population_history(self, filename):
#     #     with open(filename, 'wb') as f:
#     #         joblib.dump(self, f)
#
#
# # def get_fitness(x, f):
# #     fitness = f(x)
# #     return fitness
# #
# # def get_gradient(x, f_gradient):
# #     grad = f_gradient(x)
# #     return grad
# #
# # class ESGD():
# #     def __init__(self, population_size, mutate_strength, learning_rate, seed,
# #                  budget, gd_budget, evo_budget, dimension):
# #
# #         self.x_fitness = None
# #         self.population_size = population_size
# #         self.mutate_strength = mutate_strength
# #         self.learning_rate = learning_rate
# #         self.seed = seed
# #         self.M = int(self.population_size / 2)
# #         np.random.seed(self.seed)
# #         self.budget = budget
# #         self.gd_budget = gd_budget
# #         self.evo_budget = evo_budget
# #         self.dimension = dimension
# #         self.t = 0
# #         self.t_gd = 0
# #         self.t_evo = 0
# #         self.optimizer = None
# #
# #
# #     def initial(self, x_min, x_max):
# #         population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension])
# #         return population
# #
# #     def Fitness(self, population, f):
# #         fitness = [self.get_fitness_esgd(population[p], f) for p in range(len(population))]
# #         return fitness
# #
# #     def get_fitness_esgd(self, x, f):
# #         return get_fitness(x, f)
# #
# #     def get_gradient_esgd(self, x, f_gradient):
# #         return get_gradient(x, f_gradient)
# #
# #     def esgd_gradient(self, x, f_gradient):
# #         grad = self.get_gradient_esgd(x, f_gradient)
# #         N = len(x)
# #         x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
# #         return x
# #
# #     def esgd_mutate(self, x):
# #         return x + np.random.uniform(-self.mutate_strength, self.mutate_strength, [self.dimension])
# #
# #     def p_distance(self, x, y):
# #         dist = (np.square(np.sum((np.array(x) - np.array(y))))) ** 0.5
# #         return dist
# #
# #     def is_domin(self, x, x_min, x_max):
# #         for i in range(len(x)):
# #             if x[i] < x_min or x[i] > x_max:
# #                 if self.p_distance(x[i], x_min) < self.p_distance(x[i], x_max):
# #                     x[i] = x_min
# #                 else:
# #                     x[i] = x_max
# #         return x
# #
# #     def esgd_generate_offspring(self, population, f, x_min, x_max):
# #         fitness = self.Fitness(population, f)
# #         best_idx = np.argmin(fitness)
# #         offspring = np.zeros([self.population_size, self.dimension])
# #         for i in range(self.population_size):
# #             offspring[i] = self.is_domin(self.esgd_mutate(population[best_idx]), x_min, x_max)
# #         return offspring
# #
# #     def selection(self, population, offspring, f):
# #         new_x = np.zeros([self.population_size, self.dimension])
# #         fitness = self.Fitness(population, f)
# #         x_idx = np.argsort(fitness)
# #         offspring_fitness = self.Fitness(offspring, f)
# #         off_idx = np.argsort(offspring_fitness)
# #         x_ite = 0
# #         off_ite = 0
# #         for i in range(self.M):
# #             if offspring_fitness[off_idx[off_ite]] < fitness[x_idx[x_ite]]:
# #                 new_x[i] = copy.deepcopy(offspring[off_idx[off_ite]])
# #                 off_ite += 1
# #             else:
# #                 new_x[i] = copy.deepcopy(population[x_idx[x_ite]])
# #                 x_ite += 1
# #         for j in range(self.population_size - self.M):
# #             if np.random.rand() < 0.5:
# #                 idx = np.random.randint(x_ite + 1, self.population_size)
# #                 new_x[self.M + j] = copy.deepcopy(population[x_idx[idx]])
# #             else:
# #                 idx = np.random.randint(off_ite + 1, self.population_size)
# #                 new_x[self.M + j] = copy.deepcopy(offspring[off_idx[idx]])
# #         population = copy.deepcopy(new_x)
# #         population_backup = copy.deepcopy(population)
# #         return population, population_backup
# #
# #     def ESGD(self, x_min, x_max, f, f_gradient):
# #         population = self.initial(x_min, x_max)
# #         population_backup = copy.deepcopy(population)
# #
# #         fitness = self.Fitness(population, f)
# #         Fitness_all = [fitness]
# #         Population_all = [population]
# #         fitness_backup = copy.deepcopy(fitness)
# #
# #         history = [np.min(fitness)]
# #         best_history = [np.min(fitness)]
# #         while self.t < self.budget:
# #             while self.t_gd < self.gd_budget and self.t < self.budget:
# #                 for p in range(self.population_size):
# #                     # x_update = self.esgd_gradient(population[p], f_gradient)
# #                     population[p] = self.is_domin(self.esgd_gradient(population[p], f_gradient) ,x_min, x_max)
# #                     #self.is_domin(x_update, x_min, x_max)
# #                     fitness[p] = self.get_fitness_esgd(population[p], f)
# #                     if fitness[p] > fitness_backup[p]:
# #                         population[p] = copy.deepcopy(population_backup[p])
# #                         fitness[p] = copy.deepcopy(fitness_backup[p])
# #                     else:
# #                         population_backup[p] = copy.deepcopy(population[p])
# #                         fitness_backup[p] = copy.deepcopy(fitness[p])
# #                 fitness = self.Fitness(population, f)
# #                 Fitness_all.append(fitness)
# #                 Population_all.append(population)
# #                 history.append(np.min(fitness))
# #                 best_history.append(min(np.min(fitness), best_history[-1]))
# #                 self.t += 1
# #                 self.t_gd += 1
# #             self.t_gd = 0
# #             while self.t_evo < self.evo_budget and self.t < self.budget:
# #                 offspring = self.esgd_generate_offspring(population, f, x_min, x_max)
# #                 population, population_backup = self.selection(population, offspring, f)
# #                 self.t += 1
# #                 self.t_evo += 1
# #                 fitness = self.Fitness(population, f)
# #                 Fitness_all.append(fitness)
# #                 Population_all.append(population)
# #                 history.append(np.min(fitness))
# #                 best_history.append(min(np.min(fitness), best_history[-1]))
# #             self.t_evo = 0
# #         return history, best_history, Fitness_all, Population_all
# # # #
# # # if __name__ == '__main__':
# # #
# # #     function_name = "Ackley"
# # #     dimension = 10
# # #     population_size = 10
# # #     mutate_strength = 0.5
# # #     learning_rate = 0.01
# # #     seed = 2022
# # #     budget = 100
# # #     gd_budget = 5
# # #     evo_budget = 5
# # #
# # #
# # #     f, f_gradient, x_min, x_max = select_function(function_name)
# # #     ESGD = ESGD(population_size, mutate_strength, learning_rate, seed, budget, gd_budget, evo_budget, dimension)
# # #
# # #     history_ESGD, best_history_ESGD = ESGD.min(x_min, x_max, f, f_gradient)
# # #     print(history_ESGD)
# # #     print(len(history_ESGD))
# # #
# # #
