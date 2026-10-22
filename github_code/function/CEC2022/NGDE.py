import numpy as np
import random
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

class NGDE(Optimizer):
    def __init__(self, population_size, dimension, niche_size, budget, F, fitness_p, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.niche_size = niche_size
        self.dimension = dimension
        self.budget = budget
        self.F = F
        self.M = int(self.population_size / 2)
        self.fitness_p = fitness_p
        self.learning_rate = learning_rate
        self.t = 0
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

    def Gradient(self, x, f_gradient):
        grad = get_gradient(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def is_domin(self, x, x_min, x_max):
        for i in range(len(x)):
            if x[i] < x_min or x[i] > x_max:
                if self.p_distance(x[i], x_min) < self.p_distance(x[i], x_max):
                    x[i] = x_min
                else:
                    x[i] = x_max
        return x

    def fitness_sort(self, population, f):
        Fitness = self.Fitness(population, f)
        sort = sorted(zip(population, Fitness), key=lambda x: x[1], reverse=False)
        best_p = sort[0][0]
        return best_p, Fitness, sort

    def p_distance(self, x, y):
        dist = (np.square(np.sum((np.array(x) - np.array(y))))) ** 0.5
        return dist

    def distance_sort(self, x, population):
        Distance = np.zeros(len(population))
        for i in range(len(population)):
            Distance[i] = self.p_distance(x, population[i])
        sort = sorted(zip(population, Distance), key=lambda x: x[1], reverse=False)
        return sort

    def get_niche(self, population, f):
        Niche = []
        for i in range(int(self.population_size/self.niche_size)):
            best_p, Fitness, Fitness_sort = self.fitness_sort(population, f)
            sort = self.distance_sort(best_p, population)
            population1, niche = [], []
            for j in range(len(sort)):
                population1.append(sort[j][0])
            for k in range(self.niche_size):
                niche.append(sort[k][0])
            Niche.append(niche)
            del population1[:self.niche_size]
            population = population1
        if len(population) != 0:
            Niche.append(population)
        return Niche

    def update1(self, x, f_gradient):
        grad = self.Gradient(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

    def update2(self, x, best_x, f_gradient):
        grad = self.Gradient(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1] + self.F * (best_x[0:N:1]-x[0:N:1])
        return x

    def update3(self, x, population):
        k = np.random.randint(0, len(population), 2)
        N = len(x)
        x[0:N:1] = x[0:N:1] + self.F * (population[k[0]][0:N:1] - population[k[1]][0:N:1])
        return x

    def prob(self, x, Fitness, f):
        p = (f(x) - min(Fitness)) / (max(Fitness) - min(Fitness) + 10e-8)
        return p

    def update(self, x, best_x, population, Fitness, f, f_gradient, x_min, x_max):
        if self.p_distance(x, best_x) == 0:
            x = self.is_domin(self.update1(x, f_gradient), x_min, x_max)
        else:
            p = self.prob(x, Fitness, f)
            if p > self.fitness_p:
                x = self.is_domin(self.update2(x, best_x, f_gradient), x_min, x_max)
            else:
                x = self.is_domin(self.update3(x, population), x_min, x_max)
        return x

    def selection(self, population, offspring, f):
        candidate_population = np.array(population + offspring)
        candidate_Fitness = self.Fitness(candidate_population, f)
        sorted_indices = np.argsort(candidate_Fitness)

        new_x = []
        for i in range(self.M):
            best_individual_index = sorted_indices[i]
            best_individual = candidate_population[best_individual_index]
            new_x.append(best_individual)
            candidate_population = np.delete(candidate_population, best_individual_index, axis=0)
            sorted_indices = [index if index < best_individual_index else index - 1 for index in sorted_indices]

        for j in range(self.population_size - self.M):
            index = np.random.randint(0, len(candidate_population))
            new_x.append(candidate_population[index])
            candidate_population = np.delete(candidate_population, index, axis=0)

        return new_x

    def generate_target(self, fitness):
        return Target(fitness)

    def NGDE(self, x_min, x_max, f, f_gradient):
        population = self.initial(x_min, x_max)
        fitness = self.Fitness(population, f)
        history = [np.min(fitness)]
        best_history = [np.min(fitness)]
        population = list(population)
        niche_history = []
        niche_info_history = []

        while self.t < self.budget:
            Niche = self.get_niche(population, f)
            offspring_population = []
            niche_info = []

            for i, niche in enumerate(Niche):
                best_x, Fitness, Fitness_sort = self.fitness_sort(niche, f)
                niche_info.append({
                    "niche_index": i,
                    "individuals": [ind.tolist() for ind in niche]
                })

                for x in niche:
                    x1 = self.update(x, best_x, niche, Fitness, f, f_gradient, x_min, x_max)
                    offspring_population.append(x1)

            population = self.selection(population, offspring_population, f)

            self.t += 1
            fitness = self.Fitness(population, f)
            history.append(np.min(fitness))
            self.best_history.append(min(np.min(fitness), best_history[-1]))
            self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])
            self.population_all.append(population)

            niche_history.append(len(Niche))
            niche_info_history.append(niche_info)

        self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in
                    range(self.population_size)]
        self.g_best = self.get_sorted_population(self.pop, "min")[0]
        self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])

        self.niche_history = niche_history
        self.niche_info_history = niche_info_history

    # def NGDE(self, x_min, x_max, f, f_gradient):
    #     population = self.initial(x_min, x_max)
    #     fitness = self.Fitness(population, f)
    #     history = [np.min(fitness)]
    #     best_history = [np.min(fitness)]
    #     population = list(population)
    #     while self.t < self.budget:
    #         Niche = self.get_niche(population, f)
    #         offspring_population = []
    #         for i in range(len(Niche)):
    #             niche = Niche[i]
    #             best_x, Fitness, Fitness_sort = self.fitness_sort(niche, f)
    #             for x in niche:
    #                 x1 = self.update(x, best_x, niche, Fitness, f, f_gradient, x_min, x_max)
    #                 offspring_population.append(x1)
    #
    #         population = self.selection(population, offspring_population, f)
    #
    #         self.t += 1
    #         fitness = self.Fitness(population, f)
    #         history.append(np.min(fitness))
    #         best_history.append(min(np.min(fitness), best_history[-1]))
    #         self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])
    #         self.population_all.append(population)
    #
    #     self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in
    #                 range(self.population_size)]
    #     self.g_best = self.get_sorted_population(self.pop, "min")[0]
    #     self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])






# def get_fitness(x, f):
#     fitness = f(x)
#     return fitness
#
# def get_gradient(x, f_gradient):
#     grad = f_gradient(x)
#     return grad
#
# class NGDE():
#     def __init__(self, population_size, dimension, niche_size, budget, F, fitness_p, learning_rate):
#         self.population_size = population_size
#         self.niche_size = niche_size
#         self.dimension = dimension
#         self.budget = budget
#         self.F = F
#         self.M = int(self.population_size / 2)
#         self.fitness_p = fitness_p
#         self.learning_rate = learning_rate
#         self.t = 0
#
#     def initial(self, x_min, x_max):
#         population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension])
#         return population
#
#     def Fitness(self, population, f):
#         fitness = [get_fitness(population[p], f) for p in range(len(population))]
#         return fitness
#
#     def Gradient(self, x, f_gradient):
#         grad = get_gradient(x, f_gradient)
#         N = len(x)
#         x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
#         return x
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
#     def fitness_sort(self, population, f):
#         Fitness = self.Fitness(population, f)
#         sort = sorted(zip(population, Fitness), key=lambda x: x[1], reverse=False)
#         best_p = sort[0][0]
#         return best_p, Fitness, sort
#
#     def p_distance(self, x, y):
#         dist = (np.square(np.sum((np.array(x) - np.array(y))))) ** 0.5
#         return dist
#
#     def distance_sort(self, x, population):
#         Distance = np.zeros(len(population))
#         for i in range(len(population)):
#             Distance[i] = self.p_distance(x, population[i])
#         sort = sorted(zip(population, Distance), key=lambda x: x[1], reverse=False)
#         return sort
#
#     def get_niche(self, population, f):
#         Niche = []
#         for i in range(int(self.population_size/self.niche_size)):
#             best_p, Fitness, Fitness_sort = self.fitness_sort(population, f)
#             sort = self.distance_sort(best_p, population)
#             population1, niche = [], []
#             for j in range(len(sort)):
#                 population1.append(sort[j][0])
#             for k in range(self.niche_size):
#                 niche.append(sort[k][0])
#             Niche.append(niche)
#             del population1[:self.niche_size]
#             population = population1
#         if len(population) != 0:
#             Niche.append(population)
#         return Niche
#
#     def update1(self, x, f_gradient):
#         grad = self.Gradient(x, f_gradient)
#         N = len(x)
#         x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
#         return x
#
#     def update2(self, x, best_x, f_gradient):
#         grad = self.Gradient(x, f_gradient)
#         N = len(x)
#         x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1] + self.F * (best_x[0:N:1]-x[0:N:1]) #+ self.F * (niche[k[0]][0:N:1] - niche[k[1]][0:N:1])
#         return x
#
#     def update3(self, x, population):
#         k = np.random.randint(0, len(population), 2)
#         N = len(x)
#         x[0:N:1] = x[0:N:1] + self.F * (population[k[0]][0:N:1] - population[k[1]][0:N:1])
#         return x
#
#     def prob(self, x, Fitness, f):
#         p = (f(x) - min(Fitness)) / (max(Fitness) - min(Fitness) + 10e-8)
#         return p
#
#     def update(self, x, best_x, population, Fitness, f, f_gradient, x_min, x_max):
#         if self.p_distance(x, best_x) == 0:
#                 x = self.is_domin(self.update1(x, f_gradient), x_min, x_max)
#         else:
#             p = self.prob(x, Fitness, f)
#             if p > self.fitness_p:
#                 x = self.is_domin(self.update2(x, best_x, f_gradient), x_min, x_max)
#             else:
#                 x = self.is_domin(self.update3(x, population), x_min, x_max)
#         return x
#
#
#     def selection(self, population, offspring, f):
#         candidate_population = np.array(population + offspring)
#         candidate_Fitness = self.Fitness(candidate_population, f)
#         sorted_indices = np.argsort(candidate_Fitness)
#
#         new_x = []
#         for i in range(self.M):
#             best_individual_index = sorted_indices[i]
#             best_individual = candidate_population[best_individual_index]
#             new_x.append(best_individual)
#             candidate_population = np.delete(candidate_population, best_individual_index, axis=0)
#             sorted_indices = [index if index < best_individual_index else index - 1 for index in sorted_indices]
#
#         for j in range(self.population_size - self.M):
#             index = np.random.randint(0, len(candidate_population))
#             new_x.append(candidate_population[index])
#             candidate_population = np.delete(candidate_population, index, axis=0)
#
#         return new_x
#
#
#     def NGDE(self, x_min, x_max, f, f_gradient):
#         population = self.initial(x_min, x_max)
#         fitness = self.Fitness(population, f)
#         Fitness_all = [fitness]
#         Population_all = [population]
#         history = [np.min(fitness)]
#         best_history = [np.min(fitness)]
#         population = list(population)
#         while self.t < self.budget:
#             Niche = self.get_niche(population, f)
#             offspring_population = []
#             for i in range(len(Niche)):
#                 niche = Niche[i]
#                 best_x, Fitness, Fitness_sort = self.fitness_sort(niche, f)
#                 for x in niche:
#                     x1 = self.update(x, best_x, niche, Fitness, f, f_gradient, x_min, x_max)
#                     offspring_population.append(x1)
#
#             population = self.selection(population, offspring_population, f)
#
#             self.t += 1
#             fitness = self.Fitness(population, f)
#             Fitness_all.append(fitness)
#             Population_all.append(population)
#             history.append(np.min(fitness))
#             best_history.append(min(np.min(fitness), best_history[-1]))
#         return history, best_history, Fitness_all, Population_all
#
#
#
#
