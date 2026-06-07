# coding:utf-8
import numpy as np
import joblib
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from mealpy.utils.target import Target

def get_fitness(x, f):
    return f(x)

def get_gradient(x, f_gradient):
    return f_gradient(x)

class MGD(Optimizer):
    def __init__(self, learning_rate, budget, population_size, dimension, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.budget = budget
        self.population_size = population_size
        self.dimension = dimension
        self.t = 0
        self.population_history = []
        self.best_history = []
        self.sort_flag = False

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
        return Target(fitness) # 修改部分

    def MGD(self, x_min, x_max, f, f_gradient):
        population = self.initial(x_min, x_max)
        fitness = self.Fitness(population, f)

        history = [np.min(fitness)]
        best_history = [np.min(fitness)]

        while self.t < self.budget:
            for p in range(self.population_size):
                x_update = self.gd_gradient(population[p], f_gradient)
                population[p] = self.is_domin(x_update, x_min, x_max)
            fitness = self.Fitness(population, f)

            history.append(np.min(fitness))
            self.best_history.append(min(np.min(fitness), best_history[-1]))
            self.t += 1
            self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])

        self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in range(self.population_size)]
        self.g_best = self.get_sorted_population(self.pop, "min")[0]

        self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])

    #     return self
    #
    # def save_population_history(self, filename):
    #     with open(filename, 'wb') as f:
    #         joblib.dump(self, f)

            # def save_population_history(self, filename):
    #     with open(filename, 'wb') as f:
    #         joblib.dump(self.population_history, f)



class MPGD(Optimizer):
    def __init__(self, learning_rate, budget, population_size, dimension,
                 mutate_thresh, mutate_strength, gradient_thresh, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.budget = budget
        self.population_size = population_size
        self.dimension = dimension
        self.mutate_thresh = mutate_thresh
        self.mutate_strength = mutate_strength
        self.gradient_thresh = gradient_thresh
        self.t = 0
        self.population_history = []
        self.best_history = []
        self.sort_flag = False

    def initial(self, x_min, x_max):
        population = np.random.uniform(x_min, x_max, [self.population_size, self.dimension])
        return population

    def Fitness(self, population, f):
        fitness = [get_fitness(population[p], f) for p in range(len(population))]
        return fitness

    def mutate(self, x):
        return x + np.random.uniform(-self.mutate_strength, self.mutate_strength, [self.dimension])

    def gd_gradient(self, x, f_gradient):
        grad = get_gradient(x, f_gradient)
        N = len(x)
        x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
        return x

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

    def MPGD(self, x_min, x_max, f, f_gradient):
        population = self.initial(x_min, x_max)
        fitness = self.Fitness(population, f)

        history = [np.min(fitness)]
        best_history = [np.min(fitness)]

        mutate_index = 0
        mutate_count = 0
        while self.t < self.budget:
            for p in range(self.population_size):
                x_gradient = get_gradient(population[p], f_gradient)
                x_gradient_norm = np.linalg.norm(x_gradient, 2)
                if (x_gradient_norm < self.gradient_thresh) & (self.t - mutate_index > self.mutate_thresh):
                    mutate_index = self.t
                    mutate_count += 1
                    population[p] = self.is_domin(self.mutate(population[p]), x_min, x_max)
                else:
                    population[p] = self.is_domin(self.gd_gradient(population[p], f_gradient), x_min, x_max)
            fitness = self.Fitness(population, f)

            history.append(np.min(fitness))
            self.best_history.append(min(np.min(fitness), best_history[-1]))
            self.t += 1
            self.population_history.append([(population[p].tolist(), fitness[p]) for p in range(self.population_size)])

        self.pop = [Agent(solution=population[p], target=self.generate_target(fitness[p])) for p in range(self.population_size)]
        self.g_best = self.get_sorted_population(self.pop, "min")[0]

        self.population_history.append([(agent.solution, agent.target.fitness) for agent in self.pop])


        # return self

    # def save_population_history(self, filename):
    #     with open(filename, 'wb') as f:
    #         joblib.dump(self.population_history, f)
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
# budget = 100
# dimension = 30
# x_min = -10.0
# x_max = 10.0
# learning_rate = 0.01
# population_size = 5


# figure_save_path = "./pickle_GD/"
# if not os.path.exists(figure_save_path):
#     os.makedirs(figure_save_path)
# # ------------------
# print("=====Running OriginalDE_model======")
# model = MGD(learning_rate=learning_rate, budget=budget, population_size=population_size, dimension=dimension)
# model.MGD(x_min, x_max, objective_function, gradient_function)
# model.save_population_history(figure_save_path + 'MGD_model_population.pkl')
#
# with open(figure_save_path + 'MGD_model.pkl', 'wb') as f:
#     joblib.dump(model, f)
#
# with open(figure_save_path + 'MGD_model.pkl', 'rb') as f:
#     loaded_model = joblib.load(f)

# print(f"Best Fitness: {loaded_model.g_best.target.fitness}")
# print("Final Population:")
# for agent in loaded_model.pop:
#     print(f"Solution: {agent.solution}, Fitness: {agent.target.fitness}")
#
# print("Population History (Individuals):")
# for generation_individuals in loaded_model.population_history:
#     for individual in generation_individuals:
#         print(f"Solution: {individual[0]}, Fitness: {individual[1]}")
#     print("-" * 20)
# #
# #
#
# def get_fitness(x, f):
#     fitness = f(x)
#     return fitness
#
# def get_gradient(x, f_gradient):
#     grad = f_gradient(x)
#     return grad
#
#
# class MGD:
#     def __init__(self, learning_rate, budget, population_size, dimension):
#         self.learning_rate = learning_rate
#         self.budget = budget
#         self.population_size = population_size
#         self.dimension = dimension
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
#     def gd_gradient(self, x, f_gradient):
#         grad = get_gradient(x, f_gradient)
#         N = len(x)
#         x[0:N:1] = x[0:N:1] - self.learning_rate * grad[0:N:1]
#         return x
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
#     def MGD(self, x_min, x_max, f, f_gradient):
#         population = self.initial(x_min, x_max)
#         fitness = self.Fitness(population, f)
#
#         Fitness_all = [fitness]
#         Population_all = [population]
#         history = [np.min(fitness)]
#         best_history = [np.min(fitness)]
#
#         while self.t < self.budget:
#             for p in range(self.population_size):
#                 x_update = self.gd_gradient(population[p], f_gradient)
#                 # population[p] = self.gd_gradient(population[p], f_gradient)
#                 population[p] = self.is_domin(x_update, x_min, x_max)
#                 # fitness[p] = get_fitness(population[p], f)
#             # Population_all.append(population)
#             fitness = self.Fitness(population, f)
#
#             Fitness_all.append(fitness)
#             history.append(np.min(fitness))
#             best_history.append(min(np.min(fitness), best_history[-1]))
#             self.t += 1
#
#         return history, best_history, Fitness_all, Population_all
#
#




