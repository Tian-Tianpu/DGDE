import random
import numpy as np



def himmelblau(x):
    return 200 - (x[0] ** 2 + x[1] - 11) ** 2 - (x[0] + x[1] ** 2 - 7) ** 2

def himmelblau_gradient(x):
    return np.array([-4 * x[0] * (x[0] ** 2 + x[1] - 11) - 2 * x[0] - 2 * x[1] ** 2 + 14,
                     -2 * x[0] ** 2 - 4 * x[1] * (x[0] + x[1] ** 2 - 7) - 2 * x[1] + 22])

def six_hump_camel_back(x):
    return -4 * ((4 - 2.1 * (x[0] ** 2) + ((x[0] ** 4) / 3)) * (x[0] ** 2) + x[0] * x[1] + (4 * (x[1] ** 2) - 4) * (x[1] ** 2))

def six_hump_camel_back_gradient(x):
    return np.array([-4 * x[0] ** 2 * (4 * x[0] ** 3 / 3 - 4.2 * x[0]) - 8 * x[0] * (x[0] ** 4 / 3 - 2.1 * x[0] ** 2 + 4) - 4 * x[1],
                     -4 * x[0] - 32 * x[1] ** 3 - 8 * x[1] * (4 * x[1] ** 2 - 4)])

def three_hump_camel(x):
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2

def three_hump_camel_gradient(x):
    return np.array([4 * x[0] - 4.2 * x[0] ** 3 + x[0] ** 5 + x[1], x[0] + 2 * x[1]])

def equal_maxima(x):
    return np.sin(5 * np.pi * x) ** 6

def equal_maxima_gradient(x):
    return 30 * np.pi * np.sin(5 * np.pi * x) ** 5 * np.cos(5 * np.pi * x)

def uneven_decreasing_maxima(x):
    return x * np.sin(5 * np.pi * x) ** 6

def uneven_decreasing_maxima_gradient(x):
    return 30 * np.pi * x * np.sin(5 * np.pi * x) ** 5 * np.cos(5 * np.pi * x) + np.sin(5 * np.pi * x) ** 6

def shubert(x):
    return sum(np.cos(2 * x + 1) + 2 * np.cos(3 * x + 2) + 3 * np.cos(4 * x + 3) +
               4 * np.cos(5 * x + 4) + 5 * np.cos(6 * x + 5))

def shubert_gradient(x):
    return -2 * np.sin(2 * x + 1) - 6 * np.sin(3 * x + 2) - 12 * np.sin(4 * x + 3) - 20 * np.sin(5 * x + 4) - 30 * np.sin(6 * x + 5)

def vincent(x):
    return sum(np.sin(10 * np.log(x))) / len(x)

def vincent_gradient(x):
    return (10 * np.cos(10 * np.log(x))) / (len(x) * x)

def bent_cigar(x):
    return x[0] ** 2 + sum(100 * (x[1:]) ** 2)

def bent_cigar_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * x[0]
    grad[1:] = 200 * x[1:]
    return grad

def sphere(x):
    return sum(x ** 2)

def sphere_gradient(x):
    return 2 * x

def zakharov(x):
    f21 = sum(0.5 * i * xi for i, xi in enumerate(x))
    return sum(xi ** 2 for xi in x) + f21 ** 2 + f21 ** 4

def zakharov_gradient(x):
    f21 = sum(0.5 * i * xi for i, xi in enumerate(x))
    return np.array([2 * xi + i * xi * f21 + 2 * f21 ** 3 for i, xi in enumerate(x)])

def rosenbrock(x):
    return sum((x[:-1] - 1) ** 2) + sum(100 * (x[:-1] ** 2 - x[1:]) ** 2)

def rosenbrock_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
    grad[1:-1] = -200 * (x[:-2] ** 2 - x[1:-1]) + 2 * (x[1:-1] - 1) + 400 * x[1:-1] * (x[1:-1] ** 2 - x[2:])
    grad[-1] = -200 * (x[-2] ** 2 - x[-1])
    return grad

def ackley(x):
    N = len(x)
    inx = 1 / N
    return -20 * np.exp(-0.2 * np.sqrt(inx * sum(x ** 2))) - np.exp(inx * sum(np.cos(2 * np.pi * x))) + 20 + np.e

def ackley_gradient(x):
    N = len(x)
    inx = 1 / N
    return 4 * inx * x * np.exp(-0.2 * np.sqrt(inx * sum(x ** 2))) * (1 / (np.sqrt(inx * sum(x ** 2)))) + inx * 2 * np.pi * np.sin(2 * np.pi * x) * np.exp(inx * sum(np.cos(2 * np.pi * x)))

def griewank(x):
    N = len(x)
    index = np.arange(1, N + 1)
    return sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(index))) + 1

def griewank_gradient(x):
    N = len(x)
    index = np.arange(1, N + 1)
    return x / 2000 + np.sin(x / np.sqrt(index)) / (2 * np.sqrt(index)) * np.prod(np.cos(x / np.sqrt(index))) / np.cos(x / np.sqrt(index))

def rastrigin(x):
    return sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

def rastrigin_gradient(x):
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

def schwefel(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))

def schwefel_gradient(x):
    grad = -np.sin(np.sqrt(np.abs(x))) - x * np.cos(np.sqrt(np.abs(x))) / (2 * np.sqrt(np.abs(x)))
    return grad

# def schwefel_gradient(x):
#     eps = 1e-12
#     N = len(x)
#     EPSILON = eps * np.eye(N)
#     fitness = schwefel(x)
#     grad = np.zeros(N)
#     for i in range(N):
#         try:
#             grad[i] = (schwefel(x + EPSILON[i]) - fitness) / eps
#         except Exception as e:
#             print(f"Error calculating gradient for dimension {i}: {e}")
#             grad[i] = 0
#     return grad

def styblinski_tang(x):
    N = len(x)
    return sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2

def styblinski_tang_gradient(x):
    return (4 * x ** 3 - 32 * x + 5) / 2

def michalewicz(x):
    N = len(x)
    f_value = 0
    for i in range(N):
        f_value -= np.sin(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi))**20
    return f_value

def michalewicz_gradient(x):
    N = len(x)
    grad = np.zeros(N)
    for i in range(N):
        grad[i] = - (np.cos(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi))**20 +
                     np.sin(x[i]) * 20 * (np.sin((i + 1) * x[i]**2 / np.pi))**19 * 2 * (i + 1) * x[i] / np.pi)
    return grad

def quartic_function(x):
    N = len(x)
    return sum(i * x[i] ** 4 + random.random() for i in range(N))

def quartic_function_gradient(x):
    return np.array([i * 4 * xi ** 3 for i, xi in enumerate(x)])


def dixon_price(x):
    f = (x[0] - 1)**2
    for i in range(1, len(x)):
        f += (i + 1) * (2 * x[i]**2 - x[i-1])**2
    return f

def dixon_price_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * (x[0] - 1) - 4 * x[1]
    for i in range(1, len(x) - 1):
        grad[i] = 8 * (i + 1) * x[i] - 2 * x[i-1] - 4 * (i + 2) * x[i+1]
    grad[-1] = 8 * len(x) * x[-1] - 2 * x[-2]
    return grad

def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def levy_gradient(x):
    w = 1 + (x - 1) / 4
    grad = np.zeros_like(x)
    grad[0] = np.pi * np.sin(2 * np.pi * w[0]) * np.cos(np.pi * w[0]) / 2 - 2 * (w[1] - 1) * (1 + 10 * np.sin(np.pi * w[1] + 1)**2) / 4
    for i in range(1, len(x) - 1):
        grad[i] = 2 * (w[i] - 1) * (1 + 10 * np.sin(np.pi * w[i] + 1)**2) / 4 - 2 * (w[i+1] - 1) * (1 + 10 * np.sin(np.pi * w[i+1] + 1)**2) / 4 + 5 * np.pi * (w[i-1] - 1)**2 * np.sin(2 * np.pi * w[i] + 2) / 2
    grad[-1] = 2 * (w[-1] - 1) * (1 + np.sin(2 * np.pi * w[-1])**2) / 4 + np.pi * (w[-2] - 1)**2 * np.sin(4 * np.pi * w[-1])
    return grad


def select_function(function_name):
    if function_name == "Equal_maxima":
        return equal_maxima, equal_maxima_gradient, 0, 1
    elif function_name == "Uneven_decreasing_maxima":
        return uneven_decreasing_maxima, uneven_decreasing_maxima_gradient, 0, 1
    elif function_name == "Himmelblau":
        return himmelblau, himmelblau_gradient, -6, 6
    elif function_name == "Six_hump_camel_back":
        return six_hump_camel_back, six_hump_camel_back_gradient, -1, 1
    elif function_name == "Three-Hump Camel":
        return three_hump_camel, three_hump_camel_gradient, -1, 1
    elif function_name == "Shubert":
        return shubert, shubert_gradient, -10, 10
    elif function_name == "Vincent":
        return vincent, vincent_gradient, 0.25, 10
    elif function_name == "Bent Cigar":
        return bent_cigar, bent_cigar_gradient, -100, 100
    elif function_name == "Sphere":
        return sphere, sphere_gradient, -10, 10
    elif function_name == "Zakharov":
        return zakharov, zakharov_gradient, -10, 10
    elif function_name == "Rosenbrock":
        return rosenbrock, rosenbrock_gradient, -100, 100
    elif function_name == "Ackley":
        return ackley, ackley_gradient, -100, 100
    elif function_name == "Griewank":
        return griewank, griewank_gradient, -100, 100
    elif function_name == "Restrigin":
        return rastrigin, rastrigin_gradient, -100, 100
    elif function_name == "Schwefel":
        return schwefel, schwefel_gradient, -10, 10
    elif function_name == "Michalewicz":
        return michalewicz, michalewicz_gradient, 0, 4
    elif function_name == "Styblinski-Tang":
        return styblinski_tang, styblinski_tang_gradient, -5, 5
    elif function_name == "Quartic Function":
        return quartic_function, quartic_function_gradient, -1, 1
    elif function_name == "dixon_price":
        return dixon_price, dixon_price_gradient, -10, 10
    elif function_name == "levy":
        return levy, levy_gradient, -100, 100

    else:
        raise ValueError(f"Function {function_name} not found.")


    # elif function_name == "Beale":
    #     return beale, beale_gradient, -4.5, 4.5
    # elif function_name == "Goldstein-Price":
    #     return goldstein_price, goldstein_price_gradient, -2, 2
    # elif function_name == "Booth":
    #     return booth, booth_gradient, -10, 10

# Beale 函数 (E.M.L. Beale, 1958)
# def beale(x):
#     N = len(x)
#     f_value = 0
#     for i in range(0, N - 1, 2):
#         if i + 1 < N:
#             f_value += (1.5 - x[i] + x[i] * x[i + 1]) ** 2 + \
#                        (2.25 - x[i] + x[i] * x[i + 1] ** 2) ** 2 + \
#                        (2.625 - x[i] + x[i] * x[i + 1] ** 3) ** 2
#     return f_value
#     # return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

# def beale_gradient(x):
#     N = len(x)
#     grad = np.zeros(N)
#     for i in range(0, N - 1, 2):
#         if i + 1 < N:
#             df_dx1 = 2 * (1.5 - x[i] + x[i] * x[i + 1]) * (-1 + x[i + 1]) + \
#                      2 * (2.25 - x[i] + x[i] * x[i + 1] ** 2) * (-1 + x[i + 1] ** 2) + \
#                      2 * (2.625 - x[i] + x[i] * x[i + 1] ** 3) * (-1 + x[i + 1] ** 3)
#             df_dx2 = 2 * (1.5 - x[i] + x[i] * x[i + 1]) * x[i] + \
#                      4 * x[i + 1] * (2.25 - x[i] + x[i] * x[i + 1] ** 2) * x[i] + \
#                      6 * x[i + 1] ** 2 * (2.625 - x[i] + x[i] * x[i + 1] ** 3) * x[i]
#             grad[i] += df_dx1
#             grad[i + 1] += df_dx2
#     return grad
#
#     # df_dx1 = 2 * (1.5 - x[0] + x[0] * x[1]) * (-1 + x[1]) + 2 * (2.25 - x[0] + x[0] * x[1]**2) * (-1 + x[1]**2) + 2 * (2.625 - x[0] + x[0] * x[1]**3) * (-1 + x[1]**3)
#     # df_dx2 = 2 * (1.5 - x[0] + x[0] * x[1]) * x[0] + 4 * x[1] * (2.25 - x[0] + x[0] * x[1]**2) * x[0] + 6 * x[1]**2 * (2.625 - x[0] + x[0] * x[1]**3) * x[0]
#     # return np.array([df_dx1, df_dx2])
#
# # Goldstein-Price 函数 (A.A. Goldstein and J.F. Price, 1968)
# def goldstein_price(x):
#     N = len(x)
#     # if N < 2:
#     #     raise ValueError("Input vector must have at least 2 dimensions.")
#     f_value = 0
#     for i in range(0, N - 1, 2):
#         if i + 1 < N:
#             term1 = 1 + (x[i] + x[i + 1] + 1) ** 2 * (
#                         19 - 14 * x[i] + 3 * x[i] ** 2 - 14 * x[i + 1] + 6 * x[i] * x[i + 1] + 3 * x[i + 1] ** 2)
#             term2 = 30 + (2 * x[i] - 3 * x[i + 1]) ** 2 * (
#                         18 - 32 * x[i] + 12 * x[i] ** 2 + 48 * x[i + 1] - 36 * x[i] * x[i + 1] + 27 * x[i + 1] ** 2)
#             f_value += term1 * term2
#     return f_value
#
#     # term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)
#     # term2 = 30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2)
#     # return term1 * term2
#
# def goldstein_price_gradient(x):
#     N = len(x)
#     # if N < 2:
#     #     raise ValueError("Input vector must have at least 2 dimensions.")
#
#     grad = np.zeros(N)
#     for i in range(0, N - 1, 2):
#         if i + 1 < N:
#             t1 = 1 + (x[i] + x[i + 1] + 1) ** 2 * (
#                         19 - 14 * x[i] + 3 * x[i] ** 2 - 14 * x[i + 1] + 6 * x[i] * x[i + 1] + 3 * x[i + 1] ** 2)
#             t2 = 30 + (2 * x[i] - 3 * x[i + 1]) ** 2 * (
#                         18 - 32 * x[i] + 12 * x[i] ** 2 + 48 * x[i + 1] - 36 * x[i] * x[i + 1] + 27 * x[i + 1] ** 2)
#             dt1_dx1 = 2 * (x[i] + x[i + 1] + 1) * (
#                         19 - 14 * x[i] + 3 * x[i] ** 2 - 14 * x[i + 1] + 6 * x[i] * x[i + 1] + 3 * x[i + 1] ** 2) + (
#                                   x[i] + x[i + 1] + 1) ** 2 * (-14 + 6 * x[i] + 6 * x[i + 1])
#             dt1_dx2 = 2 * (x[i] + x[i + 1] + 1) * (
#                         19 - 14 * x[i] + 3 * x[i] ** 2 - 14 * x[i + 1] + 6 * x[i] * x[i + 1] + 3 * x[i + 1] ** 2) + (
#                                   x[i] + x[i + 1] + 1) ** 2 * (-14 + 6 * x[i] + 6 * x[i + 1])
#             dt2_dx1 = 4 * (2 * x[i] - 3 * x[i + 1]) * (
#                         18 - 32 * x[i] + 12 * x[i] ** 2 + 48 * x[i + 1] - 36 * x[i] * x[i + 1] + 27 * x[i + 1] ** 2) + (
#                                   2 * x[i] - 3 * x[i + 1]) ** 2 * (-32 + 24 * x[i] - 36 * x[i + 1])
#             dt2_dx2 = -6 * (2 * x[i] - 3 * x[i + 1]) * (
#                         18 - 32 * x[i] + 12 * x[i] ** 2 + 48 * x[i + 1] - 36 * x[i] * x[i + 1] + 27 * x[i + 1] ** 2) + (
#                                   2 * x[i] - 3 * x[i + 1]) ** 2 * (48 - 36 * x[i] + 54 * x[i + 1])
#             grad[i] += dt1_dx1 * t2 + t1 * dt2_dx1
#             grad[i + 1] += dt1_dx2 * t2 + t1 * dt2_dx2
#     return grad
#     # t1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)
#     # t2 = 30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2)
#     # dt1_dx1 = 2 * (x[0] + x[1] + 1) * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2) + (x[0] + x[1] + 1)**2 * (-14 + 6 * x[0] + 6 * x[1])
#     # dt1_dx2 = 2 * (x[0] + x[1] + 1) * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2) + (x[0] + x[1] + 1)**2 * (-14 + 6 * x[0] + 6 * x[1])
#     # dt2_dx1 = 4 * (2 * x[0] - 3 * x[1]) * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2) + (2 * x[0] - 3 * x[1])**2 * (-32 + 24 * x[0] - 36 * x[1])
#     # dt2_dx2 = -6 * (2 * x[0] - 3 * x[1]) * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2) + (2 * x[0] - 3 * x[1])**2 * (48 - 36 * x[0] + 54 * x[1])
#     # return np.array([dt1_dx1 * t2 + t1 * dt2_dx1, dt1_dx2 * t2 + t1 * dt2_dx2])
#
# # Booth 函数 (G.W. Booth, 1969)
# def booth(x):
#     N = len(x)
#     # if N < 2:
#     #     raise ValueError("Input vector must have at least 2 dimensions.")
#     f_value = 0
#     for i in range(0, N - 1, 2):
#         if i + 1 < N:
#             f_value += (x[i] + 2 * x[i + 1] - 7) ** 2 + (2 * x[i] + x[i + 1] - 5) ** 2
#     return f_value
#     # return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2
#
# def booth_gradient(x):
#     N = len(x)
#     # if N < 2:
#     #     raise ValueError("Input vector must have at least 2 dimensions.")
#
#     grad = np.zeros(N)
#     for i in range(0, N - 1, 2):
#         if i + 1 < N:
#             df_dx1 = 2 * (x[i] + 2 * x[i + 1] - 7) + 4 * (2 * x[i] + x[i + 1] - 5)
#             df_dx2 = 4 * (x[i] + 2 * x[i + 1] - 7) + 2 * (2 * x[i] + x[i + 1] - 5)
#             grad[i] += df_dx1
#             grad[i + 1] += df_dx2
#     return grad
#     # df_dx1 = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
#     # df_dx2 = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
#     # return np.array([df_dx1, df_dx2])
#
#





#
# # from new_gradient import monte_carlo_gradient_estimate
#
# def select_function(function_name):
#
#     if function_name == "Equal_maxima":
#         x_min, x_max = 0, 1
#         ##目标函数
#         def f(x):
#             y = np.sin(5 * np.pi * x) ** 6
#             return y
#         ##目标函数梯度计算
#         def f_gradient(x):
#             grad = 30 * np.pi * np.sin(5 * np.pi * x) ** 5 * np.cos(5 * np.pi * x)
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#
#     elif function_name == "Uneven_decreasing_maxima":
#         x_min ,x_max = 0, 1
#         def f(x):
#             y = (x * np.sin(5 * np.pi * x) ** 6)
#             return y
#         def f_gradient(x):
#             grad = 30 * np.pi * x * np.sin(5 * np.pi * x) ** 5 * np.cos(5 * np.pi * x) + np.sin(5 * np.pi * x) ** 6
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#     elif function_name == "Himmelblau":
#         x_min, x_max =  -6, 6
#         def f(x):
#             f_value = 200 - (x[0] ** 2 + x[1] - 11) ** 2 - (x[0] + x[1] ** 2 - 7) ** 2
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0]= -4*x[0]*(x[0]**2 + x[1] - 11) - 2*x[0] - 2*x[1]**2 + 14
#             grad[1] = -2*x[0]**2 - 4*x[1]*(x[0] + x[1]**2 - 7) - 2*x[1] + 22
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Six_hump_camel_back":
#         x_min, x_max = -1, 1
#         def f(x):
#             f_value = -4 * ((4 - 2.1 * (x[0] ** 2) + ((x[0] ** 4) / 3)) * (x[0] ** 2) + x[0] * x[1] + (4 * (x[1] ** 2) - 4) * (x[1] ** 2))
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0] = -4*x[0]**2*(4*x[0]**3/3 - 4.2*x[0]) - 8*x[0]*(x[0]**4/3 - 2.1*x[0]**2 + 4) - 4*x[1]
#             grad[1] = -4*x[0] - 32*x[1]**3 - 8*x[1]*(4*x[1]**2 - 4)
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#     elif function_name == "Three-Hump Camel":
#         x_min, x_max =  -1, 1
#         def f(x):
#             f_value = 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0]= 4*x[0] - 4.2*x[0]**3 + x[0]**5 + x[1]
#             grad[1] = x[0] + 2*x[1]
#             return grad
#         #
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Shubert":
#         x_min, x_max = -10, 10
#         def f(x):
#             N = len(x)
#             f_value = sum(np.cos(2*x[0:N:1]+1) + 2*np.cos(3*x[0:N:1]+2) + 3*np.cos(4*x[0:N:1]+3) +
#                           4*np.cos(5*x[0:N:1]+4) + 5*np.cos(6*x[0:N:1]+5))
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0:N:1] = -2*np.sin(2*x[0:N:1]+1) - 6*np.sin(3*x[0:N:1]+2) \
#                           - 12*np.sin(4*x[0:N:1]+3) - 20*np.sin(5*x[0:N:1]+4) - 30*np.sin(6*x[0:N:1]+5)
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Vincent":
#         x_min, x_max = 0.25, 10
#         def f(x):
#             N = len(x)
#             f_value = sum(np.sin(10 * np.log(x[0:N:1])))/N
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0:N:1] = (10 * np.cos(10 * np.log(x[0:N:1])))/(N * x[0:N:1])
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#
#     elif function_name == "Bent Cigar":
#         x_min, x_max = -2, 2
#         def f(x):
#             N = len(x)
#             f_value = x[0]**2 + sum(100 * (x[1:N:1])**2)
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0] = 2 * x[0]
#             grad[1:N:1] = 200 * np.array(x[1:N:1])
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Sphere":
#         x_min, x_max = -3, 3
#         def f(x):
#             N = len(x)
#             f_value = sum(np.array(x[0:N:1]) ** 2)
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0:N:1] = 2 * np.array(x[0:N:1])
#             return grad
#         #
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#     elif function_name == "Zakharov":
#         x_min, x_max = -3, 3
#         def f(x):
#             N = len(x)
#             f1 = sum(np.array(x[0:N:1])**2)
#             f21 = 0
#             for i in range(N):
#                 f21 += 0.5 * i * np.array(x[i])
#             f2 = f21 ** 2
#             f3 = f21 ** 4
#             f_value = f1 + f2 + f3
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             f21 = 0
#             for i in range(N):
#                 f21 += 0.5 * i * np.array(x[i])
#
#             for i in range(N):
#                 grad[i] = 2 * x[i] + i * x[i] * f21 + 2 * f21**3
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#     elif function_name == "Rosenbrock":
#         x_min, x_max = -2.048, 2.048
#         def f(x):
#             N = len(x)
#             f_value = sum((np.array(x[0:N-1:1])-1)**2) + sum(100 * (np.array(x[0:N-1:1])**2 - np.array(x[1:N:1]))**2)
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0:1:1] = 400 * np.array(x[0:1:1]) * (np.array(x[0:1:1])**2 - np.array(x[1:2:1])) + 2 * (np.array(x[0:1:1])-1)
#             grad[1:N-1:1] = -200 * (np.array(x[0:N-2:1])**2 - np.array(x[1:N-1:1])) + 2 * (np.array(x[1:N-1:1])-1) + 400 * np.array(x[1:N-1:1]) * (np.array(x[1:N-1:1])**2-x[2:N:1])
#             grad[N-1:N:1] = -200 * (np.array(x[N-2:N-1:1])**2 - np.array(x[N-1:N:1]))
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#     elif function_name == "Ackley":
#         x_min, x_max = -10, 10
#         def f(x):
#             N = len(x)
#             inx = 1/N
#             f_value = -20 * np.exp(-0.2 * np.sqrt(inx * sum(np.array(x[0:N:1])**2))) - np.exp(inx * sum(np.cos(2 * np.pi * np.array(x[0:N:1])))) + 20 + np.e
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             inx = 1/N
#             grad = np.zeros(N)
#             grad[0:N:1] = 4 * inx * np.array(x[0:N:1]) * np.exp(-0.2 * np.sqrt(inx * sum(np.array(x[0:N:1])**2))) * (1/(np.sqrt(inx * sum(np.array(x[0:N:1])**2)))) + inx * 2 * np.pi * np.sin(2 * np.pi * np.array(x[0:N:1])) * np.exp(inx * sum(np.cos(2 * np.pi * np.array(x[0:N:1]))))
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#     elif function_name == "Griewank":
#         x_min, x_max = -10, 10
#         def f(x):
#             N = len(x)
#             index = []
#             for i in range(N):
#                 index.append(i + 1)
#             f_value = sum(np.array(x[0:N:1])**2 / 4000) - np.dot(np.cos(np.array(x)/np.sqrt(index)), np.cos(np.array(x)/np.sqrt(index))) + 1
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             index = []
#             for i in range(N):
#                 index.append(i + 1)
#             grad[0:N:1] = (np.array(x[0:N:1])**2 / 2000) - (1/2) * (np.array(index)[0:N:1]**(-3/2)) * np.sin(np.array(x)/np.sqrt(index)) * np.dot(np.cos(np.array(x)/np.sqrt(index)), np.cos(np.array(x)/np.sqrt(index))) / (np.cos(np.array(x)[0:N:1]/np.array(index)[0:N:1]))
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Restrigin":
#         x_min, x_max = -5.12, 5.12
#         def f(x):
#             N = len(x)
#             f_value = sum(np.array(x[0:N:1])**2 - 10 * np.cos(2 * np.pi * np.array(x[0:N:1])) + 10)
#             return f_value
#
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0:N:1] = 2 * np.array(x[0:N:1]) + 20 * np.pi * np.sin(2 * np.pi * np.array(x[0:N:1]))
#             return grad
#         #
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Schwefel":
#         x_min, x_max = -10, 10
#         def f(x):
#             N = len(x)
#             f_value = - np.sum(x[0:N:1] * np.sin(np.sqrt(abs(x[0:N:1]))))
#             return f_value
#         def f_gradient(x):
#             eps = 1e-12
#             N = len(x)
#             EPSILON = eps * np.eye(N)
#             grad = np.zeros([N])
#             fitness = f(x)
#             for i in range(N):
#                 try:
#                     grad[i] = (f(x + EPSILON[i]) - fitness) / (eps)
#                 except Exception as e:
#                     print(e)
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#
#     elif function_name == "Michalewicz":
#         x_min, x_max = 0, 4
#         def f(x):
#             N = len(x)
#             f_value = 0
#             for i in range(N):
#                 f_value = - np.sin(x[i]) * (np.sin(i * x[i]**2 / np.pi))**20
#                 f_value += f_value
#             return f_value
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             for i in range(N):
#                 grad[i] = - (np.cos(x[i])*(np.sin(i * x[i]**2 / np.pi))**20 +
#                              np.sin(x[i])*20*((np.sin(i * x[i]**2 / np.pi))**19)*2*i*x[i]/np.pi)
#             return grad
#
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Styblinski-Tang":
#         x_min, x_max = -5, 5
#         def f(x):
#             N = len(x)
#             f_value = sum(np.array(x[0:N:1]) ** 4 - 16 * np.array(x[0:N:1]) ** 2 + 5 * np.array(x[0:N:1])) / 2
#             return f_value
#
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             grad[0:N:1] = (4 * np.array(x[0:N:1])**3 - 32 * np.array(x[0:N:1]) + 5) / 2
#             return grad
#         #
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     elif function_name == "Quartic Function":
#         x_min, x_max = -1, 1
#         def f(x):
#             N = len(x)
#             f_value = 0
#             for i in range(N):
#                 f_value = np.sum(i * x[i]**4 + random.random())
#                 f_value += f_value
#             return f_value
#
#         def f_gradient(x):
#             N = len(x)
#             grad = np.zeros(N)
#             for i in range(N):
#                 grad[i] = i * 4 * x[i]**3
#             return grad
#         #
#         # def f_estimate_gradient(x):
#         #     estimated_gradient = monte_carlo_gradient_estimate(f, x, epsilon=1e-6, num_samples=1000)
#         #     return estimated_gradient
#
#     else:
#         print("输入函数名称有误")
#
#
#     return f, f_gradient, x_min, x_max
