import numpy as np
import os
import scipy
from scipy.stats import ortho_group
import scipy.sparse as sp
import pickle
from CEC2022_0923 import *

# 全局缓存
_CACHED = {}

CEC_NAME = {
    1:"zakharov",  2:"rosenbrock",   3:"schwefel",    4:"step_rastrigin",
    5:"levy",      6:"hf02",         7:"hf10",        8:"hf06",
    9:"cf01",     10:"cf02",        11:"cf06",       12:"cf07",
    13:"zakharov",14:"rosenbrock",  15:"escaffer6",  16:"rastrigin",
    17:"levy",    18:"bent_cigar",  19:"hgbat",      20:"ellips",
    21:"katsuura",22:"happycat",    23:"grie_rosen", 24:"schwefel",
    25:"ackley",  26:"discus",      27:"griewank",   28:"schaffer_F7"
}


def find_block_size_for_n(n):
    if n == 100:
        return 2
    elif n == 1000:
        return 20
    elif n == 10000:
        return 200
    else:
        print("没有设置对应的维度")

# def generate_rotation_matrix(n):
#     """
#     生成块稀疏正交矩阵
#     n: 维度
#     """
#     block_size = find_block_size_for_n(n)
#     num_blocks = n // block_size
#
#     # 生成每个块的正交矩阵
#     blocks = []
#     for _ in range(num_blocks):
#         # 每个块使用随机正交矩阵
#         block = ortho_group.rvs(dim=block_size)
#         blocks.append(block)
#
#     # 构建块对角矩阵
#     M = scipy.linalg.block_diag(*blocks)
#     return M

def generate_rotation_matrix(n):
    """
    生成块稀疏正交矩阵，返回稀疏矩阵格式
    n: 维度
    """
    block_size = find_block_size_for_n(n)
    num_blocks = n // block_size

    # 生成每个块的正交矩阵
    blocks = []
    for _ in range(num_blocks):
        # 每个块使用随机正交矩阵
        block = ortho_group.rvs(dim=block_size)
        blocks.append(block)

    # 构建块对角稀疏矩阵，使用CSR格式（适合矩阵乘法）
    M = sp.block_diag(blocks, format='csr')
    return M

def generate_shift_vector(n, low=-80, high=80):
    return np.random.uniform(low, high, n)

def generate_shuffle_data(n):
    return np.random.permutation(n) + 1

def save_as_pkl(data, filepath):
    """保存数据为PKL文件"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pkl(filepath):
    """从PKL文件加载数据"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_shift_and_rot(func_num: int, nx: int, data_dir: str = "pkl_data_sparsity"):
    """
    对于复合函数：
      - 子函数总数 cf_num
      - 偏移向量长度 cf_num * nx
      - 旋转矩阵拼接后形状 (cf_num*nx, nx)
    """
    os.makedirs(data_dir, exist_ok=True)
    key = (func_num, nx)
    if key in _CACHED:
        return _CACHED[key]

    cf_map = {9:5, 10:3, 11:5, 12:6}
    cf_num = cf_map.get(func_num, None)

    # 1. 偏移向量
    shift_pkl = os.path.join(data_dir, f"shift_data_{func_num}_D{nx}.pkl")
    if os.path.exists(shift_pkl):
        OShift_temp = load_from_pkl(shift_pkl)
    else:
        # 生成新的偏移向量
        length = nx if cf_num is None else cf_num * nx
        OShift_temp = generate_shift_vector(length)
        save_as_pkl(OShift_temp, shift_pkl)
        print(f"生成新的偏移向量: {shift_pkl}")

    # 2. 旋转矩阵
    M_pkl = os.path.join(data_dir, f"M_{func_num}_D{nx}.pkl")
    if os.path.exists(M_pkl):
        Mr = load_from_pkl(M_pkl)
    else:
        # 生成新的旋转矩阵
        if cf_num is None:
            Mr = generate_rotation_matrix(nx)
        else:
            # 对于复合函数，先生成多个块对角矩阵，然后垂直堆叠
            blocks = [generate_rotation_matrix(nx) for _ in range(cf_num)]
            # 垂直堆叠稀疏矩阵
            Mr = sp.vstack(blocks, format='csr')
        save_as_pkl(Mr, M_pkl)
        print(f"生成新的旋转矩阵: {M_pkl}")

    # 3. Shuffle数据（仅特定函数需要）
    SS = None
    if 6 <= func_num <= 8:
        shuffle_pkl = os.path.join(data_dir, f"shuffle_data_{func_num}_D{nx}.pkl")
        if os.path.exists(shuffle_pkl):
            SS = load_from_pkl(shuffle_pkl)
        else:
            SS = generate_shuffle_data(nx)
            save_as_pkl(SS, shuffle_pkl)
            print(f"生成新的shuffle数据: {shuffle_pkl}")

    _CACHED[key] = (OShift_temp, Mr, SS)
    return OShift_temp, Mr, SS

class ObjectiveFunction:
    """可 pickle 的目标函数封装"""
    def __init__(self, func_num, nx, Os, Mr, SS=None):
        self.func_num = func_num
        self.nx = nx
        self.Os = Os
        self.Mr = Mr
        self.SS = SS
        self.func = globals()[CEC_NAME[func_num] + "_func"]


    def __call__(self, x):
        if self.func_num == 1:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 1) + 300
        elif self.func_num == 2:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 2.048 / 100.0) + 400
        elif self.func_num == 3:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 1) + 600
        elif self.func_num == 4:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 5.12 / 100.0) + 800
        elif self.func_num == 5:
            return self.func(x, self.nx, self.Os, self.Mr,  1, 1, 1) + 900
        elif self.func_num == 6:
            return self.func(x, self.nx, self.Os, self.Mr, self.SS, 1, 1, 1) + 1800
        elif self.func_num == 7:
            return self.func(x, self.nx, self.Os, self.Mr, self.SS, 1, 1, 1) + 2000
        elif self.func_num == 8:
            return self.func(x, self.nx, self.Os, self.Mr, self.SS, 1, 1, 1) + 2200
        elif self.func_num == 9:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 1) + 2300
        elif self.func_num == 10:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 1) + 2400
        elif self.func_num == 11:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 1) + 2600
        elif self.func_num == 12:
            return self.func(x, self.nx, self.Os, self.Mr, 1, 1, 1) + 2700
        else:
            return self.func(x, self.nx, self.Os, self.Mr, 0, 0, 1)

class GradientFunction:
    """可 pickle 的梯度函数封装"""
    def __init__(self, func_num, nx, Os, Mr, SS=None):
        self.func_num = func_num
        self.nx = nx
        self.Os = Os
        self.Mr = Mr
        self.SS = SS
        self.grad = globals()[CEC_NAME[func_num] + "_grad"]
    def __call__(self, x):
        if self.func_num == 1:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 1)
        elif self.func_num == 2:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 2.048 / 100.0)
        elif self.func_num == 3:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 1)
        elif self.func_num == 4:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 5.12 / 100.0)
        elif self.func_num == 5:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 1)
        elif self.func_num == 6:
            return self.grad(x, self.nx, self.Os, self.Mr, self.SS, 1, 1, 1)
        elif self.func_num == 7:
            return self.grad(x, self.nx, self.Os, self.Mr, self.SS, 1, 1, 1)
        elif self.func_num == 8:
            return self.grad(x, self.nx, self.Os, self.Mr, self.SS, 1, 1, 1)
        elif self.func_num == 9:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 1)
        elif self.func_num == 10:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 1)
        elif self.func_num == 11:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 1)
        elif self.func_num == 12:
            return self.grad(x, self.nx, self.Os, self.Mr, 1, 1, 1)
        else:
            return self.grad(x, self.nx, self.Os, self.Mr, 0, 0, 1)


def select_function(func_num: int, nx: int):
    if 1 <= func_num <= 12:
        Os, Mr, SS = load_shift_and_rot(func_num, nx)
    else:
        Os = np.zeros(nx)
        Mr = np.eye(nx)
        SS = None
    f = ObjectiveFunction(func_num, nx, Os, Mr, SS)
    grad = GradientFunction(func_num, nx, Os, Mr, SS)
    return f, grad, -100, 100
