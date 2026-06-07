import numpy as np
import os
from scipy.stats import ortho_group
from CEC2014_sparsity import *
import scipy.sparse as sp
import pickle
np.random.seed(42)
import numpy as np
from scipy.sparse import issparse



# 全局缓存
_CACHED = {}

CEC_NAME = {
    1:"Rotated_High_Conditioned_Elliptic",  2:"Rotated_Bent_Cigar",   3:"Rotated_Discus",    4:"Shifted_Rotated_Rosenbrock",
    5:"Shifted_Rotated_Ackley",      6:"Shifted_Rotated_Weierstrass",         7:"Shifted_Rotated_Griewank",        8:"Shifted_Rastrigin",
    9:"Shifted_Rotated_Rastrigin",     10:"Shifted_Schwefel",        11:"Shifted_Rotated_Schwefel",       12:"Shifted_Rotated_Katsuura",
    13:"Shifted_Rotated_HappyCat",14:"Shifted_Rotated_HGBat",  15:"Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock",  16:"Shifted_Rotated_Expanded_Scaffer_F6",
    17:"Hybrid_1",    18:"Hybrid_2",  19:"Hybrid_3",      20:"Hybrid_4",
    21:"Hybrid_5",22:"Hybrid_6",    23:"Composition_1", 24:"Composition_2",
    25:"Composition_3",  26:"Composition_4",      27:"Composition_5",   28:"Composition_6", 29:"Composition_7", 30:"Composition_8"
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
    修正版本：I/O 层强制 dense，计算层可 sparse
    """
    os.makedirs(data_dir, exist_ok=True)
    key = (func_num, nx)
    if key in _CACHED:
        return _CACHED[key]

    cf_map = {
        17: 3, 18: 3, 19: 4, 20: 4, 21: 5, 22: 5,
        23: 5, 24: 3, 25: 5, 26: 5, 27: 5, 28: 5,
        29: 3, 30: 3
    }
    cf_num = cf_map.get(func_num, None)
    #
    # # 1. shift vector
    # shift_path = os.path.join(data_dir, f"shift_data_{func_num}_D{nx}.txt")
    # if os.path.exists(shift_path):
    #     OShift_temp = np.loadtxt(shift_path)
    # else:
    #     if cf_num is not None:
    #         OShift_temp = generate_shift_vector(cf_num * nx)
    #     else:
    #         OShift_temp = generate_shift_vector(nx)
    #     np.savetxt(shift_path, OShift_temp)

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

    # # 2. rotation matrix
    # M_path = os.path.join(data_dir, f"M_{func_num}_D{nx}.txt")
    # if os.path.exists(M_path):
    #     M = np.loadtxt(M_path)
    # else:
    #     if cf_num is not None:
    #         blocks = [generate_rotation_matrix(nx) for _ in range(cf_num)]
    #         M = np.vstack(blocks)
    #     else:
    #         M = generate_rotation_matrix(nx)
    #
    #     # ⭐ 核心修复：保存前确保是 dense
    #     if issparse(M):
    #         M = M.toarray()
    #
    #     np.savetxt(M_path, M)
    #
    # _CACHED[key] = (OShift_temp, M)
    #
    # print(f"Function {func_num}, Dim {nx}: o shape {OShift_temp.shape}, M shape {M.shape}")

    # 2. 旋转矩阵
    M_pkl = os.path.join(data_dir, f"M_{func_num}_D{nx}.pkl")
    if os.path.exists(M_pkl):
        M = load_from_pkl(M_pkl)
    else:
        # 生成新的旋转矩阵
        if cf_num is None:
            M = generate_rotation_matrix(nx)
        else:
            # 对于复合函数，先生成多个块对角矩阵，然后垂直堆叠
            blocks = [generate_rotation_matrix(nx) for _ in range(cf_num)]
            # 垂直堆叠稀疏矩阵
            M = sp.vstack(blocks, format='csr')
        save_as_pkl(M, M_pkl)
        print(f"生成新的旋转矩阵: {M_pkl}")

    _CACHED[key] = (OShift_temp, M)
    return OShift_temp, M


class ObjectiveFunction:
    def __init__(self, func_num, nx, o, M):
        self.func_num = func_num
        self.nx = nx
        self.o = o
        self.M = M
        self.func = globals()[CEC_NAME[func_num]]

    def __call__(self, x):
        x = np.asarray(x).flatten()
        dim = len(x)

        # 保持稀疏性，直接引用
        M_to_use = self.M

        # 特殊处理复合函数 (17-30)
        if self.func_num in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
            cf_num_map = {17: 3, 18: 3, 19: 4, 20: 4, 21: 5, 22: 5, 23: 5, 24: 3, 25: 3, 26: 5, 27: 5, 28: 5, 29: 3,
                          30: 3}
            cf_num = cf_num_map.get(self.func_num, 5)

            # 处理 o
            if len(self.o) == cf_num * dim:
                o_reshaped = self.o.reshape(cf_num, dim)
            else:
                o_reshaped = np.zeros((cf_num, dim))

            # 兼容稀疏矩阵的形状检查
            # 注意：稀疏矩阵的 .shape 依然返回 (rows, cols) 元组
            m_shape = M_to_use.shape

            if m_shape == (cf_num * dim, dim):
                M_reshaped = M_to_use
            else:
                # 如果不匹配，这里必须保持稀疏 eye，否则 10000D 会内存溢出
                M_reshaped = sp.vstack([sp.eye(dim, format='csr') for _ in range(cf_num)])
                print(f"Warning: M dimension mismatch. Created sparse Identity.")

            return self.func(x, M_reshaped, o_reshaped)

        elif self.func_num in [8, 10]:
            return self.func(x, self.o)
        else:
            return self.func(x, M_to_use, self.o)
# class GradientFunction:
#     """可 pickle 的梯度函数封装"""
#     def __init__(self, func_num, o, M):
#         self.func_num = func_num
#         self.o = o
#         self.M = M
#         self.grad = globals()[CEC_NAME[func_num] + "_grad"]
#
#     def __call__(self, x):
#         if self.func_num in [8, 10]:
#             return self.grad(x, self.o)
#         else:
#             return self.grad(x, self.M, self.o)

class GradientFunction:
    """可 pickle 的梯度函数封装"""

    def __init__(self, func_num, nx, o, M):
        self.func_num = func_num
        self.nx = nx
        self.o = o
        self.M = M
        self.grad = globals()[CEC_NAME[func_num] + "_grad"]

    def __call__(self, x):
        x = np.asarray(x).flatten()
        dim = len(x)
        M_to_use = self.M

        if self.func_num in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
            cf_num_map = {17: 3, 18: 3, 19: 4, 20: 4, 21: 5, 22: 5, 23: 5, 24: 3, 25: 3, 26: 5, 27: 5, 28: 5, 29: 3,
                          30: 3}
            cf_num = cf_num_map.get(self.func_num, 5)

            if len(self.o) == cf_num * dim:
                o_reshaped = self.o.reshape(cf_num, dim)
            else:
                o_reshaped = np.zeros((cf_num, dim))

            m_shape = M_to_use.shape
            if m_shape == (cf_num * dim, dim):
                M_reshaped = M_to_use
            else:
                M_reshaped = sp.vstack([sp.eye(dim, format='csr') for _ in range(cf_num)])

            return self.grad(x, M_reshaped, o_reshaped)

        elif self.func_num in [8, 10]:
            return self.grad(x, self.o)
        else:
            return self.grad(x, M_to_use, self.o)

def select_function(func_num: int, nx: int):
    if 1 <= func_num <= 30:
        o, M = load_shift_and_rot(func_num, nx)
    f = ObjectiveFunction(func_num, nx, o, M)
    grad = GradientFunction(func_num, nx, o, M)
    return f, grad, -100, 100
