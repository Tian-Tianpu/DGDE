# -*- coding: utf-8 -*-
import numpy as np
import autograd.numpy as anp
from estimated_gradient import S_est_gradient
import scipy.sparse as sp
# 检查Mr是否为稀疏矩阵
from scipy.sparse import issparse, csr_matrix

INF = 1.0e99
EPS = 1.0e-14
E = 2.7182818284590452353602874713526625
PI = np.pi


# 4-----------------------none
def step_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    f = 0.0
    y = x.copy()  # 应该使用输入的x
    for i in range(nx):
        if (np.fabs(y[i] - Os[i]) > 0.5):
            y[i] = Os[i] + np.floor(2 * (y[i] - Os[i]) + 0.5) / 2
    z = sr_func(y, nx, Os, Mr, scale, s_flag, r_flag)  # 应该使用处理后的y
    # 混合中 scale = 5.12 / 100.0
    for i in range(nx):
        f += (z[i] * z[i] - 10.0 * np.cos(2.0 * np.pi * z[i]) + 10.0)
    return f


def step_rastrigin_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    return S_est_gradient(step_rastrigin_func, x, nx, Os, Mr, s_flag, r_flag, scale)


# 15------------------------none
def escaffer6_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    f = 0.0
    for i in range(nx - 1):
        temp1 = np.sin(np.sqrt(z[i] * z[i] + z[i + 1] * z[i + 1]))
        temp1 = temp1 * temp1
        temp2 = 1.0 + 0.001 * (z[i] * z[i] + z[i + 1] * z[i + 1])
        f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    temp1 = np.sin(np.sqrt(z[nx - 1] * z[nx - 1] + z[0] * z[0]))
    temp1 = temp1 * temp1
    temp2 = 1.0 + 0.001 * (z[nx - 1] * z[nx - 1] + z[0] * z[0])
    f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    return f


def escaffer6_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    zp1 = np.roll(z, -1)
    zm1 = np.roll(z, 1)
    sum1 = z ** 2 + zp1 ** 2
    sqrt1 = np.sqrt(sum1)
    sin_sqrt1 = np.sin(sqrt1)
    cos_sqrt1 = np.cos(sqrt1)
    sin_sq1 = sin_sqrt1 ** 2
    denom1 = (1 + 0.001 * sum1) ** 2

    dtemp1_dzi = 2 * sin_sqrt1 * cos_sqrt1 * z / sqrt1
    ddenom1_dzi = 4 * 0.001 * z * (1 + 0.001 * sum1)
    df1_dzi = (dtemp1_dzi * denom1 - (sin_sq1 - 0.5) * ddenom1_dzi) / (denom1 ** 2)

    sum2 = zm1 ** 2 + z ** 2
    sqrt2 = np.sqrt(sum2)
    sin_sqrt2 = np.sin(sqrt2)
    cos_sqrt2 = np.cos(sqrt2)
    sin_sq2 = sin_sqrt2 ** 2
    denom2 = (1 + 0.001 * sum2) ** 2

    dtemp2_dzi = 2 * sin_sqrt2 * cos_sqrt2 * z / sqrt2
    ddenom2_dzi = 4 * 0.001 * z * (1 + 0.001 * sum2)
    df2_dzi = (dtemp2_dzi * denom2 - (sin_sq2 - 0.5) * ddenom2_dzi) / (denom2 ** 2)

    grad_z = df1_dzi + df2_dzi
    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)

    return grad_x


# 23-----------------------none不是11但是确当成了11
def grie_rosen_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    f = 0.0
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)  # scale = 5.0/100.0

    z[0] += 1.0
    for i in range(nx - 1):
        z[i + 1] += 1.0
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        f += (temp * temp) / 4000.0 - np.cos(temp) + 1.0

    tmp1 = z[nx - 1] * z[nx - 1] - z[0]
    tmp2 = z[nx - 1] - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    f += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    return f


def grie_rosen_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    z = z.copy()

    z_shifted = z.copy()
    z_shifted[0] += 1.0
    z_shifted[1:] += 1.0

    zi = z_shifted[:-1]  # z[i]
    zip1 = z_shifted[1:]  # z[i+1]

    # 构建 temp 向量
    temp = 100.0 * (zi ** 2 - zip1) ** 2 + (zi - 1.0) ** 2
    dfi_dtemp = (2 * temp / 4000.0) + np.sin(temp)

    # 求导分量
    dtemp_dzi = 400.0 * zi * (zi ** 2 - zip1) + 2.0 * (zi - 1.0)
    dtemp_dzip1 = -200.0 * (zi ** 2 - zip1)

    grad_z = np.zeros(nx)
    grad_z[:-1] += dfi_dtemp * dtemp_dzi
    grad_z[1:] += dfi_dtemp * dtemp_dzip1

    # 末尾环绕项
    zi_last = z_shifted[-1]
    zip1_last = z_shifted[0]
    temp_last = 100.0 * (zi_last ** 2 - zip1_last) ** 2 + (zi_last - 1.0) ** 2
    dfi_dtemp_last = (2 * temp_last / 4000.0) + np.sin(temp_last)
    dtemp_dzi_last = 400.0 * zi_last * (zi_last ** 2 - zip1_last) + 2.0 * (zi_last - 1.0)
    dtemp_dzip1_last = -200.0 * (zi_last ** 2 - zip1_last)
    grad_z[-1] += dfi_dtemp_last * dtemp_dzi_last
    grad_z[0] += dfi_dtemp_last * dtemp_dzip1_last

    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


# 13--------------------pdf1
def zakharov_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    indices = np.arange(1, nx + 1)
    sum1 = np.sum(z ** 2)
    sum2 = 0.5 * np.dot(indices, z)
    f = sum1 + sum2 ** 2 + sum2 ** 4
    return f


def zakharov_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    indices = np.arange(1, nx + 1)
    S = 0.5 * np.dot(indices, z)
    grad = 2 * z + indices * S * (1 + 2 * S ** 2)
    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)
    return grad_x


# 14----------------------------pdf2
def rosenbrock_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    # scale = 2.048/100.0,
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    f = np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2)
    return f


def rosenbrock_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    grad = np.zeros(D)
    grad[0] = 400 * z[0] * (z[0] ** 2 - z[1]) + 2 * (z[0] - 1)
    for i in range(1, D - 1):
        grad[i] = 200 * (z[i] - z[i - 1] ** 2) - 400 * z[i] * (z[i + 1] - z[i] ** 2) - 2 * (1 - z[i])
    grad[D - 1] = 200 * (z[D - 1] - z[D - 2] ** 2)
    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)
    return grad_x


# 16-------------------------pdf4
def rastrigin_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    # scale = 5.12 / 100.0
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    f = np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)
    return f


def rastrigin_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    grad_z = 2 * z + 20 * np.pi * np.sin(2 * np.pi * z)
    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


# 17------------------------------pdf5
def levy_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    w = 1 + (z - 1) / 4.0
    term1 = anp.sin(anp.pi * w[0]) ** 2
    term2 = anp.sum((w[:-1] - 1) ** 2 * (1 + 10 * anp.sin(anp.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + anp.sin(2 * anp.pi * w[-1]) ** 2)
    f = term1 + term2 + term3
    return f


def levy_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    w = 1 + (z - 1) / 4
    grad_w = np.zeros(D)

    # 第一个元素
    term1 = 2 * np.pi * np.sin(np.pi * w[0]) * np.cos(np.pi * w[0])
    term2 = 2 * (w[0] - 1) * (1 + 10 * np.sin(np.pi * w[0] + 1) ** 2)
    term3 = (w[0] - 1) ** 2 * 20 * np.pi * np.sin(np.pi * w[0] + 1) * np.cos(np.pi * w[0] + 1)
    grad_w[0] = term1 + term2 + term3

    # 中间元素
    for i in range(1, D - 1):
        term1 = 2 * (w[i] - 1) * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2)
        term2 = (w[i] - 1) ** 2 * 20 * np.pi * np.sin(np.pi * w[i] + 1) * np.cos(np.pi * w[i] + 1)
        grad_w[i] = term1 + term2

    # 最后一个元素
    if D > 1:
        term1 = 2 * (w[D - 1] - 1) * (1 + np.sin(2 * np.pi * w[D - 1]) ** 2)
        term2 = (w[D - 1] - 1) ** 2 * 4 * np.pi * np.sin(2 * np.pi * w[D - 1]) * np.cos(2 * np.pi * w[D - 1])
        grad_w[D - 1] = term1 + term2

    grad_z = grad_w / 4
    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


# 18-----------------------------------pdf6
def bent_cigar_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    f = z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)
    return f


def bent_cigar_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    grad = np.empty_like(z)
    grad[0] = 2 * z[0]
    grad[1:] = 2e6 * z[1:]
    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)
    return grad_x


# 19---------------------------------------pdf7
def hgbat_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    # scale = 5.0/100.0
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    s1 = np.sum(z ** 2)
    s2 = np.sum(z)
    f = np.abs(s1 ** 2 - s2 ** 2) ** 0.5 + (0.5 * s1 + s2) / D + 0.5
    return f


def hgbat_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    s1 = np.sum(z ** 2)
    s2 = np.sum(z)
    T = s1 ** 2 - s2 ** 2
    sign_T = 1 if T >= 0 else -1
    grad_z = (4 * z * s1 - 2 * s2) / (2 * np.abs(T) ** 0.5) * sign_T + (z + 1) / D
    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


# 20---------------------------------------pdf8
def ellips_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    exponents = np.arange(D) / (D - 1)
    weights = (1e6) ** exponents
    f = np.sum(weights * z ** 2)
    return f


def ellips_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    exponents = np.arange(D) / (D - 1)
    weights = (1e6) ** exponents
    grad_z = 2 * weights * z
    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


# 21---------------------------------------pdf9
def katsuura_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    # scale = 5.0/100.0
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)

    def inner_term(xi, i):
        frac = np.array([np.abs((2 ** j) * xi - np.round((2 ** j) * xi)) / 2 ** j for j in range(1, 33)])
        return (1 + i * np.sum(frac)) ** (10 / (D ** 1.2))

    prod = np.prod([inner_term(z[i], i + 1) for i in range(D)])
    f = 10 / (D ** 2) * prod - 10 / (D ** 2)
    return f


def katsuura_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    nx = len(x)
    b_val = 10.0 / np.power(nx, 1.2)

    two_power_j = np.power(2.0, np.arange(1, 33))
    z_exp = z[:, np.newaxis]
    tmp2_matrix = two_power_j * z_exp

    floored = np.floor(tmp2_matrix + 0.5)
    fractional = np.abs(tmp2_matrix - floored)
    sign_val = np.sign(tmp2_matrix - floored)

    temp = np.sum(fractional / (two_power_j), axis=1)
    sign_sums = np.sum(sign_val, axis=1)
    i_factors = np.arange(1, nx + 1)

    D = 1.0 + i_factors * temp
    D_b = np.power(D, b_val)
    P = np.prod(D_b)
    dD_dz = i_factors * sign_sums
    dD_b_dz = b_val * np.power(D, b_val - 1) * dD_dz

    partial_prods = P / D_b
    grad_z = 10.0 / (nx * nx) * partial_prods * dD_b_dz

    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


# 22---------------------------------------------pdf10
def happycat_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    # scale = 5.0/100.0
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    s1 = np.sum(z)
    s2 = np.sum(z ** 2)
    f = np.abs(s2 - D) ** 0.25 + (0.5 * s2 + s1) / D + 0.5
    return f


def happycat_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    s2 = np.sum(z ** 2)
    T = s2 - D
    sign_T = 1 if T >= 0 else -1
    grad_z = (0.25 * 2 * z / (np.abs(T)) ** 0.75) * sign_T + (z + 1) / D
    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


# 24--------------------------------------------pdf12
def schwefel_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    # scale = 1000.0/100.0
    def g12(z, D):
        if abs(z) <= 500:
            return z * np.sin(np.sqrt(abs(z)))
        elif z > 500:
            m = (500 - np.mod(z, 500))
            return m * np.sin(np.sqrt(m)) - (z - 500) ** 2 / (10000 * D)
        else:
            m = (np.mod(abs(z), 500) - 500)
            return m * np.sin(np.sqrt(abs(m))) - (z + 500) ** 2 / (10000 * D)

    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    zs = z + 420.9687462275036
    f = 418.9829 * D - np.sum([g12(i, D) for i in zs])
    return f


def schwefel_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    n = z.size
    r = z + 420.9687462275036  # same shift as function

    grad_r = np.zeros_like(r)

    # masks
    mask_pos = r > 500.0
    mask_neg = r < -500.0
    mask_mid = (~mask_pos) & (~mask_neg)  # -500 <= r <= 500

    # --- mid: |r| <= 500
    if np.any(mask_mid):
        rm = r[mask_mid]
        s = np.sqrt(np.abs(rm))  # sqrt(|r|)
        # unified formula: -sin(s) - 0.5 * s * cos(s)
        grad_r[mask_mid] = -np.sin(s) - 0.5 * s * np.cos(s)

    # --- r > 500
    if np.any(mask_pos):
        rp = r[mask_pos]
        u = 500.0 - np.fmod(rp, 500.0)  # u in (0,500]
        su = np.sqrt(u)
        grad_r[mask_pos] = (np.sin(su) + 0.5 * su * np.cos(su)
                            + 2.0 * (rp - 500.0) / (10000.0 * n))

    # --- r < -500
    if np.any(mask_neg):
        rn = r[mask_neg]
        c = 500.0 - np.fmod(np.abs(rn), 500.0)
        sc = np.sqrt(c)
        grad_r[mask_neg] = (np.sin(sc) + 0.5 * sc * np.cos(sc)
                            + 2.0 * (rn + 500.0) / (10000.0 * n))

    # Map gradient (w.r.t r) back to gradient w.r.t z:
    # r = z + constant => dr/dz = I so grad_z = grad_r
    grad = grad_r.copy()

    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)
    return grad_x


# 25-------------------------------------------------pdf13
def ackley_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    s2 = np.sum(z ** 2)
    exp1 = np.exp(-0.2 * np.sqrt(s2 / D))
    exp2 = np.exp(np.sum(np.cos(2 * np.pi * z)) / D)
    f = -20 * exp1 - exp2 + 20 + np.e
    return f


def ackley_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    s2 = np.sum(z ** 2)
    r = np.sqrt(s2 / D)
    exp1 = np.exp(-0.2 * r)
    exp2 = np.exp(np.sum(np.cos(2 * np.pi * z)) / D)
    term1 = -20 * exp1 * (-0.2) * (z / (D * r))
    term2 = -exp2 * ((-2 * np.pi * np.sin(2 * np.pi * z)) / D)
    grad = term1 + term2
    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)
    return grad_x


# 26---------------------------------------------pdf14
def discus_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    f = 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2)
    return f


def discus_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    grad = np.empty_like(z)
    grad[0] = 2e6 * z[0]
    grad[1:] = 2 * z[1:]
    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)
    return grad_x


# 27----------------------------------------------------pdf15
def griewank_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    # scale = 600.0 / 100.0
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    sum_val = np.sum(z ** 2) / 4000.0
    prod_val = np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))
    f = sum_val - prod_val + 1
    return f


def griewank_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    D = len(z)
    cos_terms = np.cos(z / np.sqrt(np.arange(1, D + 1)))
    prod_all = np.prod(cos_terms)
    grad = 2 * z / 4000 + (prod_all / cos_terms) * (
            np.sin(z / np.sqrt(np.arange(1, D + 1))) / np.sqrt(np.arange(1, D + 1)))
    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)
    return grad_x


# 28-------------------------------------------------pdf16
def schaffer_F7_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    xs = anp.stack([z, anp.roll(z, -1)], axis=1)
    si = anp.sqrt(anp.sum(xs ** 2, axis=1))[:-1]
    terms = anp.sqrt(si) * (anp.sin(50 * si ** 0.2) + 1)
    f = anp.mean(terms) ** 2
    return f


def schaffer_F7_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    n = len(x)
    if n < 2:
        return np.zeros_like(z)

    # r_i = sqrt(x_i^2 + x_{i+1}^2)
    x_pairs = z[:-1] ** 2 + z[1:] ** 2
    r = np.sqrt(x_pairs)  # shape (n-1,)

    # terms t_i
    r05 = np.sqrt(r)
    r02 = np.power(r, 0.2)
    sin_term = np.sin(50.0 * r02)
    term = r05 * (1.0 + sin_term)

    # sum of terms
    f_sum = np.sum(term)

    # d t_i / d r_i
    dterm_dr = (1.0 + sin_term) / (2.0 * np.sqrt(r)) \
               + np.sqrt(r) * (10.0 * np.power(r, -0.8) * np.cos(50.0 * r02))

    # dr/dx
    dr_dxi = z[:-1] / r
    dr_dxip1 = z[1:] / r

    # accumulate gradient
    grad = np.zeros_like(z)
    grad[:-1] += dterm_dr * dr_dxi
    grad[1:] += dterm_dr * dr_dxip1

    # final chain rule
    grad *= 2.0 * f_sum / ((n - 1) ** 2)
    grad_x = transform_grad(grad, Mr, scale, s_flag, r_flag)

    return grad_x


##==============================================复合========================================================
def transform_grad(grad_z, Mr, sh_rate, s_flag, r_flag):
    # 平移不影响梯度 s_flag此处无用
    # 第一步：旋转逆变换
    if r_flag:
        # 支持稀疏矩阵乘法
        if issparse(Mr):
            grad_temp = Mr.T.dot(grad_z)
        else:
            grad_temp = Mr.T @ grad_z
    else:
        grad_temp = grad_z

    # 第二步：总是应用缩放因子
    grad_x = sh_rate * grad_temp
    return grad_x


# 6---------------------------------------------------
def hf02_func(x, nx, Os, Mr, S, s_flag, r_flag, scale):
    # scale=1

    cf_num = 3
    fit = [None] * 3
    G = [None] * 3
    G_nx = [None] * 3
    Gp = [0.4, 0.4, 0.2]
    y = [0] * nx

    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = np.ceil(Gp[i] * nx)
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G_nx = np.int64(G_nx)
    G[0] = 0
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]

    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    S = list(map(int, S))
    for i in range(nx):
        y[i] = z[S[i] - 1]
    i = 0
    fit[i] = bent_cigar_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 1)
    i = 1
    fit[i] = hgbat_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 5.0 / 100.0)
    i = 2
    fit[i] = rastrigin_func(y[G[i]:nx], G_nx[i], Os, Mr, 0, 0, 5.12 / 100.0)

    f = 0.0
    for i in range(cf_num):
        f += fit[i]
    return f


def hf02_grad(x, nx, Os, Mr, S, s_flag, r_flag, scale):
    # return S_est_gradient(hf02_func, x, nx, Os, Mr, S, s_flag, r_flag, scale)
    cf_num = 3
    Gp = np.array([0.4, 0.4, 0.2])
    G_nx = [0] * cf_num
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G = [0] * cf_num
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]
    # 坐标变换
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    S = np.array(S, dtype=int)
    y = z[S - 1]  # 置换
    # 初始化梯度
    grad_y = np.zeros_like(y)

    # 分段计算梯度
    grad_y[G[0]:G[1]] = bent_cigar_grad(y[G[0]:G[1]], G_nx[0], Os, Mr, 0, 0, 1)
    grad_y[G[1]:G[2]] = hgbat_grad(y[G[1]:G[2]], G_nx[1], Os, Mr, 0, 0, 5.0 / 100.0)
    grad_y[G[2]:] = rastrigin_grad(y[G[2]:], G_nx[2], Os, Mr, 0, 0, 5.12 / 100.0)

    # 逆置换
    grad_z = np.zeros_like(z)
    np.add.at(grad_z, S - 1, grad_y)  # 逆置换

    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


def hf10_func(x, nx, Os, Mr, S, s_flag, r_flag, scale):
    # scale=1

    cf_num = 6
    fit = [None] * 6
    G = [None] * 6
    G_nx = [None] * 6
    Gp = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
    y = [0] * nx

    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = np.ceil(Gp[i] * nx)
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G_nx = np.int64(G_nx)
    G[0] = 0
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]

    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    S = list(map(int, S))
    for i in range(nx):
        y[i] = z[S[i] - 1]

    i = 0
    fit[i] = hgbat_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 5.0 / 100.0)
    i = 1
    fit[i] = katsuura_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 5.0 / 100.0)
    i = 2
    fit[i] = ackley_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 1)
    i = 3
    fit[i] = rastrigin_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 5.12 / 100.0)
    i = 4
    fit[i] = schwefel_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 1000.0 / 100.0)
    i = 5
    fit[i] = schaffer_F7_func(y[G[i]:nx], G_nx[i], Os, Mr, 0, 0, 1)

    f = 0.0
    for i in range(cf_num):
        f += fit[i]
    return f


def hf10_grad(x, nx, Os, Mr, S, s_flag, r_flag, scale):
    # return S_est_gradient(hf10_func, x, nx, Os, Mr, S, s_flag, r_flag, scale)

    cf_num = 6
    Gp = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.2])
    G_nx = [0] * cf_num
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp

    G = [0] * cf_num
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]

    # 坐标变换
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    S = np.array(S, dtype=int)
    y = z[S - 1]  # 置换

    # 分段计算梯度
    grad_parts = []
    grad_parts.append(hgbat_grad(y[G[0]:G[1]], G_nx[0], Os, Mr, 0, 0, 5.0 / 100.0))
    grad_parts.append(katsuura_grad(y[G[1]:G[2]], G_nx[1], Os, Mr, 0, 0, 5.0 / 100.0))
    grad_parts.append(ackley_grad(y[G[2]:G[3]], G_nx[2], Os, Mr, 0, 0, 1))
    grad_parts.append(rastrigin_grad(y[G[3]:G[4]], G_nx[3], Os, Mr, 0, 0, 5.12 / 100.0))
    grad_parts.append(schwefel_grad(y[G[4]:G[5]], G_nx[4], Os, Mr, 0, 0, 1000.0 / 100.0))
    grad_parts.append(schaffer_F7_grad(y[G[5]:], G_nx[5], Os, Mr, 0, 0, 1))

    # 拼接梯度
    grad_y = np.concatenate(grad_parts)

    # 逆置换
    grad_z = np.zeros_like(z)
    np.add.at(grad_z, S - 1, grad_y)

    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x


def hf06_func(x, nx, Os, Mr, S, s_flag, r_flag, scale):
    cf_num = 5
    fit = [None] * 5
    G = [None] * 5
    G_nx = [None] * 5
    Gp = [0.3, 0.2, 0.2, 0.1, 0.2]
    y = [0] * nx

    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = np.ceil(Gp[i] * nx)
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G_nx = np.int64(G_nx)
    G[0] = 0
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]

    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    S = list(map(int, S))
    for i in range(nx):
        y[i] = z[S[i] - 1]
    i = 0
    fit[i] = katsuura_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 5.0 / 100.0)
    i = 1
    fit[i] = happycat_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 5.0 / 100.0)
    i = 2
    fit[i] = grie_rosen_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 5.0 / 100.0)
    i = 3
    fit[i] = schwefel_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0, 1000.0 / 100.0)
    i = 4
    fit[i] = ackley_func(y[G[i]:nx], G_nx[i], Os, Mr, 0, 0, 1)
    f = 0.0
    for i in range(cf_num):
        f += fit[i]
    return f


# def hf06_func(x, nx, Os, Mr, S, s_flag, r_flag):
#     cf_num = 5
#     Gp = np.array([0.3, 0.2, 0.2, 0.1, 0.2])
#     # safe partition (floor + distribute)
#     G_nx = np.floor(Gp * nx).astype(int)
#     remain = nx - np.sum(G_nx)
#     i = 0
#     while remain > 0:
#         G_nx[i % cf_num] += 1
#         i += 1
#         remain -= 1
#     G = np.insert(np.cumsum(G_nx[:-1]), 0, 0)
#
#     # shift-rotate + permutation
#     z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
#     S = np.array(S, dtype=int)
#     y = z[S - 1]
#
#     fit = [0.0] * cf_num
#     for i in range(cf_num):
#         g = G_nx[i]
#         start = G[i]
#         if g <= 0:
#             # 该组没有维度 -> 贡献为 0，跳过调用
#             fit[i] = 0.0
#             continue
#         seg = y[start:start + g]
#         if i == 0:
#             fit[i] = katsuura_func(seg, g, Os, Mr, 0, 0)
#         elif i == 1:
#             fit[i] = happycat_func(seg, g, Os, Mr, 0, 0)
#         elif i == 2:
#             fit[i] = grie_rosen_func(seg, g, Os, Mr, 0, 0)
#         elif i == 3:
#             fit[i] = schwefel_func(seg, g, Os, Mr, 0, 0)
#         elif i == 4:
#             fit[i] = ackley_func(seg, g, Os, Mr, 0, 0)
#
#     return float(sum(fit))

def hf06_grad(x, nx, Os, Mr, S, s_flag, r_flag, scale):
    cf_num = 5
    Gp = np.array([0.3, 0.2, 0.2, 0.1, 0.2])
    G_nx = [0] * cf_num
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G = [0] * cf_num
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]

    # 坐标变换
    z = sr_func(x, nx, Os, Mr, scale, s_flag, r_flag)
    S = np.array(S, dtype=int)
    y = z[S - 1]  # 置换
    # 分段计算梯度
    grad_parts = []

    if G_nx[0] > 0:
        grad_parts.append(katsuura_grad(y[G[0]:G[1]], G_nx[0], Os, Mr, 0, 0, 5.0 / 100.0))
    else:
        grad_parts.append(np.array([]))

    if G_nx[1] > 0:
        grad_parts.append(happycat_grad(y[G[1]:G[2]], G_nx[1], Os, Mr, 0, 0, 5.0 / 100.0))
    else:
        grad_parts.append(np.array([]))

    if G_nx[2] > 0:
        grad_parts.append(grie_rosen_grad(y[G[2]:G[3]], G_nx[2], Os, Mr, 0, 0, 5.0 / 100.0))
    else:
        grad_parts.append(np.array([]))

    if G_nx[3] > 0:
        grad_parts.append(schwefel_grad(y[G[3]:G[4]], G_nx[3], Os, Mr, 0, 0, 1000.0 / 100.0))
    else:
        grad_parts.append(np.array([]))

    if G_nx[4] > 0:
        grad_parts.append(ackley_grad(y[G[4]:], G_nx[4], Os, Mr, 0, 0, 1))
    else:
        grad_parts.append(np.array([]))

    # 拼接梯度
    grad_y = np.concatenate(grad_parts)

    # 逆置换
    grad_z = np.zeros_like(z)
    np.add.at(grad_z, S - 1, grad_y)

    grad_x = transform_grad(grad_z, Mr, scale, s_flag, r_flag)
    return grad_x

def cf01_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 5
    fit = [None] * 5
    delta = [10, 20, 30, 40, 50]
    bias = [0, 200, 300, 100, 400]

    i = 0
    if issparse(Mr):
        # 稀疏矩阵：使用切片获取子矩阵
        Mr0 = Mr[0 * nx:1 * nx, :]
    else:
        # 密集矩阵：使用原来的切片方式
        Mr0 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = rosenbrock_func(x, nx, Os[i * nx:(i + 1) * nx], Mr0, 1, r_flag, 2.048 / 100.0)
    fit[i] = 10000 * fit[i] / 1e+4

    i = 1
    if issparse(Mr):
        Mr1 = Mr[1 * nx:2 * nx, :]
    else:
        Mr1 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = ellips_func(x, nx, Os[i * nx:(i + 1) * nx], Mr1, 1, r_flag, 1)
    fit[i] = 10000 * fit[i] / 1e+10

    i = 2
    if issparse(Mr):
        Mr2 = Mr[2 * nx:3 * nx, :]
    else:
        Mr2 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = bent_cigar_func(x, nx, Os[i * nx:(i + 1) * nx], Mr2, 1, r_flag, 1)
    fit[i] = 10000 * fit[i] / 1e+30

    i = 3
    if issparse(Mr):
        Mr3 = Mr[3 * nx:4 * nx, :]
    else:
        Mr3 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = discus_func(x, nx, Os[i * nx:(i + 1) * nx], Mr3, 1, r_flag, 1)
    fit[i] = 10000 * fit[i] / 1e+10

    i = 4
    if issparse(Mr):
        Mr4 = Mr[4 * nx:5 * nx, :]
    else:
        Mr4 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = ellips_func(x, nx, Os[i * nx:(i + 1) * nx], Mr4, 1, 0, 1)
    fit[i] = 10000 * fit[i] / 1e+10

    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def cf01_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 5
    delta = np.array([10, 20, 30, 40, 50], dtype=np.float64)
    bias = np.array([0, 200, 300, 100, 400], dtype=np.float64)

    Os = np.asarray(Os)
    blocks_Mr = Mr.shape[0] // nx
    # 计算Os的段数
    blocks_Os = Os.size // nx
    total_blocks = min(blocks_Os, blocks_Mr)

    if total_blocks < cf_num:
        raise ValueError(
            f"数据段不足：Os 段数={blocks_Os}, Mr 段数={blocks_Mr}, "
            f"req cf_num={cf_num}"
        )

    # 分段函数值和梯度
    f = np.zeros(cf_num)
    grads = np.zeros((cf_num, nx))

    scales = np.array([1e4, 1e10, 1e30, 1e10, 1e10])
    scales = 10000.0 / scales
    funcs = [rosenbrock_func, ellips_func, bent_cigar_func, discus_func, ellips_func]
    gfuncs = [rosenbrock_grad, ellips_grad, bent_cigar_grad, discus_grad, ellips_grad]
    scale_list = [2.048 / 100.0, 1, 1, 1, 1]

    for i in range(cf_num):
        # 提取Mr的子矩阵
        if issparse(Mr):
            Mr_i = Mr[i * nx:(i + 1) * nx, :]
        else:
            Mr_i = Mr[i * nx:(i + 1) * nx, 0:nx]

        # 提取Os的子向量
        Os_i = Os[i * nx:(i + 1) * nx]

        # 计算函数值和梯度
        f[i] = scales[i] * funcs[i](x, nx, Os_i, Mr_i, 1, r_flag, scale_list[i])
        grads[i, :] = scales[i] * gfuncs[i](x, nx, Os_i, Mr_i, 1, r_flag, scale_list[i])

    # 加偏置
    f += bias

    # 计算权重 w_i
    diff = x[None, :] - Os.reshape(blocks_Os, nx)[:cf_num]  # (cf_num, nx)
    w_sq = np.sum(diff ** 2, axis=1)  # (cf_num,)
    w = np.where(
        w_sq != 0,
        (1.0 / np.sqrt(w_sq)) * np.exp(-w_sq / (2.0 * nx * delta ** 2)),
        np.inf
    )
    finite = w[np.isfinite(w)]
    if finite.size > 0:
        w[np.isinf(w)] = np.max(finite)
    w /= np.sum(w)  # 归一化

    # 加权合成梯度
    grad_total = np.tensordot(w, grads, axes=(0, 0))  # (nx,)

    return grad_total


def cf02_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 3
    fit = [None] * 3
    delta = [20, 10, 10]
    bias = [0, 200, 100]

    i = 0
    if issparse(Mr):
        Mr0 = Mr[0 * nx:1 * nx, :]
    else:
        Mr0 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = schwefel_func(x, nx, Os[i * nx:(i + 1) * nx], Mr0, 1, 0, 1000.0 / 100.0)

    i = 1
    if issparse(Mr):
        Mr1 = Mr[1 * nx:2 * nx, :]
    else:
        Mr1 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = rastrigin_func(x, nx, Os[i * nx:(i + 1) * nx], Mr1, 1, r_flag, 5.12 / 100.0)

    i = 2
    if issparse(Mr):
        Mr2 = Mr[2 * nx:3 * nx, :]
    else:
        Mr2 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = hgbat_func(x, nx, Os[i * nx:(i + 1) * nx], Mr2, 1, r_flag, 5.0 / 100.0)

    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def cf02_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 3
    delta = np.array([20.0, 10.0, 10.0], dtype=np.float64)
    bias = np.array([0.0, 200.0, 100.0], dtype=np.float64)

    Os = np.asarray(Os)
    blocks_Mr = Mr.shape[0] // nx
    blocks_Os = Os.size // nx
    total_blocks = min(blocks_Os, blocks_Mr)

    if total_blocks < cf_num:
        raise ValueError(
            f"可用段数不足：Os 段数={blocks_Os}, Mr 段数={blocks_Mr}, 需要={cf_num}"
        )

    # 计算每段函数值和梯度
    funcs = np.stack([
        schwefel_func(x, nx, Os[0 * nx:1 * nx], Mr[0 * nx:1 * nx, :] if issparse(Mr) else Mr[0 * nx:1 * nx, 0:nx], 1, 0,
                      1000.0 / 100.0),
        rastrigin_func(x, nx, Os[1 * nx:2 * nx], Mr[1 * nx:2 * nx, :] if issparse(Mr) else Mr[1 * nx:2 * nx, 0:nx], 1,
                       r_flag, 5.12 / 100.0),
        hgbat_func(x, nx, Os[2 * nx:3 * nx], Mr[2 * nx:3 * nx, :] if issparse(Mr) else Mr[2 * nx:3 * nx, 0:nx], 1,
                   r_flag, 5.0 / 100.0)
    ])

    grads = np.stack([
        schwefel_grad(x, nx, Os[0 * nx:1 * nx], Mr[0 * nx:1 * nx, :] if issparse(Mr) else Mr[0 * nx:1 * nx, 0:nx], 1, 0,
                      1000.0 / 100.0),
        rastrigin_grad(x, nx, Os[1 * nx:2 * nx], Mr[1 * nx:2 * nx, :] if issparse(Mr) else Mr[1 * nx:2 * nx, 0:nx], 1,
                       r_flag, 5.12 / 100.0),
        hgbat_grad(x, nx, Os[2 * nx:3 * nx], Mr[2 * nx:3 * nx, :] if issparse(Mr) else Mr[2 * nx:3 * nx, 0:nx], 1,
                   r_flag, 5.0 / 100.0)
    ])

    # 加偏置
    f_offset = funcs + bias

    # 计算权重 w_i
    Os_blocks = Os.reshape(blocks_Os, nx)[:cf_num]  # (3, nx)
    diff = x[None, :] - Os_blocks
    w_sq = np.sum(diff ** 2, axis=1)
    w = np.where(
        w_sq > 0,
        (1.0 / np.sqrt(w_sq)) * np.exp(-w_sq / (2.0 * nx * delta ** 2)),
        np.inf
    )
    finite = w[np.isfinite(w)]
    if finite.size > 0:
        w[np.isinf(w)] = np.max(finite)
    W = w / np.sum(w)

    # 分两项合成梯度
    term1 = np.tensordot(W, grads, axes=(0, 0))

    # 计算 ∇W 项
    sum_w = np.sum(w)
    coeff = -1.0 - w_sq / (nx * delta ** 2)
    grad_w = (w[:, None] * diff) * coeff[:, None] / w_sq[:, None]
    grad_W = (grad_w * sum_w - w[:, None] * np.sum(grad_w, axis=0)[None, :]) / (sum_w ** 2)
    term2 = np.tensordot(f_offset, grad_W, axes=(0, 0))

    grad_z = term1 + term2

    # 使用第一块的Mr进行逆旋转
    if issparse(Mr):
        Mr0 = Mr[0 * nx:1 * nx, :]
        grad_x = Mr0.T.dot(grad_z)
    else:
        Mr0 = Mr[0 * nx:1 * nx, 0:nx]
        grad_x = Mr0.T @ grad_z

    return grad_x


def cf06_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 5
    fit = [None] * 5
    delta = [20, 20, 30, 30, 20]
    bias = [0, 200, 300, 400, 200]

    i = 0
    if issparse(Mr):
        Mr0 = Mr[0 * nx:1 * nx, :]
    else:
        Mr0 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = escaffer6_func(x, nx, Os[i * nx:(i + 1) * nx], Mr0, 1, r_flag, 1)
    fit[i] = 10000 * fit[i] / 2e+7

    i = 1
    if issparse(Mr):
        Mr1 = Mr[1 * nx:2 * nx, :]
    else:
        Mr1 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = schwefel_func(x, nx, Os[i * nx:(i + 1) * nx], Mr1, 1, r_flag, 1000.0 / 100.0)

    i = 2
    if issparse(Mr):
        Mr2 = Mr[2 * nx:3 * nx, :]
    else:
        Mr2 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = griewank_func(x, nx, Os[i * nx:(i + 1) * nx], Mr2, 1, r_flag, 600.0 / 100.0)
    fit[i] = 1000 * fit[i] / 100

    i = 3
    if issparse(Mr):
        Mr3 = Mr[3 * nx:4 * nx, :]
    else:
        Mr3 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = rosenbrock_func(x, nx, Os[i * nx:(i + 1) * nx], Mr3, 1, r_flag, 2.048 / 100.0)

    i = 4
    if issparse(Mr):
        Mr4 = Mr[4 * nx:5 * nx, :]
    else:
        Mr4 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = rastrigin_func(x, nx, Os[i * nx:(i + 1) * nx], Mr4, 1, r_flag, 512 / 100.0)
    fit[i] = 10000 * fit[i] / 1e+3

    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def cf06_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 5
    delta = np.array([20, 20, 30, 30, 20], dtype=np.float64)
    bias = np.array([0, 200, 300, 400, 200], dtype=np.float64)

    Os = np.asarray(Os)
    blocks_Mr = Mr.shape[0] // nx
    blocks_Os = Os.size // nx
    total_blocks = min(blocks_Os, blocks_Mr)

    if total_blocks < cf_num:
        raise ValueError(
            f"可用段数不足：Os 段数={blocks_Os}, Mr 段数={blocks_Mr}, 需要={cf_num}"
        )

    # 基函数与梯度函数列表及缩放系数
    f_funcs = [escaffer6_func, schwefel_func, griewank_func, rosenbrock_func, rastrigin_func]
    g_funcs = [escaffer6_grad, schwefel_grad, griewank_grad, rosenbrock_grad, rastrigin_grad]
    scales = np.array([10000 / 2e7, 1.0, 1000 / 100.0, 1.0, 10000 / 1e3], dtype=np.float64)
    scale_list = [1, 1000.0 / 100.0, 600.0 / 100.0, 2.048 / 100.0, 5.12 / 100.0]

    # 计算每段函数值和梯度
    fit = np.zeros(cf_num)
    grads = np.zeros((cf_num, nx))

    for i in range(cf_num):
        # 提取Mr和Os的子块
        Os_i = Os[i * nx:(i + 1) * nx]
        if issparse(Mr):
            Mr_i = Mr[i * nx:(i + 1) * nx, :]
        else:
            Mr_i = Mr[i * nx:(i + 1) * nx, 0:nx]

        fit[i] = scales[i] * f_funcs[i](x, nx, Os_i, Mr_i, 1, r_flag, scale_list[i])
        grads[i] = scales[i] * g_funcs[i](x, nx, Os_i, Mr_i, 1, r_flag, scale_list[i])

    # 加偏置
    fit += bias

    # 计算权重 w_i
    Os_blocks = Os.reshape(blocks_Os, nx)[:cf_num]
    diff = x[None, :] - Os_blocks
    w_sq = np.sum(diff ** 2, axis=1)
    w = np.where(
        w_sq != 0,
        (1.0 / np.sqrt(w_sq)) * np.exp(-w_sq / (2.0 * nx * delta ** 2)),
        np.inf
    )
    finite = w[np.isfinite(w)]
    if finite.size > 0:
        w[np.isinf(w)] = np.max(finite)
    w_normalized = w / np.sum(w)

    # 加权合成梯度
    grad_cf06 = np.tensordot(w_normalized, grads, axes=(0, 0))

    return grad_cf06


def cf07_func(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 6
    fit = [None] * 6
    delta = [10, 20, 30, 40, 50, 60]
    bias = [0, 300, 500, 100, 400, 200]

    i = 0
    if issparse(Mr):
        Mr0 = Mr[0 * nx:1 * nx, :]
    else:
        Mr0 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = hgbat_func(x, nx, Os[i * nx:(i + 1) * nx], Mr0, 1, r_flag, 5.0 / 100.0)
    fit[i] = 10000 * fit[i] / 1000

    i = 1
    if issparse(Mr):
        Mr1 = Mr[1 * nx:2 * nx, :]
    else:
        Mr1 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = rastrigin_func(x, nx, Os[i * nx:(i + 1) * nx], Mr1, 1, r_flag, 5.12 / 100.0)
    fit[i] = 10000 * fit[i] / 1e+3

    i = 2
    if issparse(Mr):
        Mr2 = Mr[2 * nx:3 * nx, :]
    else:
        Mr2 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = schwefel_func(x, nx, Os[i * nx:(i + 1) * nx], Mr2, 1, r_flag, 1000.0 / 100.0)
    fit[i] = 10000 * fit[i] / 4e+3

    i = 3
    if issparse(Mr):
        Mr3 = Mr[3 * nx:4 * nx, :]
    else:
        Mr3 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = bent_cigar_func(x, nx, Os[i * nx:(i + 1) * nx], Mr3, 1, r_flag, 1)
    fit[i] = 10000 * fit[i] / 1e+30

    i = 4
    if issparse(Mr):
        Mr4 = Mr[4 * nx:5 * nx, :]
    else:
        Mr4 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = ellips_func(x, nx, Os[i * nx:(i + 1) * nx], Mr4, 1, r_flag, 1)
    fit[i] = 10000 * fit[i] / 1e+10

    i = 5
    if issparse(Mr):
        Mr5 = Mr[5 * nx:6 * nx, :]
    else:
        Mr5 = Mr[i * nx:(i + 1) * nx, 0:nx]
    fit[i] = escaffer6_func(x, nx, Os[i * nx:(i + 1) * nx], Mr5, 1, r_flag, 1)
    fit[i] = 10000 * fit[i] / 2e+7

    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def cf07_grad(x, nx, Os, Mr, s_flag, r_flag, scale):
    cf_num = 6
    delta = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    bias = np.array([0, 300, 500, 100, 400, 200], dtype=np.float64)

    Os = np.asarray(Os)
    blocks_Mr = Mr.shape[0] // nx
    blocks_Os = Os.size // nx
    total_blocks = min(blocks_Os, blocks_Mr)

    if total_blocks < cf_num:
        raise ValueError(
            f"可用段数不足：Os 段数={blocks_Os}, Mr 段数={blocks_Mr}, 需要={cf_num}"
        )

    # 基函数与梯度函数及缩放系数
    f_funcs = [
        hgbat_func, rastrigin_func, schwefel_func,
        bent_cigar_func, ellips_func, escaffer6_func
    ]
    g_funcs = [
        hgbat_grad, rastrigin_grad, schwefel_grad,
        bent_cigar_grad, ellips_grad, escaffer6_grad
    ]
    scales = np.array([1000.0, 1e3, 4e3, 1e30, 1e10, 2e7], dtype=np.float64)
    scales = 10000.0 / scales
    scale_list = [5.0 / 100.0, 5.12 / 100.0, 1000.0 / 100.0, 1, 1, 1]

    # 计算每段函数值与梯度
    fit = np.zeros(cf_num)
    grads = np.zeros((cf_num, nx))

    for i in range(cf_num):
        # 提取Mr和Os的子块
        Os_i = Os[i * nx:(i + 1) * nx]
        if issparse(Mr):
            Mr_i = Mr[i * nx:(i + 1) * nx, :]
        else:
            Mr_i = Mr[i * nx:(i + 1) * nx, 0:nx]

        fit[i] = scales[i] * f_funcs[i](x, nx, Os_i, Mr_i, 1, r_flag, scale_list[i])
        grads[i] = scales[i] * g_funcs[i](x, nx, Os_i, Mr_i, 1, r_flag, scale_list[i])

    # 加偏置
    fit += bias

    # 计算权重
    Os_blocks = Os.reshape(blocks_Os, nx)[:cf_num]
    diff = x[None, :] - Os_blocks
    w_sq = np.sum(diff ** 2, axis=1)
    w = np.where(
        w_sq != 0,
        (1.0 / np.sqrt(w_sq)) * np.exp(-w_sq / (2.0 * nx * delta ** 2)),
        np.inf
    )
    finite = w[np.isfinite(w)]
    if finite.size > 0:
        w[np.isinf(w)] = np.max(finite)
    w_normalized = w / np.sum(w)

    # 加权合成梯度
    grad_cf07 = np.tensordot(w_normalized, grads, axes=(0, 0))

    return grad_cf07


def shiftfunc(x, nx, Os):
    x = np.asarray(x)
    Os = np.asarray(Os)
    return x - Os


def rotatefunc(x, nx, Mr):
    x = np.asarray(x)
    Mr = np.asarray(Mr)
    return Mr @ x  # 矩阵乘向量，返回 numpy ndarray

def sr_func(x, nx, Os, Mr, sh_rate, s_flag, r_flag):
    nx = int(nx)
    x = np.asarray(x)
    Os = np.asarray(Os)

    if s_flag == 1:
        y = shiftfunc(x, nx, Os)  # 平移变换-缩放变换-旋转变换
        y = y * sh_rate
        if r_flag == 1:
            # 支持稀疏矩阵乘法
            if issparse(Mr):
                sr_x = Mr.dot(y)
            else:
                sr_x = Mr @ y
        else:
            sr_x = y
    else:
        y = x * sh_rate  # 缩放变换-旋转变换
        if r_flag == 1:
            # 支持稀疏矩阵乘法
            if issparse(Mr):
                sr_x = Mr.dot(y)
            else:
                sr_x = Mr @ y
        else:
            sr_x = y
    return sr_x


def asyfunc(x, xasy, nx, beta):
    for i in range(nx):
        if (x[i] > 0):
            xasy[i] = pow(x[i], 1.0 + beta * i / (nx - 1) * pow(x[i], 0.5))



def oszfunc(x, xosz, nx):
    for i in range(nx):
        if (i == 0 | i == nx - 1):
            if (x[i] != 0):
                xx = np.log(np.fabs(x[i]))
            if (x[i] > 0):
                c1 = 10
                c2 = 7.9
            else:
                c1 = 5.5
                c2 = 3.1
            if (x[i] > 0):
                sx = 1
            elif (x[i] == 0):
                sx = 0
            else:
                sx = -1
            xosz[i] = sx * np.exp(xx + 0.049 * (np.sin(c1 * xx) + np.sin(c2 * xx)))
        else:
            xosz[i] = x[i]


def cf_cal(x, nx, Os, delta, bias, fit, cf_num):
    w_max = 0
    w_sum = 0
    w = [None] * cf_num
    for i in range(cf_num):
        fit[i] += bias[i]
        w[i] = 0
        for j in range(nx):
            w[i] += pow(x[j] - Os[i * nx + j], 2.0)
        if (w[i] != 0):
            w[i] = pow(1.0 / w[i], 0.5) * np.exp(-w[i] / 2.0 / nx / pow(delta[i], 2.0))
        else:
            w[i] = INF
        if (w[i] > w_max):
            w_max = w[i]

    for i in range(cf_num):
        w_sum = w_sum + w[i]
    if (w_max == 0):
        for i in range(cf_num):
            w[i] = 1
        w_sum = cf_num
    f = 0.0
    for i in range(cf_num):
        f = f + w[i] / w_sum * fit[i]
    del (w)
    return f

