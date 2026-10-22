import numpy as np
import numba

# @numba.njit(nogil=True, fastmath=True)
def High_Conditioned_Elliptic(x):
  D = len(x)
  exponents = np.arange(D) / (D - 1)
  weights = (1e6) ** exponents
  f = np.sum(weights * x ** 2)
  return f

# @numba.njit(nogil=True, fastmath=True)
def High_Conditioned_Elliptic_grad(x):
    nx = len(x)
    indices = np.arange(nx)
    weights = 10.0 ** (6.0 * indices / (nx - 1))
    grad = 2.0 * weights * x
    return grad

# @numba.njit(nogil=True, fastmath=True)
def Bent_Cigar(x):
    dim = x.shape[0]
    return x[0]*x[0] + 1e+6 * np.sum(np.square(x[1:dim]))

# @numba.njit(nogil=True, fastmath=True)
def Bent_Cigar_grad(x):
    grad = np.empty_like(x)
    grad[0] = 2 * x[0]
    grad[1:] = 2e6 * x[1:]
    return grad


# @numba.njit(nogil=True, fastmath=True)
def Discus(x):
    dim = x.shape[0]
    return 1e+6 * x[0]*x[0] + np.sum(np.square(x[1:dim]))

# @numba.njit(nogil=True, fastmath=True)
def Discus_grad(x):
  grad = np.empty_like(x)
  grad[0] = 2e6 * x[0]
  grad[1:] = 2 * x[1:]
  return grad


# @numba.njit(nogil=True, fastmath=True)
def Rosenbrock(x):
    return sum((x[:-1] - 1) ** 2) + sum(100 * (x[:-1] ** 2 - x[1:]) ** 2)

# @numba.njit(nogil=True, fastmath=True)
def Rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
    grad[1:-1] = -200 * (x[:-2] ** 2 - x[1:-1]) + 2 * (x[1:-1] - 1) + 400 * x[1:-1] * (x[1:-1] ** 2 - x[2:])
    grad[-1] = -200 * (x[-2] ** 2 - x[-1])
    return grad


# @numba.njit(nogil=True, fastmath=True)
def Ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x)))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e


# @numba.njit(nogil=True, fastmath=True)
def Ackley_grad(x):
    D = len(x)
    s2 = np.sum(x ** 2)
    r = np.sqrt(s2 / D)
    exp1 = np.exp(-0.2 * r)
    exp2 = np.exp(np.sum(np.cos(2 * np.pi * x)) / D)
    term1 = -20 * exp1 * (-0.2) * (x / (D * r))
    term2 = -exp2 * ((-2 * np.pi * np.sin(2 * np.pi * x)) / D)
    grad = term1 + term2
    return grad

# @numba.njit(nogil=True, fastmath=True)
def Weierstrass(x):
    kmax = 20
    x = np.asarray(x)
    dim = x.shape[0]

    k = np.arange(kmax+1)[:, None]        # shape (kmax,1)
    a = (0.5 ** k)                      # shape (kmax,1)
    b = (3 ** k)                        # shape (kmax,1)

    # broadcast: (kmax,dim)
    cos_term = np.cos(2*np.pi * b * (x[None,:] + 0.5))
    t1 = (a * cos_term).sum(axis=0)     # shape (dim,)
    t2 = (a * np.cos(np.pi * b)).sum()  # scalar

    res = t1.sum() - dim * t2
    return res

# @numba.njit(nogil=True, fastmath=True)
def Weierstrass_grad(x):
    kmax = 20
    x = np.asarray(x)
    dim = x.shape[0]
    k = np.arange(kmax+1)[:, None]        # shape (kmax,1)
    a = (0.5 ** k)                      # shape (kmax,1)
    b = (3 ** k)                        # shape (kmax,1)
    sin_term = np.sin(2*np.pi * b * (x[None,:] + 0.5))  # shape (kmax,dim)
    grad = (-2 * np.pi) * (a * b * sin_term).sum(axis=0)  # shape (dim,)
    return grad

# @numba.njit(nogil=True, fastmath=True)
def Griewank(x):
    x = np.ascontiguousarray(x)
    y = np.arange(1, x.shape[0]+1, 1)
    return 0.00025 * np.dot(x, x) - np.prod(np.cos(x / np.sqrt(y))) + 1


# @numba.njit(nogil=True, fastmath=True)
def Griewank_grad(x):
    nx = len(x)
    index = np.arange(1, nx + 1, dtype=np.float64)
    cos_terms = np.cos(x / np.sqrt(index))
    prod_all = np.prod(cos_terms)
    grad = x / 2000.0 + (np.sin(x / np.sqrt(index)) / np.sqrt(index)) * (prod_all / cos_terms)
    return grad


# @numba.njit(nogil=True, fastmath=True)
def Rastrigin(x):
    return np.sum(np.square(x) - 10*np.cos(2*np.pi*x) + 10)

# @numba.njit(nogil=True, fastmath=True)
def Rastrigin_grad(x):
    grad = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    return grad

# @numba.njit(nogil=True, fastmath=True)
def Modified_Schwefel(x):
    n = x.size
    r = x + 420.9687462275036  # shift
    g = np.zeros_like(r)
    mask1 = np.abs(r) <= 500
    g[mask1] = -r[mask1] * np.sin(np.sqrt(np.abs(r[mask1])))
    # r > 500
    mask2 = r > 500
    u = 500 - np.fmod(r[mask2], 500)
    g[mask2] = -u * np.sin(np.sqrt(u)) + ((r[mask2] - 500)**2) / (10000 * n)
    # r < -500
    mask3 = r < -500
    c = 500 - np.fmod(np.abs(r[mask3]), 500)
    g[mask3] = -c * np.sin(np.sqrt(c)) + ((r[mask3] + 500)**2) / (10000 * n)
    return np.sum(g) + 418.9829 * n

# @numba.njit(nogil=True, fastmath=True)
def Modified_Schwefel_grad(x):
    SHIFT = 420.9687462275036
    x = np.asarray(x, dtype=float)
    n = x.size
    r = x + SHIFT
    grad = np.zeros_like(r)
    eps = 1e-12
    # mask1: |r| <= 500
    m1 = np.abs(r) <= 500
    if np.any(m1):
        rr = r[m1]
        # split sign
        pos = rr >= 0
        if np.any(pos):
            t = rr[pos]
            # dg/dr = -sin(sqrt(r)) - sqrt(r)/2 * cos(sqrt(r))
            s = np.sqrt(t)
            dg = -np.sin(s) - 0.5 * s * np.cos(s)
            grad[np.nonzero(m1)[0][pos]] = dg
        if np.any(~pos):
            t = rr[~pos]            # negative values
            s = np.sqrt(-t)         # sqrt(|r|)
            # dg/dr = -sin(sqrt(|r|)) + (r / (2 sqrt(|r|))) * cos(sqrt(|r|))
            dg = -np.sin(s) + (t / (2.0 * s)) * np.cos(s)
            grad[np.nonzero(m1)[0][~pos]] = dg

    # mask2: r > 500
    m2 = r > 500
    if np.any(m2):
        rm = r[m2]
        u = 500.0 - np.fmod(rm, 500.0)
        # avoid u==0 (rare): treat via limit
        u_safe = np.maximum(u, eps)
        s = np.sqrt(u_safe)
        dg = np.sin(s) + 0.5 * s * np.cos(s) + 2.0 * (rm - 500.0) / (10000.0 * n)
        grad[m2] = dg

    # mask3: r < -500
    m3 = r < -500
    if np.any(m3):
        rm = r[m3]
        s_abs = np.abs(rm)
        c = 500.0 - np.fmod(s_abs, 500.0)
        c_safe = np.maximum(c, eps)
        s = np.sqrt(c_safe)
        dg = -np.sin(s) - 0.5 * s * np.cos(s) + 2.0 * (rm + 500.0) / (10000.0 * n)
        grad[m3] = dg

    # dr/dx = 1 so grad wrt x is same
    return grad

# @numba.njit(nogil=True, fastmath=True)
def Katsuura(x):
    dim = x.shape[0]
    d_1_2 = 10 / pow(dim, 1.2)
    d_2 = 10 / (dim*dim)
    D = numba.prange(dim)
    J = numba.prange(1, 33)
    prod = 1
    for i in D:
        t_sum = 0
        for j in J:
            t1 = pow(2, j)
            t2 = t1 * x[i]
            t_sum += (np.abs(t2 - round(t2)) / t1)
        prod *= pow(i * t_sum + 1, d_1_2)
    return d_2 * (prod - 1)


# @numba.njit(nogil=True, fastmath=True)
def Katsuura_grad(x):
    nx = len(x)
    b_val = 10.0 / np.power(nx, 1.2)

    two_power_j = np.power(2.0, np.arange(1, 33))
    x_exp = x[:, np.newaxis]
    tmp2_matrix = two_power_j * x_exp

    floored = np.floor(tmp2_matrix + 0.5)
    fractional = np.abs(tmp2_matrix - floored)
    sign_val = np.sign(tmp2_matrix - floored)

    temp = np.sum(fractional / (two_power_j ), axis=1)
    sign_sums = np.sum(sign_val / two_power_j, axis=1)
    i_factors = np.arange(1, nx + 1)

    D = 1.0 + i_factors * temp
    D_b = np.power(D, b_val)
    P = np.prod(D_b)
    dD_dx = i_factors * sign_sums
    dD_b_dx = b_val * np.power(D, b_val - 1) * dD_dx

    partial_prods = P / D_b
    grad = 10.0 / (nx * nx) * partial_prods * dD_b_dx
    return grad


# @numba.njit(nogil=True, fastmath=True)
def HappyCat(x):
    x = np.ascontiguousarray(x)
    dim = x.shape[0]
    t = np.dot(x, x)
    return pow(np.abs(t - dim), 0.25) + (0.5*t + np.sum(x))/dim + 0.5


# @numba.njit(nogil=True, fastmath=True)
def HappyCat_grad(x):
    eps = 1e-6
    x = np.asarray(x, dtype=float)
    D = x.size
    s2 = np.dot(x, x)
    T = s2 - D
    # compute base term safely
    absT = np.abs(T)
    grad = np.empty_like(x)
    if absT <= eps:
        # near-singular: gradient magnitude is extremely large;
        # choose to return np.sign(T)*large or np.inf, here np.inf with appropriate sign
        # but sign(T) undefined at exactly 0, we'll return np.full with np.inf
        grad[:] = np.nan  # or np.inf
    else:
        coeff = 0.5 * (np.sign(T)) * (absT ** (-0.75))
        grad = coeff * x + (x + 1.0) / D
    return grad


# @numba.njit(nogil=True, fastmath=True)
def HGBat(x):
    x = np.ascontiguousarray(x)
    dim = x.shape[0]
    t1 = np.sum(x)
    t2 = np.dot(x, x)
    return np.sqrt(np.abs(t2*t2 - t1*t1)) + (0.5*t2 + t1)/dim + 0.5


# @numba.njit(nogil=True, fastmath=True)
def HGBat_grad(x):
    D = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(x)
    T = s1 ** 2 - s2 ** 2
    sign_T = 1 if T >= 0 else -1
    grad = (4 * x * s1 - 2 * s2) / (2 * np.abs(T) ** 0.5) * sign_T + (x + 1) / D
    return grad


# @numba.njit(nogil=True, fastmath=True)
def Expended_Griewank_plus_Rosenbrock(x):
    y = x  # y_i = tilde z_i in math above
    n = x.size
    y_ip1 = np.roll(y, -1)  # y_{i+1}
    # compute a_i, b_i
    a = y * y - y_ip1  # a_i = y_i^2 - y_{i+1}
    b = y
    i_idx = np.arange(1, n + 1)
    sigma = np.sqrt(i_idx)

    # t_i = 100 a_i^2 + b_i^2 / sigma_i
    t = 100.0 * a ** 2 + (b ** 2)

    # function value
    phi = t ** 2 / 4000.0 - np.cos(t / sigma) + 1.0
    f = np.sum(phi)
    return float(f)

# @numba.njit(nogil=True, fastmath=True)
def Expended_Griewank_plus_Rosenbrock_grad(x):
    n = x.size
    y = x
    # Step 2: circular neighbor
    y_ip1 = np.roll(y, -1)  # y_{i+1}
    y_im1 = np.roll(y, 1)  # y_{i-1}
    # Step 3: a_i and t_i
    a = y ** 2 - y_ip1
    b = y
    t = 100.0 * a ** 2 + b ** 2
    # Step 4: sigma
    sigma = np.sqrt(np.arange(1, n + 1))
    # Step 5: dphi/dt
    dphi_dt = t / 2000.0 + (1.0 / (sigma)) * np.sin(t / (sigma))
    # Step 6: ∂t_i/∂y_i and ∂t_i/∂y_{i+1}
    dt_dyi = 400 * y * a + 2 * y
    dt_dyip1 = -200 * a
    # Gradient wrt y: two contributions
    grad_y = dphi_dt * dt_dyi + np.roll(dphi_dt * dt_dyip1, 1)
    return grad_y

# @numba.njit(nogil=True, fastmath=True)
def Expanded_Scaffer_F6(x):

    def Scaffer_F6(x1, x2):
        t = x1*x1 + x2*x2
        return 0.5 + (pow(np.sin(np.sqrt(t)), 2)-0.5) / (1 + 0.001 * t*t)

    dim = x.shape[0]
    D_1 = numba.prange(dim-1)
    res = Scaffer_F6(x[dim-1], x[0])
    for i in D_1:
        res += Scaffer_F6(x[i], x[i+1])
    return res


# @numba.njit(nogil=True, fastmath=True)
def Expanded_Scaffer_F6_grad(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    a = x
    b = np.roll(x, -1)    # b[i] = x[i+1]
    t = a*a + b*b
    u = np.sqrt(t)

    # A and B
    A = np.sin(u)**2 - 0.5
    B = 1.0 + 0.001 * (t**2)

    # safe factor for sin(2u)/u: handle u==0 by limit -> 2.0
    # use where to avoid division by zero
    factor = np.empty_like(u)
    small = u == 0
    # when u==0, limit sin(2u)/u -> 2
    factor[small] = 2.0
    factor[~small] = np.sin(2.0 * u[~small]) / u[~small]

    # partial derivatives for each pair (w.r.t first and second arg)
    # dA/da = a * factor
    dA_da = a * factor
    dA_db = b * factor

    # dB/da = 0.004 * a * t  (since dB/dt = 0.002 t, dt/da = 2a => 0.002 t * 2a = 0.004 a t)
    dB_da = 0.004 * a * t
    dB_db = 0.004 * b * t

    # df/da = (dA_da * B - A * dB_da) / B^2
    denom = B * B
    df_da = (dA_da * B - A * dB_da) / denom
    df_db = (dA_db * B - A * dB_db) / denom

    # accumulate contributions:
    # each pair i contributes df_da[i] to grad[i] and df_db[i] to grad[i+1]
    grad = np.zeros_like(x)
    grad += df_da
    grad += np.roll(df_db, 1)   # shift so df_db[i] adds to grad[i+1] -> roll right by 1

    return grad

#-----------------------------------------正式函数-----------------------------------------------------------
# @numba.njit(nogil=True, fastmath=True)
def Rotated_High_Conditioned_Elliptic(x, M, o):
    M = np.ascontiguousarray(M)
    return High_Conditioned_Elliptic(np.dot(M, x - o)) + 100

# @numba.njit(nogil=True, fastmath=True)
def Rotated_High_Conditioned_Elliptic_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = M.T @ High_Conditioned_Elliptic_grad(np.dot(M, x - o))
    return np.atleast_1d(grad).reshape(-1)

# @numba.njit(nogil=True, fastmath=True)
def Rotated_Bent_Cigar(x, M, o):
    M = np.ascontiguousarray(M)
    return Bent_Cigar(np.dot(M, x - o)) + 200

# @numba.njit(nogil=True, fastmath=True)
def Rotated_Bent_Cigar_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = M.T @  Bent_Cigar_grad(np.dot(M, x - o))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Rotated_Discus(x, M, o):
    M = np.ascontiguousarray(M)
    return Discus(np.dot(M, x - o)) + 300

# @numba.njit(nogil=True, fastmath=True)
def Rotated_Discus_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = M.T @ Discus_grad(np.dot(M, x - o))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Rosenbrock(x, M, o):
    M = np.ascontiguousarray(M)
    return Rosenbrock(np.dot(M, 0.02048*(x - o)) + 1) + 400

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Rosenbrock_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 0.02048 * (M.T @ Rosenbrock_grad(np.dot(M, 0.02048*(x - o)) + 1))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Ackley(x, M, o):
    M = np.ascontiguousarray(M)
    return Ackley(np.dot(M, x-o)) + 500

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Ackley_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = M.T @ Ackley_grad(np.dot(M, x-o))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Weierstrass(x, M, o):
    M = np.ascontiguousarray(M)
    return Weierstrass(np.dot(M, 0.005*(x-o))) + 600

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Weierstrass_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 0.005 * (M.T @ Weierstrass_grad(np.dot(M, 0.005*(x-o))))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Griewank(x, M, o):
    M = np.ascontiguousarray(M)
    return Griewank(np.dot(M, 6*(x-o))) + 700

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Griewank_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 6 * (M.T @ Griewank_grad(np.dot(M, 6*(x-o))))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rastrigin(x, o):
    return Rastrigin(0.0512*(x-o)) + 800

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rastrigin_grad(x, o):
    return 0.0512 * Rastrigin_grad(0.0512*(x-o))


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Rastrigin(x, M, o):
    M = np.ascontiguousarray(M)
    return Rastrigin(np.dot(M, 0.0512*(x-o))) + 900

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Rastrigin_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 0.0512 * (M.T @ Rastrigin_grad(np.dot(M, 0.0512*(x-o))))
    return np.atleast_1d(grad).reshape(-1)

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Schwefel(x, o):
    return Modified_Schwefel(10*(x-o)) + 1000

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Schwefel_grad(x, o):
    return 10 * Modified_Schwefel_grad(10*(x-o))


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Schwefel(x, M, o):
    M = np.ascontiguousarray(M)
    return Modified_Schwefel(np.dot(M, 10*(x - o))) + 1100

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Schwefel_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 10 * (M.T @  Modified_Schwefel_grad(np.dot(M, 10*(x - o))))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Katsuura(x, M, o):
    M = np.ascontiguousarray(M)
    return Katsuura(np.dot(M, 0.05*(x-o))) + 1200

def Shifted_Rotated_Katsuura_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 0.05 * (M.T @ Katsuura_grad(np.dot(M, 0.05*(x-o))))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_HappyCat(x, M, o):
    M = np.ascontiguousarray(M)
    return HappyCat(np.dot(M, 0.05*(x-o))) + 1300

def Shifted_Rotated_HappyCat_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 0.05 * (M.T @ HappyCat_grad(np.dot(M, 0.05*(x-o))))
    return np.atleast_1d(grad).reshape(-1)

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_HGBat(x, M, o):
    M = np.ascontiguousarray(M)
    return HGBat(np.dot(M, 0.05*(x-o))) + 1400

def Shifted_Rotated_HGBat_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 0.05 * (M.T @ HGBat_grad(np.dot(M, 0.05*(x-o))))
    return np.atleast_1d(grad).reshape(-1)


# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock(x, M, o):
    M = np.ascontiguousarray(M)
    return Expended_Griewank_plus_Rosenbrock(np.dot(M, 0.05*(x-o))+1) + 1500

def Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = 0.05 * (M.T @ Expended_Griewank_plus_Rosenbrock_grad(np.dot(M, 0.05*(x-o))+1))
    return np.atleast_1d(grad).reshape(-1)

# @numba.njit(nogil=True, fastmath=True)
def Shifted_Rotated_Expanded_Scaffer_F6(x, M, o):
    M = np.ascontiguousarray(M)
    return Expanded_Scaffer_F6(np.dot(M, x-o)+1) + 1600

def Shifted_Rotated_Expanded_Scaffer_F6_grad(x, M, o):
    M = np.ascontiguousarray(M)
    grad = M.T @ Expanded_Scaffer_F6_grad(np.dot(M, x-o)+1)
    return np.atleast_1d(grad).reshape(-1)

# ================= Hybrid1 =================
class Hybrid1:
    def __init__(self, M, o, seed=2):
        """
        支持 M 为:
          - (nx, nx) 方阵
          - (cf_num * nx, nx) 垂直堆叠的 blocks（这是你当前生成方式）
          - 或者 (cf_num, nx, nx) 三维 blocks
        o 支持:
          - (cf_num * nx,) 平坦向量
          - (cf_num, nx) 矩阵
          - (nx,)（在 cf_num>1 时会复制到每个 block）
        """
        self.rng = np.random.default_rng(seed)
        M = np.asarray(M)
        o = np.asarray(o)

        # 保存原始形状，方便后面判断是否需要重建实例
        self.orig_M_shape = M.shape
        self.orig_o_shape = o.shape
        self.seed = seed

        # 识别并构建 blocks（每个 block 至少是 nx x nx）
        if M.ndim == 3:
            # 直接是 (cf_num, nx, nx)
            self.blocks = [M[i].copy() for i in range(M.shape[0])]
            self.cf_num = len(self.blocks)
            self.nx = self.blocks[0].shape[0]
        elif M.ndim == 2:
            rows, cols = M.shape
            if rows == cols:
                # 单 block 方阵
                self.cf_num = 1
                self.nx = rows
                self.blocks = [M.copy()]
            elif rows % cols == 0:
                # 垂直堆叠 (cf_num * nx, nx)
                self.cf_num = rows // cols
                self.nx = cols
                self.blocks = [M[i*self.nx:(i+1)*self.nx, :].copy() for i in range(self.cf_num)]
            else:
                raise ValueError(f"Unexpected M shape {M.shape} for Hybrid1")
        else:
            raise ValueError("M must be 2D or 3D array for Hybrid1")

        # 处理 o，统一成 (cf_num, nx)
        if o.ndim == 1 and o.size == self.cf_num * self.nx:
            self.o_blocks = o.reshape(self.cf_num, self.nx).copy()
        elif o.ndim == 2 and o.shape == (self.cf_num, self.nx):
            self.o_blocks = o.copy()
        elif o.ndim == 1 and o.size == self.nx and self.cf_num > 1:
            # 同一 shift 应用于所有子函数
            self.o_blocks = np.tile(o.reshape(1, self.nx), (self.cf_num, 1))
        elif self.cf_num == 1 and o.ndim == 1 and o.size == self.nx:
            self.o_blocks = o.reshape(1, self.nx).copy()
        else:
            # 兜底：用零向量（并提示）
            print(f"Warning: unexpected o shape {o.shape}. Using zeros for shifts in Hybrid1.")
            self.o_blocks = np.zeros((self.cf_num, self.nx))

        # 决策维度（真实的 D）
        self.dim = self.nx

        # permutation 仅基于真实维度 self.dim（避免越界）
        self.perm = self.rng.permutation(self.dim)

    def _get_slices(self):
        """返回三个切片（与原实现一致：0.3D, 0.3D, 剩余）"""
        idx1 = slice(0, int(0.3 * self.dim))
        idx2 = slice(int(0.3 * self.dim), int(0.6 * self.dim))
        idx3 = slice(int(0.6 * self.dim), self.dim)
        return idx1, idx2, idx3

    def f(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"Input x has wrong size {x.size}, expected {self.dim} for Hybrid1")

        x_perm = x[self.perm]
        idx1, idx2, idx3 = self._get_slices()
        res = 0.0

        # 每个子区间使用对应 block（若 block 数 < 子函数数，重复使用最后一个 block）
        blocks = self.blocks
        o_blocks = self.o_blocks

        # 子函数 1: Modified_Schwefel
        m = idx1.stop - idx1.start
        if m > 0:
            block = blocks[0] if len(blocks) > 0 else np.eye(self.nx)
            R = block[:m, :m]
            o_sub = o_blocks[0, :m]
            y = R @ (x_perm[idx1] - o_sub)
            res += Modified_Schwefel(y)

        # 子函数 2: Rastrigin
        m = idx2.stop - idx2.start
        if m > 0:
            block = blocks[1] if len(blocks) > 1 else blocks[-1]
            R = block[:m, :m]
            o_sub = o_blocks[1 if o_blocks.shape[0] > 1 else 0, :m]
            y = R @ (x_perm[idx2] - o_sub)
            res += Rastrigin(y)

        # 子函数 3: High_Conditioned_Elliptic
        m = idx3.stop - idx3.start
        if m > 0:
            block = blocks[2] if len(blocks) > 2 else blocks[-1]
            R = block[:m, :m]
            o_sub = o_blocks[2 if o_blocks.shape[0] > 2 else -1, :m]
            y = R @ (x_perm[idx3] - o_sub)
            res += High_Conditioned_Elliptic(y)

        return res + 1700

    def grad(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"Input x has wrong size {x.size}, expected {self.dim} for Hybrid1")

        x_perm = x[self.perm]
        idx1, idx2, idx3 = self._get_slices()

        grad_perm = np.zeros_like(x_perm)

        # 子函数 1 grad
        m = idx1.stop - idx1.start
        if m > 0:
            block = self.blocks[0] if len(self.blocks) > 0 else np.eye(self.nx)
            R = block[:m, :m]
            o_sub = self.o_blocks[0, :m]
            y = R @ (x_perm[idx1] - o_sub)
            g_sub = Modified_Schwefel_grad(y)
            grad_perm[idx1] = R.T @ g_sub

        # 子函数 2 grad
        m = idx2.stop - idx2.start
        if m > 0:
            block = self.blocks[1] if len(self.blocks) > 1 else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[1 if self.o_blocks.shape[0] > 1 else 0, :m]
            y = R @ (x_perm[idx2] - o_sub)
            g_sub = Rastrigin_grad(y)
            grad_perm[idx2] = R.T @ g_sub

        # 子函数 3 grad
        m = idx3.stop - idx3.start
        if m > 0:
            block = self.blocks[2] if len(self.blocks) > 2 else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[2 if self.o_blocks.shape[0] > 2 else -1, :m]
            y = R @ (x_perm[idx3] - o_sub)
            g_sub = High_Conditioned_Elliptic_grad(y)
            grad_perm[idx3] = R.T @ g_sub

        # 将 permuted gradient 恢复到原始维度顺序
        grad = np.zeros_like(x)
        grad[self.perm] = grad_perm
        return grad


# 全局单例，但在参数变化时会重建（避免旧实例 shape 残留导致的问题）
_hybrid1_instance = None
def Hybrid_1(x, M, o, seed=2):
    global _hybrid1_instance
    M = np.asarray(M); o = np.asarray(o)
    need_new = False
    if _hybrid1_instance is None:
        need_new = True
    else:
        if _hybrid1_instance.orig_M_shape != M.shape or _hybrid1_instance.orig_o_shape != o.shape or _hybrid1_instance.seed != seed:
            need_new = True
    if need_new:
        _hybrid1_instance = Hybrid1(M, o, seed=seed)
    return _hybrid1_instance.f(x)

def Hybrid_1_grad(x, M, o, seed=2):
    global _hybrid1_instance
    M = np.asarray(M); o = np.asarray(o)
    need_new = False
    if _hybrid1_instance is None:
        need_new = True
    else:
        if _hybrid1_instance.orig_M_shape != M.shape or _hybrid1_instance.orig_o_shape != o.shape or _hybrid1_instance.seed != seed:
            need_new = True
    if need_new:
        _hybrid1_instance = Hybrid1(M, o, seed=seed)
    return _hybrid1_instance.grad(x)

# ---------- 工具：从 M/o 构建 blocks 与 o_blocks ----------
def _parse_blocks_and_shifts(M, o):
    M = np.asarray(M)
    o = np.asarray(o)
    # parse M into list of blocks and determine cf_num and nx
    if M.ndim == 3:
        blocks = [M[i].copy() for i in range(M.shape[0])]
        cf_num = M.shape[0]
        nx = M.shape[1]
    elif M.ndim == 2:
        rows, cols = M.shape
        if rows == cols:
            cf_num = 1
            nx = rows
            blocks = [M.copy()]
        elif rows % cols == 0:
            cf_num = rows // cols
            nx = cols
            blocks = [M[i*nx:(i+1)*nx, :].copy() for i in range(cf_num)]
        else:
            raise ValueError(f"Unexpected M shape {M.shape}")
    else:
        raise ValueError("M must be 2D or 3D array")

    # parse o into (cf_num, nx)
    if o.ndim == 1 and o.size == cf_num * nx:
        o_blocks = o.reshape(cf_num, nx).copy()
    elif o.ndim == 2 and o.shape == (cf_num, nx):
        o_blocks = o.copy()
    elif o.ndim == 1 and o.size == nx and cf_num > 1:
        o_blocks = np.tile(o.reshape(1, nx), (cf_num, 1))
    elif cf_num == 1 and o.ndim == 1 and o.size == nx:
        o_blocks = o.reshape(1, nx).copy()
    else:
        # 兜底：用零 shift（并打印警告）
        print(f"Warning: unexpected o shape {o.shape} for parsed blocks (cf_num={cf_num}, nx={nx}). Using zeros.")
        o_blocks = np.zeros((cf_num, nx))

    return blocks, o_blocks, cf_num, nx

# ================= Hybrid2 =================
class Hybrid2:
    def __init__(self, M, o, seed=2):
        self.blocks, self.o_blocks, self.cf_num, self.nx = _parse_blocks_and_shifts(M, o)
        self.dim = self.nx
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.dim)

    def _slice_indices(self):
        return (slice(0, int(0.3*self.dim)),
                slice(int(0.3*self.dim), int(0.6*self.dim)),
                slice(int(0.6*self.dim), self.dim))

    def f(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid2")

        x_perm = x[self.perm]
        idx1, idx2, idx3 = self._slice_indices()
        res = 0.0

        # 子区间 1
        m = idx1.stop - idx1.start
        if m > 0:
            block = self.blocks[0] if len(self.blocks) > 0 else np.eye(self.nx)
            R = block[:m, :m]
            o_sub = self.o_blocks[0, :m]
            y = R @ (x_perm[idx1] - o_sub)
            res += Bent_Cigar(y)

        # 子区间 2
        m = idx2.stop - idx2.start
        if m > 0:
            block = self.blocks[1] if len(self.blocks) > 1 else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[1 if self.o_blocks.shape[0] > 1 else 0, :m]
            y = R @ (x_perm[idx2] - o_sub)
            res += HGBat(y)

        # 子区间 3
        m = idx3.stop - idx3.start
        if m > 0:
            block = self.blocks[2] if len(self.blocks) > 2 else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[2 if self.o_blocks.shape[0] > 2 else -1, :m]
            y = R @ (x_perm[idx3] - o_sub)
            res += Rastrigin(y)

        return res + 1800

    def grad(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid2")
        x_perm = x[self.perm]
        idx1, idx2, idx3 = self._slice_indices()
        grad_perm = np.zeros_like(x_perm)

        # grad 子1
        m = idx1.stop - idx1.start
        if m > 0:
            block = self.blocks[0] if len(self.blocks) > 0 else np.eye(self.nx)
            R = block[:m, :m]
            o_sub = self.o_blocks[0, :m]
            y = R @ (x_perm[idx1] - o_sub)
            grad_perm[idx1] = R.T @ Bent_Cigar_grad(y)

        # grad 子2
        m = idx2.stop - idx2.start
        if m > 0:
            block = self.blocks[1] if len(self.blocks) > 1 else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[1 if self.o_blocks.shape[0] > 1 else 0, :m]
            y = R @ (x_perm[idx2] - o_sub)
            grad_perm[idx2] = R.T @ HGBat_grad(y)

        # grad 子3
        m = idx3.stop - idx3.start
        if m > 0:
            block = self.blocks[2] if len(self.blocks) > 2 else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[2 if self.o_blocks.shape[0] > 2 else -1, :m]
            y = R @ (x_perm[idx3] - o_sub)
            grad_perm[idx3] = R.T @ Rastrigin_grad(y)

        grad = np.zeros_like(x)
        grad[self.perm] = grad_perm
        return grad

def Hybrid_2(x, M, o, seed=2):
    return Hybrid2(M, o, seed=seed).f(x)
def Hybrid_2_grad(x, M, o, seed=2):
    return Hybrid2(M, o, seed=seed).grad(x)


# ================= Hybrid3 =================
class Hybrid3:
    def __init__(self, M, o, seed=2):
        self.blocks, self.o_blocks, self.cf_num, self.nx = _parse_blocks_and_shifts(M, o)
        self.dim = self.nx
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.dim)

    def _slice_indices(self):
        return (slice(0, int(0.2*self.dim)),
                slice(int(0.2*self.dim), int(0.4*self.dim)),
                slice(int(0.4*self.dim), int(0.7*self.dim)),
                slice(int(0.7*self.dim), self.dim))

    def f(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid3")
        x_perm = x[self.perm]
        idx1, idx2, idx3, idx4 = self._slice_indices()
        res = 0.0

        # child 1
        m = idx1.stop - idx1.start
        if m > 0:
            block = self.blocks[0] if len(self.blocks) > 0 else np.eye(self.nx)
            R = block[:m, :m]; o_sub = self.o_blocks[0, :m]
            res += Griewank(R @ (x_perm[idx1] - o_sub))

        # child 2
        m = idx2.stop - idx2.start
        if m > 0:
            block = self.blocks[1] if len(self.blocks) > 1 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[1 if self.o_blocks.shape[0]>1 else 0, :m]
            res += Weierstrass(R @ (x_perm[idx2] - o_sub))

        # child 3
        m = idx3.stop - idx3.start
        if m > 0:
            block = self.blocks[2] if len(self.blocks) > 2 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[2 if self.o_blocks.shape[0]>2 else -1, :m]
            res += Rosenbrock(R @ (x_perm[idx3] - o_sub))

        # child 4
        m = idx4.stop - idx4.start
        if m > 0:
            block = self.blocks[3] if len(self.blocks) > 3 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[3 if self.o_blocks.shape[0]>3 else -1, :m]
            res += Expanded_Scaffer_F6(R @ (x_perm[idx4] - o_sub))

        return res + 1900

    def grad(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid3")
        x_perm = x[self.perm]
        idx1, idx2, idx3, idx4 = self._slice_indices()
        grad_perm = np.zeros_like(x_perm)

        # grad parts analogous to f
        m = idx1.stop - idx1.start
        if m > 0:
            block = self.blocks[0] if len(self.blocks) > 0 else np.eye(self.nx)
            R = block[:m,:m]; o_sub = self.o_blocks[0,:m]
            y = R @ (x_perm[idx1] - o_sub)
            grad_perm[idx1] = R.T @ Griewank_grad(y)

        m = idx2.stop - idx2.start
        if m > 0:
            block = self.blocks[1] if len(self.blocks) > 1 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[1 if self.o_blocks.shape[0]>1 else 0,:m]
            y = R @ (x_perm[idx2] - o_sub)
            grad_perm[idx2] = R.T @ Weierstrass_grad(y)

        m = idx3.stop - idx3.start
        if m > 0:
            block = self.blocks[2] if len(self.blocks) > 2 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[2 if self.o_blocks.shape[0]>2 else -1,:m]
            y = R @ (x_perm[idx3] - o_sub)
            grad_perm[idx3] = R.T @ Rosenbrock_grad(y)

        m = idx4.stop - idx4.start
        if m > 0:
            block = self.blocks[3] if len(self.blocks) > 3 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[3 if self.o_blocks.shape[0]>3 else -1,:m]
            y = R @ (x_perm[idx4] - o_sub)
            grad_perm[idx4] = R.T @ Expanded_Scaffer_F6_grad(y)

        grad = np.zeros_like(x)
        grad[self.perm] = grad_perm
        return grad

def Hybrid_3(x, M, o, seed=2):
    return Hybrid3(M, o, seed=seed).f(x)
def Hybrid_3_grad(x, M, o, seed=2):
    return Hybrid3(M, o, seed=seed).grad(x)


# ================= Hybrid4 =================
class Hybrid4:
    def __init__(self, M, o, seed=2):
        self.blocks, self.o_blocks, self.cf_num, self.nx = _parse_blocks_and_shifts(M, o)
        self.dim = self.nx
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.dim)

    def _slice_indices(self):
        return (slice(0, int(0.2*self.dim)),
                slice(int(0.2*self.dim), int(0.4*self.dim)),
                slice(int(0.4*self.dim), int(0.7*self.dim)),
                slice(int(0.7*self.dim), self.dim))

    def f(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid4")
        x_perm = x[self.perm]
        idx1, idx2, idx3, idx4 = self._slice_indices()
        res = 0.0

        # 子1
        m = idx1.stop - idx1.start
        if m > 0:
            block = self.blocks[0] if len(self.blocks)>0 else np.eye(self.nx)
            R = block[:m,:m]; o_sub = self.o_blocks[0,:m]
            res += HGBat(R @ (x_perm[idx1] - o_sub))

        # 子2
        m = idx2.stop - idx2.start
        if m > 0:
            block = self.blocks[1] if len(self.blocks)>1 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[1 if self.o_blocks.shape[0]>1 else 0,:m]
            res += Discus(R @ (x_perm[idx2] - o_sub))

        # 子3
        m = idx3.stop - idx3.start
        if m > 0:
            block = self.blocks[2] if len(self.blocks)>2 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[2 if self.o_blocks.shape[0]>2 else -1,:m]
            res += Expended_Griewank_plus_Rosenbrock(R @ (x_perm[idx3] - o_sub))

        # 子4
        m = idx4.stop - idx4.start
        if m > 0:
            block = self.blocks[3] if len(self.blocks)>3 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[3 if self.o_blocks.shape[0]>3 else -1,:m]
            res += Rastrigin(R @ (x_perm[idx4] - o_sub))

        return res + 2000

    def grad(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid4")
        x_perm = x[self.perm]
        idx1, idx2, idx3, idx4 = self._slice_indices()
        grad_perm = np.zeros_like(x_perm)

        m = idx1.stop - idx1.start
        if m > 0:
            block = self.blocks[0] if len(self.blocks)>0 else np.eye(self.nx)
            R = block[:m,:m]; o_sub = self.o_blocks[0,:m]
            y = R @ (x_perm[idx1] - o_sub); grad_perm[idx1] = R.T @ HGBat_grad(y)

        m = idx2.stop - idx2.start
        if m > 0:
            block = self.blocks[1] if len(self.blocks)>1 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[1 if self.o_blocks.shape[0]>1 else 0,:m]
            y = R @ (x_perm[idx2] - o_sub); grad_perm[idx2] = R.T @ Discus_grad(y)

        m = idx3.stop - idx3.start
        if m > 0:
            block = self.blocks[2] if len(self.blocks)>2 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[2 if self.o_blocks.shape[0]>2 else -1,:m]
            y = R @ (x_perm[idx3] - o_sub); grad_perm[idx3] = R.T @ Expended_Griewank_plus_Rosenbrock_grad(y)

        m = idx4.stop - idx4.start
        if m > 0:
            block = self.blocks[3] if len(self.blocks)>3 else self.blocks[-1]
            R = block[:m,:m]; o_sub = self.o_blocks[3 if self.o_blocks.shape[0]>3 else -1,:m]
            y = R @ (x_perm[idx4] - o_sub); grad_perm[idx4] = R.T @ Rastrigin_grad(y)

        grad = np.zeros_like(x)
        grad[self.perm] = grad_perm
        return grad

def Hybrid_4(x, M, o, seed=2):
    return Hybrid4(M, o, seed=seed).f(x)
def Hybrid_4_grad(x, M, o, seed=2):
    return Hybrid4(M, o, seed=seed).grad(x)


# ================= Hybrid5 =================
class Hybrid5:
    def __init__(self, M, o, seed=2):
        self.blocks, self.o_blocks, self.cf_num, self.nx = _parse_blocks_and_shifts(M, o)
        self.dim = self.nx
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.dim)

    def _slice_indices(self):
        return (slice(0, int(0.1*self.dim)),
                slice(int(0.1*self.dim), int(0.3*self.dim)),
                slice(int(0.3*self.dim), int(0.5*self.dim)),
                slice(int(0.5*self.dim), int(0.7*self.dim)),
                slice(int(0.7*self.dim), self.dim))

    def f(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid5")
        x_perm = x[self.perm]
        idxs = self._slice_indices()
        res = 0.0
        # loop through parts for brevity
        funcs = [Expanded_Scaffer_F6, HGBat, Rosenbrock, Modified_Schwefel, High_Conditioned_Elliptic]
        for i, idx in enumerate(idxs):
            m = idx.stop - idx.start
            if m <= 0:
                continue
            block = self.blocks[i] if i < len(self.blocks) else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[i if i < self.o_blocks.shape[0] else -1, :m]
            y = R @ (x_perm[idx] - o_sub)
            res += funcs[i](y)
        return res + 2100

    def grad(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid5")
        x_perm = x[self.perm]
        idxs = self._slice_indices()
        grad_perm = np.zeros_like(x_perm)
        grads = [Expanded_Scaffer_F6_grad, HGBat_grad, Rosenbrock_grad, Modified_Schwefel_grad, High_Conditioned_Elliptic_grad]
        for i, idx in enumerate(idxs):
            m = idx.stop - idx.start
            if m <= 0:
                continue
            block = self.blocks[i] if i < len(self.blocks) else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[i if i < self.o_blocks.shape[0] else -1, :m]
            y = R @ (x_perm[idx] - o_sub)
            grad_perm[idx] = R.T @ grads[i](y)
        grad = np.zeros_like(x)
        grad[self.perm] = grad_perm
        return grad

def Hybrid_5(x, M, o, seed=2):
    return Hybrid5(M, o, seed=seed).f(x)
def Hybrid_5_grad(x, M, o, seed=2):
    return Hybrid5(M, o, seed=seed).grad(x)


# ================= Hybrid6 =================
class Hybrid6:
    def __init__(self, M, o, seed=2):
        self.blocks, self.o_blocks, self.cf_num, self.nx = _parse_blocks_and_shifts(M, o)
        self.dim = self.nx
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.dim)

    def _slice_indices(self):
        return (slice(0, int(0.1*self.dim)),
                slice(int(0.1*self.dim), int(0.3*self.dim)),
                slice(int(0.3*self.dim), int(0.5*self.dim)),
                slice(int(0.5*self.dim), int(0.7*self.dim)),
                slice(int(0.7*self.dim), self.dim))

    def f(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid6")
        x_perm = x[self.perm]
        idxs = self._slice_indices()
        res = 0.0
        funcs = [Katsuura, HappyCat, Expended_Griewank_plus_Rosenbrock, Modified_Schwefel, Ackley]
        for i, idx in enumerate(idxs):
            m = idx.stop - idx.start
            if m <= 0:
                continue
            block = self.blocks[i] if i < len(self.blocks) else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[i if i < self.o_blocks.shape[0] else -1, :m]
            y = R @ (x_perm[idx] - o_sub)
            res += funcs[i](y)
        return res + 2200

    def grad(self, x):
        x = np.asarray(x).flatten()
        if x.size != self.dim:
            raise ValueError(f"x size {x.size} != expected {self.dim} in Hybrid6")
        x_perm = x[self.perm]
        idxs = self._slice_indices()
        grad_perm = np.zeros_like(x_perm)
        grads = [Katsuura_grad, HappyCat_grad, Expended_Griewank_plus_Rosenbrock_grad, Modified_Schwefel_grad, Ackley_grad]
        for i, idx in enumerate(idxs):
            m = idx.stop - idx.start
            if m <= 0:
                continue
            block = self.blocks[i] if i < len(self.blocks) else self.blocks[-1]
            R = block[:m, :m]
            o_sub = self.o_blocks[i if i < self.o_blocks.shape[0] else -1, :m]
            y = R @ (x_perm[idx] - o_sub)
            grad_perm[idx] = R.T @ grads[i](y)
        grad = np.zeros_like(x)
        grad[self.perm] = grad_perm
        return grad

def Hybrid_6(x, M, o, seed=2):
    return Hybrid6(M, o, seed=seed).f(x)
def Hybrid_6_grad(x, M, o, seed=2):
    return Hybrid6(M, o, seed=seed).grad(x)


# ---------------------------------------------------------------------------
# @numba.njit(nogil=True, fastmath=True)
def composition_omega(x_o, sig):
    n, dim = x_o.shape
    w = np.zeros(n)
    N = numba.prange(n)
    for i in N:
        c_x = np.ascontiguousarray(x_o[i, :])
        t = np.dot(c_x, c_x)
        w[i] = 1/np.sqrt(t)*np.exp(-t / (2 * dim * sig[i]))

    return w/np.sum(w)


def composition_omega_and_grad(x, o, sig, eps=1e-12, debug=False):
    """
    稳健版本：计算组合权重 w 和它们关于 x 的梯度 grad_w。
    支持 o 为 1D (n*d,) 或 2D (n, d)。
    返回:
      w: shape (n,)
      grad_w: shape (n, d)
    """
    x = np.asarray(x, dtype=float).flatten()
    dim = x.size

    # 规范化并验证 o
    o = np.asarray(o, dtype=float)
    if o.ndim == 1:
        if o.size % dim != 0:
            raise ValueError(f"o length {o.size} is not divisible by x dimension {dim}")
        n = o.size // dim
        o_2d = o.reshape(n, dim)
    elif o.ndim == 2:
        o_2d = o
        n = o_2d.shape[0]
        if o_2d.shape[1] != dim:
            raise ValueError(f"o has shape {o.shape} but x has dim {dim}")
    else:
        raise ValueError("o must be 1D or 2D array")

    # 规范化 sig
    sig = np.asarray(sig, dtype=float).flatten()
    if sig.size == 1:
        sig = np.full(n, float(sig))
    if sig.size != n:
        raise ValueError(f"sig length {sig.size} does not match number of components {n}")
    # 防止 sig 中为 0 或负值
    sig_safe = np.where(sig <= 0, eps, sig)

    # 计算 diff 和 t = ||x - o_i||^2
    diffs = x[None, :] - o_2d    # shape (n, dim)
    t = np.sum(diffs * diffs, axis=1)   # shape (n,)
    # 防止 t 为 0
    t_safe = np.where(t <= eps, eps, t)

    # 计算 log(alpha)：log(alpha) = -0.5*log(t) - t/(2*d*sig)
    # 使用 log 空间做数值稳定化（再做 shift）
    log_alpha = -0.5 * np.log(t_safe) - (t_safe / (2.0 * dim * sig_safe))

    # 稳定化：减去最大值再 exp
    max_log = np.max(log_alpha)
    scaled_alpha = np.exp(log_alpha - max_log)   # 仍为 >=0，按比例缩放的 alpha

    Z = np.sum(scaled_alpha)
    if not np.isfinite(Z) or Z <= eps:
        # 极端退化情况：返回均匀权重与零梯度（比返回 NaN 更稳健）
        if debug:
            print("composition_omega_and_grad: degenerate Z, using fallback. Z =", Z)
        w = np.ones(n) / n
        grad_w = np.zeros((n, dim), dtype=float)
        return w, grad_w

    # 计算 grad_alpha (按同一缩放比例)
    # 理论上 grad_alpha = alpha * (-diff * (1/t + 1/(d*sig)))
    inv_t = 1.0 / t_safe                     # shape (n,)
    inv_ds = 1.0 / (dim * sig_safe)         # shape (n,)
    factor = inv_t + inv_ds                 # shape (n,)
    h = - diffs * factor[:, None]           # shape (n, dim)
    grad_alpha_scaled = scaled_alpha[:, None] * h   # shape (n, dim)

    sum_grad_alpha_scaled = np.sum(grad_alpha_scaled, axis=0)   # shape (dim,)

    # 计算 w 和 grad_w（注意我们使用 scaled_alpha，缩放对最终 grad_w 无影响）
    w = scaled_alpha / Z                     # shape (n,)
    # 公式：grad_w = (grad_alpha * Z - alpha[:,None] * sum_grad_alpha[None,:]) / (Z*Z)
    # 使用 scaled_alpha / scaled quantities：
    Z_safe = Z if np.isfinite(Z) and Z > eps else eps
    grad_w = (grad_alpha_scaled * Z_safe - (scaled_alpha[:, None] * sum_grad_alpha_scaled[None, :])) / (Z_safe * Z_safe)

    # 防护：将非常小或非有限的元素替换为 0
    grad_w = np.where(np.isfinite(grad_w), grad_w, 0.0)

    if debug:
        print("composition_omega_and_grad debug:")
        print(" n =", n, "dim =", dim)
        print(" min(t)=", np.min(t), "max(t)=", np.max(t))
        print(" min(sig)=", np.min(sig_safe), "max(sig)=", np.max(sig_safe))
        print(" Z =", Z, "w.min,max =", w.min(), w.max())
        print(" any nan in grad_w?", np.any(np.isnan(grad_w)))

    return w.astype(float), grad_w.astype(float)


def Composition_1(x, M, o):
    """
    修正版本：处理正确形状的输入
    """
    x = np.asarray(x).flatten()
    dim = len(x)

    # 确保 o 是 2D 数组 (5, dim)
    if o.ndim == 1:
        if len(o) == 5 * dim:
            o = o.reshape(5, dim)
        else:
            # 如果维度不匹配，创建默认值
            o = np.zeros((5, dim))

    # 确保 M 是 2D 数组 (5*dim, dim)
    if M.shape != (5 * dim, dim):
        # 如果维度不匹配，创建默认值
        blocks = [np.eye(dim) for _ in range(5)]
        M = np.vstack(blocks)

    # 修正：使用正确的维度计算
    # composition_omega 需要 2D 输入
    x_2d = x.reshape(1, -1)  # (1, dim)

    # 计算每个子函数的偏移差异
    diffs = x_2d - o  # 广播到 (5, dim)

    omega = composition_omega(diffs, np.square(np.array([10, 20, 30, 40, 50])))

    # 计算加权和
    res = omega[0] * (Shifted_Rotated_Rosenbrock(x, M[0:dim, :], o[0, :]) - 400)
    res += omega[1] * (1e-6 * Rotated_High_Conditioned_Elliptic(x, M[dim:dim * 2, :], o[1, :]) + 99.9999)
    res += omega[2] * (1e-26 * Rotated_Bent_Cigar(x, M[dim * 2:dim * 3, :], o[2, :]) - 2e-24 + 200)
    res += omega[3] * (1e-6 * Rotated_Discus(x, M[dim * 3:dim * 4, :], o[3, :]) - 3e-4 + 300)
    res += omega[4] * (1e-6 * Rotated_High_Conditioned_Elliptic(x, M[dim * 4:dim * 5, :], o[4, :]) + 399.9999)

    return res + 2300

# def Composition_1_grad(x, M, o):
#     dim = x.shape[0]
#     sig = np.square(np.array([10, 20, 30, 40, 50]))
#     w, grad_w = composition_omega_and_grad(x, o, sig)
#
#     funcs = []
#     grads = []
#
#     # f1
#     f1 = Shifted_Rotated_Rosenbrock(x, M[0:dim, :], o[0]) - 400
#     g1 = Shifted_Rotated_Rosenbrock_grad(x, M[0:dim, :], o[0])
#     funcs.append(f1); grads.append(g1)
#
#     # f2
#     f2 = 1e-6 * Rotated_High_Conditioned_Elliptic(x, M[dim:2*dim, :], o[1]) + 99.9999
#     g2 = 1e-6 * Rotated_High_Conditioned_Elliptic_grad(x, M[dim:2*dim, :], o[1])
#     funcs.append(f2); grads.append(g2)
#
#     # f3
#     f3 = 1e-26 * Rotated_Bent_Cigar(x, M[2*dim:3*dim, :], o[2]) - 2e-24 + 200
#     g3 = 1e-26 * Rotated_Bent_Cigar_grad(x, M[2*dim:3*dim, :], o[2])
#     funcs.append(f3); grads.append(g3)
#
#     # f4
#     f4 = 1e-6 * Rotated_Discus(x, M[3*dim:4*dim, :], o[3]) - 3e-4 + 300
#     g4 = 1e-6 * Rotated_Discus_grad(x, M[3*dim:4*dim, :], o[3])
#     funcs.append(f4); grads.append(g4)
#
#     # f5
#     f5 = 1e-6 * Rotated_High_Conditioned_Elliptic(x, M[4*dim:5*dim, :], o[4]) + 399.9999
#     g5 = 1e-6 * Rotated_High_Conditioned_Elliptic_grad(x, M[4*dim:5*dim, :], o[4])
#     funcs.append(f5); grads.append(g5)
#
#     grad_total = np.zeros(dim)
#     for i in range(5):
#         grad_total += grad_w[i] * funcs[i] + w[i] * grads[i]

    # return grad_total
def Composition_1_grad(x, M, o):
    dim = x.shape[0]

    # 确保 o 是 2D 数组 (5, dim)
    if o.ndim == 1:
        if len(o) == 5 * dim:
            o_2d = o.reshape(5, dim)
        else:
            # 如果维度不匹配，创建默认值
            o_2d = np.zeros((5, dim))
    else:
        o_2d = o

    # 确保 M 是 2D 数组 (5*dim, dim)
    if M.shape != (5 * dim, dim):
        # 如果维度不匹配，创建默认值
        blocks = [np.eye(dim) for _ in range(5)]
        M_reshaped = np.vstack(blocks)
    else:
        M_reshaped = M

    sig = np.square(np.array([10, 20, 30, 40, 50]))
    w, grad_w = composition_omega_and_grad(x, o_2d, sig)

    funcs = []
    grads = []

    # f1
    f1 = Shifted_Rotated_Rosenbrock(x, M_reshaped[0:dim, :], o_2d[0]) - 400
    g1 = Shifted_Rotated_Rosenbrock_grad(x, M_reshaped[0:dim, :], o_2d[0])
    funcs.append(f1);
    grads.append(g1)

    # f2
    f2 = 1e-6 * Rotated_High_Conditioned_Elliptic(x, M_reshaped[dim:2 * dim, :], o_2d[1]) + 99.9999
    g2 = 1e-6 * Rotated_High_Conditioned_Elliptic_grad(x, M_reshaped[dim:2 * dim, :], o_2d[1])
    funcs.append(f2);
    grads.append(g2)

    # f3
    f3 = 1e-26 * Rotated_Bent_Cigar(x, M_reshaped[2 * dim:3 * dim, :], o_2d[2]) - 2e-24 + 200
    g3 = 1e-26 * Rotated_Bent_Cigar_grad(x, M_reshaped[2 * dim:3 * dim, :], o_2d[2])
    funcs.append(f3);
    grads.append(g3)

    # f4
    f4 = 1e-6 * Rotated_Discus(x, M_reshaped[3 * dim:4 * dim, :], o_2d[3]) - 3e-4 + 300
    g4 = 1e-6 * Rotated_Discus_grad(x, M_reshaped[3 * dim:4 * dim, :], o_2d[3])
    funcs.append(f4);
    grads.append(g4)

    # f5
    f5 = 1e-6 * Rotated_High_Conditioned_Elliptic(x, M_reshaped[4 * dim:5 * dim, :], o_2d[4]) + 399.9999
    g5 = 1e-6 * Rotated_High_Conditioned_Elliptic_grad(x, M_reshaped[4 * dim:5 * dim, :], o_2d[4])
    funcs.append(f5);
    grads.append(g5)

    grad_total = np.zeros(dim)
    for i in range(5):
        grad_total += grad_w[i] * funcs[i] + w[i] * grads[i]

    return grad_total

# @numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_2(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([20, 20, 20])))
    res = omega[0] * (Shifted_Schwefel(x, o[0, :]) - 1000)
    res += omega[1] * (Shifted_Rotated_Rastrigin(x, M[0:dim, :], o[1, :]) - 800)
    res += omega[2] * (Shifted_Rotated_HGBat(x, M[dim:dim*2, :], o[2, :]) - 1200)
    return res + 2400


def Composition_2_grad(x, M, o):
    x = np.asarray(x, dtype=float)
    dim = x.size
    # sig for Composition_2: [20,20,20] squared in original call -> 400 each
    sig = np.array([20.0, 20.0, 20.0])**2

    # weights and their gradients
    w, grad_w = composition_omega_and_grad(x, o, sig)
    # NOTE: your original composition_omega call used x - o as first arg; here composition_omega_and_grad expects x and o,
    # so we pass x and o directly; composition_omega_and_grad uses diff = x - o[i] internally.
    # If your composition_omega expects x_o = x-o stacked, adjust accordingly.

    # compute f_i and ∇f_i
    grads = [None]*3
    fvals = np.zeros(3, dtype=float)

    # f1: Shifted_Schwefel(x, o[0]) - 1000
    diff0 = x - o[0]
    f1 = Modified_Schwefel(diff0) - 1000.0
    g1 = Modified_Schwefel_grad(diff0)   # gradient wrt diff0, dr/dx = 1
    fvals[0] = f1
    grads[0] = g1

    # f2: Shifted_Rotated_Rastrigin -> y = M_block @ (x - o[1]); f = Rastrigin(y) - 800
    M2 = np.ascontiguousarray(M[0:dim, :])
    diff1 = x - o[1]
    y2 = M2 @ diff1
    f2 = Rastrigin(y2) - 800.0
    g_y2 = Rastrigin_grad(y2)            # gradient wrt y
    g2 = M2.T @ g_y2                     # chain rule
    fvals[1] = f2
    grads[1] = g2

    # f3: Shifted_Rotated_HGBat -> y = M_block @ (x - o[2]); f = HGBat(y) - 1200
    M3 = np.ascontiguousarray(M[dim:2*dim, :])
    diff2 = x - o[2]
    y3 = M3 @ diff2
    f3 = HGBat(y3) - 1200.0
    g_y3 = HGBat_grad(y3)                 # gradient wrt y
    g3 = M3.T @ g_y3
    fvals[2] = f3
    grads[2] = g3

    # assemble total gradient
    grad_total = np.zeros(dim, dtype=float)
    for i in range(3):
        grad_total += grad_w[i] * fvals[i] + w[i] * grads[i]

    return grad_total



# @numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_3(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0] * (0.25*Shifted_Rotated_Schwefel(x, M[0:dim, :], o[0, :])-275)
    res += omega[1] * (Shifted_Rotated_Rastrigin(x, M[dim:dim*2, :], o[1, :])-800)
    res += omega[2] * (1e-7*Rotated_High_Conditioned_Elliptic(x, M[dim*2:dim*3, :], o[2, :])-1e-5 + 200)
    return res + 2500


def Composition_3_grad(x, M, o):
    """
    x: (d,)
    M: stacked blocks, expected shape (3*d, d) where blocks are:
       M[0:d, :]   -> block for component 0 (Schwefel)
       M[d:2*d, :] -> block for component 1 (Rastrigin)
       M[2*d:3*d, :] -> block for component 2 (Elliptic)
    o: (3, d)
    """
    x = np.asarray(x, dtype=float)
    dim = x.size

    # sig = [10^2, 30^2, 50^2]
    sig = np.array([10.0, 30.0, 50.0])**2

    # compute weights and their gradients
    w, grad_w = composition_omega_and_grad(x, o, sig)

    # prepare arrays
    fvals = np.zeros(3, dtype=float)
    grads = [None, None, None]   # each is a vector of length dim

    # --- component 0 : 0.25 * Shifted_Rotated_Schwefel - 275 ---
    M0 = np.ascontiguousarray(M[0:dim, :])
    diff0 = x - o[0]
    y0 = M0 @ diff0
    f0 = 0.25 * Modified_Schwefel(y0) - 275.0
    # gradient wrt y0:
    gy0 = 0.25 * Modified_Schwefel_grad(y0)   # shape (d,)
    # chain rule back to x:
    g0 = M0.T @ gy0
    fvals[0] = f0
    grads[0] = g0

    # --- component 1 : Shifted_Rotated_Rastrigin - 800 ---
    M1 = np.ascontiguousarray(M[dim:2*dim, :])
    diff1 = x - o[1]
    y1 = M1 @ diff1
    f1 = Rastrigin(y1) - 800.0
    gy1 = Rastrigin_grad(y1)     # gradient wrt y1
    g1 = M1.T @ gy1
    fvals[1] = f1
    grads[1] = g1

    # --- component 2 : 1e-7 * Rotated High Conditioned Elliptic -1e-5 +200 ---
    M2 = np.ascontiguousarray(M[2*dim:3*dim, :])
    diff2 = x - o[2]
    y2 = M2 @ diff2
    f2 = 1e-7 * High_Conditioned_Elliptic(y2) - 1e-5 + 200.0
    gy2 = 1e-7 * High_Conditioned_Elliptic_grad(y2)
    g2 = M2.T @ gy2
    fvals[2] = f2
    grads[2] = g2

    # assemble total gradient
    grad_total = np.zeros(dim, dtype=float)
    for i in range(3):
        # grad_w[i] is vector shape (d,)
        grad_total += grad_w[i] * fvals[i] + w[i] * grads[i]

    return grad_total


# @numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_4(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 10, 10, 10, 10])))
    res = omega[0] * (0.25*Shifted_Rotated_Schwefel(x, M[0:dim, :], o[0, :])-275)
    res += omega[1] * (Shifted_Rotated_HappyCat(x, M[dim:dim*2, :], o[1, :])-1200)
    res += omega[2] * (1e-7*Rotated_High_Conditioned_Elliptic(x, M[dim*2:dim*3, :], o[2, :])+199.99999)
    res += omega[3] * (2.5*Shifted_Rotated_Weierstrass(x, M[dim*3:dim*4, :], o[3, :])-1200)
    res += omega[4] * (10*Shifted_Rotated_Griewank(x, M[dim*4:dim*5, :], o[4, :])-6600)
    return res + 2600


def Composition_4_grad(x, M, o):
    """
    x: (d,)
    M: stacked blocks, shape at least (5*d, d), blocks M[i] = M[i*d:(i+1)*d, :]
    o: (5, d)
    """
    x = np.asarray(x, dtype=float)
    dim = x.size

    # sigs: all 10^2
    sig = np.array([10.0]*5)**2

    # weights and their gradients (composition_omega uses x-o internally; here we pass x and o)
    w, grad_w = composition_omega_and_grad(x, o, sig)

    # prepare container for f_i and grad_i
    fvals = np.zeros(5, dtype=float)
    grads = [None]*5

    # block utility
    def M_block(i):
        return np.ascontiguousarray(M[i*dim:(i+1)*dim, :])

    # ---- component 0: 0.25 * Shifted_Rotated_Schwefel - 275 ----
    M0 = M_block(0)
    diff0 = x - o[0]
    y0 = M0 @ diff0
    fvals[0] = 0.25 * Modified_Schwefel(y0) - 275.0
    gy0 = 0.25 * Modified_Schwefel_grad(y0)   # gradient wrt y0
    grads[0] = M0.T @ gy0                      # back to x

    # ---- component 1: Shifted_Rotated_HappyCat - 1200 ----
    M1 = M_block(1)
    diff1 = x - o[1]
    y1 = M1 @ diff1
    fvals[1] = HappyCat(y1) - 1200.0
    gy1 = HappyCat_grad(y1)
    grads[1] = M1.T @ gy1

    # ---- component 2: 1e-7 * Rotated_HighConditionedElliptic + 199.99999 ----
    M2 = M_block(2)
    diff2 = x - o[2]
    y2 = M2 @ diff2
    fvals[2] = 1e-7 * High_Conditioned_Elliptic(y2) + 199.99999
    gy2 = 1e-7 * High_Conditioned_Elliptic_grad(y2)
    grads[2] = M2.T @ gy2

    # ---- component 3: 2.5 * Shifted_Rotated_Weierstrass - 1200 ----
    M3 = M_block(3)
    diff3 = x - o[3]
    y3 = M3 @ diff3
    fvals[3] = 2.5 * Weierstrass(y3) - 1200.0
    gy3 = 2.5 * Weierstrass_grad(y3)
    grads[3] = M3.T @ gy3

    # ---- component 4: 10 * Shifted_Rotated_Griewank - 6600 ----
    M4 = M_block(4)
    diff4 = x - o[4]
    y4 = M4 @ diff4
    fvals[4] = 10.0 * Griewank(y4) - 6600.0
    gy4 = 10.0 * Griewank_grad(y4)
    grads[4] = M4.T @ gy4

    # assemble total gradient
    grad_total = np.zeros(dim, dtype=float)
    for i in range(5):
        # grad_w[i] is shape (d,), fvals[i] scalar, grads[i] shape (d,)
        grad_total += grad_w[i] * fvals[i] + w[i] * grads[i]

    return grad_total

# @numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_5(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.array([10, 10, 10, 20, 20]))
    res = omega[0] * (10*Shifted_Rotated_HGBat(x, M[0:dim, :], o[0, :])-14000)
    res += omega[1] * (10*Shifted_Rotated_Rastrigin(x, M[dim:dim*2, :], o[1, :])-8900)
    res += omega[2] * (2.5*Shifted_Rotated_Schwefel(x, M[dim*2:dim*3, :], o[2, :])-2550)
    res += omega[3] * (25*Shifted_Rotated_Weierstrass(x, M[dim*3:dim*4, :], o[3, :])-14700)
    res += omega[4] * (1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim*4:dim*5, :], o[4, :]) + 399.9999)
    return res + 2700


def Composition_5_grad(x, M, o):
    """
    Compute gradient of Composition_5 at x.
    M is assumed to contain 5 stacked blocks, each block of shape (d, d),
    laid out as rows: M[i*d:(i+1)*d, :].
    o is shape (5, d).
    sig used as in original call: np.array([10,10,10,20,20])
    """
    x = np.asarray(x, dtype=float)
    dim = x.size

    # sig array as used in your original composition_omega call
    sig = np.array([10.0, 10.0, 10.0, 20.0, 20.0])

    # compute weights and their gradients
    w, grad_w = composition_omega_and_grad(x, o, sig)

    # helper to extract block i
    def M_block(i):
        return np.ascontiguousarray(M[i*dim:(i+1)*dim, :])

    # prepare containers
    fvals = np.zeros(5, dtype=float)
    grads = [None] * 5   # each element is gradient vector length d

    # component 0: 10 * Shifted_Rotated_HGBat - 14000
    M0 = M_block(0)
    diff0 = x - o[0]
    y0 = M0 @ diff0
    fvals[0] = 10.0 * HGBat(y0) - 14000.0
    gy0 = 10.0 * HGBat_grad(y0)   # gradient wrt y0
    grads[0] = M0.T @ gy0         # back to x

    # component 1: 10 * Shifted_Rotated_Rastrigin - 8900
    M1 = M_block(1)
    diff1 = x - o[1]
    y1 = M1 @ diff1
    fvals[1] = 10.0 * Rastrigin(y1) - 8900.0
    gy1 = 10.0 * Rastrigin_grad(y1)
    grads[1] = M1.T @ gy1

    # component 2: 2.5 * Shifted_Rotated_Schwefel - 2550
    M2 = M_block(2)
    diff2 = x - o[2]
    y2 = M2 @ diff2
    fvals[2] = 2.5 * Modified_Schwefel(y2) - 2550.0
    gy2 = 2.5 * Modified_Schwefel_grad(y2)
    grads[2] = M2.T @ gy2

    # component 3: 25 * Shifted_Rotated_Weierstrass - 14700
    M3 = M_block(3)
    diff3 = x - o[3]
    y3 = M3 @ diff3
    fvals[3] = 25.0 * Weierstrass(y3) - 14700.0
    gy3 = 25.0 * Weierstrass_grad(y3)
    grads[3] = M3.T @ gy3

    # component 4: 1e-6 * Rotated_High_Conditioned_Elliptic + 399.9999
    M4 = M_block(4)
    diff4 = x - o[4]
    y4 = M4 @ diff4
    fvals[4] = 1e-6 * High_Conditioned_Elliptic(y4) + 399.9999
    gy4 = 1e-6 * High_Conditioned_Elliptic_grad(y4)
    grads[4] = M4.T @ gy4

    # assemble total gradient
    grad_total = np.zeros(dim, dtype=float)
    for i in range(5):
        # grad_w[i] shape (d,), fvals[i] scalar, grads[i] shape (d,)
        grad_total += grad_w[i] * fvals[i] + w[i] * grads[i]

    return grad_total


# @numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_6(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 20, 30, 40, 50])))
    res = omega[0] * (2.5*Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock(x, M[0:dim, :], o[0, :])-3750)
    res += omega[1] * (10*Shifted_Rotated_HappyCat(x, M[dim:dim*2, :], o[1, :])-12900)
    res += omega[2] * (2.5*Shifted_Rotated_Schwefel(x, M[dim*2:dim*3, :], o[2, :])-2550)
    res += omega[3] * (5e-4*Shifted_Rotated_Expanded_Scaffer_F6(x, M[dim*3:dim*4, :], o[3, :])+299.2)
    res += omega[4] * (1e-6*Rotated_High_Conditioned_Elliptic(x, M[dim*4:dim*5, :], o[4, :])+399.9999)
    return res + 2800

def Composition_6_grad(x, M, o):
    """
    Gradient of Composition_6.
    M is assumed stacked as 5 blocks of shape (d,d):
      block i is M[i*dim:(i+1)*dim, :]
    o is shape (5, d)
    """
    x = np.asarray(x, dtype=float)
    dim = x.size

    # sig array = squares of [10,20,30,40,50]
    sig = np.square(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))

    # compute weights and their gradients
    w, grad_w = composition_omega_and_grad(x, o, sig)

    # helper to get block i
    def M_block(i):
        return np.ascontiguousarray(M[i*dim:(i+1)*dim, :])

    fvals = np.zeros(5, dtype=float)
    grads = [None] * 5

    identity_M = np.eye(dim)
    zero_o = np.zeros(dim)

    # component 0
    M0 = M_block(0)
    diff0 = x - o[0]
    y0 = M0 @ diff0
    fvals[0] = 2.5 * Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock(y0, identity_M, zero_o) - 3750.0
    gy0 = 2.5 * Shifted_Rotated_Expanded_Griewank_plus_Rosenbrock_grad(y0, identity_M, zero_o)  # grad wrt y0
    grads[0] = M0.T @ gy0

    # component 1
    M1 = M_block(1)
    diff1 = x - o[1]
    y1 = M1 @ diff1
    fvals[1] = 10.0 * Shifted_Rotated_HappyCat(y1, identity_M, zero_o) - 12900.0
    gy1 = 10.0 * Shifted_Rotated_HappyCat_grad(y1, identity_M, zero_o)
    grads[1] = M1.T @ gy1

    # component 2
    M2 = M_block(2)
    diff2 = x - o[2]
    y2 = M2 @ diff2
    fvals[2] = 2.5 * Shifted_Rotated_Schwefel(y2, identity_M, zero_o) - 2550.0
    gy2 = 2.5 * Shifted_Rotated_Schwefel_grad(y2, identity_M, zero_o)
    grads[2] = M2.T @ gy2

    # component 3
    M3 = M_block(3)
    diff3 = x - o[3]
    y3 = M3 @ diff3
    fvals[3] = 5e-4 * Shifted_Rotated_Expanded_Scaffer_F6(y3, identity_M, zero_o) + 299.2
    gy3 = 5e-4 * Shifted_Rotated_Expanded_Scaffer_F6_grad(y3, identity_M, zero_o)
    grads[3] = M3.T @ gy3

    # component 4
    M4 = M_block(4)
    diff4 = x - o[4]
    y4 = M4 @ diff4
    fvals[4] = 1e-6 * Rotated_High_Conditioned_Elliptic(y4, identity_M, zero_o) + 399.9999
    gy4 = 1e-6 * Rotated_High_Conditioned_Elliptic_grad(y4, identity_M, zero_o)
    grads[4] = M4.T @ gy4

    # total gradient
    grad_total = np.zeros(dim, dtype=float)
    for i in range(5):
        grad_total += grad_w[i] * fvals[i] + w[i] * grads[i]

    return grad_total


# @numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_7(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0] * (Hybrid_1(x, M[0:dim, :], o[0, :])-1700)
    res += omega[1] * (Hybrid_2(x, M[dim:dim*2, :], o[1, :])-1700)
    res += omega[2] * (Hybrid_3(x, M[dim*2:dim*3, :], o[2, :])-1700)
    return res + 2900

def Composition_7_grad(x, M, o):
    """
    Gradient of Composition_7:
      - x: input vector (dim,)
      - M: stacked block matrices for Hybrid_1, Hybrid_2, Hybrid_3
      - o: array of shifts (3, dim)
    """
    x = np.asarray(x, dtype=float)
    dim = x.size
    sig = np.square(np.array([10.0, 30.0, 50.0]))

    # compute weights and their gradients
    w, grad_w = composition_omega_and_grad(x, o, sig)

    # helper to get block i of M
    def M_block(i):
        return np.ascontiguousarray(M[i*dim:(i+1)*dim, :])

    fvals = np.zeros(3, dtype=float)
    grads = [None] * 3

    # component 0: Hybrid_1
    M0 = M_block(0)
    diff0 = x - o[0]
    y0 = M0 @ diff0
    fvals[0] = Hybrid_1(x, M0, o[0]) - 1700.0
    grads[0] = Hybrid_1_grad(x, M0, o[0])  # grad wrt x

    # component 1: Hybrid_2
    M1 = M_block(1)
    diff1 = x - o[1]
    y1 = M1 @ diff1
    fvals[1] = Hybrid_2(x, M1, o[1]) - 1700.0
    grads[1] = Hybrid_2_grad(x, M1, o[1])

    # component 2: Hybrid_3
    M2 = M_block(2)
    diff2 = x - o[2]
    y2 = M2 @ diff2
    fvals[2] = Hybrid_3(x, M2, o[2]) - 1700.0
    grads[2] = Hybrid_3_grad(x, M2, o[2])

    # total gradient with product rule
    grad_total = np.zeros_like(x)
    for i in range(3):
        grad_total += grad_w[i] * fvals[i] + w[i] * grads[i]

    return grad_total



# @numba.njit("f8(f8[:],f8[:,:],f8[:,:])", nogil=True, fastmath=True)
def Composition_8(x, M, o):
    dim = x.shape[0]
    omega = composition_omega(x-o, np.square(np.array([10, 30, 50])))
    res = omega[0] * (Hybrid_4(x, M[0:dim, :], o[0, :])-2000)
    res += omega[1] * (Hybrid_5(x, M[dim:dim*2, :], o[1, :])-2000)
    res += omega[2] * (Hybrid_6(x, M[dim*2:dim*3, :], o[2, :])-2000)
    return res + 3000

def Composition_8_grad(x, M, o):
    """
    Gradient of Composition_8:
      - x: input vector (dim,)
      - M: stacked block matrices for Hybrid_4, Hybrid_5, Hybrid_6
      - o: array of shifts (3, dim)
    """
    x = np.asarray(x, dtype=float)
    dim = x.size
    sig = np.square(np.array([10.0, 30.0, 50.0]))  # same as in omega

    # compute weights and their gradients
    w, grad_w = composition_omega_and_grad(x, o, sig)

    # helper to get block i of M
    def M_block(i):
        return np.ascontiguousarray(M[i*dim:(i+1)*dim, :])

    fvals = np.zeros(3, dtype=float)
    grads = [None] * 3

    # component 0: Hybrid_4
    M0 = M_block(0)
    diff0 = x - o[0]
    y0 = M0 @ diff0
    fvals[0] = Hybrid_4(x, M0, o[0]) - 2000.0
    grads[0] = Hybrid_4_grad(x, M0, o[0])

    # component 1: Hybrid_5
    M1 = M_block(1)
    diff1 = x - o[1]
    y1 = M1 @ diff1
    fvals[1] = Hybrid_5(x, M1, o[1]) - 2000.0
    grads[1] = Hybrid_5_grad(x, M1, o[1])

    # component 2: Hybrid_6
    M2 = M_block(2)
    fvals[2] = Hybrid_6(x, M2, o[2]) - 2000.0
    grads[2] = Hybrid_6_grad(x, M2, o[2])

    # total gradient using product rule
    grad_total = np.zeros_like(x)
    for i in range(3):
        grad_total += grad_w[i] * fvals[i] + w[i] * grads[i]

    return grad_total




















