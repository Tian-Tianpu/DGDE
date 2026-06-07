import numpy as np
from mealpy.optimizer import Optimizer
import math
import joblib
import scipy
from scipy.stats import cauchy
from copy import deepcopy
from mealpy.utils.agent import Agent


class OriginalES(Optimizer):
    """
    The original version of: Evolution Strategies (ES)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction.
        Natural computing, 1(1), pp.3-52.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, lamda: float = 0.75, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.lamda = self.validator.check_float("lamda", lamda, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "lamda"])
        self.n_child = int(self.lamda * self.pop_size)
        self.sort_flag = True

    def initialize_variables(self):
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)
        # 存储每一代种群历史 (解 + 适应度)
        self.population_history = []
        self.best_history = []

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        strategy = self.generator.uniform(0, self.distance)
        return Agent(solution=solution, strategy=strategy)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx].solution + self.pop[idx].strategy * self.generator.normal(
                0, 1.0, self.problem.n_dims
            )
            pos_new = self.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            strategy = np.exp(
                tau_p * self.generator.normal(0, 1.0, self.problem.n_dims)
                + tau * self.generator.normal(0, 1.0, self.problem.n_dims)
            )
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=strategy)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)
        self.pop = self.get_sorted_and_trimmed_population(child + self.pop, self.pop_size, self.problem.minmax)

        self.population_history.append(
            [(agent.solution, agent.target.fitness) for agent in self.pop]
        )
        self.best_history.append(min(agent.target.fitness for agent in self.pop))



class LevyES(OriginalES):
    """
    The developed Levy-flight version: Evolution Strategies (ES)

    Notes:
        + The Levy-flight is applied, the flow and equations is changed
        + Link: https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction.
        Natural computing, 1(1), pp.3-52.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, lamda: float = 0.75, **kwargs: object) -> None:
        super().__init__(epoch, pop_size, lamda, **kwargs)

    def initialize_variables(self):
        super().initialize_variables()
        self.population_history = []
        self.best_history = []

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        """
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx].solution + self.pop[idx].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            strategy = np.exp(
                tau_p * self.generator.normal(0, 1.0, self.problem.n_dims) +
                tau * self.generator.normal(0, 1.0, self.problem.n_dims)
            )
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=strategy)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)

        child_levy = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx].solution + self.get_levy_flight_step(multiplier=0.001, size=self.problem.n_dims, case=-1)
            pos_new = self.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            stdevs = np.array([
                np.exp(tau_p * self.generator.normal(0, 1.0) + tau * self.generator.normal(0, 1.0))
                for _ in range(self.problem.n_dims)
            ])
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=stdevs)
            child_levy.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child_levy[-1].target = self.get_target(pos_new)
        child_levy = self.update_target_for_population(child_levy)

        self.pop = self.get_sorted_and_trimmed_population(child + child_levy + self.pop, self.pop_size, self.problem.minmax)

        self.population_history.append(
            [(agent.solution, agent.target.fitness) for agent in self.pop]
        )
        self.best_history.append(min(agent.target.fitness for agent in self.pop))


class CMA_ES(Optimizer):
    """
    The simple version of: Covariance Matrix Adaptation Evolution Strategy (Simple-CMA-ES)

    Links:
        1. Inspired from this version: https://github.com/jenkspt/CMA-ES
        2. https://ieeexplore.ieee.org/abstract/document/6790628/

    References
    ~~~~~~~~~~
    [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies.
        Evolutionary computation, 9(2), 159-195.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def before_main_loop(self):
        self.mu = int(np.round(self.pop_size / 2))
        self.population_history = []
        self.best_history = []

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        """
        pos_list = np.array([agent.solution for agent in self.pop]).T
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pos_topk = np.array([agent.solution for agent in pop_sorted[:self.mu]]).T

        # Covariance of top k but using mean of entire population
        centered = pos_list - pos_topk.mean(1, keepdims=True)
        C = (centered @ centered.T) / (self.mu - 1)

        # Eigenvalue decomposition
        w, E = np.linalg.eigh(C)
        if np.any(np.diag(w) < 0):
            w[w < 0] = 0

        # Generate new population
        N = self.generator.normal(size=(self.problem.n_dims, self.pop_size))
        X = pos_topk.mean(1, keepdims=True) + (E @ np.diag(np.sqrt(w)) @ N)
        X = X.T

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.correct_solution(X[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[-1], self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        self.population_history.append(
            [(agent.solution, agent.target.fitness) for agent in self.pop]
        )
        self.best_history.append(min(agent.target.fitness for agent in self.pop))


class MAP_CMA_ES(Optimizer):
    """
    MAP-CMA-ES
    """

    def __init__(self, epoch: int = 100, pop_size: int = 10, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False  # 与 simple-CMA-ES 类一致

    # ------------------ MAP-CMA 相关内部状态初始化 ------------------
    def before_main_loop(self):
        n = self.problem.n_dims
        self.mu = int(np.floor(self.pop_size / 2))
        self.population_history = []
        self.best_history = []

        # 正权重（只对前 mu 个体有效，其余为 0）
        weights_prime = np.array([
            np.log((self.pop_size + 1) / 2.0) - np.log(i + 1) if i < self.mu else 0.0
            for i in range(self.pop_size)
        ])
        self.weights = weights_prime / (np.sum(weights_prime) + 1e-32)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # 学习率与常数（与参考实现一致）
        alpha_cov = 2.0
        self.c1 = alpha_cov / ((n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1.0 - self.c1 - 1e-8,
            alpha_cov * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0) ** 2 + alpha_cov * self.mu_eff / 2.0),
        )
        # 步长控制参数
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        # rank-1 路径累积
        self.cc = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)

        # 动量缩放 r（paper 建议 ~ n）
        self.r = float(n)

        # cm 取值：确保 cm + cm*c1/(r*cmu) = 1
        # => cm = 1 / (1 + c1/(r*cmu))
        self.cm = 1.0 / (1.0 + self.c1 / (self.r * max(self.cmu, 1e-32)))

        # 期望范数 E||N(0,I)||
        self.chi_n = np.sqrt(n) * (1.0 - (1.0 / (4.0 * n)) + 1.0 / (21.0 * (n ** 2)))

        # 进化路径
        self.p_sigma = np.zeros(n)
        self.pc = np.zeros(n)

        # 初始化分布参数：均值/步长/协方差
        # 均值：取初始种群均值
        pos_list = np.array([agent.solution for agent in self.pop])
        self.mean = np.mean(pos_list, axis=0)

        # 步长：基于边界跨度或初始种群方差的一个经验初始化
        try:
            lb = np.array(self.problem.lb, dtype=float)
            ub = np.array(self.problem.ub, dtype=float)
            span = np.maximum(ub - lb, 1e-8)
            self.sigma = 0.3 * float(np.mean(span))
        except Exception:
            # 回退：用初始群体标准差
            self.sigma = float(np.mean(np.std(pos_list, axis=0))) + 1e-8

        # 协方差矩阵
        self.C = np.eye(n)
        self.B = None   # 特征向量
        self.D = None   # 特征值开方（对角）

    # 特征分解（缓存）
    def _eigen_decomposition(self):
        if self.B is not None and self.D is not None:
            return self.B, self.D
        # 保对称
        self.C = (self.C + self.C.T) / 2.0
        D2, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.clip(D2, 1e-32, None))
        # 回写正定版本
        self.C = B @ np.diag(D ** 2) @ B.T
        self.B, self.D = B, D
        return B, D

    # ------------------ 单代演化（与 CMA_ES 接口一致） ------------------
    def evolve(self, epoch):
        """
        与 simple-CMA-ES 相同接口：
        - 读取当前 self.pop
        - 基于 MAP-CMA 的规则生成新群体并进行贪婪选择
        - 更新内部分布参数 mean / C / sigma
        """

        n = self.problem.n_dims

        # 按适应度排序，取前 mu
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pos_topk = np.array([agent.solution for agent in pop_sorted[:self.mu]])

        # ====== 估计当前分布下的采样并生成新解 ======
        # 先做一次特征分解（供步长路径等使用）
        B, D = self._eigen_decomposition()

        # 使用 MAP-CMA 的当前均值/协方差/步长采样
        Z = self.generator.normal(size=(n, self.pop_size))  # ~ N(0, I)
        X = (self.mean.reshape(-1, 1) + self.sigma * (B @ np.diag(D) @ Z)).T  # pop_size x n

        pop_new = []
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(X[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            # 同步/异步两种更新兼容
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[-1], self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # ====== 按 MAP-CMA 规则更新分布参数 ======
        # 重新计算按新一代评估后的排序与权重组合
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        X_k = np.array([agent.solution for agent in pop_sorted[:self.pop_size]])       # (pop_size, n)
        # 基于当前均值与步长的标准化偏移
        Y_k = (X_k - self.mean) / (self.sigma + 1e-32)                                # (pop_size, n)
        # 选择+重组的加权方向
        y_w = np.sum((Y_k[:self.mu].T * self.weights[:self.mu]), axis=1)              # (n,)

        # C^(-1/2) = B * D^{-1} * B^T
        B, D = self._eigen_decomposition()
        self.B, self.D = None, None  # 强制下次重分解
        C_inv_sqrt = B @ np.diag(1.0 / (D + 1e-32)) @ B.T

        # 路径更新
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * (C_inv_sqrt @ y_w)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * y_w

        # 均值更新（带动量项）
        # m_{t+1} = m_t + cm * ( sigma * y_w + (c1 / (r * cmu)) * sigma * pc )
        self.mean = self.mean + self.cm * (
            self.sigma * y_w + (self.c1 / (self.r * max(self.cmu, 1e-32))) * self.sigma * self.pc
        )

        # 协方差更新（rank-one + rank-μ）
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.sum(
            [w * np.outer(y, y) for w, y in zip(self.weights[:self.mu], Y_k[:self.mu])],
            axis=0
        )
        # 注意：这里 weights 为正权重，论文中 cmu 前会乘以 sum(weights)=1，因此直接 cmu*rank_mu
        self.C = (1.0 - self.c1 - self.cmu) * self.C + self.c1 * rank_one + self.cmu * rank_mu

        # 步长控制
        norm_ps = np.linalg.norm(self.p_sigma)
        self.sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma) * (norm_ps / (self.chi_n + 1e-32) - 1.0))
        # 防发散：简单裁剪
        self.sigma = float(np.clip(self.sigma, 1e-16, 1e16))

        # 记录群体历史（解与适应度），以便外部可视化
        self.population_history.append(
            [(agent.solution, agent.target.fitness) for agent in self.pop]
        )
        self.best_history.append(min(agent.target.fitness for agent in self.pop))


class LRA_CMA_ES(Optimizer):
    """
    Learning Rate Adaptation CMA-ES (LRA-CMA-ES)
    Variant of CMA-ES with momentum-based mean update.
    Features:
        - Momentum-based mean update: x_mean += cm * (sigma * y_w + (c1/(r*cmu)) * sigma * pc)
        - Covariance matrix update: rank-one + rank-mu
        - Step size control and evolution paths (ps, pc)
        - Population history recording for visualization
    """

    def __init__(self, epoch=100, pop_size=10, cm=0.5, r=None, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.cm = self.validator.check_float("cm", cm, (0, 1.0))
        self.r = r
        if self.r is not None:
            self.r = self.validator.check_float("r", self.r, (0, np.inf))
        self.set_parameters(["epoch", "pop_size", "cm", "r"])
        self.sort_flag = True
        self.population_history = []
        self.best_history = []

    # ---------------- Utility ----------------
    def make_pos_definite(self, C, eps=1e-10):
        """确保协方差矩阵正定"""
        # 确保对称
        C = (C + C.T) / 2

        # 添加小的正则化项到对角线
        n = C.shape[0]
        C += eps * np.eye(n)

        # 检查并修正负特征值
        try:
            # 尝试Cholesky分解
            np.linalg.cholesky(C)
            return C
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用特征值分解修正
            eigvals, eigvecs = np.linalg.eigh(C)
            # 确保所有特征值都为正
            eigvals = np.maximum(eigvals, eps)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        step = np.zeros(self.problem.n_dims)  # 初始化 step 为零
        return Agent(solution=solution, step=step)

    def update_step__(self, pop, C):
        # 确保协方差矩阵正定
        C = self.make_pos_definite(C)

        for idx in range(len(pop)):
            try:
                # 尝试使用更稳定的方法采样
                pop[idx].step = self.generator.multivariate_normal(
                    np.zeros(self.problem.n_dims), C, method='eigh'
                ).real
            except np.linalg.LinAlgError:
                # 如果仍然失败，使用对角矩阵作为后备
                pop[idx].step = self.generator.normal(
                    0, 1, self.problem.n_dims
                )
        return pop

    # ---------------- CMA-ES 参数初始化 ----------------
    def before_main_loop(self):
        if self.r is None:
            self.r = float(self.problem.n_dims)

        # 按适应度排序
        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)

        self.mu = int(np.round(self.pop_size / 2))
        self.ps = np.zeros(self.problem.n_dims)
        self.pc = np.zeros(self.problem.n_dims)
        self.C = np.eye(self.problem.n_dims)

        # recombination weights
        w = np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size + 1))
        self.w = w / np.sum(w)
        self.mu_eff = 1.0 / np.sum(self.w ** 2)

        # Step-size control
        sigma0 = 0.1 * (self.problem.ub - self.problem.lb)
        self.sigma = sigma0
        self.cs = (self.mu_eff + 2) / (self.problem.n_dims + self.mu_eff + 5)
        self.ds = 1 + self.cs + 2 * max(np.sqrt((self.mu_eff - 1) / (self.problem.n_dims + 1)), 0)
        self.ENN = np.sqrt(self.problem.n_dims) * (
                    1 - 1 / (4 * self.problem.n_dims) + 1 / (21 * self.problem.n_dims ** 2))

        # Covariance parameters
        self.cc = (4 + self.mu_eff / self.problem.n_dims) / (
                    4 + self.problem.n_dims + 2 * self.mu_eff / self.problem.n_dims)
        self.c1 = 2.0 / ((self.problem.n_dims + 1.3) ** 2 + self.mu_eff)
        alpha_mu = 2.0
        self.cmu = min(1.0 - self.c1,
                       alpha_mu * (self.mu_eff - 2 + 1 / self.mu_eff) / (
                               (self.problem.n_dims + 2) ** 2 + alpha_mu * self.mu_eff / 2))
        self.hth = (1.4 + 2 / (self.problem.n_dims + 1)) * self.ENN

        # Initial mean
        self.x_mean = np.mean([agent.solution for agent in self.pop[:self.mu]], axis=0)

        # 初始化步长
        self.pop = self.update_step__(self.pop, self.C)

    # ---------------- Evolution ----------------
    def evolve(self, epoch: int):
        # 保存当前种群用于后续更新
        old_pop = self.pop.copy()

        # 生成子代
        pop_child = []
        for idx in range(self.pop_size):
            pos_new = self.x_mean + self.sigma * self.pop[idx].step
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            # 保留父代的步长信息
            agent.step = self.pop[idx].step.copy()
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_child[-1].target = self.get_target(pos_new)

        # 评估子代
        pop_child = self.update_target_for_population(pop_child)

        # 合并父代和子代
        combined_pop = self.pop + pop_child
        combined_pop = self.get_sorted_population(combined_pop, self.problem.minmax)

        # 选择最好的个体
        self.pop = combined_pop[:self.pop_size]

        # 计算加权步长和
        x_step = np.zeros(self.problem.n_dims)
        for i in range(self.mu):
            x_step += self.w[i] * self.pop[i].step

        # 动量-based mean update
        mean_update = self.sigma * x_step + (self.c1 / (self.r * self.cmu)) * self.sigma * self.pc
        self.x_mean = self.x_mean + self.cm * mean_update

        # 进化路径更新
        try:
            # 使用更稳定的矩阵求逆方法
            C_inv = np.linalg.pinv(self.C)
            inv_sqrt_C = scipy.linalg.sqrtm(C_inv)
        except:
            # 如果失败，使用单位矩阵作为后备
            inv_sqrt_C = np.eye(self.problem.n_dims)

        t_ps = inv_sqrt_C @ x_step
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * t_ps

        # Step-size update
        ps_norm = np.linalg.norm(self.ps)
        self.sigma = self.sigma * np.exp((self.cs / self.ds) * (ps_norm / self.ENN - 1))

        # pc update
        norm_ps = np.linalg.norm(self.ps)
        hs = 1.0 if norm_ps / np.sqrt(1 - (1 - self.cs) ** (2 * (epoch + 1))) < self.hth else 0.0
        delta = (1 - hs) * self.cc * (2 - self.cc)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * x_step

        # 协方差矩阵更新
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.zeros((self.problem.n_dims, self.problem.n_dims))
        for i in range(self.mu):
            rank_mu += self.w[i] * np.outer(self.pop[i].step, self.pop[i].step)

        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * rank_one + self.cmu * rank_mu

        # 确保协方差矩阵正定
        self.C = self.make_pos_definite(self.C)

        # 更新步长
        self.pop = self.update_step__(self.pop, self.C)

        # 记录种群历史
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        self.population_history.append(
            [(agent.solution.copy(), agent.target.fitness) for agent in pop_sorted]
        )
        self.best_history.append(min(agent.target.fitness for agent in self.pop))

# class LRA_CMA_ES(Optimizer):
#     """
#     Learning Rate Adaptation CMA-ES (LRA-CMA-ES)
#     Variant of CMA-ES with momentum-based mean update.
#     Features:
#         - Momentum-based mean update: x_mean += cm * (sigma * y_w + (c1/(r*cmu)) * sigma * pc)
#         - Covariance matrix update: rank-one + rank-mu
#         - Step size control and evolution paths (ps, pc)
#         - Population history recording for visualization
#     """
#
#     def __init__(self, epoch=100, pop_size=10, cm=0.5, r=None, **kwargs):
#         super().__init__(**kwargs)
#         self.epoch    = self.validator.check_int("epoch", epoch, [1, 100000])
#         self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
#         self.cm       = self.validator.check_float("cm", cm, (0, 1.0))
#         self.r        = r
#         if self.r is not None:
#             self.r = self.validator.check_float("r", self.r, (0, np.inf))
#         self.set_parameters(["epoch", "pop_size", "cm", "r"])
#         self.sort_flag = True
#         self.population_history = []
#
#     # ---------------- Utility ----------------
#     def make_pos_definite(self, C, eps=1e-10):
#         C = (C + C.T) / 2
#         eigvals, eigvecs = np.linalg.eigh(C)
#         eigvals = np.clip(eigvals, eps, None)
#         return eigvecs @ np.diag(eigvals) @ eigvecs.T
#
#     def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
#         if solution is None:
#             solution = self.problem.generate_solution(encoded=True)
#         step = np.zeros(self.problem.n_dims)  # 初始化 step 为零
#         return Agent(solution=solution, step=step)
#
#     def update_step__(self, pop, C):
#         C = self.make_pos_definite(C)
#         for idx in range(len(pop)):
#             pop[idx].step = self.generator.multivariate_normal(
#                 np.zeros(self.problem.n_dims), C
#             ).real  # 保证为实数
#         return pop
#
#     # ---------------- CMA-ES 参数初始化 ----------------
#     def before_main_loop(self):
#         if self.r is None:
#             self.r = float(self.problem.n_dims)
#         # 按适应度排序
#         self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
#
#         self.mu   = int(np.round(self.pop_size / 2))
#         self.ps   = np.zeros(self.problem.n_dims)
#         self.pc   = np.zeros(self.problem.n_dims)
#         self.C    = np.eye(self.problem.n_dims)
#
#         # recombination weights
#         w = np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size+1))
#         self.w = w / np.sum(w)
#         self.mu_eff = 1.0 / np.sum(self.w ** 2)
#
#         # Step-size control
#         sigma0 = 0.1 * (self.problem.ub - self.problem.lb)
#         self.sigma = sigma0
#         self.cs = (self.mu_eff + 2) / (self.problem.n_dims + self.mu_eff + 5)
#         self.ds = 1 + self.cs + 2 * max(np.sqrt((self.mu_eff - 1) / (self.problem.n_dims + 1)), 0)
#         self.ENN = np.sqrt(self.problem.n_dims) * (1 - 1/(4*self.problem.n_dims) + 1/(21*self.problem.n_dims**2))
#
#         # Covariance parameters
#         self.cc = (4 + self.mu_eff / self.problem.n_dims) / (4 + self.problem.n_dims + 2 * self.mu_eff / self.problem.n_dims)
#         self.c1 = 2.0 / ((self.problem.n_dims + 1.3)**2 + self.mu_eff)
#         alpha_mu = 2.0
#         self.cmu = min(1.0 - self.c1,
#                        alpha_mu * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.problem.n_dims + 2)**2 + alpha_mu * self.mu_eff / 2))
#         self.hth = (1.4 + 2/(self.problem.n_dims + 1)) * self.ENN
#
#         # Initial mean
#         self.x_mean = np.mean([agent.solution for agent in self.pop[:self.mu]], axis=0)
#
#     # ---------------- Evolution ----------------
#     def evolve(self, epoch: int):
#         # Generate offspring
#         pop_child = []
#         for idx in range(self.pop_size):
#             pos_new = self.x_mean + self.sigma * self.pop[idx].step
#             pos_new = self.correct_solution(pos_new)
#             agent = self.generate_empty_agent(pos_new)
#             pop_child.append(agent)
#             if self.mode not in self.AVAILABLE_MODES:
#                 pop_child[-1].target = self.get_target(pos_new)
#
#         # Evaluate offspring
#         pop_child = self.update_target_for_population(pop_child)
#
#         # Combine via greedy selection
#         pop_sorted   = self.get_sorted_population(self.pop, self.problem.minmax)
#         child_sorted = self.get_sorted_population(pop_child, self.problem.minmax)
#         self.pop = self.greedy_selection_population(pop_sorted, child_sorted, self.problem.minmax)
#         self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
#
#         # Update steps
#         self.pop = self.update_step__(self.pop, self.C)
#
#         # Weighted step sum
#         x_step = np.zeros(self.problem.n_dims)
#         for i in range(self.mu):
#             x_step += self.w[i] * self.pop[i].step
#
#         # Momentum-based mean update
#         mean_update = self.sigma * x_step + (self.c1 / (self.r * self.cmu)) * self.sigma * self.pc
#         self.x_mean = self.x_mean + self.cm * mean_update
#
#         # Evolution path update
#         try:
#             inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(self.make_pos_definite(self.C)).T)
#         except np.linalg.LinAlgError:
#             inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(self.make_pos_definite(self.C, eps=1e-8)).T)
#         t_ps = inv_sqrt_C @ x_step
#         self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * t_ps
#
#         # Step-size update
#         self.sigma = self.sigma * np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.ENN - 1)) ** 0.3
#
#         # pc update
#         norm_ps = np.linalg.norm(self.ps)
#         hs = 1.0 if norm_ps / np.sqrt(1 - (1 - self.cs)**(2 * epoch)) < self.hth else 0.0
#         delta = (1 - hs) * self.cc * (2 - self.cc)
#         self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * x_step
#
#         # Covariance update
#         self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc) + delta * self.C
#         for i in range(self.mu):
#             self.C += self.cmu * self.w[i] * np.outer(self.pop[i].step, self.pop[i].step)
#
#         # Ensure positive definite
#         self.C = self.make_pos_definite(self.C)
#
#         # Record population history
#         pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
#         self.population_history.append(
#             [(agent.solution.copy(), agent.target.fitness) for agent in pop_sorted]
#         )


# class LRA_CMA_ES(Optimizer):
#     """
#     Learning Rate Adaptation CMA-ES (LRA-CMA-ES) - variant of CMA-ES with momentum-based mean update.
#     Core features:
#         - Momentum-based mean update: mean += cm * (sigma * y_w + (c1/(r * cmu)) * sigma * pc)
#         - Covariance matrix update: rank-one + rank-mu updates
#         - Step size control and evolution paths (p_sigma, pc) as in original CMA-ES
#     """
#     def __init__(self, epoch=10000, pop_size=100, cm=0.5, r=None, **kwargs):
#         """
#         Args:
#             epoch (int): maximum number of iterations
#             pop_size (int): population size (lambda)
#             cm (float): momentum coefficient for mean update (0 < cm <= 1)
#             r (float): momentum scaling parameter (default: problem dimension)
#         """
#         super().__init__(**kwargs)
#         self.epoch    = self.validator.check_int("epoch", epoch,    [1, 100000])
#         self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
#         self.cm       = self.validator.check_float("cm", cm, (0, 1.0))
#         self.r        = r
#         if self.r is not None:
#             self.r = self.validator.check_float("r", self.r, (0, np.inf))
#         self.set_parameters(["epoch", "pop_size", "cm", "r"])
#         self.sort_flag = True
#         self.population_history = []
#
#     def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
#         """
#         Generate a new agent with given solution (random if None) and a Gaussian step.
#         """
#         if solution is None:
#             solution = self.problem.generate_solution(encoded=True)
#         # initial step vector (identity covariance)
#         step = self.generator.multivariate_normal(
#             np.zeros(self.problem.n_dims), np.eye(self.problem.n_dims)
#         )
#         return Agent(solution=solution, step=step)
#
#     # def update_step__(self, pop, C):
#     #     """
#     #     Update each agent's step by sampling from N(0, C).
#     #     """
#     #     for idx in range(len(pop)):
#     #         pop[idx].step = self.generator.multivariate_normal(
#     #             np.zeros(self.problem.n_dims), C
#     #         )
#     #     return pop
#     def update_step__(self, pop, C):
#         """
#         Update each agent's step by sampling from N(0, C).
#         Ensure step is real.
#         """
#         # Symmetrize C
#         C = (C + C.T) / 2
#         # Eigen-decomposition
#         D2, B = np.linalg.eigh(C)
#         D2 = np.clip(D2, 0, None)  # clip negative eigenvalues
#         sqrt_D = np.sqrt(D2)
#         for idx in range(len(pop)):
#             z = self.generator.normal(size=self.problem.n_dims)  # standard normal
#             step = B @ (sqrt_D * z)
#             pop[idx].step = np.real(step)  # 强制取实部，避免 complex
#         return pop
#
#     def before_main_loop(self):
#         # If momentum parameter not given, default to dimension
#         if self.r is None:
#             self.r = float(self.problem.n_dims)
#         # Sort initial population by fitness
#         self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
#         # Initialize CMA-ES internal parameters
#         self.mu   = int(np.round(self.pop_size / 2))
#         self.ps   = np.zeros(self.problem.n_dims)  # evolution path for sigma
#         self.C    = np.eye(self.problem.n_dims)    # covariance matrix
#         self.pc   = np.zeros(self.problem.n_dims)  # evolution path for C
#         # Recombination weights
#         self.w = np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size+1))
#         self.w = self.w / np.sum(self.w)
#         self.mu_eff = 1.0 / np.sum(self.w**2)
#         # Step-size control parameters
#         sigma0 = 0.1 * (self.problem.ub - self.problem.lb)
#         self.cs  = (self.mu_eff + 2) / (self.problem.n_dims + self.mu_eff + 5)
#         self.ds  = 1 + self.cs + 2 * max(np.sqrt((self.mu_eff - 1) / (self.problem.n_dims + 1)), 0)
#         self.ENN = np.sqrt(self.problem.n_dims) * (1 - 1.0/(4*self.problem.n_dims) + 1.0/(21*self.problem.n_dims**2))
#         # Covariance update parameters
#         self.cc  = (4 + self.mu_eff/self.problem.n_dims) / (4 + self.problem.n_dims + 2 * self.mu_eff/self.problem.n_dims)
#         self.c1  = 2.0 / ((self.problem.n_dims + 1.3)**2 + self.mu_eff)
#         alpha_mu = 2.0
#         self.cmu = min(1.0 - self.c1,
#                        alpha_mu * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.problem.n_dims + 2)**2 + alpha_mu * self.mu_eff / 2))
#         self.hth = (1.4 + 2/(self.problem.n_dims + 1)) * self.ENN
#         # Initial step-size
#         self.sigma = sigma0
#         # Initial mean vector: average of top mu solutions
#         self.x_mean = np.mean([agent.solution for agent in self.pop[:self.mu]], axis=0)
#
#     def evolve(self, epoch: int):
#         # Generate offspring population
#         pop_child = []
#         for idx in range(self.pop_size):
#             # Sample candidate from multivariate normal: mean + sigma * step
#             pos_new = self.x_mean + self.sigma * self.pop[idx].step
#             pos_new = self.correct_solution(pos_new)
#             agent = self.generate_empty_agent(pos_new)
#             pop_child.append(agent)
#             # If in single-thread mode, evaluate immediately
#             if self.mode not in self.AVAILABLE_MODES:
#                 pop_child[-1].target = self.get_target(pos_new)
#         # Evaluate all offspring
#         pop_child = self.update_target_for_population(pop_child)
#         # Combine with parents via greedy (mu + lambda strategy)
#         pop_sorted   = self.get_sorted_population(self.pop, self.problem.minmax)
#         child_sorted = self.get_sorted_population(pop_child, self.problem.minmax)
#         # Greedy selection: pick better individual at each index
#         self.pop = self.greedy_selection_population(pop_sorted, child_sorted, self.problem.minmax)
#         # Sort population after selection
#         self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
#         # Generate new steps from covariance C
#         self.pop = self.update_step__(self.pop, self.C)
#         # Compute weighted sum of steps (y_w)
#         x_step = np.zeros(self.problem.n_dims)
#         for i in range(self.mu):
#             x_step += self.w[i] * self.pop[i].step
#         # Momentum-based mean update: mean += cm * (sigma*y_w + (c1/(r*cmu))*sigma*pc)
#         mean_update = self.sigma * x_step + (self.c1 / (self.r * self.cmu)) * self.sigma * self.pc
#         self.x_mean = self.x_mean + self.cm * mean_update
#         # Update evolution path p_sigma
#         try:
#             inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(self.C).T)
#         except np.linalg.LinAlgError:
#             # add small regularization if needed
#             inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(self.C + 1e-8*np.eye(self.problem.n_dims)).T)
#         t_ps = np.dot(x_step, inv_sqrt_C)
#         self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs*(2 - self.cs)*self.mu_eff) * t_ps
#         # Update step-size sigma
#         self.sigma = self.sigma * np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.ENN - 1))**0.3
#         # Update evolution path pc
#         norm_ps = np.linalg.norm(self.ps)
#         if norm_ps / np.sqrt(1 - (1 - self.cs)**(2 * epoch)) < self.hth:
#             hs = 1.0
#         else:
#             hs = 0.0
#         delta = (1 - hs) * self.cc * (2 - self.cc)
#         self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * x_step
#         # Covariance matrix update (rank-one and rank-mu)
#         self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc) + delta * self.C
#         for i in range(self.mu):
#             self.C += self.cmu * self.w[i] * np.outer(self.pop[i].step, self.pop[i].step)
#         # Ensure C is positive semi-definite
#         eigvals, eigvecs = np.linalg.eig(self.C)
#         eigvals[eigvals < 0] = 0
#         self.C = eigvecs.dot(np.diag(eigvals)).dot(eigvecs.T)
#         # Record current population (solutions and fitness)
#         pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
#         self.population_history.append(
#             [(agent.solution.copy(), agent.target.fitness) for agent in pop_sorted]
#         )




# class LRA_CMA_ES(Optimizer):
#     """
#     Learning Rate Adaptation CMA-ES (LRA-CMA-ES) implementing MAP_CMA_ES architecture.
#     """
#     def __init__(self, epoch=10000, pop_size=100, **kwargs):
#         super().__init__(**kwargs)
#         self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
#         self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
#         self.set_parameters(["epoch", "pop_size"])
#         self.sort_flag = True
#         # Record history of populations
#         self.population_history = []
#
#     def before_main_loop(self):
#         # Number of parents (mu)
#         self.mu = int(np.round(self.pop_size / 2))
#         # Evolution paths
#         self.ps = np.zeros(self.problem.n_dims)
#         self.pc = np.zeros(self.problem.n_dims)
#         # Initial covariance matrix
#         self.C = np.eye(self.problem.n_dims)
#         # Recombination weights (logarithmic)
#         weights = np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size + 1))
#         self.w = weights / np.sum(weights)
#         # Effective number of solutions
#         self.mu_eff = 1.0 / np.sum(self.w**2)
#         # Step-size control parameters
#         sigma0 = 0.1 * (self.problem.ub - self.problem.lb)
#         self.cs = (self.mu_eff + 2) / (self.problem.n_dims + self.mu_eff + 5)
#         self.ds = 1 + self.cs + 2 * max(np.sqrt((self.mu_eff - 1) / (self.problem.n_dims + 1)) - 1, 0)
#         # Expected length of N(0,I) vector
#         self.ENN = np.sqrt(self.problem.n_dims) * (1.0 - 1.0/(4*self.problem.n_dims) + 1.0/(21*self.problem.n_dims**2))
#         # Covariance update parameters
#         self.cc = (4 + self.mu_eff / self.problem.n_dims) / (4 + self.problem.n_dims + 2 * self.mu_eff / self.problem.n_dims)
#         self.c1 = 2.0 / ((self.problem.n_dims + 1.3)**2 + self.mu_eff)
#         alpha_mu = 2.0
#         self.cmu = min(1 - self.c1,
#                        alpha_mu * (self.mu_eff - 2 + 1.0/self.mu_eff)
#                        / ((self.problem.n_dims + 2)**2 + alpha_mu * self.mu_eff / 2))
#         self.hth = (1.4 + 2.0/(self.problem.n_dims+1)) * self.ENN
#         # Initial global step size (sigma)
#         self.sigma = sigma0
#         # Initialize mean as average of top mu individuals (sort initial population by fitness)
#         self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
#         self.x_mean = np.mean([agent.solution for agent in self.pop[:self.mu]], axis=0)
#
#     def evolve(self, epoch):
#         # Generate offspring population
#         pop_new = []
#         for idx in range(self.pop_size):
#             # Sample new solution from multivariate normal distribution
#             pos_new = self.x_mean + self.sigma * self.pop[idx].step
#             pos_new = self.correct_solution(pos_new)
#             agent = self.generate_empty_agent(pos_new)
#             pop_new.append(agent)
#             if self.mode not in self.AVAILABLE_MODES:
#                 # Asynchronous update: evaluate and replace immediately if better
#                 pop_new[-1].target = self.get_target(pos_new)
#                 self.pop[idx] = self.get_better_agent(pop_new[-1], self.pop[idx], self.problem.minmax)
#         if self.mode in self.AVAILABLE_MODES:
#             # Synchronous evaluation of all offspring, then greedy selection
#             pop_new = self.update_target_for_population(pop_new)
#             self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
#         else:
#             # After asynchronous updates, sort the current population
#             self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
#         # Update strategy (step-size) for each individual from the current covariance
#         for idx in range(self.pop_size):
#             self.pop[idx].step = self.generator.multivariate_normal(
#                 np.zeros(self.problem.n_dims), self.C)
#         # Compute weighted step (for mean update)
#         self.x_step = np.zeros(self.problem.n_dims)
#         for idx in range(self.mu):
#             self.x_step += self.w[idx] * self.pop[idx].step
#         # Update mean
#         self.x_mean = self.x_mean + self.sigma * self.x_step
#         # Update evolution path ps
#         inv_chol = np.linalg.inv(np.linalg.cholesky(self.C).T)
#         t11 = np.dot(self.x_step, inv_chol)
#         self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * t11
#         # Adapt step size sigma
#         self.sigma = self.sigma * (np.exp(self.cs/self.ds * (np.linalg.norm(self.ps)/self.ENN - 1)))**0.3
#         # Covariance update
#         # Heaviside step flag
#         hsig_cond = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (epoch+1)))
#         hs = 1.0 if hsig_cond < self.hth else 0.0
#         # Update evolution path pc
#         self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc*(2-self.cc)*self.mu_eff) * self.x_step
#         # Rank-one update
#         delta = (1 - hs) * self.cc * (2 - self.cc)
#         self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc) + delta * self.C
#         # Rank-mu update
#         for idx in range(self.mu):
#             self.C += self.cmu * self.w[idx] * np.outer(self.pop[idx].step, self.pop[idx].step)
#         # Ensure covariance is positive semi-definite
#         eigen_vals, eigen_vecs = np.linalg.eigh(self.C)
#         eigen_vals_clipped = np.clip(eigen_vals, a_min=0, a_max=None)
#         self.C = eigen_vecs @ np.diag(eigen_vals_clipped) @ eigen_vecs.T
#         # Record population history: (solution, fitness)
#         self.population_history.append([
#             (np.copy(agent.solution), agent.target.fitness) for agent in self.pop
#         ])






