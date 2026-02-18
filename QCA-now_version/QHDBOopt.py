# coding: utf-8
from __future__ import annotations

import copy
import math
import os
import time
import random
from typing import Any, Callable, Sequence, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import pickle
# ---------------------------------------------------------------------------
# -------------------------  CORE  QHDBO  IMPLEMENTATION  --------------------
# ---------------------------------------------------------------------------

WEIGHT_E: float = 0.43   # energy_FNDT 权重
WEIGHT_R: float = 0.57   # robustness  权重
MAX_E:    float = 300.0  # energy_FNDT 理论上限
MAX_R:    float = 0.5    # robustness  理论上限


def _make_fobj(topo, ID_Chrome_in_complete, Len_Interval) -> Callable[[np.ndarray], float]:
    """Return a single‑objective cost function (minimise) for QHDBO."""

    def fobj(chrom: np.ndarray) -> float:
        #在这里加一步对量子概率幅的测量
        binary_chrom = (chrom >= 0).astype(int)
        topo.G = topo.reconnect_G(binary_chrom, ID_Chrome_in_complete, Len_Interval)
        energy_FNDT = topo.calculate_FNDT()
        robustness = topo.calculate_robustness()
        E_norm = energy_FNDT / MAX_E
        R_norm = robustness / MAX_R
        score = WEIGHT_E * E_norm + WEIGHT_R * R_norm
        return -score  # QHDBO expects a cost to minimise

    return fobj


class _CoreQHDBO:
    """Algorithmic core. `pop_size` is the number of dung‑beetle agents."""

    def __init__(self, pop_size: int, super_node_percent: float, *,
                 seed: Optional[int] = None, max_iteration: int = 500, N: int) -> None:
        self.pop_size = int(pop_size)
        self.N = N
        self.super_node_percent = float(super_node_percent)
        self.max_iteration = int(max_iteration)
        self.seed = seed
        self.best_x: Optional[np.ndarray] = None
        self.best_f: Optional[float] = None
        self.curve: Optional[np.ndarray] = None
        
        # 路径设置
        self.base_dir = './log'  # 基础日志目录
        self.results_dir = './results'  # 结果保存目录
        self.picture_dir = './picture'  # 图片保存目录
        self.algorithm_name = 'QHDBO'  # 算法名称

    def _log_initial_state(self, log_dir, FNDT, robustness):
        """记录初始状态"""
        with open(f'{log_dir}/initial_state.txt', 'w', encoding='utf-8') as f:
            f.write(f"节点数量: {self.N}\n")
            f.write(f"超级节点比例: {self.super_node_percent}\n")
            f.write(f"初始FNDT: {FNDT}\n")
            f.write(f"初始鲁棒性: {robustness}\n")
            f.write(f"最大迭代次数: {self.max_iteration}\n")
            f.write(f"种群大小: {self.pop_size}\n")

        # 清空generation_all.txt文件
        with open(f'{log_dir}/generation_all.txt', 'w', encoding='utf-8') as f:
            pass

    def _log_generation_state(self, log_dir, gen, fit, fndt_values, robustness_values):
        """记录每一代的状态"""
        # 使用追加模式打开文件
        with open(f'{log_dir}/generation_all.txt', 'a', encoding='utf-8') as f:
            f.write(f"Generation {gen}\n")

            for i, fv in enumerate(fit):
                f.write(f"Solution {i}:  FNDT={fndt_values[i]:.4f}, robustness={robustness_values[i]:.4f}, fitness={-fv:.6f}\n")

            # Record the state of the best individual
            best_idx = np.argmin(fit)
            best_f = fit[best_idx]
            best_fndt = fndt_values[best_idx]
            best_robustness = robustness_values[best_idx]
            f.write(f"Best individual: fitness={best_f:.6f}, FNDT={best_fndt:.4f}, robustness={best_robustness:.4f}\n")
            f.write("\n") # Add line breaks to separate information from different generations
    def _log_final_results(self, log_dir, topo, best_solution, best_objectives):
        """记录最终结果"""
        with open(f'{log_dir}/final_results.txt', 'w', encoding='utf-8') as f:
            f.write("Optimization completed\n")
            f.write(f"Best solution (binary): {best_solution}\n")
            f.write(f"FNDT of the best solution: {best_objectives[0]:.4f}\n")
            f.write(f"Robustness of the best solution: {best_objectives[1]:.4f}\n")

        # 保存最终的拓扑结构和结果
        with open(f'{log_dir}/final_results.pkl', 'wb') as fp:
            pickle.dump({
                'best_solution': best_solution,
                'best_objectives': best_objectives,
                'topo': topo,
            }, fp)

    def _save_pareto_front(self, best_individuals, iteration=None):
        plt.figure(figsize=(10, 8))
        # 提取FNDT和鲁棒性值
        fndt_values = [ind[0] for ind in best_individuals]
        robustness_values = [ind[1] for ind in best_individuals]
        
        # 绘制Pareto前沿
        plt.scatter(fndt_values, robustness_values, c='red', marker='*', s=100, label='Pareto前沿')
        
        plt.xlabel('FNDT')
        plt.ylabel('Robustness')
        plt.title('Pareto Front')
        plt.legend()
        plt.grid(True)
        
        # 如果提供了迭代次数，则保存到特定的迭代文件夹中
        if iteration is not None:
            # 构建文件名：TEAM_N节点数_超级节点比例_迭代次数
            folder_path = f'{self.picture_dir}/{self.algorithm_name}/pareto_front_{self.algorithm_name}_N{self.N}_SP{self.super_node_percent*100}'
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(f'{folder_path}/iter_{iteration}.png')
        else:
            # 保存到results目录
            plt.savefig(f'{self.results_dir}/{self.algorithm_name}/pareto_front_{self.N}_{self.super_node_percent}.png')
        
        plt.close()
    # ---------- public API ----------
    # 在主循环中增加支配检查和替换步骤
    def optimize(self, log_dir: str, final_dir:str,fobj: Callable[[np.ndarray], float], paramlen: int,
                bound: Tuple[float, float] = (-1.0, 1.0), *, verbose: bool = False,
                Reduce_GStart: Optional[Sequence[float]] = None,
                topo: Any = None, ID_Chrome_in_complete: Any = None, Len_Interval: Any = None) -> Tuple[np.ndarray, float, np.ndarray]:
        lb = np.full(paramlen, bound[0])
        ub = np.full(paramlen, bound[1])

        rng = np.random.default_rng(self.seed)
        # -- initialise population ------------------------------------------
        x = self._gd_initialization(self.pop_size, paramlen, ub, lb)
        fit = np.apply_along_axis(fobj, 1, x)
        p_x = x.copy()
        p_fit = fit.copy()
        best_idx = fit.argmin()
        best_x = x[best_idx].copy()
        best_f = float(fit[best_idx])
        curve = np.empty(self.max_iteration)
        XX = p_x.copy()

        # 预先计算所有个体的 FNDT 和 Robustness 值
        fndt_values = np.zeros(self.pop_size)
        robustness_values = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            binary_individual = (x[i] >= 0).astype(int)
            topo.G = topo.reconnect_G(binary_individual, ID_Chrome_in_complete, Len_Interval)
            fndt_values[i] = topo.calculate_FNDT()
            robustness_values[i] = topo.calculate_robustness()

        # -- Log initial state -------------------------------------------------
        self._log_initial_state(log_dir, fndt_values[best_idx], robustness_values[best_idx])

        # -- main loop -------------------------------------------------------
        for t in range(1, self.max_iteration + 1):
            worse = x[fit.argmax()]
            # 1) producers phase -----------------------------------------
            self._producers_phase(rng, x, fit, p_x, worse, lb, ub, fobj, XX)
            best_idx = fit.argmin()
            best_xx = x[best_idx]
            # 2) dynamic radius -----------------------------------------
            R = 0.5 * (math.cos(math.pi * t / self.max_iteration) + 1)
            Xnew11 = self._bounds(best_x * (1 - R), lb, ub)
            Xnew22 = self._bounds(best_x * (1 + R), lb, ub)
            # 3) predator avoidance ------------------------------------
            self._predator_avoidance(rng, x, fit, p_x, lb, ub, fobj, R, Xnew11, Xnew22)
            # 4) wandering ----------------------------------------------
            self._wandering(rng, x, fit, p_x, lb, ub, fobj, best_x, best_xx)
            # 5) quantum mutation ---------------------------------------
            best_x, best_f = self._quantum_mutation(rng, best_x, best_f, lb, ub, fobj, t)
            # 6) update personal / global best --------------------------
            XX = p_x.copy()
            best_f = self._update_personal_best(x, fit, p_x, p_fit, best_x, best_f)
            curve[t - 1] = best_f

            # 重新计算所有个体的 FNDT 和 Robustness 值
            for i in range(self.pop_size):
                binary_individual = (x[i] >= 0).astype(int)
                topo.G = topo.reconnect_G(binary_individual, ID_Chrome_in_complete, Len_Interval)
                fndt_values[i] = topo.calculate_FNDT()
                robustness_values[i] = topo.calculate_robustness()

            # -- Log generation state -----------------------------------------
            self._log_generation_state(log_dir, t, fit, fndt_values, robustness_values)

            # -- Domination check and replacement ------------------------
            # 检查当前种群中的解是否被其他解支配
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i != j and self._is_dominated(x[j], x[i], fndt_values[j], robustness_values[j], fndt_values[i], robustness_values[i]):
                        # 如果 x[i] 被 x[j] 支配，则用 x[j] 替换 x[i]
                        x[i] = x[j].copy()
                        fit[i] = fit[j]

            # -- Log and print current iteration results -----------------
            if verbose:
                # Calculate FNDT and Robustness for the best individual
                best_binary = (best_x >= 0).astype(int)
                topo.G = topo.reconnect_G(best_binary, ID_Chrome_in_complete, Len_Interval)
                best_FNDT = topo.calculate_FNDT()
                best_robustness = topo.calculate_robustness()

                # Calculate FNDT and Robustness for all individuals
                all_FNDT = fndt_values.copy()
                all_robustness = robustness_values.copy()
                all_fit_abs = abs(fit).copy()

                print(f"Iteration {t}/{self.max_iteration}:")
                print(f"  Best Individual: FNDT={best_FNDT:.4f}, Robustness={best_robustness:.4f}, Score={-best_f:.6f}")
                bstind = []
                # Print each individual's FNDT, Robustness, and absolute fit
                for i in range(self.pop_size):
                    bstind.append([all_FNDT[i], all_robustness[i], all_fit_abs[i]])
                    print(f"  Individual {i+1}: FNDT={all_FNDT[i]:.4f}, Robustness={all_robustness[i]:.4f}, Fit={all_fit_abs[i]:.6f}")
                self._save_pareto_front(bstind, t)
                self._log_generation_state('./log', t, fit, fndt_values, robustness_values)
        # -- Log final results ---------------------------------------------
        self._log_final_results(final_dir, topo, best_x, np.array([best_FNDT, best_robustness]))

        self.best_x, self.best_f, self.curve = best_x, best_f, curve
        if verbose:
            print(f"QHDBO: best_cost={best_f:.6f}  (best_score = {-best_f:.6f})")
        return best_x, best_f, curve

    # ----------------- helper methods (same as earlier version) -------------
    # (Paste full definitions of _bounds, _gd_initialization, _producers_phase,
    #  _predator_avoidance, _wandering, _quantum_mutation, _update_personal_best
    #  here. They are unchanged and omitted for brevity.)
        # --------------------- 工具函数 ---------------------
    @staticmethod
    def _bounds(vec: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(vec, lb), ub)

    @classmethod
    def _gd_initialization(
        cls, pop: int, dim: int, ub: np.ndarray, lb: np.ndarray
    ) -> np.ndarray:
        gp = cls._goodenode(pop, dim)
        return gp * (ub - lb) + lb
    import numpy as np

    @classmethod
    def _goodenode(cls, pop: int, dim: int) -> np.ndarray:
        """
        生成佳点集

        :param pop: 种群大小
        :param dim: 维度
        :return: 佳点集
        """
        # 生成索引
        k = np.arange(1, pop + 1)[:, None] * np.ones(dim)
        Ind = np.arange(1, dim + 1)
        
        # 产生素数
        prime1 = cls._primes(100 * dim)
        
        # 找到满足条件的最小素数
        q = np.where(prime1 >= (2 * dim + 3))[0][0]
        p = prime1[q]
        
        # 计算 tmp2
        tmp2 = 2 * np.cos((2 * np.pi * Ind) / p)
        
        # 计算 r
        r = np.ones(pop)[:, None] * tmp2
        
        # 计算 G_p
        G_p = k * r
        
        # 计算 GD
        GD = np.mod(G_p, 1)
        
        return GD

    @classmethod
    def _primes(cls, n: int) -> np.ndarray:
        """
        生成小于等于 n 的素数

        :param n: 上限
        :return: 素数数组
        """
        if n < 2:
            return np.array([])
        
        sieve = np.ones(n + 1, dtype=bool)
        sieve[:2] = False
        for i in range(2, int(np.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i * i:n + 1:i] = False
        
        return np.where(sieve)[0]
    def _producers_phase(
        self,
        rng: np.random.Generator,
        x: np.ndarray,
        fit: np.ndarray,
        p_x: np.ndarray,
        worse: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        XX: np.ndarray,
    ) -> None:
        p_num = self.pop_size
        for i in range(p_num):
            r2 = rng.random()
            if r2 < 0.9:
                a = 1 if rng.random() > 0.1 else -1
                x[i] = p_x[i] + 0.3 * np.abs(p_x[i] - worse) + a * 0.1 * XX[i]
            else:
                theta = math.radians(rng.integers(1, 181))
                x[i] = p_x[i] + math.tan(theta) * np.abs(p_x[i] - XX[i])
            x[i] = self._bounds(x[i], lb, ub)
            fit[i] = fobj(x[i])

    def _predator_avoidance(
        self,
        rng: np.random.Generator,
        x: np.ndarray,
        fit: np.ndarray,
        p_x: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        R: float,
        Xnew11: np.ndarray,
        Xnew22: np.ndarray,
    ) -> None:
        start = self.pop_size
        end = int(self.pop_size * 0.38)
        for i in range(start, end):
            if rng.random() < 0.2 + R * 0.6:
                radius = (ub - lb) / 2 * R
                sin_term = math.sin(rng.random() * math.pi)
                angle = rng.random() * 2 * math.pi
                perturb = radius * rng.random() * sin_term * np.array(
                    [math.cos(angle), math.sin(angle)] * (len(lb) // 2 + 1)
                )[: len(lb)]
                x[i] = p_x[i] + perturb
            else:
                x[i] = p_x[i] + (
                    rng.standard_normal(len(lb)) * (p_x[i] - Xnew11)
                    + rng.random(len(lb)) * (p_x[i] - Xnew22)
                )
            x[i] = self._bounds(x[i], lb, ub)
            fit[i] = fobj(x[i])

    def _wandering(
        self,
        rng: np.random.Generator,
        x: np.ndarray,
        fit: np.ndarray,
        p_x: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        best_x: np.ndarray,
        best_xx: np.ndarray,
    ) -> None:
        start = int(self.pop_size * 0.38)
        for j in range(start, self.pop_size):
            # 生成新的解
            new_x = best_x + rng.standard_normal(len(lb)) * (
                (np.abs(p_x[j] - best_xx) + np.abs(p_x[j] - best_x)) / 2
            )
            new_x = self._bounds(new_x, lb, ub)

            # 计算新解的适应度
            new_fit = fobj(new_x)

            # 只有当新解的适应度更优时，才更新解和适应度
            if new_fit < fit[j]:
                x[j] = new_x
                fit[j] = new_fit
        # 定义支配关系函数
    # 定义支配关系函数
    def _is_dominated(self, a: np.ndarray, b: np.ndarray, fndt_a: float, robustness_a: float, fndt_b: float, robustness_b: float) -> bool:
        # 判断 a 是否支配 b
        if (fndt_a <= fndt_b and robustness_a >= robustness_b) and (fndt_a < fndt_b or robustness_a > robustness_b):
            return True
        return False
    def _quantum_mutation(
        self,
        rng: np.random.Generator,
        best_x: np.ndarray,
        best_f: float,
        lb: np.ndarray,
        ub: np.ndarray,
        fobj: Callable[[np.ndarray], float],
        t: int,
    ):
        norm_best = np.linalg.norm(best_x) or 1e-16
        alpha = best_x / norm_best
        beta = np.where(rng.random(len(lb)) < 0.5, 1, -1) * np.sqrt(1 - alpha ** 2)
        Z_R = np.vstack((alpha, beta))
        rot = rng.random() * 2 * math.pi
        Rm = np.array([[math.cos(rot), -math.sin(rot)],
                       [math.sin(rot),  math.cos(rot)]])
        Z_R2 = Rm @ Z_R
        cands = [
            best_x + rng.standard_t(t) * alpha * norm_best,
            best_x + rng.standard_t(t) * beta  * norm_best,
            best_x + rng.standard_t(t) * Z_R2[0] * norm_best,
            best_x + rng.standard_t(t) * Z_R2[1] * norm_best,
        ]
        f_cands = [fobj(self._bounds(c, lb, ub)) for c in cands]
        idx = int(np.argmin(f_cands))
        if f_cands[idx] < best_f:
            best_f = f_cands[idx]
            best_x = self._bounds(cands[idx], lb, ub)
        return best_x, best_f
    
    def _update_personal_best(
        self,
        x: np.ndarray,
        fit: np.ndarray,
        p_x: np.ndarray,
        p_fit: np.ndarray,
        best_x: np.ndarray,
        best_f: float,
    ):
        for i in range(self.pop_size):
            if fit[i] < p_fit[i]:
                p_fit[i] = fit[i]
                p_x[i] = x[i]
            if p_fit[i] < best_f:
                best_f = p_fit[i]
                best_x = p_x[i]
        self.best_x = best_x
        return best_f

class QHDBOopt:
    """Wrapper mirroring `QPSOopt` where *N* is node count."""

    def __init__(self, N: int, super_node_percent: float, topo: Any = None,
                 max_iteration: int = 100, need_dynamic: bool = False,
                 draw_w: bool = False):
        self.node_num = N                       # 节点数量 (与 QPSOopt 一致)
        self.super_node_percent = super_node_percent
        self.topo = topo
        self.max_iteration = max_iteration
        self.need_dynamic = need_dynamic
        self.draw_w = draw_w

        # ------- population size (fixed) -----------------------------------
        self.beetle_num = 30                   # 等价于 QPSOopt 的 particle_num

        # directory structure
        self.base_dir = './log'
        self.results_dir = './results'
        self.picture_dir = './picture'
        self.algorithm_name = 'QHDBO'
        os.makedirs(f"{self.base_dir}/{self.algorithm_name}", exist_ok=True)
        self.log_dir = f'{self.base_dir}/{self.algorithm_name}/{N}_{int(100*self.super_node_percent)}'
        self.final_dir = f'{self.results_dir}/{self.algorithm_name}'

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/{self.algorithm_name}", exist_ok=True)
        os.makedirs(f"{self.picture_dir}/{self.algorithm_name}", exist_ok=True)
        # bookkeeping attributes
        self.Reduce_GStart = None
        self.paramlen = 0
        self.seed = 12
        self.ID_Chrome_in_complete = None
        self.Len_Interval = None

        # results
        self.best_solution: Optional[np.ndarray] = None
        self.best_objectives: Optional[np.ndarray] = None
        self.G_total: list[np.ndarray] = []

        # internal core optimiser uses beetle_num, not node count
        self._core = _CoreQHDBO(self.beetle_num, super_node_percent, seed=self.seed,  max_iteration=self.max_iteration, N=N)

    # ---------------------------------------------------------------------
    def optimize(self, Reduce_GStart, paramlen, ID_Chrome_in_complete, Len_Interval, verbose: bool = True):
        print(f"开始QHDBO优化，节点数量: {self.node_num}, 超级节点比例: {self.super_node_percent}")

        # store inputs
        self.Reduce_GStart = Reduce_GStart
        self.paramlen = paramlen
        self.ID_Chrome_in_complete = ID_Chrome_in_complete
        self.Len_Interval = Len_Interval

        # objective function for core optimiser
        fobj = _make_fobj(self.topo, self.ID_Chrome_in_complete, self.Len_Interval)

        # run optimisation ---------------------------------------------------
        best_x, best_cost, curve = self._core.optimize(self.log_dir,self.final_dir,
            Reduce_GStart=self.Reduce_GStart,
            fobj=fobj, paramlen=self.paramlen, verbose=verbose,
            topo=self.topo, ID_Chrome_in_complete=self.ID_Chrome_in_complete, Len_Interval=self.Len_Interval)

        # convert real‑valued beetle position to binary chromosome
        best_binary = (best_x >= 0).astype(int)
        self.topo.G = self.topo.reconnect_G(best_binary, self.ID_Chrome_in_complete, self.Len_Interval)
        best_FNDT = self.topo.calculate_FNDT()
        best_robustness = self.topo.calculate_robustness()

        self.best_solution = best_binary
        self.best_objectives = np.array([best_FNDT, best_robustness])
        self.G_total.append(copy.deepcopy(self.topo.G))

        print(f"QHDBO优化完成: FNDT={best_FNDT:.4f}, Robustness={best_robustness:.4f}, Score={-best_cost:.6f}")
        return self.best_solution, self.best_objectives, self.G_total