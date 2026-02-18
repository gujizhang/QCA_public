"""
QPSOopt：多目标量子粒子群优化算法(MQPSO)
实现了多目标量子粒子群算法的核心功能，用于网络拓扑优化
"""
import os
import numpy as np
import time
import copy
import random
import math
import matplotlib.pyplot as plt
from typing import List, Any, Tuple, Dict
import pickle
class QPSOopt:
    def __init__(self, N, super_node_percent, topo=None, max_iteration=100, need_dynamic=False, draw_w=False):
        """
        初始化QPSO优化器

        参数:
            N: 节点数量
            super_node_percent: 超级节点百分比
            topo: 拓扑结构对象，如果为None则使用默认参数生成
            max_iteration: 最大迭代次数
            need_dynamic: 是否需要动态显示
            draw_w: 是否绘制权重图
        """
        self.N = N  # 普通节点总数
        self.super_node_percent = super_node_percent  # 超级节点所占的比例
        self.topo = topo  # 拓扑结构对象
        self.max_iteration = max_iteration  # 最大迭代次数
        self.need_dynamic = need_dynamic  # 是否需要动态显示
        self.draw_w = draw_w  # 是否绘制权重图
    
        # 默认参数设置
        self.seed = 12  # 随机种子
        np.random.seed(self.seed)
        random.seed(self.seed)
    
        # QPSO算法参数
        self.particle_num = 20  # 粒子数量
        self.alpha1 = 1.0  # 初始收缩扩张系数
        self.alpha2 = 0.5  # 最终收缩扩张系数
        self.grid_num = 10  # 网格等分数量
        self.archive_size = 50  # 外部存档阈值
        self.max_value = 1.0  # 参数最大值
        self.min_value = 0.0  # 参数最小值
    
        # 路径设置
        self.base_dir = './log'  # 基础日志目录
        self.results_dir = './results'  # 结果保存目录
        self.picture_dir = './picture'  # 图片保存目录
        self.algorithm_name = 'QPSO'  # 算法名称
        
        # 创建必要的目录
        for directory in [
            f'{self.base_dir}/{self.algorithm_name}',
            f'{self.results_dir}/{self.algorithm_name}',
            f'{self.picture_dir}/{self.algorithm_name}'
        ]:
            os.makedirs(directory, exist_ok=True)
    
        # 存储优化过程中的数据
        self.ID_Chrome_in_complete = None  # 完整的ID_Chrome
        self.Len_Interval = None  # 每个区间的长度
        self.paramlen = 0  # 编码后的ID_Chrome长度
        self.Reduce_GStart = None  # 编码后的ID_Chrome
        self.best_solution = None  # 最优解
        self.best_objectives = None  # 最优解的目标函数值
        
        # 粒子群相关变量
        self.PS = None  # 当前粒子群位置
        self.Pbest = None  # 个体历史最优位置
        self.fit_Pbest = None  # 个体历史最优适应度
        self.Gbest = None  # 全局最优位置
        self.fit_Gbest = None  # 全局最优适应度
        self.archive = []  # 外部存档
        self.fit_archive = []  # 外部存档适应度
        self.G_total = []  # 记录每一代中全局最优个体所对应的邻接矩阵

    def cal_obj(self, Chrom):
        """
        计算目标函数值
        
        参数:
            Chrom: 染色体
            
        返回:
            目标函数值列表 [FNDT, robustness]
        """
        # 根据染色体重新连接拓扑图的超级节点
        self.topo.G = self.topo.reconnect_G(Chrom, self.ID_Chrome_in_complete, self.Len_Interval)
        # 计算FNDT和鲁棒性这两个目标函数的值
        energy_FNDT = self.topo.calculate_FNDT()
        robustness = self.topo.calculate_robustness()
        return [energy_FNDT, robustness]

    def measure(self, particle):
        """
        对量子粒子进行测量，转化为二进制编码
        
        参数:
            particle: 量子粒子
            
        返回:
            测量后的二进制编码
        """
        # 将连续值转换为二进制编码
        binary_code = []
        for i in range(len(particle)):
            # 使用概率阈值将连续值转换为0或1
            if random.random() < particle[i]:
                binary_code.append(1)
            else:
                binary_code.append(0)
        
        return binary_code

    def get_noninferior(self, fit_values):
        """
        获取非劣解集的索引
        
        参数:
            fit_values: 适应度值列表
            
        返回:
            非劣解集的索引列表
        """
        fit_values = np.array(fit_values)
        p_num = fit_values.shape[0]  # 粒子数量
        index = []  # 存放非劣解集的索引
        
        for i in range(p_num):
            dominated = False
            for j in range(p_num):
                if i != j:
                    # 判断j是否支配i
                    if (fit_values[j, 0] >= fit_values[i, 0] and fit_values[j, 1] > fit_values[i, 1]) or \
                       (fit_values[j, 0] > fit_values[i, 0] and fit_values[j, 1] >= fit_values[i, 1]):
                        dominated = True
                        break
            if not dominated:
                index.append(i)
        
        return index

    def grid_divide(self, fit_values):
        """
        使用自适应网格法确定每个非劣解所属的网格
        
        参数:
            fit_values: 适应度值列表
            
        返回:
            网格信息
        """
        fit_values = np.array(fit_values)
        p_num = fit_values.shape[0]  # 粒子数量
        
        # 找出每个目标函数的最大值和最小值
        f1_min = np.min(fit_values[:, 0])
        f1_max = np.max(fit_values[:, 0])
        f2_min = np.min(fit_values[:, 1])
        f2_max = np.max(fit_values[:, 1])
        
        # 计算每个网格的大小
        if f1_max == f1_min:
            d1 = 0.1  # 避免除以零
        else:
            d1 = (f1_max - f1_min) / self.grid_num
            
        if f2_max == f2_min:
            d2 = 0.1  # 避免除以零
        else:
            d2 = (f2_max - f2_min) / self.grid_num
        
        # 确定每个非劣解所属的网格
        grid_inform = np.zeros((p_num, 3))  # 第一列和第二列分别表示横纵坐标，第三列表示该网格中解的数量
        # 计算每个解所属的网格，并计算每个网格中解的数量，
        # math.ceil((fit_values[i, 0] - f1_min) / d1)是取整，这个数代表这个解在哪个网格里
        for i in range(p_num):
            if f1_max == f1_min:
                grid_inform[i, 0] = 1
            else:
                grid_inform[i, 0] = math.ceil((fit_values[i, 0] - f1_min) / d1)
                if grid_inform[i, 0] == 0:
                    grid_inform[i, 0] = 1
                    
            if f2_max == f2_min:
                grid_inform[i, 1] = 1
            else:
                grid_inform[i, 1] = math.ceil((fit_values[i, 1] - f2_min) / d2)
                if grid_inform[i, 1] == 0:
                    grid_inform[i, 1] = 1
        
        # 计算每个网格中解的数量
        for i in range(p_num):
            grid_x = grid_inform[i, 0]
            grid_y = grid_inform[i, 1]
            count = 0
            for j in range(p_num):
                if grid_inform[j, 0] == grid_x and grid_inform[j, 1] == grid_y:
                    count += 1
            grid_inform[i, 2] = count
        
        return grid_inform

    def cut_noninferior(self, archive, fit_archive, grid_inform):
        """
        当外部存档超过阈值时，删减非劣解集
        
        参数:
            archive: 外部存档
            fit_archive: 外部存档适应度
            grid_inform: 网格信息
            
        返回:
            删减后的外部存档和适应度
        """
        archive = copy.deepcopy(archive)
        fit_archive = copy.deepcopy(fit_archive)
        
        while len(archive) > self.archive_size:
            # 找出解最多的网格
            max_count = 0
            max_grid_x = 0
            max_grid_y = 0
            
            for i in range(len(grid_inform)):
                if grid_inform[i, 2] > max_count:
                    max_count = grid_inform[i, 2]
                    max_grid_x = grid_inform[i, 0]
                    max_grid_y = grid_inform[i, 1]
            
            # 找出该网格中的所有解
            grid_solutions = []
            for i in range(len(grid_inform)):
                if grid_inform[i, 0] == max_grid_x and grid_inform[i, 1] == max_grid_y:
                    grid_solutions.append(i)
            
            # 随机选择一个解删除
            if grid_solutions:
                delete_index = random.choice(grid_solutions)
                archive.pop(delete_index)
                fit_archive.pop(delete_index)
                grid_inform = np.delete(grid_inform, delete_index, axis=0)
                
                # 更新网格中解的数量
                for i in range(len(grid_inform)):
                    if grid_inform[i, 0] == max_grid_x and grid_inform[i, 1] == max_grid_y:
                        grid_inform[i, 2] -= 1
        
        return archive, fit_archive

    def select_Gbest(self, grid_inform):
        """
        从外部存档中选择全局最优解
        
        参数:
            grid_inform: 网格信息
            
        返回:
            全局最优解和适应度
        """
        # 计算每个网格的选择概率
        p_num = len(self.archive)
        prob = np.zeros(p_num)
        
        for i in range(p_num):
            prob[i] = 10 / grid_inform[i, 2]  # 网格中解越少，被选中的概率越大
        
        # 归一化概率
        prob_sum = np.sum(prob)
        if prob_sum > 0:
            prob = prob / prob_sum
        else:
            prob = np.ones(p_num) / p_num
        
        # 轮盘赌选择
        cumsum = np.cumsum(prob)
        r = random.random()
        for i in range(p_num):
            if r <= cumsum[i]:
                return self.archive[i], self.fit_archive[i]
        
        # 如果没有选中，返回第一个解
        return self.archive[0], self.fit_archive[0]

    def MQPSO(self, Gbesti, alpha):
        """
        多目标量子粒子群算法核心函数
        
        参数:
            Gbesti: 全局最优位置
            alpha: 收缩扩张系数
            
        返回:
            更新后的粒子群位置、个体最优位置和适应度
        """
        p_num = len(self.Pbest)  # 粒子群的行数
        p_vd = len(self.Pbest[0])  # 粒子群的列数，既变量的个数
        Mbest = np.sum(self.Pbest, axis=0) / p_num  # Pbest的平均值，表示平均粒子历史最好位置

        # 更新粒子位置
        for i in range(p_num):
            b = random.uniform(0, 1)  # 在(0,1)上均匀分布的随机数值
            u = random.uniform(0, 1)  # 在(0,1)上均匀分布的随机数值
            P = b * self.Pbest[i, :] + (1 - b) * Gbesti  # 粒子位置更新的中间量
            if random.random() > 0.5:
                self.PS[i, :] = P + alpha * abs(Mbest - self.PS[i, :]) * math.log(1 / u)
            else:
                self.PS[i, :] = P - alpha * abs(Mbest - self.PS[i, :]) * math.log(1 / u)

        # 将新结果中每个参数的值限制在最大值和最小值之间
        for k in range(p_vd):
            max_temp = max(self.PS[:, k])
            min_temp = min(self.PS[:, k])
            if max_temp == min_temp:
                self.PS[:, k] = min_temp
            else:
                for ik in range(p_num):
                    self.PS[ik, k] = (self.PS[ik, k] - min_temp) / (max_temp - min_temp) * (
                                self.max_value - self.min_value) + self.min_value
        
        # 更新个体最优
        for i in range(p_num):
            # 对粒子进行测量，并转化为二进制编码
            binary_code = self.measure(self.PS[i, :])
            # 计算适应度值
            fit_1 = self.cal_obj(binary_code)
            
            # 比较新旧适应度
            if fit_1[0] >= self.fit_Pbest[i, 0] and fit_1[1] >= self.fit_Pbest[i, 1]:  # 新个体支配旧个体
                self.fit_Pbest[i, :] = fit_1  # 更新个体最优所对应的适应度值
                self.Pbest[i, :] = self.PS[i, :]
            elif fit_1[0] < self.fit_Pbest[i, 0] and fit_1[1] < self.fit_Pbest[i, 1]:  # 旧个体支配新个体
                continue
            else:  # 两个个体之间不存在支配关系，随机选择一个
                rd = random.random()
                if rd <= 0.5:  # 选择新个体
                    self.fit_Pbest[i, :] = fit_1  # 更新个体最优所对应的适应度值
                    self.Pbest[i, :] = self.PS[i, :]
        
        return self.PS, self.Pbest, self.fit_Pbest

    def draw_pareto_front(self, fit_Pbest, fit_archive, iteration=None):
        """
        绘制Pareto前沿
        
        参数:
            fit_Pbest: 个体历史最优适应度
            fit_archive: 外部存档适应度
            iteration: 当前迭代次数，如果提供则保存特定迭代的图像
        """
        plt.figure(figsize=(10, 8))
        # 绘制当前种群的适应度值
        plt.scatter(fit_Pbest[:, 0], fit_Pbest[:, 1], c='blue', marker='o', label='当前种群')
        # 绘制外部存档中的非劣解
        plt.scatter(np.array(fit_archive)[:, 0], np.array(fit_archive)[:, 1], c='red', marker='*', s=100, label='Pareto前沿')
        
        plt.xlabel('FNDT')
        plt.ylabel('Robustness')
        plt.title('Pareto Front')
        plt.legend()
        plt.grid(True)
        
        # 如果提供了迭代次数，则保存到特定的迭代文件夹中
        if iteration is not None:
            # 构建文件名：QPSO_N节点数_超级节点比例_迭代次数
            folder_path = f'{self.picture_dir}/{self.algorithm_name}/pareto_front_{self.algorithm_name}_N{self.N}_SP{self.super_node_percent*100}'
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(f'{folder_path}/iter_{iteration}.png')
        else:
            # 保存到results目录
            plt.savefig(f'{self.results_dir}/{self.algorithm_name}/pareto_front_{self.N}_{self.super_node_percent}.png')
        
        plt.close()
    def _log_initial_state(self, log_dir):
        """记录初始状态"""
        with open(f'{log_dir}/initial_state.txt', 'w', encoding='utf-8') as f:
            f.write(f"节点数量: {self.N}\n")
            f.write(f"超级节点比例: {self.super_node_percent}\n")
            f.write(f"初始FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"初始鲁棒性: {self.topo.calculate_robustness()}\n")
            f.write(f"粒子数量: {self.particle_num}\n")
            f.write(f"最大迭代次数: {self.max_iteration}\n")
            f.write(f"初始收缩扩张系数: {self.alpha1}\n")
            f.write(f"最终收缩扩张系数: {self.alpha2}\n")
            f.write(f"网格等分数量: {self.grid_num}\n")
            f.write(f"外部存档阈值: {self.archive_size}\n")
        
        # 清空generation_all.txt文件
        with open(f'{log_dir}/generation_all.txt', 'w', encoding='utf-8') as f:
            pass

    def _log_generation_state(self, log_dir, gen):
        """记录每一代的状态"""
        # 使用追加模式打开文件
        with open(f'{log_dir}/generation_all.txt', 'a', encoding='utf-8') as f:
            f.write(f"第 {gen} 代\n")
            f.write(f"全局最优适应度: {self.fit_Gbest[gen, :]}\n")
            f.write(f"外部存档大小: {len(self.archive)}\n")
            f.write("\n外部存档适应度值:\n")
            for i, fv in enumerate(self.fit_archive):
                f.write(f"解 {i}: {fv}\n")
            f.write("\n")  # 添加换行以分隔不同代的信息

    def _log_final_results(self, log_dir):
        """记录最终结果"""
        with open(f'{log_dir}/final_results.txt', 'w', encoding='utf-8') as f:
            f.write("优化完成\n")
            f.write(f"最终外部存档: {self.archive}\n")
            f.write(f"最终外部存档适应度: {self.fit_archive}\n")
            f.write(f"最优解目标函数值: {self.best_objectives}\n")
            f.write(f"最终FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"最终鲁棒性: {self.topo.calculate_robustness()}\n")
        
        # 保存最终的拓扑结构和结果
        with open(f'{log_dir}/final_results.pkl', 'wb') as fp:
            pickle.dump({
                'best_solution': self.best_solution,
                'best_objectives': self.best_objectives,
                'G_total': self.G_total,
                'archive': self.archive,
                'fit_archive': self.fit_archive,
                'topo': self.topo
            }, fp)
    def optimize(self, Reduce_GStart, paramlen, ID_Chrome_in_complete, Len_Interval):
        """
        执行QPSO优化算法
        
        参数:
            Reduce_GStart: 编码后的ID_Chrome
            paramlen: 编码后的ID_Chrome长度
            ID_Chrome_in_complete: 完整的ID_Chrome
            Len_Interval: 每个区间的长度
            
        返回:
            最优解和目标函数值
        """
        print(f"开始QPSO优化，节点数量: {self.N}, 超级节点比例: {self.super_node_percent}")
        
        # 创建日志目录
        log_dir = f'{self.base_dir}/{self.algorithm_name}/{self.N}_{int(100*self.super_node_percent)}'
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 存储参数
        self.Reduce_GStart = Reduce_GStart
        self.paramlen = paramlen
        self.ID_Chrome_in_complete = ID_Chrome_in_complete
        self.Len_Interval = Len_Interval
        
        # 初始化粒子群
        self.PS = np.random.random((self.particle_num, self.paramlen))  # 初始化粒子群位置
        self.Pbest = copy.deepcopy(self.PS)  # 初始化个体历史最优位置
        self.fit_Pbest = np.zeros((self.particle_num, 2))  # 初始化个体历史最优适应度
        
        # 计算初始适应度
        for i in range(self.particle_num):
            binary_code = self.measure(self.PS[i, :])
            self.fit_Pbest[i, :] = self.cal_obj(binary_code)
        
        # 初始化全局最优
        self.Gbest = np.zeros((self.max_iteration + 1, self.paramlen))
        self.fit_Gbest = np.zeros((self.max_iteration + 1, 2))
        
        # 初始化外部存档
        index = self.get_noninferior(self.fit_Pbest)
        self.archive = []
        self.fit_archive = []
        
        for i in range(len(index)):
            ix = index[i]
            self.archive.append(self.Pbest[ix, :])
            self.fit_archive.append(self.fit_Pbest[ix, :])
        
        # 选择初始全局最优
        grid_inform = self.grid_divide(self.fit_archive)
        self.Gbest[0, :], self.fit_Gbest[0, :] = self.select_Gbest(grid_inform)
        
        # 记录初始状态
        self._log_initial_state(log_dir)
        
        # 开始迭代优化
        start_time = time.time()
        
        for gen in range(self.max_iteration):
            # 根据迭代次数线性动态减小alpha
            alpha = (self.alpha1 - self.alpha2) * (self.max_iteration - gen) / self.max_iteration + self.alpha2
            
            # 执行MQPSO算法更新粒子位置
            self.PS, self.Pbest, self.fit_Pbest = self.MQPSO(self.Gbest[gen, :], alpha)
            
            # 第一次筛选：根据当前代粒子的适应度值，筛选出非劣解集
            index1 = self.get_noninferior(self.fit_Pbest)
            for i in range(len(index1)):
                ix = index1[i]
                self.archive.append(self.Pbest[ix, :])
                self.fit_archive.append(self.fit_Pbest[ix, :])
            
            # 第二次筛选：对合并后的解集重新筛选非劣解
            index2 = self.get_noninferior(self.fit_archive)
            archive2 = []
            fit_archive2 = []
            
            for i in range(len(index2)):
                ix = index2[i]
                archive2.append(self.archive[ix])
                fit_archive2.append(self.fit_archive[ix])
            
            # 使用自适应网格法确定每个非劣解所属的网格
            grid_inform = self.grid_divide(fit_archive2)
            
            # 如果外部存档超过阈值，执行删减操作
            if len(archive2) > self.archive_size:
                archive2, fit_archive2 = self.cut_noninferior(archive2, fit_archive2, grid_inform)
                grid_inform = self.grid_divide(fit_archive2)
            
            # 更新外部存档
            self.archive = archive2
            self.fit_archive = fit_archive2
            
            # 选择全局最优
            self.Gbest[gen + 1, :], self.fit_Gbest[gen + 1, :] = self.select_Gbest(grid_inform)
            
            # 记录全局最优个体对应的邻接矩阵
            binary_code = self.measure(self.Gbest[gen + 1, :])
            self.topo.G = self.topo.reconnect_G(binary_code, self.ID_Chrome_in_complete, self.Len_Interval)
            self.G_total.append(copy.deepcopy(self.topo.G))
            
            # 输出当前迭代信息
            print(f"QPSO: 迭代 {gen + 1}/{self.max_iteration}, FNDT: {self.fit_Gbest[gen + 1, 0]:.4f}, Robustness: {self.fit_Gbest[gen + 1, 1]:.4f}")
            
            # 记录每一代的状态
            self._log_generation_state(log_dir, gen)
            
            # 绘制当前Pareto前沿并保存
            #每隔10代绘制一次
            if (gen + 1) % 10 == 0:
                self.draw_pareto_front(self.fit_Pbest, self.fit_archive, gen)
        
        # 计算优化时间
        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"QPSO优化完成，耗时: {optimization_time:.2f}秒")
        
        # 绘制最终Pareto前沿
        self.draw_pareto_front(self.fit_Pbest, self.fit_archive)
        
        # 如果需要绘制权重图，则绘制最优解对应的拓扑图
        if self.draw_w:
            self.topo.plot_topology()
        
        # 找出最优解
        best_index = np.argmax(self.fit_Gbest[:, 0] + self.fit_Gbest[:, 1])
        self.best_solution = self.Gbest[best_index, :]
        self.best_objectives = self.fit_Gbest[best_index, :]
        
        # 记录最终结果
        self._log_final_results(log_dir)
        
        # 返回最优解和目标函数值
        return self.best_solution, self.best_objectives, self.G_total