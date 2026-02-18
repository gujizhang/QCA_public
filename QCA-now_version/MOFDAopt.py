import numpy as np
import os
import random
import matplotlib.pyplot as plt
import logging
import pickle

class MOFDAopt:
    def __init__(self, N, super_node_percent, topo, max_iteration, need_dynamic, draw_w):
        """
        初始化 MOFDAopt 类。

        参数:
        Archive_size (int): 存档大小。
        alpha (int): 粒子数量。
        dim (int): 维度。
        N (int): 节点数量。
        super_node_percent (float): 超级节点比例。
        topo (object): 拓扑结构对象。
        max_iteration (int): 最大迭代次数。
        need_dynamic (bool): 是否动态展示。
        draw_w (bool): 是否画出权重图。
        """
        self.Archive_size = 20
        self.alpha = 40
        self.dim = 0
        self.N = N
        self.super_node_percent = super_node_percent
        self.topo = topo
        self.max_iteration = max_iteration
        self.need_dynamic = need_dynamic
        self.draw_w = draw_w
        self.ID_Chrome_in_complete = []
        self.Len_Interval = None
        self.Archive_costs = []
        self.Archive = []
        self.boomAlpha = 0.1
        self.nGrid = 10
        self.dim = 1
        self.gamma = 2
        self.children_num = 2
        # 初始化权重作为超参数
        self.weight1 = 0.7
        self.weight2 = 0.3
        self.ub = 1
        self.lb = 0
        # 初始化 MAX_E 和 MAX_R 为 self 属性
        self.MAX_E = 300.0
        self.MAX_R = 0.5
        # directory structure
        self.base_dir = './log'
        self.results_dir = './results'
        self.picture_dir = './picture'
        self.algorithm_name = 'MOFDA'
        os.makedirs(f"{self.base_dir}/{self.algorithm_name}", exist_ok=True)
        self.log_dir = f'{self.base_dir}/{self.algorithm_name}/{N}_{int(100*self.super_node_percent)}'
        self.final_dir = f'{self.results_dir}/{self.algorithm_name}'

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/{self.algorithm_name}", exist_ok=True)
        os.makedirs(f"{self.picture_dir}/{self.algorithm_name}", exist_ok=True)

    def CreateEmptyParticle(self, alpha):
        """
        创建指定数量的空粒子。

        参数:
        alpha (int): 粒子数量。

        返回:
        list: 包含空粒子的列表。
        """
        return [{'Velocity': 0, 'Position': np.zeros(self.dim), 'Best': {'Position': np.zeros(self.dim), 'Cost': None}} for _ in range(alpha)]

    def DetermineDominations(self, particles):
        """
        确定粒子之间的支配关系。

        参数:
        particles (list): 粒子列表。

        返回:
        list: 更新了支配关系信息的粒子列表。
        """
        n = len(particles)
        for i in range(n):
            particles[i]['Dominated'] = []
            particles[i]['DominationCount'] = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    dominates = True
                    dominated = True
                    for k in range(len(particles[i]['Cost'])):
                        if particles[i]['Cost'][k] > particles[j]['Cost'][k]:
                            dominates = False
                        if particles[i]['Cost'][k] < particles[j]['Cost'][k]:
                            dominated = False
                    if dominates and not dominated:
                        particles[i]['Dominated'].append(j)
                        particles[j]['DominationCount'] += 1
        return particles

    def GetNonDominatedParticles(self, particles):
        """
        获取非支配粒子列表。

        参数:
        particles (list): 粒子列表。

        返回:
        list: 非支配粒子列表。
        """
        non_dominated = []
        for particle in particles:
            if particle['DominationCount'] == 0:
                non_dominated.append(particle)
        return non_dominated

    def GetCosts(self, particles):
        """
        获取粒子列表中所有粒子的成本值。

        参数:
        particles (list): 粒子列表。

        返回:
        np.ndarray: 包含所有粒子成本值的数组。
        """
        costs = []
        for particle in particles:
            if 'Cost' in particle:
                costs.append(particle['Cost'])
        return np.array(costs)

    def CreateHypercubes(self, Archive_costs, nGrid, boomAlpha):
        """
        创建超立方体网格。

        参数:
        Archive_costs (np.ndarray): 存档中粒子的成本值数组。
        nGrid (int): 每个维度的网格数量。
        boomAlpha (float): 超立方体膨胀系数。

        返回:
        dict: 包含超立方体信息的字典。
        """
        if len(Archive_costs) == 0:
            return None
        min_costs = np.min(Archive_costs, axis=0)
        max_costs = np.max(Archive_costs, axis=0)
        ranges = max_costs - min_costs
        inflated_min = min_costs - boomAlpha * ranges
        inflated_max = max_costs + boomAlpha * ranges
        hypercube_size = (inflated_max - inflated_min) / nGrid
        G = {
            'min': inflated_min,
            'max': inflated_max,
            'size': hypercube_size,
            'nGrid': nGrid
        }
        return G

    def GetGridIndex(self, particle, G):
        """
        获取粒子所在的超立方体网格索引。

        参数:
        particle (dict): 粒子信息字典。
        G (dict): 超立方体信息字典。

        返回:
        tuple: 包含网格索引和子索引的元组。
        """
        if G is None or 'Cost' not in particle:
            return None, None
        
        # 检查 G['size'] 是否为零
        if np.all(G['size'] == 0):
            return 0, None
        
        indices = np.floor((particle['Cost'] - G['min']) / G['size']).astype(int)
        indices = np.clip(indices, 0, G['nGrid'] - 1)
        grid_index = 0
        multiplier = 1
        for index in indices:
            grid_index += index * multiplier
            multiplier *= G['nGrid']
        return grid_index, None

    def optimize(self, Reduce_GStart, paramlen, ID_Chrome_in_complete, Len_Interval):

        """
        执行优化过程。

        参数:
        Reduce_GStart (any): 起始缩减参数。
        paramlen (int): 参数长度，也就是dim。
        ID_Chrome_in_complete (any): 不完整的染色体 ID。
        Len_Interval (any): 间隔长度。

        返回:
        np.ndarray: 存档中所有粒子的成本值数组。
        """
        # 将传入的参数赋值给实例属性
        self.ID_Chrome_in_complete = ID_Chrome_in_complete
        self.Len_Interval = Len_Interval
        # 初始化粒子
        self.dim = paramlen

        self.flow_x = self.CreateEmptyParticle(self.alpha)
        for i in range(self.alpha):
            self.flow_x[i]['Velocity'] = 0
            self.flow_x[i]['Position'] = np.zeros(self.dim)
            self.flow_x[i]['Best'] = {}
            self.flow_x[i]['Best']['Position'] = self.flow_x[i]['Position']
            self.flow_x[i]['Best']['Cost'] = self.flow_x[i].get('Cost', None)
            # self.lb 和 self.ub 
            self.flow_x[i]['Position'] = np.random.rand(1, self.dim) * (self.ub - self.lb) + self.lb
            # 这里假设fhd函数有对应的Python实现
            self.flow_x[i]['Cost'] = self.GetCost(self.flow_x[i]['Position'].T)

        # 确定支配关系并获取非支配粒子
        self.flow_x = self.DetermineDominations(self.flow_x)
        self.Archive = self.GetNonDominatedParticles(self.flow_x)

        #Archive里有很多个粒子，每个粒子是一个字典，字典里有Position，所以这里需要先循环再计算
        # 计算Archive中所有粒子的成本值

        for particle in self.Archive:
            self.Archive_costs.append(self.GetCost(particle['Position'].T))
        self.G = self.CreateHypercubes(self.Archive_costs, self.nGrid, self.boomAlpha)

        for i in range(len(self.Archive)):
            self.Archive[i]['GridIndex'], self.Archive[i]['GridSubIndex'] = self.GetGridIndex(self.Archive[i], self.G)

        # 初始化日志和存档历史
        self.logger = self._setup_logger()
        self._log_initial_state()
        self.archive_history = []

        for iter in range(self.max_iteration):  # 修改此处
            # 选择领导者
            Leader = self.SelectLeader(self.Archive, self.dim)
            # 计算 delta
            delta = (((1 - iter / self.max_iteration + 1e-8) ** (2 * np.random.randn())) * 
                     (np.random.rand(self.dim) * iter / self.max_iteration) * 
                     np.random.rand(self.dim))

            for i in range(self.alpha):
                neighbor_x = []
                for _ in range(self.children_num):
                    # 生成随机位置
                    Xrand = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                    # 生成邻居粒子位置
                    neighbor_pos = self.flow_x[i]['Position'] + np.random.randn(self.dim) * delta * (
                        np.random.rand() * Xrand - np.random.rand() * self.flow_x[i]['Position']
                    ) * np.linalg.norm(Leader['Position'] - self.flow_x[i]['Position'])
                    # 确保位置在边界内
                    neighbor_pos = np.maximum(neighbor_pos, self.lb)
                    neighbor_pos = np.minimum(neighbor_pos, self.ub)

                    neighbor = {'Position': neighbor_pos}
                    # 计算邻居粒子的成本
                    neighbor['Cost'] = self.GetCost(neighbor['Position'].T)
                    neighbor_x.append(neighbor)

                # 对邻居粒子按成本排序

                neighbor_costs = np.array([neighbor['Cost'] for neighbor in neighbor_x])
                # 移除硬编码的权重，使用 self 属性
                # 计算加权和
                weighted_sums = []
                if neighbor_costs.ndim == 1:
                    weighted_sum = neighbor_costs[0] * self.weight1 + neighbor_costs[1] * self.weight2
                    weighted_sums.append(weighted_sum)
                else:
                    for cost in neighbor_costs:
                        weighted_sum = cost[0] * self.weight1 + cost[1] * self.weight2
                        weighted_sums.append(weighted_sum)
                
                # 自定义排序
                indx = list(range(len(weighted_sums)))
                indx.sort(key=lambda i: weighted_sums[i])

                if np.linalg.norm(neighbor_x[indx[0]]['Cost']) < np.linalg.norm(self.flow_x[i]['Cost']):
                    # 将列表转换为 numpy 数组
                    neighbor_cost = np.array(neighbor_x[indx[0]]['Cost'])
                    flow_cost = np.array(self.flow_x[i]['Cost'])
                    # 计算速度调整因子
                    Sf = (neighbor_cost - flow_cost) / np.sqrt(
                        np.linalg.norm(neighbor_x[indx[0]]['Position'] - self.flow_x[i]['Position'])
                    )
                    V = np.random.randn() * np.linalg.norm(Sf)
                    Vmax = np.max(0.1 * (self.ub - self.lb))
                    Vmin = np.min(-0.1 * (self.ub - self.lb))
                    V = np.clip(V, -Vmax, -Vmin)
                    # 更新粒子位置
                    self.flow_x[i]['Position'] = self.flow_x[i]['Position'] + V * (
                        neighbor_x[indx[0]]['Position'] - self.flow_x[i]['Position']
                    ) / np.sqrt(np.linalg.norm(neighbor_x[indx[0]]['Position'] - self.flow_x[i]['Position']))
                else:
                    r = np.random.randint(0, self.alpha)
                    if np.linalg.norm(self.flow_x[r]['Cost']) <= np.linalg.norm(self.flow_x[i]['Cost']):
                        self.flow_x[i]['Position'] = self.flow_x[i]['Position'] + np.random.randn(self.dim) * (
                            self.flow_x[r]['Position'] - self.flow_x[i]['Position']
                        )
                    else:
                        self.flow_x[i]['Position'] = self.flow_x[i]['Position'] + np.random.randn() * (
                            Leader['Position'] - self.flow_x[i]['Position']
                        )

                # 确保粒子位置在边界内
                self.flow_x[i]['Position'] = np.maximum(self.flow_x[i]['Position'], self.lb)
                self.flow_x[i]['Position'] = np.minimum(self.flow_x[i]['Position'], self.ub)
                # 更新粒子成本
                self.flow_x[i]['Cost'] = self.GetCost(self.flow_x[i]['Position'].T)

            # 确定支配关系
            self.flow_x = self.DetermineDominations(self.flow_x)
            non_dominated_flow_x = self.GetNonDominatedParticles(self.flow_x)

            # 更新存档
            self.Archive = np.concatenate((self.Archive, non_dominated_flow_x))

            # 确定存档中粒子的支配关系
            self.Archive = self.DetermineDominations(self.Archive)
            self.Archive = self.GetNonDominatedParticles(self.Archive)

            # 更新存档中粒子的网格索引
            for i in range(len(self.Archive)):
                self.Archive[i]['GridIndex'], self.Archive[i]['GridSubIndex'] = self.GetGridIndex(self.Archive[i], self.G)

            # 管理存档大小
            if len(self.Archive) > self.Archive_size:
                EXTRA = len(self.Archive) - self.Archive_size
                self.Archive = self.DeleteFromRep(self.Archive, EXTRA, self.gamma)

                self.Archive_costs = self.GetCosts(self.Archive)
                self.G = self.CreateHypercubes(self.Archive_costs, self.nGrid, self.boomAlpha)

            # 记录迭代状态
            self._log_generation_state(iter)
            self._save_pareto_front(iter)

        # 记录最终结果
        self._log_final_results()

        return self.GetCosts(self.Archive)

    def DeleteFromRep(self, Archive, EXTRA, gamma):
        """
        从存档中删除多余的粒子以控制存档大小。

        参数:
        Archive (list): 存档中的粒子列表。
        EXTRA (int): 需要删除的粒子数量。
        gamma (float): 未使用的参数，可考虑移除。

        返回:
        list: 更新后的存档粒子列表。
        """
        lenofdel = len(Archive) - EXTRA
        lenofarc = len(Archive)
        while len(Archive) > lenofarc - lenofdel:
            grid_counts = {} 
            for particle in Archive:
                if particle['GridIndex'] in grid_counts:
                    grid_counts[particle['GridIndex']] += 1
                else:
                    grid_counts[particle['GridIndex']] = 1
            max_count = max(grid_counts.values())
            overpopulated_grids = [grid for grid, count in grid_counts.items() if count == max_count]
            grid_to_remove = np.random.choice(overpopulated_grids)
            particles_in_grid = [i for i, p in enumerate(Archive) if p['GridIndex'] == grid_to_remove]
            particle_to_remove = np.random.choice(particles_in_grid)
            Archive.pop(particle_to_remove)
        return Archive

    def _setup_logger(self):
        """
        设置日志记录器。

        返回:
        logging.Logger: 配置好的日志记录器。
        """
        logger = logging.getLogger('MOFDAopt')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _log_initial_state(self):
        """
        记录算法的初始状态。
        """
        self.logger.info(f'Initializing MOFDAopt with max_iter={self.max_iteration}, Archive_size={self.Archive_size}, alpha={self.alpha}, dim={self.dim}')
        #记录初始的非支配解
        self.logger.info(f'Initial non-dominated solutions: {self.Archive}')


    def _log_generation_state(self, iter):
        """
        记录每一代的迭代状态。

        参数:
        iter (int): 当前迭代次数。
        """
        """记录每一代的状态"""
        # 使用追加模式打开文件
        with open(f'{self.log_dir}/generation_all.txt', 'a', encoding='utf-8') as f:
            f.write(f"Generation {iter}\n")
            for i, fv in enumerate(self.Archive):
                # 使用 self.MAX_E 和 self.MAX_R
                f.write(f"Solution {i}:  FNDT={(fv['Cost'][0]*self.MAX_E):.4f}, robustness={(fv['Cost'][1]*self.MAX_R):.4f}\n")
            f.write("\n") # Add line breaks to separate information from different generations

    def _log_final_results(self):
        """
        记录算法的最终结果。
        """
        with open(f'{self.log_dir}/final_results.txt', 'w', encoding='utf-8') as f:
            for i, fv in enumerate(self.Archive):
                # 使用 self.MAX_E 和 self.MAX_R
                f.write(f"Solution {i}:  FNDT={(fv['Cost'][0]*self.MAX_E):.4f}, robustness={(fv['Cost'][1]*self.MAX_R):.4f}\n")
            f.write("\n") # Add line breaks to separate information from different generations
        # 保存最终的拓扑结构和结果
        with open(f'{self.log_dir}/final_results.pkl', 'wb') as fp:
            pickle.dump({
                'best_solution': self.Archive,
                'topo': self.topo,
            }, fp)
    def _save_pareto_front(self, iteration):
        """
        保存 Pareto 前沿图。
        """
        import matplotlib.font_manager as fm  # 添加字体管理模块导入
        plt.rcParams['font.family'] = 'SimHei'  # 设置支持中文的字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        plt.figure(figsize=(10, 8))
        # 提取FNDT和鲁棒性值
        costs = self.GetCosts(self.Archive)

        fndt_values = [ind[0]*self.MAX_E for ind in costs]
        robustness_values = [ind[1]*self.MAX_R for ind in costs]
        
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
    def GetOccupiedCells(self, rep):
        """
        获取占据的单元格索引和每个单元格的成员数量。

        参数:
        rep (list): 粒子列表。

        返回:
        tuple: 包含占据单元格索引列表和成员数量列表的元组。
        """
        grid_counts = {}
        for particle in rep:
            if particle['GridIndex'] in grid_counts:
                grid_counts[particle['GridIndex']] += 1
            else:
                grid_counts[particle['GridIndex']] = 1
        occ_cell_index = list(grid_counts.keys())
        occ_cell_member_count = list(grid_counts.values())
        return occ_cell_index, occ_cell_member_count

    def RouletteWheelSelection(self, p):
        """
        轮盘赌选择法。

        参数:
        p (np.ndarray): 选择概率数组。

        返回:
        int: 选中的索引。
        """
        r = np.random.rand()
        cumulative_prob = np.cumsum(p)
        selected_index = np.where(cumulative_prob >= r)[0][0]
        return selected_index

    def SelectLeader(self, rep, beta=1):
        """
        从粒子列表中选择领导者。

        参数:
        rep (list): 粒子列表。
        beta (float): 选择概率计算参数，默认为 1。

        返回:
        dict: 选中的领导者粒子信息字典。
        """
        occ_cell_index, occ_cell_member_count = self.GetOccupiedCells(rep)
        
        # 将 occ_cell_member_count 转换为浮点数数组
        occ_cell_member_count = np.array(occ_cell_member_count, dtype=float)
        
        p = occ_cell_member_count ** (-beta)
        sum_p = np.sum(p)
        # 添加防止除零保护
        if sum_p == 0:
            p = np.ones_like(p) / len(p)  # 赋予均匀概率
        else:
            p = p / sum_p
        
        selected_cell_index = occ_cell_index[self.RouletteWheelSelection(p)]
        
        selected_cell_members = [i for i, p in enumerate(rep) if p['GridIndex'] == selected_cell_index]
        
        n = len(selected_cell_members)
        selected_membr_index = np.random.randint(0, n)
        
        h = selected_cell_members[selected_membr_index]
        
        rep_h = rep[h]
        return rep_h

    def GetCost(self, particle):
        """
        计算粒子的成本值。

        参数:
        particle (dict): 粒子信息字典。

        返回:
        float: 粒子的成本值。
        """
        MAX_E:    float = 300.0  # energy_FNDT 理论上限
        MAX_R:    float = 0.5    # robustness  理论上限
        binary_chrom = (particle >= 0.5).astype(int)
        # 确保 binary_chrom 是 NumPy 数组
        if not isinstance(binary_chrom, np.ndarray):
            binary_chrom = np.array(binary_chrom)
        self.topo.G = self.topo.reconnect_G(binary_chrom, self.ID_Chrome_in_complete, self.Len_Interval)
        energy_FNDT = self.topo.calculate_FNDT()
        robustness = self.topo.calculate_robustness()
        E_norm = energy_FNDT / MAX_E
        R_norm = robustness / MAX_R
        return [E_norm,R_norm]  # QHDBO expects a cost to minimise

