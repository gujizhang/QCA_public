import numpy as np
import time
from typing import List, Any, Tuple
import random
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt

def ant_search(Adj, nodes, m, pheromone, start, target, alpha=3, beta=1, rho=0.1, q0=0.9):
    """
    蚁群搜索算法，寻找从起点到目标的最优路径
    
    参数:
        Adj: 邻接矩阵
        nodes: 节点列表
        m: 蚂蚁数量
        pheromone: 信息素矩阵
        start: 起始节点索引
        target: 目标节点索引
        alpha: 信息素重要程度因子
        beta: 启发式因子重要程度
        rho: 信息素挥发系数
        q0: 状态转移规则的参数
        
    返回:
        path: 每只蚂蚁的路径，形状为 (m, N) 的 numpy 数组
        best_ant: 最优蚂蚁索引，整数
        path_length: 每只蚂蚁的路径长度，形状为 (m,) 的 numpy 数组
        hops: 每只蚂蚁的跳数，形状为 (m,) 的 numpy 数组
        pheromone: 更新后的信息素矩阵，形状为 (N, N) 的 numpy 数组
    """
    N = len(nodes)
    
    # 初始化路径、路径长度和跳数
    path = np.zeros((m, N), dtype=int)
    path_length = np.zeros(m)
    hops = np.zeros(m, dtype=int)
    
    # 初始化启发式信息矩阵
    heuristic = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and Adj[i, j] == 1:
                # 使用距离的倒数作为启发式信息
                dist = nodes[i].distance[nodes[j].id]
                if dist > 0:
                    heuristic[i, j] = 1.0 / dist
    
    # 每只蚂蚁寻路
    for ant in range(m):
        # 初始化访问标记
        visited = np.zeros(N, dtype=bool)
        visited[start] = True
        
        # 初始化当前节点和路径
        current = start
        path[ant, 0] = current
        hop = 1
        
        # 寻路过程
        while current != target and hop < N:
            # 获取当前节点的邻居
            neighbors = []
            for j in range(N):
                if Adj[current, j] == 1 and not visited[j]:
                    neighbors.append(j)
            
            if not neighbors:
                # 无法继续前进，路径失败
                break
            
            # 计算转移概率
            probabilities = np.zeros(len(neighbors))
            for i, neighbor in enumerate(neighbors):
                probabilities[i] = (pheromone[current, neighbor] ** alpha) * (heuristic[current, neighbor] ** beta)
            
            # 归一化概率
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
            else:
                probabilities = np.ones(len(neighbors)) / len(neighbors)
            
            # 状态转移规则
            q = random.random()
            if q < q0:
                # 贪婪选择
                next_node = neighbors[np.argmax(probabilities)]
            else:
                # 轮盘赌选择
                cumsum = np.cumsum(probabilities)
                r = random.random()
                next_node = neighbors[np.searchsorted(cumsum, r)]
            
            # 更新路径
            path[ant, hop] = next_node
            path_length[ant] += nodes[current].distance[nodes[next_node].id]
            visited[next_node] = True
            current = next_node
            hop += 1
        
        # 记录跳数
        if current == target:
            hops[ant] = hop
            # 添加从最后一个节点到目标的距离
            if hop > 1 and path[ant, hop-1] != target:
                path_length[ant] += nodes[path[ant, hop-1]].distance[nodes[target].id]
        else:
            # 路径失败，设置一个大值
            hops[ant] = N
            path_length[ant] = float('inf')
    
    # 找出最优蚂蚁
    valid_ants = np.where(hops < N)[0]
    if len(valid_ants) > 0:
        # 首先按跳数排序，然后按路径长度排序
        min_hops = np.min(hops[valid_ants])
        min_hops_ants = valid_ants[hops[valid_ants] == min_hops]
        best_ant = min_hops_ants[np.argmin(path_length[min_hops_ants])]
    else:
        # 没有有效路径
        best_ant = 0
    
    # 更新信息素
    pheromone = (1 - rho) * pheromone  # 信息素挥发
    
    # 对最优路径增加信息素
    if hops[best_ant] < N:
        delta_tau = 1.0 / path_length[best_ant]
        for i in range(hops[best_ant] - 1):
            from_node = path[best_ant, i]
            to_node = path[best_ant, i + 1]
            pheromone[from_node, to_node] += delta_tau
            pheromone[to_node, from_node] += delta_tau  # 对称更新
    
    return path, best_ant, path_length, hops, pheromone
class TOSGopt:
    def __init__(self, N: int, super_node_percent: float, topo=None, max_iteration: int = 100, 
                 need_dynamic: bool = False, draw_w: bool = False):
        """
        初始化TOSG优化器

        参数:
            N: 节点总数
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

        # 蚁群算法参数
        self.m = 20  # 蚂蚁数量
        self.alpha = 3  # 信息素重要程度因子
        self.beta = 1  # 启发式因子重要程度
        self.rho = 0.1  # 信息素挥发系数
        self.q0 = 0.9  # 状态转移规则的参数
        
        # 默认参数设置 - 从HeterogeneousTopology中获取
        self.N_S = 1  # sink节点数量，默认为1
        self.seed = 12  # 随机种子
        
        # 设置随机种子
        np.random.seed(self.seed)
        
        # 存储优化过程中的数据
        self.ID_Chrome_in_complete = None  # 完整的ID_Chrome
        self.Len_Interval = None  # 每个区间的长度
        self.paramlen = 0  # 编码后的ID_Chrome长度
        self.Reduce_GStart = None  # 编码后的ID_Chrome
        self.best_solution = None  # 最优解
        self.best_objectives = None  # 最优解的目标函数值
                
        # 路径设置
        self.base_dir = './log'  # 基础日志目录
        self.results_dir = './results'  # 结果保存目录
        self.algorithm_name = 'TOSG'  # 算法名称
        
        # 创建必要的目录
        for directory in [
            f'{self.base_dir}/{self.algorithm_name}',
            f'{self.results_dir}/{self.algorithm_name}'
        ]:
            os.makedirs(directory, exist_ok=True)
    def _log_initial_state(self, log_dir):
        """记录初始状态"""
        with open(f'{log_dir}/initial_state.txt', 'w', encoding='utf-8') as f:
            f.write(f"节点数量: {self.N}\n")
            f.write(f"超级节点比例: {self.super_node_percent}\n")
            f.write(f"初始FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"初始鲁棒性: {self.topo.calculate_robustness()}\n")
            f.write(f"最大迭代次数: {self.max_iteration}\n")

    def _log_final_results(self, log_dir, energy_FNDT, robustness):
        """记录最终结果"""
        with open(f'{log_dir}/final_results.txt', 'w', encoding='utf-8') as f:
            f.write("优化完成\n")
            f.write(f"最终FNDT: {energy_FNDT}\n")
            f.write(f"最终鲁棒性: {robustness}\n")
        
        # 保存最终的拓扑结构和结果
        with open(f'{log_dir}/final_results.pkl', 'wb') as fp:
            pickle.dump({
                'topo': self.topo
            }, fp)
    def initialize_nodes(self) -> List[Any]:
        """初始化节点属性"""
        # 从topo中获取节点信息
        nodes = []
        for node_id in self.topo.G.nodes():
            node_data = self.topo.G.nodes[node_id]
            node = type('Node', (), {})  # 创建一个简单的对象
            node.id = node_id
            node.state = node_data['type']  # 0: 普通节点, 1: 超级节点, 2: sink节点
            node.radius = self.topo.superR if node.state == 1 else self.topo.R
            node.neighbor = np.zeros(self.N)
            node.neighbor_num = 0
            node.deg_aco = 0
            
            # 计算节点间距离
            node.distance = {}
            for other_id in self.topo.G.nodes():
                if node_id != other_id:
                    # 获取节点坐标
                    pos1 = (node_data['pos'][0], node_data['pos'][1])
                    pos2 = (self.topo.G.nodes[other_id]['pos'][0], self.topo.G.nodes[other_id]['pos'][1])
                    # 计算欧氏距离
                    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    node.distance[other_id] = dist
            
            nodes.append(node)
        
        return nodes

    def determine_neighbors(self, nodes: List[Any]) -> None:
        """确定节点间的邻居关系"""
        for i in range(len(nodes)-1):
            for j in range(i+1, len(nodes)):
                if (nodes[i].distance[nodes[j].id] <= min(nodes[i].radius, nodes[j].radius) and 
                    nodes[i].state + nodes[j].state != 4):  # 两个sink节点不能相连
                    nodes[i].neighbor[j] = 1
                    nodes[i].neighbor_num += 1
                    nodes[j].neighbor[i] = 1
                    nodes[j].neighbor_num += 1

    def initialize_pheromone(self, nodes: List[Any]) -> np.ndarray:
        """初始化信息素矩阵"""
        pheromone = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)-1):
            for j in range(i+1, len(nodes)):
                if nodes[i].neighbor[j] == 1:
                    pheromone[i, j] = 1.0
                    pheromone[j, i] = 1.0
        return pheromone

    def determine_targets(self, nodes: List[Any]) -> np.ndarray:
        """确定每个节点的目标sink节点"""
        sink_nodes = [i for i, node in enumerate(nodes) if node.state == 2]
        Target = np.zeros(len(nodes), dtype=int)
        
        for i in range(len(nodes)):
            if i in sink_nodes:
                Target[i] = i
            else:
                # 找到距离最近的sink节点
                min_dist = float('inf')
                min_sink = sink_nodes[0]
                for sink in sink_nodes:
                    if nodes[i].distance[nodes[sink].id] < min_dist:
                        min_dist = nodes[i].distance[nodes[sink].id]
                        min_sink = sink
                Target[i] = min_sink
        
        return Target

    def optimize_network(self, nodes: List[Any]) -> Tuple[np.ndarray, int]:
        """
        执行TOSG优化过程

        参数:
            nodes: 节点信息列表

        返回:
            Tuple[np.ndarray, int]: (优化后的邻接矩阵, 添加的边数量)
        """
        # 初始化邻接矩阵
        Adj = np.zeros((len(nodes), len(nodes)))#设置邻接矩阵
        for i, j in self.topo.G.edges():
            Adj[i, j] = 1
            Adj[j, i] = 1
        
        # 确定邻居关系
        self.determine_neighbors(nodes)
        
        # 初始化信息素矩阵
        pheromone = self.initialize_pheromone(nodes)#初始化信息素矩阵
        
        # 确定目标sink节点
        Target = self.determine_targets(nodes)

        # 记录最优路径信息
        hops_best = np.zeros(len(nodes))#记录每一个节点多次迭代之后的最优路径的跳数
        path_length_best = np.zeros(len(nodes))#记录每一个节点多次迭代之后最优路径的欧式距离
        path_best = np.zeros((len(nodes), len(nodes)), dtype=int)#记录每一个节点多次迭代之后最优路径上的节点

        # 主循环：蚁群算法迭代
        for Nc in range(self.max_iteration):
            print(f"TOSG Iteration {Nc+1}/{self.max_iteration}")
            
            # 对每个非sink节点执行蚁群搜索
            sink_nodes = [i for i, node in enumerate(nodes) if node.state == 2]#先找出sink节点
            for i in range(len(nodes)):
                if i not in sink_nodes:
                    path, best_ant, path_length, hops, pheromone = ant_search(
                        Adj, nodes, self.m, pheromone, i, Target[i], 
                        self.alpha, self.beta, self.rho, self.q0)
                    
                    if Nc == 0: # 第一次迭代    
                        hops_best[i] = hops[best_ant] - 1
                        path_length_best[i] = path_length[best_ant]
                        path_best[i] = path[best_ant]
                    elif hops_best[i] > hops[best_ant] - 1:#如果当前的路径比之前的路径短，就更新
                        hops_best[i] = hops[best_ant] - 1
                        path_length_best[i] = path_length[best_ant]
                        path_best[i] = path[best_ant]
                    elif (hops_best[i] == hops[best_ant] - 1 and 
                          path_length_best[i] > path_length[best_ant]):
                        hops_best[i] = hops[best_ant] - 1
                        path_length_best[i] = path_length[best_ant]
                        path_best[i] = path[best_ant]

        # 计算节点重要性
        for i in range(len(nodes)):
            if nodes[i].state in [1, 2]:  # 超级节点或sink节点
                nodes[i].deg_aco = np.sum(path_best == i)

        # 构建最终邻接关系
        TOSG_count = 0
        for i in range(len(nodes)):
            if nodes[i].state == 1:  # 只对超级节点进行优化
                max_ipt = 0
                max_index = -1
                neighbor_indices = np.where(nodes[i].neighbor == 1)[0]
                
                for j in neighbor_indices:
                    if (nodes[j].state in [1, 2] and Adj[i, j] == 0):
                        if max_ipt < nodes[j].deg_aco:#如果当前的节点的重要性比之前的节点的重要性大，就更新
                            max_ipt = nodes[j].deg_aco
                            max_index = j
                
                if max_index != -1:
                    Adj[i, max_index] = 1
                    Adj[max_index, i] = 1
                    TOSG_count += 1

        return Adj, TOSG_count

    def cal_obj(self, Chrom):
        """
        计算目标函数值
        
        参数:
            Chrom: 染色体
            
        返回:
            [energy_FNDT, robustness]: 目标函数值列表
        """
        # 根据decode_result重新连接topo.G的超级节点
        self.topo.G = self.topo.reconnect_G(Chrom, self.ID_Chrome_in_complete, self.Len_Interval)
        # 计算FNDT和Energy这两个目标函数的值
        energy_FNDT = self.topo.calculate_FNDT()
        robustness = self.topo.calculate_robustness()
        return [energy_FNDT, robustness]

    def optimize(self, Reduce_GStart, paramlen, ID_Chrome_in_complete, Len_Interval):
        """
        执行TOSG优化
        
        参数:
            Reduce_GStart: 编码后的ID_Chrome
            paramlen: 编码后的ID_Chrome长度
            ID_Chrome_in_complete: 完整的ID_Chrome
            Len_Interval: 每个区间的长度
            
        返回:
            None
        """
        #输出图原本有多少条边
        print("图原本有多少条边：", len(self.topo.G.edges()))

        self.Reduce_GStart = Reduce_GStart
        self.paramlen = paramlen
        self.ID_Chrome_in_complete = ID_Chrome_in_complete
        self.Len_Interval = Len_Interval
        
        print("Starting TOSG optimization...")
        start_time = time.time()
        
        # 初始化节点
        nodes = self.initialize_nodes()
        
        # 创建日志目录
        log_dir = f'{self.base_dir}/{self.algorithm_name}/{self.N}_{int(100*self.super_node_percent)}'
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 记录初始状态
        self._log_initial_state(log_dir)
        
        # 执行TOSG优化
        optimized_adj, added_edges = self.optimize_network(nodes)
        
        # 将优化后的邻接矩阵转换为NetworkX图
        G_optimized = nx.Graph()
        for i in range(len(nodes)):
            # 复制节点属性
            G_optimized.add_node(i, **self.topo.G.nodes[i])
        
        # 添加边
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if optimized_adj[i, j] == 1:
                    # 计算边的权重（距离）
                    pos1 = (self.topo.G.nodes[i]['pos'][0], self.topo.G.nodes[i]['pos'][1])
                    pos2 = (self.topo.G.nodes[j]['pos'][0], self.topo.G.nodes[j]['pos'][1])
                    weight = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    G_optimized.add_edge(i, j, weight=weight)
        
        # 更新拓扑结构
        self.topo.G = G_optimized
        print("图原本有多少条边：", len(self.topo.G.edges()))
        # 计算优化后的目标函数值
        energy_FNDT = self.topo.calculate_FNDT()
        robustness = self.topo.calculate_robustness()
                
        # 记录最终结果
        self._log_final_results(log_dir, energy_FNDT, robustness)
        # 保存最优解
        self.best_solution = np.array(Reduce_GStart)  # 这里可以根据实际情况修改
        self.best_objectives = [energy_FNDT, robustness]
        
        print(f"TOSG optimization completed in {time.time() - start_time:.2f}s")
        print(f"Added edges: {added_edges}")
        print(f"Final objectives: FNDT = {energy_FNDT}, Robustness = {robustness}")
