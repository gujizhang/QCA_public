import numpy as np
import time
import copy
from typing import List, Any, Tuple, Dict
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt


class ChromStruct:
    def __init__(self):
        self.Population = None
        self.Index = None
        self.max_rank = 0

class TEAMopt:

    def __init__(self, N, super_node_percent, topo=None, max_iteration=100, need_dynamic=False, draw_w=False):
        """
        初始化TEAM优化器

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
    
        # 默认参数设置 - 从HeterogeneousTopology中获取
        self.N_S = 1  # sink节点数量
        self.seed = 12  # 随机种子
        # 设置随机种子
        np.random.seed(self.seed)
    
        # 种群参数设置
        self.MP = 4  # 种群的数量 4比较合适
        self.pop = 10  # 种群中个体的数量 10比较合适
        self.p_c = 0.7  # 交叉概率的基准
        self.p_mu = 0.03  # 突变概率的基准
        self.population = []  # 种群全体
        self.objnum = 2  # 目标函数的数量
        self.ID_Chrome_in_complete = None  # 完整的ID_Chrome
        self.Len_Interval = None  # 每个区间的长度
        self.paramlen = 0  # 编码后的ID_Chrome
        self.Reduce_GStart = None  # 编码后的ID_Chrome
    
        # 路径设置
        self.base_dir = './log'  # 基础日志目录
        self.results_dir = './results'  # 结果保存目录
        self.picture_dir = './picture'  # 图片保存目录
        self.algorithm_name = 'TEAM'  # 算法名称
    
        # 创建必要的目录
        for directory in [
            f'{self.base_dir}/{self.algorithm_name}',
            f'{self.results_dir}/{self.algorithm_name}',
            f'{self.picture_dir}/{self.algorithm_name}'
        ]:
            os.makedirs(directory, exist_ok=True)
    def _log_initial_state(self, log_dir):
        """记录初始状态"""
        with open(f'{log_dir}/initial_state.txt', 'w', encoding='utf-8') as f:
            f.write(f"节点数量: {self.N}\n")
            f.write(f"超级节点比例: {self.super_node_percent}\n")
            f.write(f"初始FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"初始鲁棒性: {self.topo.calculate_robustness()}\n")
            f.write(f"种群数量: {self.MP}\n")
            f.write(f"每个种群的个体数量: {self.pop}\n")
            f.write(f"交叉概率: {self.p_c}\n")
            f.write(f"变异概率: {self.p_mu}\n")
            f.write(f"最大迭代次数: {self.max_iteration}\n")
        
        # 清空generation_all.txt文件
        with open(f'{log_dir}/generation_all.txt', 'w', encoding='utf-8') as f:
            pass

    def _log_generation_state(self, log_dir, gen, best_individuals):
        """记录每一代的状态"""
        # 使用追加模式打开文件
        with open(f'{log_dir}/generation_all.txt', 'a', encoding='utf-8') as f:
            f.write(f"第 {gen} 代\n")
            f.write(f"最优个体数量: {len(best_individuals)}\n")
            f.write("\n最优个体适应度值:\n")
            for i, individual in enumerate(best_individuals):
                f.write(f"个体 {i}: FNDT={individual[self.paramlen]}, 鲁棒性={individual[self.paramlen+1]}\n")
            f.write("\n")  # 添加换行以分隔不同代的信息

    def _log_final_results(self, log_dir, best_individuals):
        """记录最终结果"""
        with open(f'{log_dir}/final_results.txt', 'w', encoding='utf-8') as f:
            f.write("优化完成\n")
            f.write(f"最终最优个体数量: {len(best_individuals)}\n")
            f.write("\n最终最优个体适应度值:\n")
            for i, individual in enumerate(best_individuals):
                f.write(f"个体 {i}: FNDT={individual[self.paramlen]}, 鲁棒性={individual[self.paramlen+1]}\n")
            
            # 记录最终的FNDT和鲁棒性
            f.write(f"\n最终FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"最终鲁棒性: {self.topo.calculate_robustness()}\n")
        
        # 保存最终的拓扑结构和结果
        with open(f'{log_dir}/final_results.pkl', 'wb') as fp:
            pickle.dump({
                'best_individuals': best_individuals,
                'topo': self.topo
            }, fp)

    def _save_pareto_front(self, best_individuals, iteration=None):
        """保存Pareto前沿图"""
        plt.figure(figsize=(10, 8))
        # 提取FNDT和鲁棒性值
        fndt_values = [ind[self.paramlen] for ind in best_individuals]
        robustness_values = [ind[self.paramlen+1] for ind in best_individuals]
        
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
    def tournament_selection(self, chromosome: np.ndarray, pool: int, tour: int, V: int, M: int) -> np.ndarray:
        """
        竞标赛选择适合繁育的父代染色体

        Args:
            chromosome: 种群矩阵
            pool: 父代数量
            tour: 每次竞标赛的参与者数量
            V: 决策变量数量
            M: 目标函数数量

        Returns:
            选择出的父代染色体矩阵
        """
        pop = chromosome.shape[0]  # 获得种群中个体的数量

        # 个体中Pareto等级和拥挤距离所在的位置
        rank = V + M
        dis = V + M + 1

        # 初始化父代染色体矩阵
        parent_chromosome = np.zeros((pool, chromosome.shape[1]))

        for i in range(pool):
            # 选择竞标赛参与者
            candidate = np.zeros(tour, dtype=int)
            for j in range(tour):
                # 随机选择一个候选者
                temp = np.random.randint(0, pop)

                # 确保不重复选择同一个个体
                while j > 0 and temp in candidate[:j]:
                    temp = np.random.randint(0, pop)

                candidate[j] = temp

            # 记录每一个参赛者的Pareto等级和拥挤距离
            c_rank = chromosome[candidate, rank]
            c_dis = chromosome[candidate, dis]

            # 找出具有最小Pareto等级的个体
            min_c = np.where(c_rank == np.min(c_rank))[0]

            if len(min_c) != 1:
                # 如果有多个个体具有相同的最小Pareto等级，选择拥挤距离最大的
                max_d = np.where(c_dis[min_c] == np.max(c_dis[min_c]))[0]
                if len(max_d) != 1:
                    max_d = max_d[0]
                parent_chromosome[i] = chromosome[candidate[min_c[max_d]]]
            else:
                # 如果只有一个个体具有最小Pareto等级，直接选择它
                parent_chromosome[i] = chromosome[candidate[min_c[0]]]

        return parent_chromosome
    def non_domination_sort_mod(self, chromosome, M, V) :
        """
        对初始种群开始排序 快速非支配排序
        
        Args:
            chromosome: 种群个体矩阵
            M: 目标函数数量
            V: 决策变量数量
            
        Returns:
            包含非支配排序和拥挤距离的种群矩阵
        """
        chromosome = np.array(chromosome)
        N = chromosome.shape[0]  # N为矩阵的行数，也是个体的数量
        
        front = 1
        F = {}
        F[front] = []
        individual = []
        
        # 初始化个体结构
        for i in range(N):
            individual.append({"n": 0, "p": []})  # n是个体i被支配的个体数量，p是被个体i支配的个体集合
        
        # 计算支配关系
        for i in range(N):
            for j in range(N):
                dom_less = 0
                dom_equal = 0
                dom_more = 0
                
                # 判断个体i和个体j的支配关系
                for k in range(M):
                    if chromosome[i, V + k] < chromosome[j, V + k]:
                        dom_less += 1 # 表示j支配i的的程度
                    elif chromosome[i, V + k] == chromosome[j, V + k]:
                        dom_equal += 1#表示i和j的目标函数值相等，没有只配关系
                    else:
                        dom_more += 1#表示i支配j的的程度
                
                # 因为我们的问题是最大化问题，所以i>=j,即i支配j
                if dom_less == 0 and dom_equal != M:  # 说明i>=j,即i支配j，把j加入i的支配合集中
                    individual[i]["p"].append(j)
                elif dom_more == 0 and dom_equal != M:  # 说明i<=j，即i受j支配,相应的n加1
                    individual[i]["n"] += 1
            
            # 个体i非支配等级排序最高，属于当前最优解集，相应的染色体中携带代表排序数的信息
            if individual[i]["n"] == 0:
                # 确保染色体矩阵有足够的列
                if chromosome.shape[1] <= M + V + 1:
                    chromosome = np.column_stack((chromosome, np.zeros((N, 1))))
                
                chromosome[i, M + V] = 1
                F[front].append(i)  # 等级为1的非支配解集
        
        # 给其他个体进行分级
        while len(F[front]) > 0:
            Q = []  # 存放下一个front集合
            
            # 循环当前支配解集中的个体
            for i in range(len(F[front])):
                # 个体i有自己所支配的解集
                if len(individual[F[front][i]]["p"]) > 0:
                    # 循环个体i所支配解集中的个体
                    for j in range(len(individual[F[front][i]]["p"])):
                        # 表示个体j的被支配个数减1
                        individual[individual[F[front][i]]["p"][j]]["n"] -= 1
                        
                        # 如果q是非支配解集，则放入集合Q中
                        if individual[individual[F[front][i]]["p"][j]]["n"] == 0:
                            # 个体染色体中加入分级信息
                            chromosome[individual[F[front][i]]["p"][j], M + V] = front + 1
                            Q.append(individual[F[front][i]]["p"][j])
            
            front += 1
            F[front] = Q

        # 对个体的代表排序等级的列向量进行升序排序
        index_of_fronts = np.argsort(chromosome[:, M + V])
        sorted_based_on_front = np.zeros_like(chromosome)

        for i in range(len(index_of_fronts)):
            sorted_based_on_front[i, :] = chromosome[index_of_fronts[i], :]

        # Crowding distance 计算每个个体的拥挤度
        current_index = 0
        # 创建结果数组，大小为N行，列数为决策变量+目标函数+等级+拥挤距离
        z = np.zeros((N, V + M + 2))

        # 这里减1是因为F的最后一个元素为空，这样才能跳出循环。所以一共有len(F)-1个排序等级
        for front in range(1, len(F)):
            if len(F[front]) == 0:
                break

            # 创建当前等级的临时数组
            y = np.zeros((len(F[front]), V + M + 2 + M))
            previous_index = current_index

            # 复制当前等级的个体到临时数组
            for i in range(len(F[front])):
                y[i, :V + M + 1] = sorted_based_on_front[current_index + i, :V + M + 1]

            current_index = current_index + len(F[front])

            # 计算每个目标函数的拥挤距离
            for i in range(M):
                # 按照目标函数值升序排列
                index_of_objectives = np.argsort(y[:, V + i])
                sorted_based_on_objective = np.zeros_like(y)

                # 按目标函数值排序
                for j in range(len(index_of_objectives)):
                    sorted_based_on_objective[j, :] = y[index_of_objectives[j], :]

                # fmax为目标函数最大值，fmin为目标函数最小值
                f_max = sorted_based_on_objective[-1, V + i]
                f_min = sorted_based_on_objective[0, V + i]

                # 对排序后的第一个个体和最后一个个体的距离设为无穷大（或者一个非常大的数）
                y[index_of_objectives[-1], V + M + 2 + i] = 10000
                y[index_of_objectives[0], V + M + 2 + i] = 10000

                # 循环集合中除了第一个和最后一个的个体
                for j in range(1, len(index_of_objectives) - 1):
                    next_obj = sorted_based_on_objective[j + 1, V + i]
                    previous_obj = sorted_based_on_objective[j - 1, V + i]

                    if f_max - f_min == 0:
                        y[index_of_objectives[j], V + M + 2 + i] = 10000  # 无穷大或者一个很大的数
                    else:
                        y[index_of_objectives[j], V + M + 2 + i] = (next_obj - previous_obj) / (f_max - f_min)

            # 计算总的拥挤距离
            for i in range(len(F[front])):
                distance = 0
                for j in range(M):
                    distance += y[i, V + M + 2 + j]
                y[i, V + M + 1] = distance

            # 将计算好的拥挤距离添加到结果中
            z[previous_index:previous_index + len(F[front]), :] = y[:, :V + M + 2]

        # 得到的是已经包含等级和拥挤度的种群矩阵，并且已经按等级排序
        return z[:N, :V + M + 2]
    def cal_obj(self, Chrom):
        # 根据decode_result重新连接topo.G，的超级节点
        self.topo.G = self.topo.reconnect_G(Chrom, self.ID_Chrome_in_complete, self.Len_Interval)
        # 计算FNDT和Energy这两个目标函数的值
        energy_FNDT = self.topo.calculate_FNDT()
        robustness = self.topo.calculate_robustness()
        return [energy_FNDT, robustness]

    def cal_fit(self, Population):
        # 计算每个个体的适应度值，并把适应度值保存到每个染色体最后两位
        result = []
        for i in range(len(Population)):
            # 获取决策变量部分
            decision_vars = Population[i][0:self.paramlen]
            # 计算目标函数值
            obj_values = self.cal_obj(decision_vars)
            # 拼接决策变量和目标函数值
            individual = np.concatenate((decision_vars, obj_values))
            result.append(individual)
        return np.array(result)
    def fitness(self, chromosome: np.ndarray, V: int, M: int, N: int, k: int) -> np.ndarray:
        """
        根据给定的M个指标值，计算不同Pareto等级中的个体的一个综合适应度
        M个指标有不同的重要等级：延迟>能效>鲁棒性，我这里只用了能效和鲁棒性，所以只对比这两个
        """
        pop, _ = chromosome.shape
        
        # 计算能效的最大值
        n1 = np.floor(N/k)  # 一个sink或一个超级节点管理的节点的数量
        Radius = self.topo.superR  # 超级节点的通信半径
        
        # 节点能耗参数设置
        E_elec = self.topo.Eelec  # 发送1bits数据的发送电路或接收电路能耗(J/bit)
        packet_bits =  self.topo.packet_size  # 数据包的大小（bits）
        epsilon_mp = self.topo.Eamp  # 多径衰减信道的放大器能耗(J/bit/m^-4)
        beta =  self.topo.beta  # 超级节点与建立连接的能耗(J)
        E_Ag = self.topo.alpha  # 超级节点数据聚合的能耗(J/bit)
        E_h = self.topo.super_node_energy  # 超级节点的初始能量(J)
        
        E_total = (n1-1)*packet_bits*E_elec + (n1-1)*packet_bits*E_Ag + \
            packet_bits*E_elec + packet_bits*Radius**4*epsilon_mp + beta
        E_max = np.ceil(E_h/E_total)  # 能效的最大值
        R_max = 0.5  # 鲁棒性的最大值
        
        # 根据重要性确定不同指标的权重系数
        r = [1.4]  # 设定相邻权重系数的比值
        W = np.zeros(M)  # 记录M个指标的权重系数
        W_t = 0
        
        for i in range(1, M):
            temp = 1
            for j in range(i, M):
                temp = temp * r[j-1]
            W_t = W_t + temp
        
        W[M-1] = 1/(1+W_t)  # 首先计算最后一个权重系数
        for i in range(M-1):
            W[M-i-2] = W[M-i-1] * r[M-i-2]
        
        # 根据权重系数，计算不同个体的综合适应度
        for i in range(pop):
            fit = W[0]*(chromosome[i, V]/E_max) + W[1]*(chromosome[i, V+1]/R_max)
            pop, cols = chromosome.shape
            if cols <= V+M+2:
                chromosome = np.column_stack((chromosome, np.zeros((pop, 1))))
            chromosome[i, V+M+2] = fit
        
        return chromosome

    def record_max_min(self, chromosome: np.ndarray, V: int, M: int, N: int, k: int) -> Tuple[np.ndarray, int]:
        """
        记录每一代中每个种群的每一个Pareto等级中最优的个体和最差的个体
        """
        # 获取最高的pareto等级
        max_rank = int(np.max(chromosome[:, V+M]))
        
        # 记录每一个等级中最优和最差个体在种群中的下标
        Index_all = np.zeros((max_rank, 2), dtype=int)
        
        previous_index = 0
        account = 0
        
        for i in range(max_rank):
            # 找出当前等级的最大索引
            current_level = i + 1  # 当前Pareto等级
            current_indices = np.where(chromosome[:, V+M] == current_level)[0]
            current_index = np.max(current_indices)
            
            # 获取当前等级的所有个体
            temp = chromosome[previous_index:current_index+1, :]
            
            # 计算同一Pareto等级中不同个体的综合得分
            temp = self.fitness(temp, V, M, N, k)
            
            # 根据最后一列（综合得分）对矩阵进行排序
            sort_indices = np.argsort(temp[:, V+M+2])
            
            # 记录最优个体和最差个体在种群中的下标
            Index_all[i, 0] = sort_indices[-1] + account  # 最优个体
            Index_all[i, 1] = sort_indices[0] + account   # 最差个体
            
            # 更新索引
            previous_index = current_index + 1
            account += len(current_indices)
        
        return Index_all, max_rank            
    def genetic_operation(self, parent_chromosome: np.ndarray, V: int, p_cross: float, p_mutation: float) -> np.ndarray:
        """
        对父代种群进行交叉和变异操作，生成后代种群
        
        Args:
            parent_chromosome: 父代种群矩阵
            V: 决策变量数量
            p_cross: 交叉概率
            p_mutation: 变异概率
            
        Returns:
            后代种群矩阵
        """
        N = parent_chromosome.shape[0]  # N为选择的父代个体数量
        offspring = []  # 存储交叉变异后的后代个体种群
        
        # 首先进行交叉操作
        for i in range(N):
            # 选择两个不同的父代
            parent_1 = np.random.randint(0, N)
            parent_2 = np.random.randint(0, N)
            
            # 确保选择的两个父代不是相同的个体
            while np.array_equal(parent_chromosome[parent_1, :V], parent_chromosome[parent_2, :V]):
                parent_2 = np.random.randint(0, N)
            
            # 复制父代
            offspring_1 = parent_chromosome[parent_1, :].copy()
            offspring_2 = parent_chromosome[parent_2, :].copy()
            
            # 根据交叉概率决定是否进行交叉
            if np.random.random() < p_cross:
                # 随机选择交叉点
                crossover_point = np.random.randint(1, V-1)
                # 交换两个子代的基因
                temp = offspring_1[crossover_point:V].copy()
                offspring_1[crossover_point:V] = offspring_2[crossover_point:V].copy()
                offspring_2[crossover_point:V] = temp
            
            # 将新产生的子代加入到子代种群中
            offspring.append(offspring_1[:V])
            offspring.append(offspring_2[:V])
        
        # 将列表转换为numpy数组
        offspring = np.array(offspring)
        num_offspring = len(offspring)
        
        # 然后进行变异操作
        for i in range(num_offspring):
            # 对每个基因位点进行变异
            for j in range(V):
                # 根据变异概率决定是否进行变异
                if np.random.random() < p_mutation:
                    # 基因取反
                    offspring[i, j] = 1 - offspring[i, j]
        
        # 只保留与种群大小相同数量的后代
        offspring = offspring[:self.pop]
        
        return offspring
    def replace_chromosome(self, combine_chromosome: np.ndarray, M: int, V: int, pop: int) -> np.ndarray:
        """
        精英选择策略，从合并的2*pop个个体中选择最优的pop个个体组成下一代种群
        
        Args:
            combine_chromosome: 合并的染色体数组
            M: 目标函数数量
            V: 决策变量数量
            pop: 种群大小
            
        Returns:
            np.ndarray: 选择后的新种群
        """
        N = len(combine_chromosome)
        # 根据分层等级排序
        index = np.argsort(combine_chromosome[:, M + V])
        
        # 按照分层等级重新排序染色体
        sorted_chromosome = combine_chromosome[index]
        
        # 获取最大分层等级
        max_rank = int(np.max(combine_chromosome[:, M + V]))
        
        # 初始化返回数组
        f = np.zeros((pop, combine_chromosome.shape[1]))
        
        previous_index = 0
        for i in range(1, max_rank + 1):
            # 找出当前等级的所有个体
            current_index = np.max(np.where(sorted_chromosome[:, M + V] == i)[0])
            
            if current_index + 1 > pop:  # 如果当前层的个体加入后超过种群大小
                remaining = pop - previous_index
                temp_pop = sorted_chromosome[previous_index:current_index + 1]
                # 根据拥挤度降序排序
                temp_sort_index = np.argsort(temp_pop[:, M + V + 1])[::-1]
                
                # 选择剩余数量的个体
                for j in range(remaining):
                    f[previous_index + j] = temp_pop[temp_sort_index[j]]
                return f
                
            elif current_index + 1 < pop:  # 如果当前层的个体数量未达到种群大小
                f[previous_index:current_index + 1] = sorted_chromosome[previous_index:current_index + 1]
            else:  # 如果刚好达到种群大小
                f[previous_index:current_index + 1] = sorted_chromosome[previous_index:current_index + 1]
                return f
            
            previous_index = current_index + 1
        
        return f   
    def Layered_Cooperation(self, Chrom: List, MP: int) -> List:
        """
        分层协作：用当前种群中同等级的最优个体去替代下一个种群中同等级的最差个体
        
        Args:
            Chrom: 存储不同种群中所有个体的列表，每个元素是一个ChromStruct对象
            MP: 种群的数量
            
        Returns:
            更新后的种群列表
        """
        for i in range(MP):
            # 确定下一个种群的索引
            next_pop = (i + 1) % MP
            
            # 求出两个相邻种群中最少的pareto等级
            Mini = min(Chrom[i].max_rank, Chrom[next_pop].max_rank)
            
            # 将当前种群中同等级的最优个体去替换下一个种群中同等级的最差个体
            for j in range(Mini):
                maxI = Chrom[i].Index[j, 0]  # 当前种群中第j等级的最优个体索引
                minI = Chrom[next_pop].Index[j, 1]  # 下一个种群中第j等级的最差个体索引
                Chrom[next_pop].Population[minI] = Chrom[i].Population[maxI].copy()
        
        return Chrom

    def Fit_Operator(self, Chrom: List, MP: int, M: int, V: int, N: int, k: int) -> np.ndarray:
        """
        适应度操作：根据适应度值从Pareto=1中选择最优的个体
        
        Args:
            Chrom: 存储不同种群中所有个体的列表
            MP: 种群数量
            M: 目标函数数量
            V: 决策变量数量
            N: 节点总数
            k: 超级节点数量
            
        Returns:
            全局最优解
        """
        # 初始化存储每个种群最优个体的矩阵
        YY_chrom = np.zeros((MP, M+V+3))
        
        # 首先从当前的每一个种群中选择最优的个体
        for i in range(MP):
            chromosome = Chrom[i].Population
            
            # 找出非支配等级为1的个体的最大下标
            current_indices = np.where(chromosome[:, M + V] == 1)[0]
            if len(current_indices) > 0:
                current_index = np.max(current_indices)
                
                # 取出非支配等级为1的所有个体
                temp_pop = chromosome[:current_index + 1, :]
                
                # 计算同一Pareto等级中不同个体的综合得分
                temp_pop = self.fitness(temp_pop, V, M, N, k)
                
                # 将这些个体按照适应度值降序排序
                temp_sort_index = np.argsort(-temp_pop[:, M + V + 2])
                
                # 选择非支配等级为1且适应度值最高的个体作为最优解
                YY_chrom[i, :] = temp_pop[temp_sort_index[0], :]
        
        # 然后对MP个最优个体组成的新种群中，进行非支配排序和拥挤距离计算
        # 从Pareto等级为1的个体中选择适应度值最高的个体作为最终的当代最优值
        YY_sorted = self.non_domination_sort_mod(YY_chrom[:, :V+M], M, V)
        
        # 找出排序后非支配等级为1的个体
        cur_indices = np.where(YY_sorted[:, M + V] == 1)[0]
        if len(cur_indices) > 0:
            curIndex = np.max(cur_indices)
            
            # 取出非支配等级为1的所有个体
            chrom_temp = YY_sorted[:curIndex + 1, :]
            
            # 计算同一Pareto等级中不同个体的综合得分
            chrom_temp = self.fitness(chrom_temp, V, M, N, k)
            
            # 按照适应度值降序排序
            sort_index = np.argsort(-chrom_temp[:, M + V + 2])
            
            # 返回所有的全局最优解
            optimal = chrom_temp[sort_index[0], :]
        else:
            # 如果没有非支配等级为1的个体，返回第一个个体
            optimal = YY_sorted[0, :]
        
        return optimal         
    def optimize(self, Reduce_GStart, paramlen, ID_Chrome_in_complete, Len_Interval):
        """
        执行TEAM优化算法
        
        参数:
            Reduce_GStart: 编码后的ID_Chrome
            paramlen: 编码后的ID_Chrome长度
            ID_Chrome_in_complete: 完整的ID_Chrome
            Len_Interval: 每个区间的长度
            
        返回:
            最优解和目标函数值
        """
        print(f"开始TEAM优化，节点数量: {self.N}, 超级节点比例: {self.super_node_percent}")
        
        # 创建日志目录
        log_dir = f'{self.base_dir}/{self.algorithm_name}/{self.N}_{int(100*self.super_node_percent)}'
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 存储参数
        self.Reduce_GStart = Reduce_GStart
        self.paramlen = paramlen
        self.ID_Chrome_in_complete = ID_Chrome_in_complete
        self.Len_Interval = Len_Interval
        V = self.paramlen
        M = self.objnum
        
        # 记录初始状态
        self._log_initial_state(log_dir)
        
        # 初始生成种群*种群个体数量多个个体，并把它们分别分配给不同的种群，每个个体的长度为Reduce_GStart
        self.population = []  # 清空种群
        for i in range(self.MP):
            pop_list = [np.random.randint(0, 2, paramlen) for _ in range(self.pop)]
            self.population.append(pop_list)
        
        Chrom = [ChromStruct() for _ in range(self.MP)]
        for i in range(self.MP):
            Chrom[i].Population = np.array(self.population[i])
        
        pc = self.p_c + (0.9 - 0.7) * np.random.rand(self.MP)  # 交叉概率
        pm = self.p_mu + (0.05 - 0.001) * np.random.rand(self.MP)  # 变异概率
        opt_result = np.zeros((self.max_iteration, paramlen + self.objnum + 3))
        pool = round(self.pop / 2)  # 交配池的大小
        tour = 2  # 锦标赛选择的参数
        
        # 开始迭代
        start_time = time.time()
        len_superNode = 0
        
        for gen in range(self.max_iteration):
            print(f"TEAM: 迭代 {gen + 1}/{self.max_iteration}")
            
            # 1、对每一个子种群进行一次新的迭代
            for j in range(self.MP):
                Ind = Chrom[j].Population.copy()
                # 1.1、计算不同个体在fm上的函数值，添加到染色体矩阵中
                Ind = self.cal_fit(Ind)
                # 进行分层排序，结果维度为pop*(V+M+2)
                Ind = self.non_domination_sort_mod(Ind, self.objnum, self.paramlen)
                parent_chromosome = self.tournament_selection(Ind, pool, tour, self.paramlen, self.objnum)
                # 对父代个体进行交叉、变异操作生成子代，结果维度为pop*(V+M+2)
                offspring = self.genetic_operation(parent_chromosome, self.paramlen, pc[j], pm[j])
                # 计算子代的目标函数值
                offspring = self.cal_fit(offspring)
                    
                # 父代个体的数量为pop; 子代个体的数量为pop
                main_num = len(Ind)
                offspring_num = len(offspring)
                
                # 合并父代和子代个体
                combine = np.zeros((main_num + offspring_num, V+M+2))
                combine[:main_num, :V+M] = Ind[:, :V+M]
                combine[main_num:, :V+M] = offspring[:, :V+M]
                
                # 对合并后的种群进行非支配排序
                combine = self.non_domination_sort_mod(combine, self.objnum, self.paramlen)
                # 精英选择，选择新一代种群
                Chrom[j].Population = self.replace_chromosome(combine, self.objnum, self.paramlen, self.pop)
                # 记录每个等级中的最优和最差个体

                len_superNode = len([node for node, data in self.topo.G.nodes(data=True) if data['type'] == 1 or data['type'] == 2])
                Chrom[j].Index, Chrom[j].max_rank = self.record_max_min(Chrom[j].Population, self.paramlen, self.objnum, self.N, self.topo.sink_node_count)
            
            # 2、进行分层协作
            if gen % 10 == 0 and gen > 0:  # 每10代进行一次分层协作
                Chrom = self.Layered_Cooperation(Chrom, self.MP)
            
            # 3、从所有子种群中选择最优个体
            opt_result[gen, :] = self.Fit_Operator(Chrom, self.MP, self.objnum, self.paramlen, self.N, len_superNode)
            
            # 4.日志记录 获取当前代的最优个体
            best_individuals = []
            for j in range(self.MP):
                # 获取第一个Pareto等级的个体
                first_rank_indices = np.where(Chrom[j].Population[:, self.paramlen + self.objnum] == 1)[0]
                for idx in first_rank_indices:
                    best_individuals.append(Chrom[j].Population[idx])
            
            # 记录每一代的状态
            self._log_generation_state(log_dir, gen, best_individuals)
            
            # 保存当前代的Pareto前沿图
            self._save_pareto_front(best_individuals, gen)    

            # 打印当前迭代信息
            print(f"Generation {gen+1}/{self.max_iteration} completed, objectives: {opt_result[gen, V:V+M]} ")
        
        print(f"TEAM optimization completed: {time.time() - start_time:.2f}s")
        
        # 获取最终的最优个体
        final_best_individuals = []
        for j in range(self.MP):
            first_rank_indices = np.where(Chrom[j].Population[:, self.paramlen + self.objnum] == 1)[0]
            for idx in first_rank_indices:
                final_best_individuals.append(Chrom[j].Population[idx])
        
        # 计算优化时间
        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"TEAM优化完成，耗时: {optimization_time:.2f}秒")
        
        # 记录最终结果
        self._log_final_results(log_dir, final_best_individuals)
        
        # 保存最终Pareto前沿图
        self._save_pareto_front(final_best_individuals)
        
        # 返回最优解
        return final_best_individuals