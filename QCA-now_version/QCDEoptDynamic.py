import math
import numpy as np
import random
import copy
import numpy as np
import random as rd
import copy
import os
import pickle
import time
import matplotlib.pyplot as plt

class ChebyshevDynamicNeighborhood:
    def __init__(self, num_weights, mu, sigma, T_size, epsilon):
        """
        初始化切比雪夫动态邻域环境

        参数:
            num_weights (int): 权重数量
            mu (float): 高斯分布的均值
            sigma (float): 高斯分布的标准差
        """
        self.num_weights = num_weights
        self.mu = mu
        self.sigma = sigma
        self.weights = self.generate_weights()
        self.adj_matrix = None
        self.W_Bi_T = []
        self.T_size = T_size
        self.epsilon = epsilon
    def generate_weights(self):
        """
        以mu为中心在0-1的范围内生成一组值，这些值在mu附近分布更密集，离它越远越稀疏。

        返回:
            list: 调整后的权重向量列表
        """
        weights = []
        while len(weights) < self.num_weights:
            lambda1 = np.random.normal(self.mu, self.sigma)
            if 0 <= lambda1 <= 1:
                lambda2 = 1 - lambda1
                weights.append((lambda1, lambda2))
        #在返回之前对权重进行排序，确保第一个权重是最小的
        weights.sort()
        return weights

    def create_quantum_circuit(self):
        weight_temp = np.array(self.weights)
        self.W_Bi_T = []
        for bi in range(weight_temp.shape[0]):
            Bi = self.weights[bi]
            DIS = np.sum((weight_temp - Bi) ** 2, axis=1)
            B_T = np.argsort(DIS)
            B_T = B_T[1:self.T_size + 1]
            self.W_Bi_T.append(B_T)
        
    def change_weight_circuit(self, G, T, FEs_success, FEs):
        '''
            G (int): 当前代
            T (int): 时间窗口大小
            FEs_success (list): 二维列表，FEs_success[k][g] 表示第k个邻域在第g代的成功更新次数
            FEs (list): 二维列表，FEs[k][g] 表示第k个邻域在第g代生成的解的总数
        '''
        # 计算每个邻域被使用的概率
        K = len(FEs)
        p_k_G_values = [self.calculate_p_k_G(i, G, T, FEs_success, FEs) for i in range(K)]
        weight_temp = np.array(self.weights)
        # 假设m是一个需要传入的参数，这里暂时设置为一个示例值，可根据实际情况调整
        m = 10 
        for k in range(K):
            if random.random() < p_k_G_values[k]:
                Bi = self.weights[k]
                DIS = np.sum((weight_temp - Bi) ** 2, axis=1)
                # 获取p_k_G_values排名在前m以内的索引
                p_k_G_indices = np.argsort(p_k_G_values)[::-1][:m]
                # 获取DIS排名前一半以内的索引
                dis_indices = np.argsort(DIS)[:len(DIS) // 2]
                # 取交集
                valid_indices = np.intersect1d(p_k_G_indices, dis_indices)
                if len(valid_indices) >= self.T_size:
                    new_neighbors = np.random.choice(valid_indices, self.T_size, replace=False)
                else:
                    # 如果有效索引不足，则从切比雪夫距离最近的索引中选择
                    # 这里需要去掉本身
                    new_neighbors = np.argsort(DIS)[1:self.T_size+1]
                self.W_Bi_T[k] = new_neighbors
    
    def calculate_R_k_G(self, k, G, T, FEs_success, FEs):
        '''
        计算第k个邻域在第G代的成功更新率 R_{k,G}

        参数:
            k (int): 邻域索引
            G (int): 当前代
            T (int): 时间窗口大小
            FEs_success (list): 二维列表，FEs_success[k][g] 表示第k个邻域在第g代的成功更新次数
            FEs (list): 二维列表，FEs[k][g] 表示第k个邻域在第g代生成的解的总数

        返回:
            float: R_{k,G} 的值
        '''
        # if G <= T:
        #     return 0
        start = G - T
        numerator = sum(FEs_success[k][g] for g in range(start, G))
        denominator = sum(FEs[k][g] for g in range(start, G)) + self.epsilon
        return numerator / denominator

    def calculate_p_k_G(self, k, G, T, FEs_success, FEs):
        '''
        计算第k个邻域在第G代被使用的概率 p_{k,G}

        参数:
            k (int): 邻域索引
            G (int): 当前代
            T (int): 时间窗口大小
            FEs_success (list): 二维列表，FEs_success[k][g] 表示第k个邻域在第g代的成功更新次数
            FEs (list): 二维列表，FEs[k][g] 表示第k个邻域在第g代生成的解的总数

        返回:
            float: p_{k,G} 的值
        '''
        K = len(FEs)
        R_k_G_values = [self.calculate_R_k_G(i, G, T, FEs_success, FEs) for i in range(K)]
        total_R = sum(R_k_G_values)
        if total_R == 0:
            return 1 / K  # 避免除零错误
        return R_k_G_values[k] / total_R


class GenQuamtumCircuit:
    def __init__(self, weight_num, population_num, gen_qubit_num, sample,adj_list, topo, len_sample,ID_Chrome_in_complete,Len_Interval, Reduce_GStart):
        """
        初始化量子电路生成器
        
        输入参数:
            weight_num: int, 权重数量
            population_num: int, 种群大小
            gen_qubit_num: int, 量子比特数量
            sample: list, 样本数据
            adj_list: dict, 邻接表
        """
        self.weight_num = weight_num
        self.population_num = population_num
        self.gen_qubit_num = gen_qubit_num
        self.sample = sample
        self.adj_list = adj_list
        self.random_seed = 123
        self.replnum = 6#原本是3
        self.population = []
        self.fitness_values = []
        #np.random.seed(self.random_seed)
        #random.seed(self.random_seed)
        self.topo = topo
        self.len_sample = len_sample
        self.ID_Chrome_in_complete = ID_Chrome_in_complete
        self.Len_Interval = Len_Interval#样本长度
        self.Reduce_GStart = Reduce_GStart#样本个体参考值
        self.EP_X_ID = [] # 支配前沿ID
        self.Pop_FV = []# 个体的函数值
        self.Pop_BTFV = []# 个体的局部最优解
        self.EP_X_FV = []# 支配前沿的函数值
        self.qc_gen = self.create_gen_circuit()

        # 初始化FEs_success和FEs为二维列表
        self.num_generations = 50  # 假设初始代数为0，后续可根据实际情况更新
        self.FEs_success = [[1] * self.num_generations for _ in range(weight_num)]
        self.FEs = [[1] * self.num_generations for _ in range(weight_num)]


        #建立二维空数组，一维长度为weight_num

    def create_gen_circuit(self):
        """
        创建个体，直接使用长度为self.len_sample的浮点数数组
        """
        weight_population_gen_qc = []
        for pop_idx in range(self.weight_num):
            population_individuals_gen = []
            for individual_idx in range(self.population_num):
                # 直接生成浮点数数组，长度为self.len_sample
                qc_individual = np.random.rand(self.len_sample).tolist()
                temp_btfv = np.random.rand(self.len_sample).tolist()

                self.Pop_BTFV.append(temp_btfv)
                self.Pop_FV.append(self.Func(qc_individual))
                population_individuals_gen.append(qc_individual)
            weight_population_gen_qc.append(population_individuals_gen)

        return weight_population_gen_qc

    def mutate_gen(self, weight_index, pi):
        """
        执行基因突变操作

        输入参数:
            weight_index: int, 权重索引
            mutation_rate: list, 突变率列表

        返回值:
            list: 突变后的量子电路参数列表
        """
        temp_btsol = self.Pop_BTFV[weight_index]
        for i in range(self.population_num):
            #print("突变前的量子电路参数列表,", pi)
            q_gen_temp = pi
            Pro = 0.4  # 判断是否为变异位点的概率
            len_chrome = int(len(q_gen_temp))  # 染色体链长度
            for j in range(len_chrome):
                pick = random.random()  # 对每个量子比特位生成随机数
                if pick < Pro:  # 该位点可进行变异
                    # 修改qc_gen[weight_index][i]的参数
                    #print("突变前的量子电路参数列表,", (temp_btsol[j] - q_gen_temp[j]))
                    test = np.random.uniform(0, 1)
                    if temp_btsol[j] - q_gen_temp[j]:
                        #防止进化到局部最优就不动了
                        q_gen_temp[j] = abs((q_gen_temp[j] + np.random.uniform(0, 1)) % 1)
                    else:
                        #产生真正意义上的新的解pi=NS=LS±(GS−LS)∗ln(1/μ)，初始化的时候可以先把LS和GS
                        q_gen_temp[j] = abs((q_gen_temp[j] + (temp_btsol[j] - q_gen_temp[j]) * math.log(1 /np.random.uniform(0, 5))) % 1)
            #print("突变后的量子电路参数列表,",q_gen_temp)

    def cross_gen(self, mutate_i, mutate_j, mutate_z):
        """
        执行基因交叉操作

        输入参数:
            mutate_i: list, 需要进行交叉的目标量子电路参数列表
            mutate_j: list, 需要进行交叉的目标量子电路参数列表
            mutate_z: list, 需要进行交叉的目标量子电路参数列表
        返回值:
            list: 交叉后的量子电路参数列表
        """
        # 执行按位置的交叉，根据图中的逻辑
        circuits = [mutate_i, mutate_j, mutate_z]
        params_len = len(circuits[0])

        # 创建新的解决方案
        new_circuits = [[], [], []]

        # 按照示例模式进行交叉
        for i in range(params_len):
            # 位置模式: 0,1,2,0,1,2,0,1,2
            pos = i % 3

            # A: A1,C2,B3,A4,C5,B6,A7,C8,B9
            if pos == 0:  # 位置1,4,7
                new_circuits[0].append(circuits[0][i])  # A取自A
            elif pos == 1:  # 位置2,5,8
                new_circuits[0].append(circuits[2][i])  # A取自C
            else:  # 位置3,6,9
                new_circuits[0].append(circuits[1][i])  # A取自B

            # B: B1,A2,C3,B4,A5,C6,B7,A8,C9
            if pos == 0:  # 位置1,4,7
                new_circuits[1].append(circuits[1][i])  # B取自B
            elif pos == 1:  # 位置2,5,8
                new_circuits[1].append(circuits[0][i])  # B取自A
            else:  # 位置3,6,9
                new_circuits[1].append(circuits[2][i])  # B取自C

            # C: C1,B2,A3,C4,B5,A6,C7,B8,A9
            if pos == 0:  # 位置1,4,7
                new_circuits[2].append(circuits[2][i])  # C取自C
            elif pos == 1:  # 位置2,5,8
                new_circuits[2].append(circuits[1][i])  # C取自B
            else:  # 位置3,6,9
                new_circuits[2].append(circuits[0][i])  # C取自A

        return new_circuits

    def init_EP(self):
        for pi in range(self.Pop_size):
            np = 0
            F_V_P = self.Pop_FV[pi]
            for ppi in range(self.Pop_size):
                F_V_PP = self.Pop_FV[ppi]
                if pi != ppi:
                    if self.is_dominate(F_V_PP, F_V_P):
                        np += 1
            if np == 0:
                self.EP_X_ID.append(pi)
                self.EP_X_FV.append(F_V_P[:]) #保存函数值  

    def Func(self, qc_temp_ij):
        """
        直接将qc_temp_ij作为输入，进行后续处理
        """
        # 直接使用qc_temp_ij作为probabilities (已经是浮点数组)
        probabilities = np.array(qc_temp_ij)

        # 确保probabilities长度等于self.len_sample
        probabilities = probabilities[:self.len_sample]

        # 归一化 probabilities 数据到[0,1]
        min_val = np.min(probabilities)
        max_val = np.max(probabilities)
        normalized_data = (probabilities - min_val) / (max_val - min_val) * 2 % 1

        rand_gen = np.random.RandomState(456)
        measurement_result = np.array([
            rand_gen.choice([0, 1], p=[1 - prob, prob]) for prob in normalized_data
        ])

        '''第二步：重新连接图当中的边计算适应度'''
        self.topo.G = self.topo.reconnect_G(
            measurement_result,
            self.ID_Chrome_in_complete,
            self.Len_Interval
        )
        energy_FNDT = self.topo.calculate_FNDT()
        robustness = self.topo.calculate_robustness()

        return [energy_FNDT, robustness]

    def cpt_tchbycheff(self, weight_list,idx, X):
        # idx：X在种群中的位置
        # 计算X的切比雪夫距离（与理想点Z的）
        max = 0
        ri = weight_list[idx]
        F_X = self.Func(X)
        for i in range(2):
            if i == 0:
                F_X[i] = 1 - (F_X[i] / 200)
            if i == 1:
                F_X[i] = 1 - (F_X[i] / 0.5)
            fi = self.Tchebycheff_dist(ri[i], F_X[i], 0)
            
            if fi > max:
                max = fi
        return max
    def Tchebycheff_dist(self, w, f, z):
        # 计算切比雪夫距离
        return w * abs(f - z)
    def EO(self, weight_list, wi, p1,max_iteration=10):#之前是10
        tp_best = copy.deepcopy(p1)  # 初始化tp_best为当前解
        qbxf_tp = self.cpt_tchbycheff(weight_list, wi, tp_best)  # 计算当前解的Tchebycheff函数值
        h = 0  # 标志位，表示是否进行了有效的更新
        while max_iteration > 0:
            max_iteration -= 1
            if h == 1:  # 如果已经找到较优解则退出
                return tp_best,qbxf_tp
            temp_best = copy.deepcopy(p1)  # 复制当前解作为候选解
            #进行变异
            self.mutate_gen(wi, temp_best)
            qbxf_te = self.cpt_tchbycheff(weight_list, wi, temp_best)  # 计算变异后的Tchebycheff函数值
            if qbxf_te < qbxf_tp:  # 如果变异后的解更优，更新解
                h = 1
                qbxf_tp = qbxf_te
                tp_best = temp_best  # 更新最优解
        
        return tp_best,qbxf_tp  # 返回最优解
    
    def regenerate_circuit(self, weight_index):
        """
        重新随机生成指定权重索引的电路参数
        
        输入参数:
            weight_index: int, 权重索引
            
        返回值:
            list: 新生成的电路参数列表
        """
        # 随机生成新的电路参数
        new_params = np.random.rand(self.len_sample).tolist()
        # 更新电路参数
        self.qc_gen[weight_index][0] = new_params
        # 更新适应度值
        self.Pop_FV[weight_index] = self.Func(new_params)
        # 更新局部最优解
        self.Pop_BTFV[weight_index] = new_params
        
        return new_params
        
    def generate_next(self, weight_list, gen, pi, ik, il):

        p0 = self.qc_gen[pi][0]
        p1 = self.qc_gen[ik][0]
        p2 = self.qc_gen[il][0]
        
        # 进化下一代个体。基于自身Xi+邻居中随机选择的2个Xk，Xl 还考虑gen 去进化下一代
        qbxf_p0 = self.cpt_tchbycheff(weight_list, pi, p0)
        qbxf_i = qbxf_p0

        # 需要深拷贝成独立的一份
        n_p0, n_p1, n_p2 = copy.deepcopy(p0), copy.deepcopy(p1), copy.deepcopy(p2)
        # 突变进化量子加上观测，有希望观测出来一个更好的
        #用上当前的局部最优解进行个体QPi的更新
        n_p0,qbxf_np0 = self.EO(weight_list, pi, n_p0)

        qbxf_1 = np.array([qbxf_p0, qbxf_np0])
        best_1 = np.argmin(qbxf_1)
        # 选中切比雪夫距离最小（最好的）个体
        Y1 = [p0, n_p0][best_1]

        # 领域三个变量一组循环交叉更新量子
        n_p1, n_p2 ,n_p3= self.cross_gen(n_p0, n_p1, n_p2)
        # 交叉后的切比雪夫距离
        qbxf_np1 = self.cpt_tchbycheff(weight_list, pi, n_p1)
        qbxf_np2 = self.cpt_tchbycheff(weight_list, pi, n_p2)
        qbxf_np3 = self.cpt_tchbycheff(weight_list, pi, n_p3)

        qbxf = np.array([qbxf_p0, qbxf_np0, qbxf_np1, qbxf_np2, qbxf_np3])
        # 上面的是交叉突变后的局部最优解，这里更新一下
        best = np.argmin(qbxf)
        # 选中切比雪夫距离最小（最好的）个体
        Y2 = [p0, n_p0, n_p1, n_p2, n_p3][best]

        # 随机选中目标中的某一个目标进行判断，目标太多，不要贪心，随机选一个目标就好
        fm = np.random.randint(0, 2)
        # 如果是极小化目标求解，以0。5的概率进行更详细的判断。（返回最优解策略不能太死板，否则容易陷入局部最优）
        if np.random.rand() < 0.5:
            FY1 = self.Func(Y1)
            FY2 = self.Func(Y2)
            # 如果随机选的这个目标Y2更好，就返回Y2的
            if FY2[fm] > FY1[fm]:
                return Y2,qbxf_i,qbxf[best]
            else:
                return Y1,qbxf_i,qbxf_1[best_1]
        #返回计算后的原指标和进化指标以及生成的下一代的结果
        return Y2,qbxf_i,qbxf[best]

    def update_EP_By_ID(self, id, F_Y):
    # 如果id存在，则更新其对应函数集合的值
        if id in self.EP_X_ID:
            # 拿到所在位置
            position_pi = self.EP_X_ID.index(id)
            # 更新函数值
            self.EP_X_FV[position_pi][:] = F_Y[:]
    def update_EP_By_Y(self, id_Y):
        # 根据Y更新前沿
        # 根据Y更新EP
        i = 0
        # 拿到id_Y的函数值
        F_Y = self.Pop_FV[id_Y]
        # 需要被删除的集合
        Delet_set = []
        # 支配前沿集合，的数量
        Len = len(self.EP_X_FV)
        for pi in range(Len):
            # F_Y是否支配pi号个体，支配？则pi被剔除
            if self.is_dominate(F_Y, self.EP_X_FV[pi]):
                # 列入被删除的集合
                Delet_set.append(pi)
                break
            if i != 0:
                break
            if self.is_dominate(self.EP_X_FV[pi], F_Y):
                # 它有被别人支配！！记下来能支配它的个数
                i += 1
        # 新的支配前沿的ID集合，种群个体ID，
        new_EP_X_ID = []
        # 新的支配前沿集合的函数值
        new_EP_X_FV = []
        for save_id in range(Len):
            if save_id not in Delet_set:
                # 不需要被删除，那就保存
                new_EP_X_ID.append(self.EP_X_ID[save_id])
                new_EP_X_FV.append(self.EP_X_FV[save_id])
        # 更新上面计算好的新的支配前沿
        self.EP_X_ID = new_EP_X_ID
        self.EP_X_FV = new_EP_X_FV
        # 如果i==0，意味着没人支配id_Y
        # 没人支配id_Y？太好了，加进支配前沿呗
        if i == 0:
            # 不在里面直接加新成员
            if id_Y not in self.EP_X_ID:
                self.EP_X_ID.append(id_Y)
                self.EP_X_FV.append(F_Y)
            else:
                # 本来就在里面的，更新它
                idy = self.EP_X_ID.index(id_Y)
                self.EP_X_FV[idy] = F_Y[:]
        # over
        return self.EP_X_ID, self.EP_X_FV
    def is_dominate(self, F_X, F_Y):
        # 判断F_X是否支配F_Y
        if type(F_Y) != list:
            F_X = self.Func(F_X)
            F_Y = self.Func(F_Y)
        i = 0
        for xv, yv in zip(F_X, F_Y):
            if xv > yv:
                i = i + 1
            if xv < yv:
                return False
        if i != 0:
            return True
        return False
    def update_BTX(self, P_B, Y, weight_list, current_generation):
        # 根据Y更新P_B集内邻居
        for j in P_B:
            Xj = self.Pop_BTFV[j]
            d_x = self.cpt_tchbycheff(weight_list, j, Xj)
            d_y = self.cpt_tchbycheff(weight_list, j, Y)
            if d_y < d_x:
                # 在这里先把领域优势解存一下，存到Pop_BTFV
                self.Pop_BTFV[j] = Y
                # 更新FEs_success
                self.FEs_success[j][current_generation] += 1
            self.FEs[j][current_generation] += 1




class QCDEopt:
    def __init__(self, N, super_node_percent, topo, max_iteration=50, need_dynamic=True, draw_w=False):
        """
        初始化QMOEAD优化器
        
        参数:
            N: 节点数量
            super_node_percent: 超级节点百分比
            topo: 拓扑结构对象
            max_iteration: 最大迭代次数
            need_dynamic: 是否需要动态显示
            draw_w: 是否绘制权重图
        """
        self.N = N
        self.super_node_percent = super_node_percent
        self.topo = copy.deepcopy(topo)
        self.max_iteration = max_iteration
        self.need_dynamic = need_dynamic
        self.draw_w = draw_w
        
        # 初始化基本参数
        self.weight_num = 30  # 30
        self.population_num = 1
        self.T_size = max(4,self.weight_num/10)
        self.Z = [0, 0]
        self.mu = 0.5
        self.sigma = 1
        self.epsilon = 0.05
        self.TimeWindow = 1
        
        # 路径设置
        self.base_dir = './log'  # 基础日志目录
        self.results_dir = './results'  # 结果保存目录
        self.picture_dir = './picture'  # 图片保存目录
        self.algorithm_name = 'QCDEDynamic_' + str(self.weight_num)  # 算法名称
        
        # 创建必要的目录
        for directory in [
            f'{self.base_dir}/{self.algorithm_name}',
            f'{self.results_dir}/{self.algorithm_name}',
            f'{self.picture_dir}/{self.algorithm_name}'
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # 初始化权重列表
        self.weight_qc = self._init_weight_list()
        self.weight_list = self.weight_qc.weights
        self.W_Bi_T = []
        
    def _init_weight_list(self):
        """初始化权重列表"""

        return ChebyshevDynamicNeighborhood(self.weight_num, self.mu, self.sigma, self.T_size, self.epsilon)
        # theta = 1.0 / self.weight_num
        # return [[1 - theta * i, i * theta] for i in range(self.weight_num)]
        
    def setup_quantum_circuit(self, sample_raw, len_sample, ID_Chrome_in_complete, Len_Interval, Reduce_GStart):
        """设置量子电路参数"""
        self.gen_qubit_num = int(np.ceil(np.log2(len_sample)))
        sample = list(sample_raw)
        diff = 2 ** self.gen_qubit_num - len_sample
        sample.extend([0] * diff)
        
        # 计算每个权重Wi的T个邻居
        self.weight_qc.create_quantum_circuit()  
        self.W_Bi_T = self.weight_qc.W_Bi_T

        # 创建量子电路计算器
        self.gen_calculator = GenQuamtumCircuit(
            self.weight_num, 
            self.population_num, 
            self.gen_qubit_num, 
            sample, 
            self.W_Bi_T, 
            self.topo, 
            len_sample,
            ID_Chrome_in_complete, 
            Len_Interval, 
            Reduce_GStart
        )
        
        # 初始化解的跟踪
        self._init_solution_tracking()
        
    def _init_solution_tracking(self):
        """初始化解的跟踪参数"""
        self.best_layer_index = []
        self.life_cycle_list = []
        self.unchanged_generations = [0] * self.gen_calculator.weight_num
        self.previous_solutions = [None] * self.gen_calculator.weight_num
        
        for i in range(self.gen_calculator.weight_num):
            life_cycle = [0] * self.gen_calculator.population_num
            self.life_cycle_list.append(life_cycle)
            self.previous_solutions[i] = copy.deepcopy(self.gen_calculator.qc_gen[i][0])
    def _check_and_update_solutions(self):
        """检查并更新解"""
        for i in range(self.gen_calculator.weight_num):
            if i not in self.gen_calculator.EP_X_ID:
                if np.array_equal(self.gen_calculator.qc_gen[i][0], self.previous_solutions[i]):
                    self.unchanged_generations[i] += 1
                else:
                    self.unchanged_generations[i] = 0
                
                if self.unchanged_generations[i] >= 3:
                    print(f"解 {i} 已经 {self.unchanged_generations[i]} 代没有改变，重新生成")
                    self.gen_calculator.regenerate_circuit(i)
                    self.unchanged_generations[i] = 0
            
            self.previous_solutions[i] = copy.deepcopy(self.gen_calculator.qc_gen[i][0])
    
    def _optimize_weights(self, step):
        """优化每个权重"""
        for weight_index in range(self.gen_calculator.weight_num):
            #print(f"weight_index: {weight_index} ++++++++++++++++++++")
            
            # 选择邻居
            pi = weight_index
            Bi = self.W_Bi_T[pi]
            k, l = rd.sample(range(1, self.T_size), 2)
            ik, il = Bi[k], Bi[l]
            
            # 生成下一代
            Y, cbxf_i, cbxf_y = self.gen_calculator.generate_next(self.weight_list, step, pi, ik, il)
            
            # 更新解
            if cbxf_y < cbxf_i:
                self.now_y = pi
                self.gen_calculator.qc_gen[pi][0] = copy.deepcopy(Y)
                F_Y = self.gen_calculator.Func(Y)[:]
                self.gen_calculator.Pop_FV[pi] = F_Y
                self.gen_calculator.update_EP_By_ID(pi, F_Y)
            
            self.gen_calculator.update_EP_By_Y(pi)
            self.gen_calculator.update_BTX(Bi, Y, self.weight_list, step)
            #print(f"gen_calculator.Pop_FV: {self.gen_calculator.Pop_FV}")

    def _log_initial_state(self, log_dir):
        """记录初始状态"""
        with open(f'{log_dir}/initial_state.txt', 'w', encoding='utf-8') as f:
            f.write(f"节点数量: {self.N}\n")
            f.write(f"超级节点比例: {self.super_node_percent}\n")
            f.write(f"初始FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"初始鲁棒性: {self.topo.calculate_robustness()}\n")
            f.write(f"权重数量: {self.weight_num}\n")
            f.write(f"最大迭代次数: {self.max_iteration}\n")
            f.write(f"邻居大小: {self.T_size}\n")
            f.write("\n初始种群适应度值:\n")
            for i, fv in enumerate(self.gen_calculator.Pop_FV):
                f.write(f"个体 {i}: {fv}\n")
        
        # 清空generation_all.txt文件
        with open(f'{log_dir}/generation_all.txt', 'w', encoding='utf-8') as f:
            pass
    
    def _log_generation_state(self, log_dir, step):
        """记录每一代的状态"""
        # 使用追加模式打开文件
        with open(f'{log_dir}/generation_all.txt', 'a', encoding='utf-8') as f:
            f.write(f"第 {step} 代\n")
            f.write(f"Pareto前沿个体数量: {len(self.gen_calculator.EP_X_ID)}\n")
            f.write("\nPareto前沿个体:\n")
            for idx in self.gen_calculator.EP_X_ID:
                f.write(f"个体 {idx}: {self.gen_calculator.Pop_FV[idx]}\n")
            f.write("\n")  # 添加换行以分隔不同代的信息
    
    def _log_final_results(self, log_dir):
        """记录最终结果"""
        with open(f'{log_dir}/final_results.txt', 'w', encoding='utf-8') as f:
            f.write("优化完成\n")
            f.write(f"最终Pareto前沿个体数量: {len(self.gen_calculator.EP_X_ID)}\n")
            f.write("\n最终Pareto前沿个体:\n")
            for idx in self.gen_calculator.EP_X_ID:
                f.write(f"个体 {idx}: {self.gen_calculator.Pop_FV[idx]}\n")
            
            # 记录最终的FNDT和鲁棒性
            f.write(f"\n最终FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"最终鲁棒性: {self.topo.calculate_robustness()}\n")
        
        # 保存最终的拓扑结构和结果
        with open(f'{log_dir}/final_results.pkl', 'wb') as fp:
            pickle.dump({
                'EP_X_ID': self.gen_calculator.EP_X_ID,
                'EP_X_FV': self.gen_calculator.EP_X_FV,
                'Pop_FV': self.gen_calculator.Pop_FV,
                'topo': self.topo
            }, fp)
    def _save_pareto_front(self, iteration):
        """
        绘制并保存Pareto前沿图
        
        参数:
            iteration: 当前迭代次数
        """
        # 创建新的图形，设置固定大小
        plt.figure(figsize=(10, 8))
        
        # 获取Pareto前沿ID和所有个体的适应度值
        Pareto_F_ID = self.gen_calculator.EP_X_ID
        Pop_F_Data = np.array(self.gen_calculator.Pop_FV)
        
        # 绘制所有个体的适应度值
        for pi, pp in enumerate(Pop_F_Data):
            plt.scatter(pp[0], pp[1], c='blue', marker='o')
        
        # 绘制Pareto前沿解
        pareto_points = []
        for pid in Pareto_F_ID:
            p = Pop_F_Data[pid]
            pareto_points.append(p)
            plt.scatter(p[0], p[1], c='red', marker='*', s=100)
        
        # 设置图表属性
        plt.grid(True)
        plt.xlabel('Function 1:FNDT')
        plt.ylabel('Function 2:Robustness')
        plt.title("Pareto Front")

        # 构建保存路径
        folder_path = f'{self.picture_dir}/{self.algorithm_name}/pareto_front_{self.algorithm_name}_N{self.N}_SP{int(self.super_node_percent*100)}'
        os.makedirs(folder_path, exist_ok=True)
        
        # 保存图像
        if iteration is not None:
            plt.savefig(f'{folder_path}/iter_{iteration}.png')
        else:
            # 保存最终结果到results目录
            plt.savefig(f'{self.results_dir}/{self.algorithm_name}/pareto_front_{self.N}_{int(self.super_node_percent*100)}.png')
        
        # 关闭图形，释放资源
        plt.close()
    def optimize(self, Reduce_GStart, paramlen, ID_Chrome_in_complete, Len_Interval):
        """执行优化过程并记录日志"""
        print(f"开始QCDE优化，节点数量: {self.N}, 超级节点比例: {self.super_node_percent}")
        
        # 设置量子电路
        sample_raw = list(range(paramlen))
        self.setup_quantum_circuit(sample_raw, paramlen, ID_Chrome_in_complete, Len_Interval, Reduce_GStart)
        
        # 创建日志目录
        log_dir = f'{self.base_dir}/{self.algorithm_name}/{self.N}_{int(100*self.super_node_percent)}'
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 记录初始状态
        self._log_initial_state(log_dir)
        
        # 开始迭代优化
        start_time = time.time()
        
        # 执行优化过程
        for step in range(self.max_iteration):
            print(f"QCDE: 迭代 {step + 1}/{self.max_iteration}")

            # 检查解的变化并更新
            self._check_and_update_solutions()

            # 对每个权重进行优化
            self._optimize_weights(step)

            # 记录每一代的状态
            self._log_generation_state(log_dir, step)
            
            # 保存当前代的Pareto前沿图
            self._save_pareto_front(step)
            
            self.weight_qc.change_weight_circuit(step, self.TimeWindow, self.gen_calculator.FEs_success, self.gen_calculator.FEs)
        # 计算优化时间
        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"QCDE优化完成，耗时: {optimization_time:.2f}秒")
        
        # 记录最终结果
        self._log_final_results(log_dir)
        
        # 保存最终Pareto前沿图 - 这里不传递iteration参数，使用默认值None
        self._save_pareto_front(None)  # 传递None表示这是最终结果

        # 返回Pareto前沿
        return self.gen_calculator.EP_X_ID, self.gen_calculator.EP_X_FV
