import math
import numpy as np
import random
import copy
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
        
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
        self.shots = 200
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


        #建立二维空数组，一维长度为weight_num

    def create_gen_circuit(self):
        """
        创建个体，直接使用长度为self.len_sample的浮点数数组
        """
        weight_population_gen_qc = []
        n = int(np.ceil(np.log2(self.len_sample)))
        
        for pop_idx in range(self.weight_num):
            population_individuals_gen = []
            for individual_idx in range(self.population_num):
                # 生成长度为n的浮点数数组
                qc_individual = np.random.rand(n).tolist()
                temp_btfv = np.random.rand(n).tolist()


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
        使用量子电路处理qc_temp_ij作为概率输入，并从测量结果中选择概率最大的结果
        """
        # 确定量子比特数量
        n = len(qc_temp_ij)
        sequence_length = 2 ** n
        # 创建量子电路
        qc = QuantumCircuit(n, n)
        
        # 根据概率对每个量子比特分别设置旋转门
        for idx in range(n):
            # 确保概率值在[0,1]范围内
            p = max(0, min(1, qc_temp_ij[idx]))
            theta = 2 * np.arccos(np.sqrt(p))
            qc.ry(theta, idx)
        
        # 测量所有量子比特
        qc.measure(range(n), range(n))
        
        # 使用Sampler原语采样
        sampler = Sampler()
        
        # 执行采样
        job = sampler.run(qc, shots=self.shots)
        result = job.result()
        
        # 获取测量结果（概率分布）
        counts = result.quasi_dists[0].binary_probabilities()
        
                # 将概率分布转换为具体序列
        states, probs = zip(*counts.items())
        probs = np.array(probs)
        probs /= probs.sum()  # 归一化概率
        #np.random.seed(42)
        rand_gen = np.random.RandomState(42)
        # 随机抽取序列
        chosen_states = rand_gen.choice(states, size=sequence_length, p=probs)
        
        # 拼接生成二进制序列
        generated_sequence = ''.join(chosen_states)
        
        # 将二进制字符串转换为0和1的列表
        measurement_result = [int(bit) for bit in generated_sequence[:self.len_sample]]


        # 确保结果长度与len_sample一致
        if len(measurement_result) < self.len_sample:
            # 如果结果太短，用随机值填充
            measurement_result.extend(rand_gen.randint(0, 2, self.len_sample - len(measurement_result)))
        else:
            # 如果结果太长，截断
            measurement_result = measurement_result[:self.len_sample]
        
        # 转换为numpy数组
        measurement_result = np.array(measurement_result)
        
        '''第二步：重新连接图当中的边计算适应度'''
        self.topo.G = self.topo.reconnect_G(
            measurement_result,
            self.ID_Chrome_in_complete,
            self.Len_Interval
        )
        energy_FNDT = self.topo.calculate_FNDT()
        robustness = self.topo.calculate_robustness()
        
        return [energy_FNDT, robustness]
    # def Func(self, qc_temp_ij):
    #     """
    #     直接将qc_temp_ij作为输入，进行后续处理
    #     """
    #     # 直接使用qc_temp_ij作为probabilities (已经是浮点数组)
    #     probabilities = np.array(qc_temp_ij)

    #     # 确保probabilities长度等于self.len_sample
    #     probabilities = probabilities[:self.len_sample]

    #     # 归一化 probabilities 数据到[0,1]
    #     min_val = np.min(probabilities)
    #     max_val = np.max(probabilities)
    #     normalized_data = (probabilities - min_val) / (max_val - min_val) * 2 % 1

    #     rand_gen = np.random.RandomState(456)
    #     measurement_result = np.array([
    #         rand_gen.choice([0, 1], p=[1 - prob, prob]) for prob in normalized_data
    #     ])

    #     '''第二步：重新连接图当中的边计算适应度'''
    #     self.topo.G = self.topo.reconnect_G(
    #         measurement_result,
    #         self.ID_Chrome_in_complete,
    #         self.Len_Interval
    #     )
    #     energy_FNDT = self.topo.calculate_FNDT()
    #     robustness = self.topo.calculate_robustness()

    #     return [energy_FNDT, robustness]
    # def Func(self, qc_temp_ij):
    #     """
    #     使用量子电路处理qc_temp_ij作为概率输入，并从测量结果中选择概率最大的结果
    #     """

    #     # 确保probabilities长度等于self.len_sample
    #     probabilities = np.array(qc_temp_ij)[:self.len_sample]
        
    #     # 确定量子比特数量，使得2^n >= len_sample
    #     n = int(np.ceil(np.log2(len(probabilities))))
        
    #     # 创建量子电路
    #     qc = QuantumCircuit(n, n)
        
    #     # 根据概率对每个量子比特分别设置旋转门
    #     for idx in range(min(n, len(probabilities))):
    #         # 确保概率值在[0,1]范围内
    #         p = max(0, min(1, probabilities[idx]))
    #         theta = 2 * np.arccos(np.sqrt(p))
    #         qc.ry(theta, idx)
        
    #     # 测量所有量子比特
    #     qc.measure(range(n), range(n))
        
    #     # 使用Sampler原语采样
    #     sampler = Sampler()
    #      # 可以根据需要调整采样次数
        
    #     # 执行采样
    #     job = sampler.run(qc, shots=self.shots)
    #     result = job.result()
        
    #     # 获取测量结果（概率分布）
    #     counts = result.quasi_dists[0].binary_probabilities()
        
    #     # 找出概率最大的测量结果
    #     max_prob_bitstring = max(counts, key=counts.get)
        
    #     # 将二进制字符串转换为0和1的列表
    #     measurement_result = [int(bit) for bit in max_prob_bitstring.zfill(n)]
        
    #     # 确保结果长度与len_sample一致
    #     if len(measurement_result) < self.len_sample:
    #         # 如果结果太短，用随机值填充
    #         measurement_result.extend(np.random.randint(0, 2, self.len_sample - len(measurement_result)))
    #     else:
    #         # 如果结果太长，截断
    #         measurement_result = measurement_result[:self.len_sample]
        
    #     # 转换为numpy数组
    #     measurement_result = np.array(measurement_result)
        
    #     '''第二步：重新连接图当中的边计算适应度'''
    #     self.topo.G = self.topo.reconnect_G(
    #         measurement_result,
    #         self.ID_Chrome_in_complete,
    #         self.Len_Interval
    #     )
    #     energy_FNDT = self.topo.calculate_FNDT()
    #     robustness = self.topo.calculate_robustness()
        
    #     return [energy_FNDT, robustness]

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
        # 计算量子比特数量
        n = int(np.ceil(np.log2(self.len_sample)))
        
        # 随机生成新的电路参数
        new_params = np.random.rand(n).tolist()
        # # 随机生成新的电路参数
        # new_params = np.random.rand(self.len_sample).tolist()

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
        #print("当前是：",id_Y,"    ",Delet_set)
        for save_id in range(Len):
            if save_id not in Delet_set:
                # 不需要被删除，那就保存
                new_EP_X_ID.append(self.EP_X_ID[save_id])
                new_EP_X_FV.append(self.EP_X_FV[save_id])
        # 更新上面计算好的新的支配前沿
        #print("当前是EP_X_ID和EP_X_FV：",self.EP_X_ID,self.EP_X_FV)
        self.EP_X_ID = new_EP_X_ID
        self.EP_X_FV = new_EP_X_FV
        # 如果i==0，意味着没人支配id_Y
        # 没人支配id_Y？太好了，加进支配前沿呗
        if i == 0:
            # 不在里面直接加新成员
            #print("当前是：",id_Y,"    ",F_Y)
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
    def update_BTX(self, P_B, Y, weight_list):
        # 根据Y更新P_B集内邻居
        for j in P_B:
            Xj = self.Pop_BTFV[j]
            d_x = self.cpt_tchbycheff(weight_list, j, Xj)
            d_y = self.cpt_tchbycheff(weight_list, j, Y)
            if d_y < d_x:
                #在这里先把领域优势解存一下，存到Pop_BTFV
                self.Pop_BTFV[j] = Y




