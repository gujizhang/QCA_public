import numpy as np
import random as rd
from gen_class_qc import GenQuamtumCircuit
import copy
import os
import pickle
import matplotlib.pyplot as plt
class QMOEADopt:
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
        self.weight_num = 30
        self.population_num = 1
        self.T_size = max(4,self.weight_num/10)
        self.Z = [0, 0]
        
        # 初始化权重列表
        self.weight_list = self._init_weight_list()
        self.W_Bi_T = []
        
        # 路径设置 - 与QCDEopt保持一致
        self.base_dir = './log'  # 基础日志目录
        self.results_dir = './results'  # 结果保存目录
        self.picture_dir = './picture'  # 图片保存目录
        self.algorithm_name = 'QMOEAD'  # 算法名称
        
        # 创建必要的目录
        for directory in [
            f'{self.base_dir}/{self.algorithm_name}',
            f'{self.results_dir}/{self.algorithm_name}',
            f'{self.picture_dir}/{self.algorithm_name}'
        ]:
            os.makedirs(directory, exist_ok=True)
    def _init_weight_list(self):
        """初始化权重列表"""
        theta = 1.0 / self.weight_num
        return [[1 - theta * i, i * theta] for i in range(self.weight_num)]
    
    def setup_quantum_circuit(self, sample_raw, len_sample, ID_Chrome_in_complete, Len_Interval, Reduce_GStart):
        """设置量子电路参数"""
        self.gen_qubit_num = int(np.ceil(np.log2(len_sample)))
        sample = list(sample_raw)
        diff = 2 ** self.gen_qubit_num - len_sample
        sample.extend([0] * diff)
        
        # 计算每个权重Wi的T个邻居
        weight_temp = np.array(self.weight_list)
        for bi in range(weight_temp.shape[0]):
            Bi = self.weight_list[bi]
            DIS = np.sum((weight_temp - Bi) ** 2, axis=1)
            B_T = np.argsort(DIS)
            B_T = B_T[1:self.T_size + 1]
            self.W_Bi_T.append(B_T)
            
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
            self.gen_calculator.update_BTX(Bi, Y, self.weight_list)
            #print(f"gen_calculator.Pop_FV: {self.gen_calculator.Pop_FV}")


    def _log_initial_state(self, log_dir):
        """记录初始状态"""
        with open(f'{log_dir}/initial_state.txt', 'w', encoding='utf-8') as f:
            f.write(f"节点数量: {self.N}\n")
            f.write(f"超级节点比例: {self.super_node_percent}\n")
            f.write(f"初始FNDT: {self.topo.calculate_FNDT()}\n")
            f.write(f"初始鲁棒性: {self.topo.calculate_robustness()}\n")
            f.write(f"权重数量: {self.weight_num}\n")
            f.write(f"种群数量: {self.population_num}\n")
            f.write(f"邻居大小: {self.T_size}\n")
            f.write(f"最大迭代次数: {self.max_iteration}\n")
            f.write("\n初始种群适应度值:\n")
            for i, fv in enumerate(self.gen_calculator.Pop_FV):
                f.write(f"个体 {i}: {fv}\n")
                
    def _log_generation_state(self, log_dir, step):
        # 追加到汇总文件
        with open(f'{log_dir}/generation_all.txt', 'a', encoding='utf-8') as f:
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write(f"第 {step} 代\n")
            f.write(f"Pareto前沿个体数量: {len(self.gen_calculator.EP_X_ID)}\n")
            f.write("\nPareto前沿个体:\n")
            for idx in self.gen_calculator.EP_X_ID:
                f.write(f"个体 {idx}: {self.gen_calculator.Pop_FV[idx]}\n")
            # 输入日志当前代的所有个体适应度值
            f.write(f"第 {step} 代所有个体适应度值:\n")
            for idx, fv in enumerate(self.gen_calculator.Pop_FV):
                f.write(f"个体 {idx}: {fv}\n")
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
                'pareto_individuals': [self.gen_calculator.qc_gen[idx][0] for idx in self.gen_calculator.EP_X_ID],
                'pareto_objectives': [self.gen_calculator.Pop_FV[idx] for idx in self.gen_calculator.EP_X_ID],
                'topo': self.topo
            }, fp)
    
    def optimize(self):
        """执行优化过程并记录日志"""
        # 创建日志目录
        log_dir = f'{self.base_dir}/{self.algorithm_name}/{self.N}_{int(100*self.super_node_percent)}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 记录初始状态
        self._log_initial_state(log_dir)
        
        # 清空generation_all.txt文件
        with open(f'{log_dir}/generation_all.txt', 'w', encoding='utf-8') as f:
            pass

        # 执行优化过程
        for step in range(self.max_iteration):
            print(f"QMOEAD: 迭代 {step+1} +++++++++++++++++++++++++++++++++++++++++++++++")

            # 检查解的变化并更新
            self._check_and_update_solutions()

            # 对每个权重进行优化
            self._optimize_weights(step)

            # 记录每一代的状态
            self._log_generation_state(log_dir, step)
            # 保存当前代的Pareto前沿图
            self._save_pareto_front(step)
            # 删除重复的输出，只保留一次
            print(f'迭代 {step+1},支配前沿个体(moead.EP_X_ID) :{self.gen_calculator.EP_X_FV}')

        # 记录最终结果
        self._log_final_results(log_dir)
        
        # 保存最终的Pareto前沿图
        self._save_pareto_front(None)