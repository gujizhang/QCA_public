import pickle
import warnings
from qiskit_algorithms.utils import algorithm_globals
from heterogeneous_top import HeterogeneousTopology
import os
import random as rd
from QMOEADopt import QMOEADopt
from TEAMopt import TEAMopt
from TOSGopt import TOSGopt
from QCDEopt import QCDEopt
from QPSOopt import QPSOopt
from QHDBOopt import QHDBOopt
from MOFDAopt import MOFDAopt

"""使用qiskit编写的一个多目标遗传算法求解问题的代码，
问题背景是有一个N个节点的无标度拓扑结构，当中的每个点是一个路由器，
设置数据集基本参数"""

algorithm_globals.random_seed = 12
rd.seed(algorithm_globals.random_seed)
Nlist = [700]#[400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
super_node_percent_list = [0.1]#[0.05, 0.1, 0.15, 0.2]
max_iteration = 50

# 是否动态展示
# need_dynamic = False
need_dynamic = False
# 是否画出权重图
draw_w = False

for N in Nlist: 
    for super_node_percent in super_node_percent_list:
        # 拼接参数形成数据集地址
        Kmeans_data_path = f'./data/Kmeans_G_{algorithm_globals.random_seed}_{N}_{super_node_percent}.pkl'
        G_encode_data_path = f'./data/G_encode_data_{algorithm_globals.random_seed}_{N}_{super_node_percent}.pkl'
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        # 创建拓扑结构
        topo = HeterogeneousTopology(N=N, sink_node_count=1, R=60, M=800, seed=12, super_node_percent=super_node_percent)

        # 加载数据
        if not os.path.exists(Kmeans_data_path):
            print("generate top use:heterogeneous_top.py first please")
            continue
            
        with open(Kmeans_data_path, 'rb') as f:
            topo.G = pickle.load(f)
            topo.k_means_class = pickle.load(f)
            print("generate top data has been loaded successfully!")
            
        if not os.path.exists(G_encode_data_path):
            print("generate top use:heterogeneous_top.py first please")
            continue
            
        with open(G_encode_data_path, 'rb') as f:
            G_adj_matrix = pickle.load(f)
            Reduce_GStart = pickle.load(f)
            G_encode_adj_matrix = pickle.load(f)
            G_nei_node = pickle.load(f)
            ID_Chrome = pickle.load(f)
            ID_Chrome_in_complete = pickle.load(f)
            Len_Interval = pickle.load(f)
            Len_C = pickle.load(f)
            print("G_encode_data has been loaded successfully!")

        # 预处理：计算初始FNDT和鲁棒性
        FNDT = topo.calculate_FNDT()
        rob = topo.calculate_robustness()
        print("FNDT:", FNDT)
        print("Robustness:", rob)


    
        #创建TEAM算法的优化器(改进的NSGA-II)，改了
        print("TEAM")
        optimizerTEAMopt = TEAMopt(N, super_node_percent, topo, 12, need_dynamic, draw_w)
        optimizerTEAMopt.optimize(Reduce_GStart, len(Reduce_GStart), ID_Chrome_in_complete, Len_Interval)

        # #创建QCDE算法的优化器，改了
        # print("QCDE")
        # optimizerQCDEopt = QCDEopt(N, super_node_percent, topo, max_iteration, need_dynamic, draw_w)
        # optimizerQCDEopt.optimize(Reduce_GStart, len(Reduce_GStart), ID_Chrome_in_complete, Len_Interval)

        # #创建算法QPSO算法的优化器(改进的网格法)，改了
        # print("QPSO")
        # optimizerQPSOopt = QPSOopt(N, super_node_percent, topo, max_iteration, need_dynamic, draw_w)
        # optimizerQPSOopt.optimize(Reduce_GStart, len(Reduce_GStart), ID_Chrome_in_complete, Len_Interval)


        # # 创建TOSG算法的优化器，正在改
        # print("TOSG")
        # optimizerTOSGopt = TOSGopt(N, super_node_percent, topo, 20, need_dynamic, draw_w)
        # optimizerTOSGopt.optimize(Reduce_GStart, len(Reduce_GStart), ID_Chrome_in_complete, Len_Interval)

        # #创建QHDBO算法的优化器，改了
        # print("QHDBO")
        # optimizerQHDBOopt = QHDBOopt(N, super_node_percent, topo, max_iteration, need_dynamic, draw_w)
        # optimizerQHDBOopt.optimize(Reduce_GStart, len(Reduce_GStart), ID_Chrome_in_complete, Len_Interval)

        # #创建MOFDA算法的优化器，改了
        # print("MOFDA")
        # optimizerMOFDAopt = MOFDAopt(N, super_node_percent, topo, max_iteration, need_dynamic, draw_w)
        # optimizerMOFDAopt.optimize(Reduce_GStart, len(Reduce_GStart), ID_Chrome_in_complete, Len_Interval)

        # #创建QMOEAD算法的优化器，有点慢，先搁置
        # print("QMOEAD")
        # optimizerQMOEADopt = QMOEADopt(N, super_node_percent, topo, max_iteration, need_dynamic, draw_w)
        # optimizerQMOEADopt.setup_quantum_circuit(Reduce_GStart, len(Reduce_GStart), ID_Chrome_in_complete, Len_Interval, Reduce_GStart)
        # optimizerQMOEADopt.optimize()