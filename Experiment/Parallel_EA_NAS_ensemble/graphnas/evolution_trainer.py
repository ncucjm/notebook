import time
import os
import json
import ray
import torch
import numpy as np
from graphnas.gnn_model_manager import CitationGNNManager
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager
from graphnas.search_space import MacroSearchSpace

@ray.remote(num_gpus=0.5)
class Evolution_Trainer(object):
    """
    This class implements the Asyncronous Aging Evolution,
    proposed by Real et. al. on:

    Regularized Evolution for Image Classifier Architecture Search

    available on: https://arxiv.org/abs/1802.01548
    """
    def __init__(self, args):
        self.cycle = 0
        self.args = args
        self.random_seed = args.random_seed
        self.population = []
        self.accuracies = []
        self.population_size = args.population_size
        self.population_history = []
        self.accuracies_history = []
        self.sample_size = args.sample_size
        self.cycles = args.cycles
        self.init_time = 0
        self.ev_train_time = []
        self.ev_acc = []
        self.ev_random_initial_time = []
        self.ev_random_acc = []

        self.build_model()

        # self.__initialize_population_Random()
        # 种子初始化初始化由初始化函数实现
        #　random初始化种群
        # if self.args.initialize_mode == "random":
        #     # 如果random_population已存在则不再初始化
        #     if not os.path.exists("random_population.txt"):
        #         self.__initialize_population_Random()
        #     else:
        #         print("*" * 35, "random_population DONE", "*" * 35)
    def build_model(self):
        if self.args.search_mode == "macro":
            search_space_cls = MacroSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(
                self.args.layers_of_child_model)
            # build RNN controller
            if self.args.dataset in ["cora", "citeseer", "pubmed"]:
                # implements based on dgl
                self.submodel_manager = CitationGNNManager(self.args)
            if self.args.dataset in ["Cora", "Citeseer", "Pubmed",
                                     "CS", "Physics", "Computers", "Photo"]:
                # implements based on pyg
                self.submodel_manager = GeoCitationManager(self.args)
        print("Search space:")
        print(self.search_space)
        print("Generated Action List: ")
        print(self.action_list)

    def _generate_random_individual(self):
        ind = []
        # 每个action operator 使用数字编码，对每个action　随机采样
        for action in self.action_list:
            ind.append(np.random.randint(0,
                                         len(self.search_space[action])))
        return ind

    def _construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            print('single_chromosome:', single_action)
            for action, action_name in zip(single_action, self.action_list):
                predicted_actions = self.search_space[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def form_gnn_info(self, gnn):
        if self.args.search_mode == "micro":
            actual_action = {}
            if self.args.predict_hyper:
                actual_action["action"] = gnn[:-4]
                actual_action["hyper_param"] = gnn[-4:]
            else:
                actual_action["action"] = gnn
                actual_action["hyper_param"] = [0.005, 0.8, 5e-5, 128]
            return actual_action
        return gnn

    @ray.method(num_returns=2)
    def initialize_population_Random(self):
        print("\n\n===== Random initialize the populations =====")#　随机初始化种群
        start_initial_population_time = time.time()

        epoch = 0
        while len(self.population) < self.population_size:

            once_random_initialize_start_time = time.time()
            individual = self._generate_random_individual()
            ind_actions = self._construct_action([individual])# 将基因编码解码为GNN结构空间list
            gnn = self.form_gnn_info(ind_actions[0])
            _, ind_acc = self.submodel_manager.train(gnn, format=self.args.format)
            print("individual:", individual, " val_score:", ind_acc)
            self.accuracies.append(ind_acc) # 将种群中每个个体的acc_scores存入accuraies　list
            self.population.append(individual) # 将种群中每个个体的基因存入　populations list

            once_random_initialize_end_time = time.time()

            print("the", epoch, "epcoh random initialize time: ",
                  once_random_initialize_end_time - once_random_initialize_start_time, 's')
            if epoch == 0:
                self.ev_random_initial_time.append(once_random_initialize_start_time)
                self.ev_random_initial_time.append(once_random_initialize_end_time)
                self.ev_random_acc.append(ind_acc)
            else:
                self.ev_random_initial_time.append(once_random_initialize_end_time)
                self.ev_random_acc.append(ind_acc)
            epoch += 1

        end_initial_pop_time = time.time()
        self.init_time = end_initial_pop_time - start_initial_population_time # 计算初始化过程所需时间
        print("Time elapsed initializing population: " + str(self.init_time), 's')
        print("===== Random initialize populations DONE ====")
        print("all random initialize time list: ", self.ev_random_initial_time)
        print("all random initialize population acc list: ", self.ev_random_acc)

        self.experiment_data_save(self.args.initialize_mode + "_" +
                                  self.args.dataset+"_" + self.args.evolution_sesd_name +".txt",
                                  self.ev_random_initial_time, self.ev_random_acc)
        self.population_save(self.args.dataset+"_" + self.args.evolution_sesd_name + "_population.txt",
                             self.population, self.accuracies)
        return self.population, self.accuracies

    def derive_from_population(self):

        population = self._construct_action(self.population)
        best_score_index, _ = self._get_best_individual_accuracy(self.accuracies)
        best_structure = self.form_gnn_info(population[best_score_index])

        print("[DERIVE] Best Structure:", str(best_structure))

        # train from scratch to get the final score
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        test_scores_list = []

        for i in range(10):  # run 10 times to get Mean and Stddev
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print("[DERIVE] Best Results: ", best_structure, ": ",
              np.mean(test_scores_list),
              "+/-", np.std(test_scores_list))

    def _get_best_individual_accuracy(self, accs):
        max_acc_index = 0
        max_acc = -1
        for index, acc in enumerate(accs):
            if acc > max_acc:
                max_acc = acc
                max_acc_index = index
        return max_acc_index, max_acc

    def experiment_data_save(self, name, time_list, acc_list):
        path = self.path_get()[1]
        with open(path + "/" + name, "w") as f:
            f.write(str(time_list))
            f.write("\n" + str(acc_list))
        print("the ", name, " have written")

    def population_save(self, name, population, accuracies):
        path = self.path_get()[1]
        with open(path + "/" + name, "w") as f:
            f.write(str(population))
            f.write("\n" + str(accuracies))
        print("the ", name, " have written")

    def population_read(self, name):
        path = self.path_get()[1]
        with open(path + "/" + name, "r") as f:
            all_data = f.readlines()
            population = all_data[0][:-1]
            accuracies = all_data[1]
            population = json.loads(population)
            accuracies = json.loads(accuracies)
            return population, accuracies

    def path_get(self):
        # 当前文件目录
        current_path = os.path.abspath('')
        # 当前文件夹父目录
        father_path = os.path.abspath(os.path.dirname(current_path))
        # corpus_path = os.path.join(father_path, corpus)
        return father_path, current_path

    @ray.method(num_returns=4)
    def train(self, history_population, history_validation_accuracy):

        # 传入history_population, history_validation_accuracy参数，为child fitness　做准备

        print("\n\n===== Evolution ====")
        # 从相应的目录中读取遗传种子自己的population与accuracies,进行本次进化操作
        population, accuracies = self.population_read(self.args.dataset + "_" + self.args.evolution_sesd_name +"_population.txt")

        child_acc_list = []

        print("the original populations:\n", population)
        print("the original accuracies:\n", accuracies)
        print("[POPULATION STATS] Mean/Median/Best: ",
              np.mean(accuracies),
              np.median(accuracies),
              np.max(accuracies))
        # 选择 获取parents
        parents_list = self._selection(population, accuracies)
        # 交叉 获取step2 child
        child_list = self._crossover(parents_list)
        # 变异 获取step2 child
        child_list = self._mutation(child_list)

        # 变异结束, 计算child_list中history_population中没有的gnn的acc
        # 已经存在history_population的gnn,直接从history_acc中获取
        # 计算 child_gnn中的gnn的fitness
        for child_gnn in child_list:
            if child_gnn in history_population:
                child_acc = history_validation_accuracy[history_population.index(child_gnn)]
            else:
                child_acc = self._fitness_computation([child_gnn])
            child_acc_list.append(child_acc)

        return child_list, child_acc_list, population, accuracies

    def _fitness_computation(self, child_list):
        """
        在acc计算前，判断history_population中是否包含了child gnn结构，如果包含，不再计算acc
        直接从history_val_acc中获取
        """
        for child in child_list:
            child_actions = self._construct_action([child])
            gnn = self.form_gnn_info(child_actions[0])
            _, child_acc = self.submodel_manager.train(gnn, format=self.args.format)
            return child_acc

    def _selection(self, population, accuracies):

        if self.args.selection_mode == "random":
            parent_list = []
            index_list = [index for index in range(len(population))]
            parent_index = np.random.choice(index_list, self.sample_size, replace=False)
            for index in parent_index:
                parent_list.append(population[index].copy())

        elif self.args.selection_mode == "wheel":
            print("wheel select")
            # 基于fitness计算采样概率:
            fitness = np.array(accuracies)
            fitness_probility = fitness / sum(fitness)
            fitness_probility = fitness_probility.tolist()
            index_list = [index for index in range(len(fitness))]
            parent_list = []
            # 如果self.sample_size不是偶数,需要处理
            if self.sample_size % 2 != 0:
                self.sample_size += 1
            #　基于fitness概率采样
            parent_index = np.random.choice(index_list, self.sample_size, replace=False, p=fitness_probility)
            for index in parent_index:
                parent_list.append(population[index].copy())

        elif self.args.selection_mode == "wheel_reverse":
            print("wheel select")
            # 基于fitness计算反向采样概率:
            fitness_reverse = 1 - np.array(accuracies)
            fitness_probility_reverse = fitness_reverse / sum(fitness_reverse)
            fitness_probility_reverse = fitness_probility_reverse.tolist()
            index_list = [index for index in range(len(fitness_reverse))]
            parent_list = []
            # 如果self.sample_size不是偶数,需要处理
            if self.sample_size % 2 != 0:
                self.sample_size += 1
            #　基于fitness概率采样
            parent_index = np.random.choice(index_list, self.sample_size, replace=False, p=fitness_probility_reverse)
            for index in parent_index:
                parent_list.append(population[index].copy())

        print("the parent_list:\n", parent_list)

        return parent_list

    def _crossover(self, parents):
        if self.args.crossover_mode =="single_point_crossover":
            print("crossover operator")
            child_list = []
            #单点交叉
            while parents:
                # step1:从parents list中无放回的取出一对父母
                parent_1 = parents.pop()
                parent_2 = parents.pop()
                # step2:测量parent长度,随机确定交叉点:
                crossover_point = np.random.randint(1, len(parent_1))
                # step3:对父母染色体进行交叉得到子代:
                #　交叉操作判断
                corssover_op = np.random.choice([True,False], 1, p=[self.args.crossover_p, 1-self.args.crossover_p])[0]

                if corssover_op:
                    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
                    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
                else:
                    child_1 = parent_1
                    child_2 = parent_2

                child_list.append(child_1)
                child_list.append(child_2)
        print("the child_list:\n", child_list)
        return child_list


    def _mutation(self, child_list):
        if self.args.mutation_mode == "single_point_mutation":
            print("mutation operator")
            for index in range(len(child_list)):
                # 对于index的child是否发生变异判断
                mutation_op = np.random.choice([True, False], 1, p=[self.args.mutation_p, 1 - self.args.mutation_p])[0]
                if mutation_op:
                    # 对索引号为Index的child　随机选择可能的变异点
                    position_to_mutate = np.random.randint(len(child_list[index]))
                    sp_list = self.search_space[self.action_list[position_to_mutate]]
                    child_list[index][position_to_mutate] = np.random.randint(0, len(sp_list))
        print("the child_list:\n", child_list)

        return child_list

