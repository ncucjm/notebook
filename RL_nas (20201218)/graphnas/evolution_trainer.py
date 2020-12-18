import time
import torch
import numpy as np
from collections import deque
from graphnas.trainer import Trainer
from graphnas.rl_trainer import RL_Trainer

class Evolution_Trainer(Trainer,RL_Trainer):
    """
    This class implements the Asyncronous Aging Evolution,
    proposed by Real et. al. on:

    Regularized Evolution for Image Classifier Architecture Search

    available on: https://arxiv.org/abs/1802.01548
    """
    def __init__(self, args):
        #　继承了Trainer类，调用Trainer中的构造方法
        super(Evolution_Trainer, self).__init__(args)
        self.cycle = 0
        self.args = args
        self.random_seed = args.random_seed
        self.population = deque()
        self.accuracies = deque()
        self.population_size = args.population_size
        self.sample_size = args.sample_size
        self.cycles = args.cycles
        self.init_time = 0
        print('initializing population on evolution_trainer init, maybe not the best strategy')
        #　random初始化种群
        #self.__initialize_population_Random()
        # rl初始化种群
        self.__initialize_population_RL()

    def __initialize_population＿RL(self):
        print("\n\n=====  Reinforcement Learning initialize the populations =====")  # 随机初始化种群
        start_initial_population_time = time.time()
        trainer = RL_Trainer(self.args)
        self.population, self.accuracies = trainer.train(self.action_list)
        end_initial_pop_time = time.time()
        self.init_time = end_initial_pop_time - start_initial_population_time  # 计算初始化过程所需时间
        print("Time elapsed initializing population: " + str(self.init_time))
        print("===== Reinforcement Learning initialize populations DONE ====")


    def __initialize_population_Random(self):
        print("\n\n===== Random initialize the populations =====")#　随机初始化种群
        start_initial_population_time = time.time()
        while len(self.population) < self.population_size:
            individual = self._generate_random_individual()
            ind_actions = self._construct_action([individual])# 将基因编码解码为GNN结构空间list
            gnn = self.form_gnn_info(ind_actions[0])
            _, ind_acc = self.submodel_manager.train(gnn, format=self.args.format)
            print("individual:", individual, " val_score:", ind_acc)
            self.accuracies.append(ind_acc) # 将种群中每个个体的acc_scores存入accuraies　list
            self.population.append(individual) # 将种群中每个个体的基因存入　populations list
        end_initial_pop_time = time.time()
        self.init_time = end_initial_pop_time - start_initial_population_time # 计算初始化过程所需时间
        print("Time elapsed initializing population: " + str(self.init_time))
        print("===== Random initialize populations DONE ====")

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

    def train(self):
        print("\n\n===== Evolution ====")
        start_evolution_time = time.time()
        print("the original populations:\n", self.population)
        print("the original accuracies:\n", self.accuracies)
        print("[POPULATION STATS] Mean/Median/Best: ",
              np.mean(self.accuracies),
              np.median(self.accuracies),
              np.max(self.accuracies))
        while self.cycles > 0:
            # 选择
            parents_list = self._selection()
            # 交叉
            child_list = self._crossover(parents_list)
            # 变异
            child_list = self._mutation(child_list)
            #　更新
            self._updating(child_list)
            self.cycles = self.cycles - 1

        end_evolution_time = time.time()
        total_evolution_time = end_evolution_time - start_evolution_time
        print('Time spent on evolution: ' +
              str(total_evolution_time))
        print('Total elapsed time: ' +
              str(total_evolution_time + self.init_time))
        print("===== Evolution DONE ====")


    def _selection(self):

        if self.args.selection_mode == "random":
            print("random select")
            sample = []  # 采样染色体列表
            sample_accs = []  # 采样染色体fitness列表
            # 随机选取候选集
            while len(sample) < self.sample_size:# self.sample_size:从population中的采样规模
                candidate = np.random.randint(0, len(self.population))  # 基于种群空间大小,　随机从种群中挑选候选一个index
                sample.append(self.population[candidate])  # 基于index获取基因编码
                sample_accs.append(self.accuracies[candidate])  # 基于index获取基因对应的　acc_score
                # 从候选集中基于fitness高低选取parents染色体
            max_sample_acc_index, max_sample_acc = self._get_best_individual_accuracy(sample_accs)
                # 确定最佳acc_score对应的index与acc_score
            parent_list = sample[max_sample_acc_index].copy()# 从sample list 中获取parent基因编码

        elif self.args.selection_mode == "wheel":
            print("wheel select")
            # 基于fitness计算采样概率:
            fitness = np.array(self.accuracies)
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
                parent_list.append(self.population[index].copy())
        print("the parent_list:\n", parent_list)
        return parent_list

    def _crossover(self, parents):
        if self.args.crossover_mode =="point":
            print("point crossover")
            child_list = []
            #单点交叉
            while parents:
                # step1:从parents list中无放回的取出一对父母
                parent_1 = parents.pop()
                parent_2 = parents.pop()
                # step2:测量parent长度,随机确定交叉点:
                cross_point = np.random.randint(1, len(parent_1))
                # step3:对父母染色体进行交叉得到子代:
                child_1 = parent_1[:cross_point] + parent_2[cross_point:]
                child_2 = parent_2[:cross_point] + parent_1[cross_point:]
                child_list.append(child_1)
                child_list.append(child_2)
        elif self.args.crossover_mode == "none":
            print("none crossover")
            child_list = [parents]
        print("the child_list:\n", child_list)
        return child_list

    def _mutation(self, child_list):
        if self.args.mutation_mode == "point_none":
            print("point_none mutation")
            for indiv in child_list:
                # 每次变异随机选择一个位置，这个位置在search　space中随机变异一个数字
                # Choose a random position on the individual to mutate
                position_to_mutate = np.random.randint(len(indiv))
                # This position will receive a randomly chosen index　of the search_spaces's list
                # for the action corresponding to that position in the individual
                sp_list = self.search_space[self.action_list[position_to_mutate]]
                indiv[position_to_mutate] = np.random.randint(0, len(sp_list))
                child_list = [indiv]

        elif self.args.mutation_mode == "point_p":
            print("point_p mutation")
            for index in range(len(child_list)):
                # 对于index的child是否发生变异判断
                mutation_op = np.random.choice([True, False], 1, p=[self.args.mutation_p, 1-self.args.mutation_p])[0]
                if mutation_op:
                    # 对索引号为Index的child　随机选择可能的变异点
                    position_to_mutate = np.random.randint(len(child_list[index]))
                    sp_list = self.search_space[self.action_list[position_to_mutate]]
                    child_list[index][position_to_mutate] = np.random.randint(0, len(sp_list))
        print("the child_list:\n", child_list)
        return child_list

    def _updating(self, child_list):
        if self.args.updating_mode == "age":
            print("age updating")
            for child in child_list:
                child_actions = self._construct_action([child])
                gnn = self.form_gnn_info(child_actions[0])
                _, child_acc = self.submodel_manager.train(gnn, format=self.args.format)#child_acc正确
                # 将child与acc_score加入种群population,accuracies list　
                self.accuracies.append(child_acc)
                self.population.append(child)
                # 　cycles结束，从population中挑选最优个体
                if self.cycles % self.args.eval_cycle == 0:
                    self.derive_from_population()
                # Remove oldest individual (Aging/Regularized evolution)
                self.population.popleft()  # 从population　list　中剔除最左边的个体基因
                self.accuracies.popleft()  # 同时剔除其acc_score 准确度
                print("cycle: ", self.cycle, "populations:\n", self.population)
                print("cycle: ", self.cycle, "accuracies:\n", self.accuracies)
                print("[POPULATION STATS] Mean/Median/Best: ",
                      np.mean(self.accuracies),
                      np.median(self.accuracies),
                      np.max(self.accuracies))
            self.cycle += 1
        elif self.args.updating_mode == "none-age":
            print("none-age updating")
            # 选择child_list中比population中fitnesss高的更新
            # 计算child_list中fitness
            for child in child_list:
                child_actions = self._construct_action([child])
                gnn = self.form_gnn_info(child_actions[0])
                _, child_acc = self.submodel_manager.train(gnn, format=self.args.format)

                if child_acc > min(self.accuracies):
                    min_index = self.accuracies.index(min(self.accuracies))
                    self.accuracies[min_index] = child_acc
                    self.population[min_index] = child


            print("cycle: ", self.cycle, "populations:\n", self.population)
            print("cycle: ", self.cycle, "accuracies:\n", self.accuracies)
            # 每训练self.eval_cycle次，从population中挑选最优个体
            if self.cycles % self.args.eval_cycle == 0:
                self.derive_from_population()

            print("[POPULATION STATS] Mean/Median/Best: ",
                  np.mean(self.accuracies),
                  np.median(self.accuracies),
                  np.max(self.accuracies))
            self.cycle += 1

    def derive(self, sample_num=None):
        self.derive_from_population()
