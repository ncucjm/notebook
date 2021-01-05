import time
import torch
import numpy as np
import graphnas.utils.tensor_utils as utils
from graphnas.evolution_trainer import Evolution_Trainer
import ray
import os
ray.shutdown()
#ray　初始化
ray.init()

class Parameter():
    def __init__(self, method):

        self.random_seed = 123
        self.cuda = True

        self.evolution_sesd_name = method["evolution_seed_name"]
        self.initialize_mode = method["initialize_mode"]
        self.selection_mode = method["selection_mode"]
        self.crossover_mode = method["crossover_mode"]
        self.crossover_p = method["crossover_p"]
        self.mutation_mode = method["mutation_mode"]
        self.mutation_p = method["mutation_p"]
        self.updating_mode = method["updating_mode"]
        # 进化轮次
        self.cycles = 1
        #　初始化种群数量
        self.population_size = 3
        # 选取parent规模
        self.sample_size = 2

        self.layers_of_child_model = 2
        self.search_mode = "macro"
        self.format = "two"
        self.dataset = method["dataset"]
        self.epochs = 300
        self.retrain_epochs = 300
        self.multi_label = False
        self.in_drop = 0.6
        self.lr = 0.005
        self.param_file = "core_test.pkl"
        self.weight_decay = 5e-4
        self.max_param = 5E6
        self.supervised = False
        self.submanager_log_file = f"sub_manager_logger_file_{time.time()}.txt"

        #历史信息
        self.history_popution = []
        self.history_val_acc = []
        self.common_child = []



# @ray.remote(num_gpus=0.5)
# def task_ev_none_age_05(dataset):
#     print("ev_none_age")
#     method = {"evolution_seed_name": "wheel_1_0.8_noneage",
#              "initialize_mode": "random",
#               "selection_mode": "wheel",
#               "crossover_mode": "point",
#               "mutation_mode": "point_p",
#               "mutation_p": 0.8,
#               "updating_mode": "none_age",
#               "dataset": "Citeseer"}
#     print("method:\n", method)
#     method["dataset"] = dataset
#     args = Parameter(method)
#     Nas(args)

def experiment_data_save(name, time_list, acc_list):
    path = path_get()[1]
    with open(path + "/" + name, "w") as f:
        f.write(str(time_list))
        f.write("\n" + str(acc_list))
    print("the ", name, " have written")

def path_get():
    # 当前文件目录
    current_path = os.path.abspath('')
    # 当前文件夹父目录
    father_path = os.path.abspath(os.path.dirname(current_path))
    # corpus_path = os.path.join(father_path, corpus)
    return father_path, current_path

def updating(args, common_child, common_child_acc, seed_population, seed_acc):
    print(args.updating_mode, " updating operating start")
    # 随机无放回的在common_child中选择child_gnn放入seed_population中,并更新相关acc
    if args.updating_mode == "random":
        index_list = [index for index in range(len(common_child))]
        child_index = np.random.choice(index_list, args.sample_size, replace=False)
        for index in child_index:
            seed_population.pop()
            seed_acc.pop()
            seed_population.append(common_child[index])
            seed_acc.append(common_child_acc[index])

    elif args.updating_mode =="wheel":
        temp_child_gnn = []
        temp_child_acc = []

        fitness = np.array(common_child_acc)
        fitness_probility = fitness / sum(fitness)
        fitness_probility = fitness_probility.tolist()
        index_list = [index for index in range(len(fitness))]
        # 　基于fitness概率采样
        child_index = np.random.choice(index_list, args.sample_size, replace=False, p=fitness_probility)

        for index in child_index:
            temp_child_gnn.append(common_child[index])
            temp_child_acc.append(common_child_acc[index])
        # 去掉population中不好的样本
        for acc in temp_child_acc:
            if acc > min(seed_acc):
                min_index = seed_acc.index(min(seed_acc))
                seed_acc[min_index] = acc
                acc_index = temp_child_acc.index(acc)
                seed_population[min_index] = temp_child_gnn[acc_index]

    elif args.updating_mode =="wheel_reverse":
        temp_child_gnn = []
        temp_child_acc = []

        fitness_reverse = 1 - np.array(common_child_acc)
        fitness_probility_reverse = fitness_reverse / sum(fitness_reverse)
        fitness_probility_reverse = fitness_probility_reverse.tolist()
        index_list = [index for index in range(len(fitness_reverse))]
        # 　基于fitness反向概率采样
        child_index = np.random.choice(index_list, args.sample_size, replace=False, p=fitness_probility_reverse)

        for index in child_index:
            temp_child_gnn.append(common_child[index])
            temp_child_acc.append(common_child_acc[index])

        #随机更新population
        index_list = [index for index in range(len(common_child))]
        child_index = np.random.choice(index_list, args.sample_size, replace=False)
        for index in child_index:
            seed_population.pop()
            seed_acc.pop()
            seed_population.append(common_child[index])
            seed_acc.append(common_child_acc[index])
    print("update over")
    return seed_population, seed_acc

def population_save(name, population, accuracies):
    path = path_get()[1]
    with open(path + "/" + name, "w") as f:
        f.write(str(population))
        f.write("\n" + str(accuracies))
        print("the ", name, " have written")

if __name__=="__main__":

    dataset = "Pubmed"
    history_population = []
    history_validation_accuracy = []
    common_child = []
    common_child_acc = []
    ev_train_time = []
    ev_acc = []
    ev_acc1 = []
    ev_acc2 = []
    ev_acc3 = []

    method1 = {"evolution_seed_name": "wheel_0.2_0.2_wheel",
              "initialize_mode": "random",
              "selection_mode": "wheel",
              "crossover_mode": "single_point_crossover",
              "crossover_p": 0.2,
              "mutation_mode": "single_point_mutation",
              "mutation_p": 0.2,
              "updating_mode": "wheel",
              "dataset": "Citeseer"}
    method1["dataset"] = dataset
    print("method:\n", method1)


    method2 = {"evolution_seed_name": "random_0.5_0.5_random",
              "initialize_mode": "random",
              "selection_mode": "random",
              "crossover_mode": "single_point_crossover",
              "crossover_p": 0.5,
              "mutation_mode": "single_point_mutation",
              "mutation_p": 0.5,
              "updating_mode": "random",
              "dataset": "Citeseer"}
    method2["dataset"] = dataset
    print("method:\n", method2)

    method3 = {"evolution_seed_name": "wheel_reverse_0.8_0.8_wheel_reverse",
               "initialize_mode": "random",
               "selection_mode": "wheel_reverse",
               "crossover_mode": "single_point_crossover",
               "crossover_p": 0.8,
               "mutation_mode": "single_point_mutation",
               "mutation_p": 0.8,
               "updating_mode": "wheel_reverse",
               "dataset": "Citeseer"}
    method3["dataset"] = dataset
    print("method:\n", method2)

    args1 = Parameter(method1)
    args2 = Parameter(method2)
    args3 = Parameter(method3)

    if args1.cuda and not torch.cuda.is_available():
        args1.cuda = False

    torch.manual_seed(args1.random_seed)
    np.random.seed(args1.random_seed)
    np.set_printoptions(precision=8)

    if args1.cuda:
        torch.cuda.manual_seed(args1.random_seed)

    utils.makedirs(args1.dataset)

    trainer1 = Evolution_Trainer.remote(args1)
    trainer2 = Evolution_Trainer.remote(args2)
    trainer3 = Evolution_Trainer.remote(args3)

    initialize_pop1, initialize_acc1 = trainer1.initialize_population_Random.remote()
    initialize_pop2, initialize_acc2 = trainer2.initialize_population_Random.remote()
    initialize_pop3, initialize_acc3 = trainer3.initialize_population_Random.remote()

    task1 = [initialize_pop1, initialize_acc1]
    task2 = [initialize_pop2, initialize_acc2]
    task3 = [initialize_pop3, initialize_acc3]

    #初始化种子
    start_initial_population_time = time.time()

    pop1, acc1 = ray.get(task1)
    pop2, acc2 = ray.get(task2)
    pop3, acc3 = ray.get(task3)

    end_initial_pop_time = time.time()
    init_time = end_initial_pop_time - start_initial_population_time


    history_population = pop1 + pop2 + pop3
    history_validation_accuracy = acc1 + acc2 + acc3

    #初始化完毕开始训练
    #main.py中更新操作后，需要保存每一种子的population到相应的文件中
    print("\n\n===== Evolution =====")
    start_evolution_time = time.time()

    for i in range(2):
        print("第 ", i, " 轮训练")
        once_evolution_train_start_time = time.time()
        print("the history populations:\n", history_population)
        print("the history accuracies:\n", history_validation_accuracy)
        print("[HISTORY POPULATION STATS] Mean/Median/Best: ",
              np.mean(history_validation_accuracy),
              np.median(history_validation_accuracy),
              np.max(history_validation_accuracy))
        # 传入history_pop,history_val,为计算child fitness做准备
        child_list1, child_acc_list1, population1, accuracy1 = trainer1.train.remote(history_population, history_validation_accuracy)
        child_list2, child_acc_list2, population2, accuracy2 = trainer2.train.remote(history_population, history_validation_accuracy)
        child_list3, child_acc_list3, population3, accuracy3 = trainer3.train.remote(history_population, history_validation_accuracy)


        task1 = [child_list1, child_acc_list1, population1, accuracy1]
        task2 = [child_list2, child_acc_list2, population2, accuracy2]
        task3 = [child_list3, child_acc_list3, population3, accuracy3]


        child_gnn_list1, child_val_acc_list1, _population1, _accuracy1 = ray.get(task1)
        child_gnn_list2, child_val_acc_list2, _population2, _accuracy2 = ray.get(task2)
        child_gnn_list3, child_val_acc_list3, _population3, _accuracy3 = ray.get(task3)

        # 组成 common　child
        common_child = child_gnn_list1 + child_gnn_list2 + child_gnn_list3
        common_child_acc = child_val_acc_list1 + child_val_acc_list2 + child_val_acc_list3

        #基于common_child,common_child_acc更新每个seed的population

        updated_population1, updated_accuracy1 = updating(args1, common_child, common_child_acc, _population1, _accuracy1)
        updated_population2, updated_accuracy2 = updating(args2, common_child, common_child_acc, _population2, _accuracy2)
        updated_population3, updated_accuracy3 = updating(args3, common_child, common_child_acc, _population3, _accuracy3)


        once_evolution_train_end_time = time.time()

        print("the", i, "epcoh evlution train time: ",
              once_evolution_train_end_time - once_evolution_train_start_time, 's')


        # 基于gnn结构是否在history中出现过更新history_population与history_acc获取best_acc前更新history_validation_accuracy

        updated_population = updated_population1 + updated_population2 + updated_population3
        updated_accuracy = updated_accuracy1 + updated_accuracy2 + updated_accuracy3
        for index in range(len(updated_population)):
            gnn = updated_population[index]
            if gnn not in history_population:
                history_population.append(gnn)
                history_validation_accuracy.append(updated_accuracy[index])

        best_acc = max(history_validation_accuracy)
        best_acc1 = max(updated_accuracy1)
        best_acc2 = max(updated_accuracy2)
        best_acc3 = max(updated_accuracy3)


        # 保存每个遗传种子种群与准确度信息
        population_save(args1.dataset + "_" + args1.evolution_sesd_name + "_population.txt",
                             updated_population1, updated_accuracy1)
        population_save(args2.dataset + "_" + args2.evolution_sesd_name + "_population.txt",
                             updated_population2, updated_accuracy2)
        population_save(args3.dataset + "_" + args3.evolution_sesd_name + "_population.txt",
                        updated_population3, updated_accuracy3)

        if i == 0:
            ev_train_time.append(once_evolution_train_start_time)
            ev_train_time.append(once_evolution_train_end_time)
            ev_acc1.append(best_acc1)
            ev_acc2.append(best_acc2)
            ev_acc3.append(best_acc3)
            ev_acc.append(best_acc)
        else:
            ev_train_time.append(once_evolution_train_end_time)
            ev_acc1.append(best_acc1)
            ev_acc2.append(best_acc2)
            ev_acc3.append(best_acc3)
            ev_acc.append(best_acc)

    print("all evalution train time list: ", ev_train_time)
    print("all best population acc list: ", ev_acc)

    experiment_data_save(args1.dataset + "_" + args1.evolution_sesd_name + ".txt", ev_train_time, ev_acc1)
    experiment_data_save(args2.dataset + "_" + args2.evolution_sesd_name + ".txt", ev_train_time, ev_acc2)
    experiment_data_save(args3.dataset + "_" + args3.evolution_sesd_name + ".txt", ev_train_time, ev_acc3)

    experiment_data_save(args1.dataset + "_" + "ensemble_evolution" + ".txt", ev_train_time, ev_acc)


    end_evolution_time = time.time()
    total_evolution_time = end_evolution_time - start_evolution_time

    print('Time spent on evolution: ' + str(total_evolution_time))
    print('Total time: ' + str(total_evolution_time + init_time))
    print("===== Evolution DONE ====")



# if __name__=="__main__":
#
#     for dataset in ["Citeseer", "Cora", "Pubmed"]:
#
#         history_population = []
#         history_validation_accuracy = []
#         common_child = []
#         common_child_acc = []
#         ev_train_time = []
#         ev_acc = []
#         ev_acc1 = []
#         ev_acc2 = []
#         ev_acc3 = []
#
#         method1 = {"evolution_seed_name": "wheel_0.2_0.2_wheel",
#                   "initialize_mode": "random",
#                   "selection_mode": "wheel",
#                   "crossover_mode": "single_point_crossover",
#                   "crossover_p": 0.2,
#                   "mutation_mode": "single_point_mutation",
#                   "mutation_p": 0.2,
#                   "updating_mode": "wheel",
#                   "dataset": "Citeseer"}
#         method1["dataset"] = dataset
#         print("method:\n", method1)
#
#
#         method2 = {"evolution_seed_name": "random_0.5_0.5_random",
#                   "initialize_mode": "random",
#                   "selection_mode": "random",
#                   "crossover_mode": "single_point_crossover",
#                   "crossover_p": 0.5,
#                   "mutation_mode": "single_point_mutation",
#                   "mutation_p": 0.5,
#                   "updating_mode": "random",
#                   "dataset": "Citeseer"}
#         method2["dataset"] = dataset
#         print("method:\n", method2)
#
#         method3 = {"evolution_seed_name": "wheel_reverse_0.8_0.8_wheel_reverse",
#                    "initialize_mode": "random",
#                    "selection_mode": "wheel_reverse",
#                    "crossover_mode": "single_point_crossover",
#                    "crossover_p": 0.8,
#                    "mutation_mode": "single_point_mutation",
#                    "mutation_p": 0.8,
#                    "updating_mode": "wheel_reverse",
#                    "dataset": "Citeseer"}
#         method3["dataset"] = dataset
#         print("method:\n", method2)
#
#         args1 = Parameter(method1)
#         args2 = Parameter(method2)
#         args3 = Parameter(method3)
#
#         if args1.cuda and not torch.cuda.is_available():
#             args1.cuda = False
#
#         torch.manual_seed(args1.random_seed)
#         np.random.seed(args1.random_seed)
#         np.set_printoptions(precision=8)
#
#         if args1.cuda:
#             torch.cuda.manual_seed(args1.random_seed)
#
#         utils.makedirs(args1.dataset)
#
#         trainer1 = Evolution_Trainer.remote(args1)
#         trainer2 = Evolution_Trainer.remote(args2)
#         trainer3 = Evolution_Trainer.remote(args3)
#
#         initialize_pop1, initialize_acc1 = trainer1.initialize_population_Random.remote()
#         initialize_pop2, initialize_acc2 = trainer2.initialize_population_Random.remote()
#         initialize_pop3, initialize_acc3 = trainer3.initialize_population_Random.remote()
#
#         task1 = [initialize_pop1, initialize_acc1]
#         task2 = [initialize_pop2, initialize_acc2]
#         task3 = [initialize_pop3, initialize_acc3]
#
#         #初始化种子
#         start_initial_population_time = time.time()
#
#         pop1, acc1 = ray.get(task1)
#         pop2, acc2 = ray.get(task2)
#         pop3, acc3 = ray.get(task3)
#
#         end_initial_pop_time = time.time()
#         init_time = end_initial_pop_time - start_initial_population_time
#
#
#         history_population = pop1 + pop2 + pop3
#         history_validation_accuracy = acc1 + acc2 + acc3
#
#         #初始化完毕开始训练
#         #main.py中更新操作后，需要保存每一种子的population到相应的文件中
#         print("\n\n===== Evolution =====")
#         start_evolution_time = time.time()
#
#         for i in range(2):
#             print("第 ", i, " 轮训练")
#             once_evolution_train_start_time = time.time()
#             print("the history populations:\n", history_population)
#             print("the history accuracies:\n", history_validation_accuracy)
#             print("[HISTORY POPULATION STATS] Mean/Median/Best: ",
#                   np.mean(history_validation_accuracy),
#                   np.median(history_validation_accuracy),
#                   np.max(history_validation_accuracy))
#
#             child_list1, child_acc_list1, population1, accuracy1 = trainer1.train.remote(history_population, history_validation_accuracy)
#             child_list2, child_acc_list2, population2, accuracy2 = trainer2.train.remote(history_population, history_validation_accuracy)
#             child_list3, child_acc_list3, population3, accuracy3 = trainer3.train.remote(history_population, history_validation_accuracy)
#
#
#             task1 = [child_list1, child_acc_list1, population1, accuracy1]
#             task2 = [child_list2, child_acc_list2, population2, accuracy2]
#             task3 = [child_list3, child_acc_list3, population3, accuracy3]
#
#
#             child_gnn_list1, child_val_acc_list1, _population1, _accuracy1 = ray.get(task1)
#             child_gnn_list2, child_val_acc_list2, _population2, _accuracy2 = ray.get(task2)
#             child_gnn_list3, child_val_acc_list3, _population3, _accuracy3 = ray.get(task3)
#
#
#             common_child = child_gnn_list1 + child_gnn_list2 + child_gnn_list3
#             common_child_acc = child_val_acc_list1 + child_val_acc_list2 + child_val_acc_list3
#
#             #基于common_child,common_child_acc更新每个seed的population
#
#             updated_population1, updated_accuracy1 = updating(args1, common_child, common_child_acc, _population1, _accuracy1)
#             updated_population2, updated_accuracy2 = updating(args2, common_child, common_child_acc, _population2, _accuracy2)
#             updated_population3, updated_accuracy3 = updating(args3, common_child, common_child_acc, _population3, _accuracy3)
#
#
#             once_evolution_train_end_time = time.time()
#
#             print("the", i, "epcoh evlution train time: ",
#                   once_evolution_train_end_time - once_evolution_train_start_time, 's')
#
#
#             # 基于gnn结构是否在history总出现过更新history_population与history_acc获取best_acc前更新history_validation_accuracy
#
#             updated_population = updated_population1 + updated_population2 + updated_population3
#             updated_accuracy = updated_accuracy1 + updated_accuracy2 + updated_accuracy3
#             for index in range(len(updated_population)):
#                 gnn = updated_population[index]
#                 if gnn not in history_population:
#                     history_population.append(gnn)
#                     history_validation_accuracy.append(updated_accuracy[index])
#
#             best_acc = max(history_validation_accuracy)
#             best_acc1 = max(updated_accuracy1)
#             best_acc2 = max(updated_accuracy2)
#             best_acc3 = max(updated_accuracy3)
#
#
#             # 保存update1_pop...等信息
#             population_save(args1.dataset + "_" + args1.evolution_sesd_name + "_population.txt",
#                                  updated_population1, updated_accuracy1)
#             population_save(args2.dataset + "_" + args2.evolution_sesd_name + "_population.txt",
#                                  updated_population2, updated_accuracy2)
#             population_save(args3.dataset + "_" + args3.evolution_sesd_name + "_population.txt",
#                             updated_population3, updated_accuracy3)
#
#             if i == 0:
#                 ev_train_time.append(once_evolution_train_start_time)
#                 ev_train_time.append(once_evolution_train_end_time)
#                 ev_acc1.append(best_acc1)
#                 ev_acc2.append(best_acc2)
#                 ev_acc3.append(best_acc3)
#                 ev_acc.append(best_acc)
#             else:
#                 ev_train_time.append(once_evolution_train_end_time)
#                 ev_acc1.append(best_acc1)
#                 ev_acc2.append(best_acc2)
#                 ev_acc3.append(best_acc3)
#                 ev_acc.append(best_acc)
#
#         print("all evalution train time list: ", ev_train_time)
#         print("all best population acc list: ", ev_acc)
#
#         experiment_data_save(args1.dataset + "_" + args1.evolution_sesd_name + ".txt", ev_train_time, ev_acc1)
#         experiment_data_save(args2.dataset + "_" + args2.evolution_sesd_name + ".txt", ev_train_time, ev_acc2)
#         experiment_data_save(args3.dataset + "_" + args3.evolution_sesd_name + ".txt", ev_train_time, ev_acc3)
#
#         experiment_data_save(args1.dataset + "_" + "ensemble_evolution" + ".txt", ev_train_time, ev_acc)
#
#
#         end_evolution_time = time.time()
#         total_evolution_time = end_evolution_time - start_evolution_time
#         print('Time spent on evolution: ' + str(total_evolution_time))
#         print('Total time: ' + str(total_evolution_time + init_time))
#         print("===== Evolution DONE ====")








