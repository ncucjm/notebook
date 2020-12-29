import time
import torch
import argparse
import numpy as np
import graphnas.utils.tensor_utils as utils
from graphnas.evolution_trainer import Evolution_Trainer
import ray

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
        self.mutation_mode = method["mutation_mode"]
        self.mutation_p = method["mutation_p"]
        self.updating_mode = method["updating_mode"]
        self.cycles = 1
        self.population_size = 5
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

class Nas():
    def __init__(self, args):
        if args.cuda and not torch.cuda.is_available():
            args.cuda = False

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        np.set_printoptions(precision=8)

        if args.cuda:
            torch.cuda.manual_seed(args.random_seed)
        utils.makedirs(args.dataset)
        trainer = Evolution_Trainer(args)
        trainer.train()

# none_age
# method = {"selection_mode": "wheel",
#           "crossover_mode": "point",
#           "mutation_mode": "point_p",
#           "mutation_p": 0.2,
#           "updating_mode": "none_age"}
# age
# method = {"selection_mode": "random",
#             "crossover_mode": "none",
#             "mutation_mode": "point_none",
#             "mutation_p": 0,
#             "updating_mode": "age"}

@ray.remote(num_gpus=0.2)
def task_ev_none_age():
    print("ev_none_age")
    method = {"evolution_seed_name": "wheel_1_0.2_noneage",
             "initialize_mode": "random",
              "selection_mode": "wheel",
              "crossover_mode": "point",
              "mutation_mode": "point_p",
              "mutation_p": 0.2,
              "updating_mode": "none_age",
              "dataset": "Citeseer"}
    print("method:\n", method)
    args = Parameter(method)
    Nas(args)

@ray.remote(num_gpus=0.3)
def task_ev_none_age_1():
    print("ev_none_age")
    method = {"evolution_seed_name": "wheel_1_0.8_noneage",
             "initialize_mode": "random",
              "selection_mode": "wheel",
              "crossover_mode": "point",
              "mutation_mode": "point_p",
              "mutation_p": 0.8,
              "updating_mode": "none_age",
              "dataset": "Citeseer"}
    print("method:\n", method)
    args = Parameter(method)
    Nas(args)

@ray.remote(num_gpus=0.5)
def task_ev_age():
    print("ev_age")
    method = {"evolution_seed_name": "random_none_none_age",
              "initialize_mode": "random",
              "selection_mode": "random",
              "crossover_mode": "none",
              "mutation_mode": "point_none",
              "mutation_p": 0,
              "updating_mode": "age",
              "dataset": "Citeseer"}
    print("method:\n", method)
    args = Parameter(method)
    Nas(args)

task_1 = task_ev_none_age.remote()
task_2 = task_ev_none_age_1.remote()
task_3 = task_ev_age.remote()


task_list = [task_1, task_2, task_3]

ray.get(task_list)


