import torch
import numpy as np
from graphnas.gnn_model_manager import CitationGNNManager
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager
from graphnas.search_space import MacroSearchSpace
from graphnas_variants.micro_graphnas.micro_search_space import IncrementSearchSpace
from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager

class Trainer(object):
    """Manage the training process"""

    def __init__(self, args):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.build_model()  # build controller and sub-model

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

    # 基于搜索空间,对结构空间进行数字编码,并使用随机初始化获得个体组建种群
    def _generate_random_individual(self):
        ind = []
        # 每个action operator 使用数字编码，对每个action　随机采样
        for action in self.action_list:
            ind.append(np.random.randint(0,
                                         len(self.search_space[action])))
        return ind

    #　基于个体编码进行解码还原成GNN结构列表，为GNN建模做准备
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

    def derive_from_history(self):
        with open(self.args.dataset + self.args.submanager_log_file) as f:
            lines = f.readlines()
        results = []
        for line in lines:
            actions = line[:line.index(";")]
            val_score = line.split(";")[-1]
            results.append((actions, val_score))
        results.sort(key=lambda x: x[-1], reverse=True)
        best_structure = ""
        best_score = 0
        for actions in results[:5]:
            actions = eval(actions[0])
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            val_scores_list = []
            for i in range(20):
                val_acc, test_acc = self.submodel_manager.evaluate(actions)
                val_scores_list.append(val_acc)

            tmp_score = np.mean(val_scores_list)
            if tmp_score > best_score:
                best_score = tmp_score
                best_structure = actions

        print("best structure:" + str(best_structure))
        # train from scratch to get the final score
        np.random.seed(123)
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        test_scores_list = []
        for i in range(100):
            # manager.shuffle_data()
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print(f"best results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")
        return best_structure