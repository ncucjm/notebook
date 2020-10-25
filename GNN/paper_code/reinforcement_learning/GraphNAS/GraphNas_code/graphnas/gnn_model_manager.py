import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data

from graphnas.gnn import GraphNet
from graphnas.utils.model_utils import EarlyStop, TopAverage, process_action


def load(args, save_file=".npy"):
    '''    '''
    save_file = args.dataset + save_file
    if os.path.exists(save_file):
        return np.load(save_file).tolist()
    else:
        datas = load_data(args)
        np.save(save_file, datas)
        return datas


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


# manager the train process of GNN on citation dataset
class CitationGNNManager(object):

    def __init__(self, args):

        self.args = args

        if hasattr(args, 'dataset') and args.dataset in ["cora", "citeseer", "pubmed"]:
            self.data = load(args)
            self.args.in_feats = self.in_feats = self.data.features.shape[1]
            self.args.num_class = self.n_classes = self.data.num_labels


        self.early_stop_manager = EarlyStop(10)

        self.reward_manager = TopAverage(10)
        """
class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        # print("Top %d average: %f" % (self.top_k, avg))
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)
        """
        self.args = args
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.retrain_epochs = args.retrain_epochs
        self.loss_fn = torch.nn.BCELoss()
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10

        self.param_file = args.param_file
        self.shared_params = None

        self.loss_fn = torch.nn.functional.nll_loss

    def load_param(self):
        # don't share param
        pass

    def save_param(self, model, update_all=False):
        # don't share param
        pass

    # train from scratch
    def evaluate(self, actions=None, format="two"):
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        if self.args.cuda:
            model.cuda()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        try:
            model, val_acc, test_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs,
                                                      cuda=self.args.cuda, return_best=True,
                                                      half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
            """
class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0
            """
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                test_acc = 0
            else:
                raise e
        return val_acc, test_acc

    # train from scratch
    def train(self, actions=None, format="two"):
        # actions = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 4]
        # format="two"
        origin_action = actions

        # 分类任务类别数修改第二层GNN模型的输出维度
        # 修改为任务类别的维度数,Citeseer数据集的类别为6,则第二层GNN最后输出维度为6
        actions = process_action(actions, format, self.args)
        # actions = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 6]

        print("train action:", actions)

        # create model
        # 基于选择出的GNN结构构建GNN
        model = self.build_gnn(actions)

        try:
            if self.args.cuda:
                model.cuda()
            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

            model, val_acc = self.run_model(model, optimizer, self.loss_fn, self.data, self.epochs, cuda=self.args.cuda,
                                            half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
            """
class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0
            """

        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e

        reward = self.reward_manager.get_reward(val_acc)

        # 当val_acc大于历史top10的val_acc的均值,则reward为正
        # reward被限制在-0.5到0.5之间
        """
class TopAverage(object):
    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        # print("Top %d average: %f" % (self.top_k, avg))
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)
        """
        self.save_param(model, update_all=(reward > 0))
        # 模型没有共享参数

        self.record_action_info(origin_action, reward, val_acc)
        # 将gnn结构,reward,val_acc信息记录到log文件中

        return reward, val_acc

    def record_action_info(self, origin_action, reward, val_acc):
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "a") as file:
            # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
            file.write(str(origin_action))

            file.write(";")
            file.write(str(reward))

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False)
        # actions = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 6]
        # self.in_feats = 3703
        # self.n_classes = 6
        # drop_out = 0.6

        return model

    # 验证模型
    def retrain(self, actions, format="two"):
        return self.train(actions, format)

    def test_with_param(self, actions=None, format="two", with_retrain=False):
        # actions = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 4]
        # format = "two"
        # with_retrain = True
        return self.train(actions, format)

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=False):

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf") # 正无穷
        min_train_loss = float("inf") # 正无穷
        model_val_acc = 0
        features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNNManager.prepare_data(data, cuda)

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()

            # forward
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # evaluate
            """
def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()
            """
            model.eval()
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, labels, mask)
            dur.append(time.time() - t0)

            val_loss = float(loss_fn(logits[val_mask], labels[val_mask]))

            val_acc = evaluate(logits, labels, val_mask)
            test_acc = evaluate(logits, labels, test_mask)

            # 当训练中本次val_loss 低于历史最低值,才会更新 model_val_acc,min_train_loss
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc
            if show_info:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

                end_time = time.time()
                print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc

    # @staticmethod
    # def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="citation_testing_2.pkl",
    #               half_stop_score=0, return_best=False, cuda=True, need_early_stop=False):
    #
    #     early_stop_manager = EarlyStop(early_stop)
    #     # initialize graph
    #     dur = []
    #     begin_time = time.time()
    #     features, g, labels, mask, val_mask, test_mask, n_edges = CitationGNNManager.prepare_data(data, cuda)
    #     saved = False
    #     best_performance = 0
    #     for epoch in range(1, epochs + 1):
    #         should_break = False
    #         t0 = time.time()
    #
    #         model.train()
    #         logits = model(features, g)
    #         logits = F.log_softmax(logits, 1)
    #         loss = loss_fn(logits[mask], labels[mask])
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         model.eval()
    #         logits = model(features, g)
    #         logits = F.log_softmax(logits, 1)
    #         train_acc = evaluate(logits, labels, mask)
    #         train_loss = float(loss)
    #         dur.append(time.time() - t0)
    #
    #         val_loss = float(loss_fn(logits[val_mask], labels[val_mask]))
    #         val_acc = evaluate(logits, labels, val_mask)
    #         test_acc = evaluate(logits, labels, test_mask)
    #
    #         print(
    #             "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
    #                 epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))
    #
    #         end_time = time.time()
    #         print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
    #         # print("Test Accuracy {:.4f}".format(acc))
    #         if early_stop_manager.should_save(train_loss, train_acc, val_loss, val_acc):
    #             saved = True
    #             torch.save(model.state_dict(), tmp_model_file)
    #             if test_acc > best_performance:
    #                 best_performance = test_acc
    #         if need_early_stop and early_stop_manager.should_stop(train_loss, train_acc, val_loss, val_acc):
    #             should_break = True
    #         if should_break and epoch > 50:
    #             print("early stop")
    #             break
    #         if half_stop_score > 0 and epoch > (epochs / 2) and val_acc < half_stop_score:
    #             print("half_stop")
    #             break
    #     if saved:
    #         model.load_state_dict(torch.load(tmp_model_file))
    #     model.eval()
    #     val_acc = evaluate(model(features, g), labels, val_mask)
    #     print(evaluate(model(features, g), labels, test_mask))
    #     if return_best:
    #         return model, val_acc, best_performance
    #     else:
    #         return model, val_acc

    @staticmethod
    def prepare_data(data, cuda=True):
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.ByteTensor(data.train_mask)
        test_mask = torch.ByteTensor(data.test_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        n_edges = data.graph.number_of_edges()
        # create DGL graph
        g = DGLGraph(data.graph)
        # add self loop
        g.add_edges(g.nodes(), g.nodes())
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        if cuda:
            features = features.cuda()
            labels = labels.cuda()
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)
        return features, g, labels, mask, val_mask, test_mask, n_edges
