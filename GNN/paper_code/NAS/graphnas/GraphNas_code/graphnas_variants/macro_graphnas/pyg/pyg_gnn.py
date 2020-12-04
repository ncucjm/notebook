import torch
import torch.nn.functional as F

from graphnas.gnn import GraphNet as BaseNet
from graphnas.search_space import act_map
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_layer import GeoLayer


class GraphNet(BaseNet):

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, state_num=5,
                 residual=False):
        # actions = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 6]
        # num_feat = 3703
        # num_label = 6
        # drop_out = 0.6
        # multi_label=False
        # batch_normal=False
        # state_num=5
        # residual=False

        self.residual = residual
        self.batch_normal = batch_normal

        super(GraphNet, self).__init__(actions, num_feat, num_label, drop_out, multi_label, batch_normal, residual,
                                       state_num)
        # pyg_gnn脚本中的 GraphNet类 是脚本gnn中 GraphNet的子类,通过super调用父类的构造方法

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        # residual=False
        # batch_normal=False

        if self.residual:
            self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()

        self.layers = torch.nn.ModuleList()

        self.acts = []

        self.gates = torch.nn.ModuleList()

        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)
        # actions = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 6]
        # batch_normal = False
        # drop_out = 0.6
        # self.layer_nums = 2
        # num_feat = 3703
        # num_label = 6a
        # state_num=5

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=6):
        # actions = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 6]
        # batch_normal = False
        # drop_out = 0.6
        # layer_nums = 2
        # num_feat = 3703
        # num_label = 6
        # state_num=5

        # build hidden layer
        for i in range(layer_nums):

            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num

            # extract layer information
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            # i = 0时,  attention_type = "gat", aggregator_type = "sum", act = "relu", head_num = "2", out_channels = "8"
            # i = 1时,  attention_type = "linear", aggregator_type = "mlp", act = "tanh", head_num = "2", out_channels = "6"

            concat = True

            if i == layer_nums - 1:
                concat = False

            if self.batch_normal:
                self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))

            self.layers.append(
                GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout,
                         att_type=attention_type, agg_type=aggregator_type, ))
            # i = 0时,  in_channels = 3703  out_channels = 8,  head_num = "2",
            #           concat = True, dropout=0.6  att_type = "gat", agg_type = "sum",

            self.acts.append(act_map(act))

            if self.residual:
                if concat:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels * head_num))
                else:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index_all):
        output = x
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

                output = act(layer(output, edge_index_all) + fc(output))
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = act(layer(output, edge_index_all))
        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = "layer_%d" % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = self.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                result[key] = self.fcs[i]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                result[key] = self.bns[i]
        return result

    def load_param(self, param):
        if param is None:
            return

        for i in range(self.layer_nums):
            self.layers[i].load_param(param["layer_%d" % i])

        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
                if key in param:
                    self.fcs[i] = param[key]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f"layer_{i}_fc_{bn.weight.size(0)}"
                if key in param:
                    self.bns[i] = param[key]
