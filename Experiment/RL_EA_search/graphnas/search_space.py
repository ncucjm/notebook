import torch
from collections import deque

class MacroSearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            # Define operators in search space
            self.search_space = {
                "attention_type": ["gat",
                                   "gcn",
                                   "cos",
                                   "const",
                                   "gat_sym",
                                   'linear',
                                   'generalized_linear'],
                'aggregator_type': ["sum", "mean", "max", "mlp", ],
                'activate_function': ["sigmoid", "tanh", "relu", "linear",
                                      "softplus", "leaky_relu",
                                      "relu6", "elu"],
                'number_of_heads': [1, 2, 4, 6, 8, 16],
                'hidden_units': [4, 8, 16, 32, 64, 128, 256],
            }

    def get_search_space(self):
        return self.search_space

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search
    # space according to operator category.
    def generate_action_list(self, num_of_layers=2):
        action_names = list(self.search_space.keys())
        action_list = action_names * num_of_layers
        return action_list


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")
# unit test
if __name__=="__main__":
    search_space = MacroSearchSpace()
    space = search_space.get_search_space()
    gnn_structure = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 6]
    action_list =['attention_type', 'aggregator_type', 'activate_function', 'number_of_heads', 'hidden_units',
                  'attention_type', 'aggregator_type', 'activate_function', 'number_of_heads', 'hidden_units']
    population = deque()
    i = 0
    for operator, action_name in zip(gnn_structure, action_list):
        if i == 9:
            operator = 8
        i+=1
        print(space[action_name].index(operator))

