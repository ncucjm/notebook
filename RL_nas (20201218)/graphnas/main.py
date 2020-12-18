import time
import torch
import argparse
import numpy as np
from graphnas.rl_trainer import RL_Trainer
import graphnas.utils.tensor_utils as utils
from graphnas.gs_trainer import GridSearch_Trainer
from graphnas.rs_trainer import RandomSearch_Trainer
from graphnas.evolution_trainer import Evolution_Trainer

def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args

def register_default_args(parser):
    parser.add_argument('--optimizer', type=str, default='EA',
                        choices=['EA', 'RL', 'RS', 'GS'],
                        help='EA: Evolutionary algorithm,\
                              RL: Reinforcement Learning algorithm\
                              RS: Random Search algorithm\
                              GS: Grid Search algorithm')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training GraphNAS,\
                              derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--max_save_num', type=int, default=3)
    # EA
    parser.add_argument('--selection_mode', type=str, default='wheel', help='random, wheel')
    parser.add_argument('--crossover_mode', type=str, default='point', help='point, none')
    parser.add_argument('--mutation_mode', type=str, default='point_p', help='point_p, point_none')
    parser.add_argument('--mutation_p', type=float, default=0.2, help='[0-1]')
    parser.add_argument('--updating_mode', type=str, default="none-age", help='age,none-age')

    # parser.add_argument('--selection_mode', type=str, default='random', help='random, wheel')
    # parser.add_argument('--crossover_mode', type=str, default='none', help='point, none')
    # parser.add_argument('--mutation_mode', type=str, default='point_none', help='point_p, point_none')
    # parser.add_argument('--mutation_p', type=float, default=0.2, help='[0-1]')
    # parser.add_argument('--updating_mode', type=str, default="age", help='age,none-age')

    # test
    parser.add_argument('--cycles', type=int, default=2, help='Evolution cycles')
    # default
    #parser.add_argument('--cycles', type=int, default=1000, help='Evolution cycles')

    #训练多少次，从population中挑选出最优个体
    parser.add_argument('--eval_cycle', type=int, default=100,
                        help='Evaluate best model every x iterations. def:100')
    # test
    parser.add_argument('--population_size', type=int, default=3)
    # default
    #parser.add_argument('--population_size', type=int, default=100)

    #　test
    parser.add_argument('--sample_size', type=int, default=2, help='Sample size for tournament selection')
    # default
    # parser.add_argument('--sample_size', type=int, default=25, help='Sample size for tournament selection')

    # controller
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')

    ######################
    # all_training epoch:
    # test
    parser.add_argument('--max_epoch', type=int, default=1)
    # exp
    #parser.add_argument('--max_epoch', type=int, default=10)
    ######################

    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward',
                        choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)

    #######################
    # training controller_max_step

    # test
    parser.add_argument('--controller_max_step', type=int, default=5,
                        help='step for controller parameters')
    # exp
    # parser.add_argument('--controller_max_step', type=int, default=100,
    #                     help='step for controller parameters')
    #######################

    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)

    ####################
    # derive the gnn num
    # test
    parser.add_argument('--derive_num_sample', type=int, default=1)
    # exp
    #parser.add_argument('--derive_num_sample', type=int, default=100)
    ####################

    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    # child model
    parser.add_argument("--dataset", type=str, default="Citeseer",
                        required=False, help="The input dataset.")
    ###################
    #gnn train epoch
    parser.add_argument("--epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300, help="number of training epochs")
    ###################

    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")

    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")

    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file',
                        type=str,
                        default=f"sub_manager_logger_file_{time.time()}.txt")
    # RL_initialize_population
    parser.add_argument('--population_k', type=int, default=2)



def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    np.set_printoptions(precision=8)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    if args.optimizer == 'EA':
        trainer = Evolution_Trainer(args)
    elif args.optimizer == 'RL':
        trainer = RL_Trainer(args)
    elif args.optimizer == 'GS':
        trainer = GridSearch_Trainer(args)
    elif args.optimizer == 'RS':
        trainer = RandomSearch_Trainer(args)
    else:
        raise Exception("[!] Optimizer not found: ", args.optimizer)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'derive':
        trainer.derive()
    else:
        raise Exception("[!] Mode not found: ", args.mode)


if __name__ == "__main__":
    args = build_args()
    main(args)
