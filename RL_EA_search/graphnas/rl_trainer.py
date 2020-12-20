from collections import deque
import os
import glob
import torch
import numpy as np
import scipy.signal
import graphnas.utils.tensor_utils as utils
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager
import  time
logger = utils.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

history = []
def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value

def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim

def experiment_data_save(name,time_list,acc_list):
    path = path_get()[1]
    with open(path+"/"+name, "w") as f:
        f.write(str(time_list))
        f.write("\n"+str(acc_list))
    print("the ", name, " have written")


def path_get():
    # 当前文件目录
    current_path = os.path.abspath('')
    # 当前文件夹父目录
    father_path = os.path.abspath(os.path.dirname(current_path))
    # corpus_path = os.path.join(father_path, corpus)
    return father_path, current_path

class RL_Trainer(object):

    def __init__(self, args):
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0
        self.max_length = self.args.shared_rnn_max_length

        self.with_retrain = False
        self.submodel_manager = None
        self.controller = None
        self.build_model()  # build controller and sub-model
        self.RL_train_time = []
        self.RL_search_time = []
        self.RL_train_acc = []
        self.RL_search_acc = []


        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)

        if self.args.mode == "derive":
            self.load_model()

    def build_model(self):
        self.args.share_param = False
        self.with_retrain = True
        self.args.shared_initial_step = 0
        if self.args.search_mode == "macro":
            # generate model description in macro way (generate entire network description)
            from graphnas.search_space import MacroSearchSpace
            search_space_cls = MacroSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)
            # build RNN controller
            from graphnas.graphnas_controller import SimpleNASController
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)
            if self.args.dataset in ["Cora", "Citeseer", "Pubmed"]:
                # implements based on pyg
                self.submodel_manager = GeoCitationManager(self.args)
        if self.cuda:
            self.controller.cuda()

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

    def train(self, action_list):
        model_path = "/home/jerry/experiment/RL_nas/graphnas/Citeseer"
        # Training the controller
        if not os.listdir(model_path):# 判断保存controler模型的文件夹是否为空，为空返回False,反之为Ture
            self.train_controller()
            print("*" * 35, "using controller search the initialize population", "*" * 35)
            populations, accuracies = self.derive(self.args.population_size, action_list)
            print("*" * 35, "the search DONE", "*" * 35)
            self.save_model()
        else:
            self.load_model() # 每次加载step序号最大controler模型search
            print("*" * 35, "using controller search the initialize population", "*" * 35)
            populations, accuracies = self.derive(self.args.population_size, action_list)
            print("*" * 35, "the search DONE", "*" * 35)
        return populations, accuracies

    def derive(self, sample_num, action_list):
        if sample_num is None and self.args.derive_from_history:
            return self.derive_from_history()
        else:
            if sample_num is None:
                sample_num = self.args.derive_num_sample
            gnn_list, _, entropies = self.controller.sample(sample_num, with_details=True)
            accuracies = []

            epoch = 0
            for action in gnn_list:
                once_RL_search_start_time = time.time()

                gnn = self.form_gnn_info(action)
                reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                               with_retrain=self.with_retrain)
                acc_score = reward[1]
                accuracies.append(acc_score)

                once_RL_search_end_time = time.time()

                print("the", epoch, "epcoh controller train time: ",
                      once_RL_search_end_time - once_RL_search_start_time, 's')

                if epoch == 0:
                    self.RL_search_time.append(once_RL_search_start_time)
                    self.RL_search_time.append(once_RL_search_end_time)
                    self.RL_search_acc.append(acc_score)
                else:
                    self.RL_search_time.append(once_RL_search_end_time)
                    self.RL_search_acc.append(acc_score)

                epoch += 1
            father_path = path_get()[0]
            experiment_data_save("controler_search.txt", self.RL_search_time, self.RL_search_acc)
            print("all RL search time list: ", self.RL_search_time)
            print("all RL search acc list: ", self.RL_search_acc)

            for individual, ind_acc in zip(gnn_list, accuracies):
                print("individual:", individual, " val_score:", ind_acc)
            # gnn_structure　基因编码
            population = []
            for gnn_structure in gnn_list:
                i = 0
                single = []
                for operator, action_name in zip(gnn_structure, action_list):
                    if i == 9:
                        operator = 8
                    i += 1
                    single.append(self.search_space[action_name].index(operator))
                population.append(single)

            return population, accuracies

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(f'[!] No checkpoint found in {self.args.dataset}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.controller_step = max(controller_steps)

        self.controller.load_state_dict(
            torch.load(self.controller_path))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path))
        logger.info(f'[*] LOADED: {self.controller_path}')

    def get_reward(self, gnn_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
            pass
        else:
            gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            reward = self.submodel_manager.test_with_param(gnn,
                                                      format=self.args.format,
                                                      with_retrain=self.with_retrain)

            if reward is None:  # cuda error hanppened
                reward = 0
            else:
                rewards = reward[0]#奖励计算正确

            reward_list.append(rewards)
            acc_validation = reward[1]

        if self.args.entropy_mode == 'reward':
            rewards = reward_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden, acc_validation

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        print("*" * 35, "training controller", "*" * 35)
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.args.batch_size)
        total_loss = 0
        for step in range(self.args.controller_max_step):
            #contraller训练一次的时间
            once_controller_train_start_time = time.time()

            # sample graphnas
            structure_list, log_probs, entropies = self.controller.sample(with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden, acc = results
            else:
                continue

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = utils.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()
            once_controller_train_end_time = time.time()
            print("the", step, "epcoh controller train time: ",
                  once_controller_train_end_time-once_controller_train_start_time, "s")

            if step == 0:
                self.RL_train_time.append(once_controller_train_start_time)
                self.RL_train_time.append(once_controller_train_end_time)
                self.RL_train_acc.append(acc)
            else:
                self.RL_train_time.append(once_controller_train_end_time)
                self.RL_train_acc.append(acc)
        print("all RL train time list: ", self.RL_train_time)
        print("all RL train acc list: ", self.RL_train_acc)
        print("*" * 35, "training controller over", "*" * 35)
        experiment_data_save("controler_train.txt", self.RL_train_time, self.RL_train_acc)

    def evaluate(self, gnn):
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()
        gnn = self.form_gnn_info(gnn)
        results = self.submodel_manager.retrain(gnn, format=self.args.format)
        if results:
            reward, scores = results
        else:
            return
        logger.info(f'eval | {gnn} | reward: {reward:8.2f} | scores: {scores:8.2f}')

    @property
    def model_info_filename(self):
        return f"{self.args.dataset}_{self.args.search_mode}_{self.args.format}_results.txt"

    @property
    def controller_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps




