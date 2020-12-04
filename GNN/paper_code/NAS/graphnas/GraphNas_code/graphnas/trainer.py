import glob
import os

import numpy as np
import scipy.signal
import torch

import graphnas.utils.tensor_utils as utils
from graphnas.gnn_model_manager import CitationGNNManager
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager

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
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0

        self.max_length = self.args.shared_rnn_max_length

        self.with_retrain = False
        self.submodel_manager = None
        self.controller = None
        # 构建controller 与 半监督模型
        self.build_model()  # build controller and sub-model

        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)

        # self.args.mode == "train"
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
            # self.search_space = {'attention':['gat','gcn',...],... }

            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)
            # self.action_list = ['attention_type', 'aggregator_type',# 'activate_function',  'number_of_heads', 'hidden_units',
            #                       'attention_type', 'aggregator_type', 'activate_function', 'number_of_heads', 'hidden_units']

            # build RNN controller
            from graphnas.graphnas_controller import SimpleNASController
            # 构建controller
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)

            # self.args.dataset = "citeseer"
            if self.args.dataset in ["cora", "citeseer", "pubmed"]:
                # implements based on dgl
                self.submodel_manager = CitationGNNManager(self.args)
            if self.args.dataset in ["Cora", "Citeseer", "Pubmed"]:
                # implements based on pyg
                # 构建GNN模型
                self.submodel_manager = GeoCitationManager(self.args)


        if self.args.search_mode == "micro":
            self.args.format = "micro"
            self.args.predict_hyper = True
            if not hasattr(self.args, "num_of_cell"):
                self.args.num_of_cell = 2
            from graphnas_variants.micro_graphnas.micro_search_space import IncrementSearchSpace
            search_space_cls = IncrementSearchSpace()
            search_space = search_space_cls.get_search_space()
            from graphnas.graphnas_controller import SimpleNASController
            from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager
            self.submodel_manager = MicroCitationManager(self.args)
            self.search_space = search_space
            action_list = search_space_cls.generate_action_list(cell=self.args.num_of_cell)
            if hasattr(self.args, "predict_hyper") and self.args.predict_hyper:
                self.action_list = action_list + ["learning_rate", "dropout", "weight_decay", "hidden_unit"]
            else:
                self.action_list = action_list
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)
            if self.cuda:
                self.controller.cuda()

        #为控制器分配cuda计算资源
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

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # self.start_epoch = 0
            # self.args.max_epoch = 10

            # 1. Training the shared parameters of the child graphnas
            self.train_shared(max_step=self.args.shared_initial_step)
            # self.args.shared_initial_step = 0

            # 2. Training the controller parameters theta
            self.train_controller()
            print("第 ", self.epoch, " epoch的100次controller_training完成")
            # 3. Derive architectures
            self.derive(sample_num=self.args.derive_num_sample)
            # self.args.derive_num_sample = 100
            print("第 ", self.epoch, " epoch的100次deriving完成")
            # 每完成两次epoch保存一次模型
            if self.epoch % self.args.save_epoch == 0:
                # self.args.save_epoch = 2
                self.save_model()

        if self.args.derive_finally:
            # self.args.derive_finally = True
            best_actions = self.derive()
            print("best structure:" + str(best_actions))
        self.save_model()

    def train_shared(self, max_step=50, gnn_list=None):
        """
        Args:
            max_step: Used to run extra training steps as a warm-up.
            gnn: If not None, is used instead of calling sample().

        """
        if max_step == 0:  # no train shared
            return

        print("*" * 35, "training model", "*" * 35)
        gnn_list = gnn_list if gnn_list else self.controller.sample(max_step)
        # 如果gnn_list不是none则gnn_list = gnn_list

        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            try:
                _, val_score = self.submodel_manager.train(gnn, format=self.args.format)
                logger.info(f"{gnn}, val_score:{val_score}")
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e

        print("*" * 35, "training over", "*" * 35)

    def get_reward(self, gnn_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        # gnn_list = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 4],--->选择出的GNN结构
        # entropies = tensor([1.9459, 1.3863, 2.0794, 1.7917, 1.9458, 1.9459, 1.3862, 2.0794, 1.7917,
        #                     1.9458], device='cuda:0', grad_fn=<CatBackward>)--->LSTM每一步输出信息熵

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

            reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                    with_retrain=self.with_retrain)
            # format = "two"
            # with_retrain = True
            # GeoCitationManager 继承了 CitationGNNManager类,所以有test_with_param方法

                
            if reward is None:  # cuda error hanppened
                reward = 0
            else:
                reward = reward[0]

            reward_list.append(reward)

        # 对reward进行处理
        if self.args.entropy_mode == 'reward':
            rewards = reward_list + self.args.entropy_coeff * entropies
        # reward_list=[0.34,...],每个选择出的GNN在验证集上产生的reward列表,每个奖reward取值范围[-0.5,0.5]
        # self.args.entropy_coeff = 1e-4
        # entropies = tensor([1.9459, 1.3863, 2.0794, 1.7917, 1.9458, 1.9459, 1.3862, 2.0794, 1.7917,
        #                     1.9458], device='cuda:0', grad_fn=<CatBackward>)--->LSTM每一步输出信息熵

        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        print("*" * 35, "training controller", "*" * 35)

        model = self.controller

        # 使pytorch定义的controller模型进入训练模式
        # 每次训练都要初始化 adv_history,entropy_history,reward_history 三个列表
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        # 初始化带batch_size参数的中 h0,c0向量,全部为0
        hidden = self.controller.init_hidden(self.args.batch_size)
        # self.args.batch_size = 64

        # 初始化控制器LSTM模型总损失值
        total_loss = 0

        for step in range(self.args.controller_max_step):
            # self.args.controller_max_step = 100
            # controller每次训练100轮,一轮选一个GNN结构

            # sample graphnas
            structure_list, log_probs, entropies = self.controller.sample(with_details=True)
            # structrue_list = ['gat', 'sum', 'relu', 2, 8, 'linear', 'mlp', 'tanh', 2, 4],--->选择出的GNN结构
            # log_probs = tensor([-1.9461, -1.3936, -2.0807, -1.7964, -1.9570, -1.9413, -1.3704, -2.0878,
            #         -1.7907, -1.9185], device='cuda:0', grad_fn=<CatBackward>)--->LSTM每一步选择的operator的自信息I
            # entropies = tensor([1.9459, 1.3863, 2.0794, 1.7917, 1.9458, 1.9459, 1.3862, 2.0794, 1.7917,
            #         1.9458], device='cuda:0', grad_fn=<CatBackward>)--->LSTM每一步输出信息熵

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()

            results = self.get_reward(structure_list, np_entropies, hidden)

            # results = (rewards, hidden) hidden原封不动的回来了

            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            # discount
            # 使用滤波器实现rewards折扣损失计算,重新计算rewards列表内的reward
            if 1 > self.args.discount > 0:
                # self.args.discount = 1
                rewards = discount(rewards, self.args.discount)
                # 每次训练都要初始化 adv_history,entropy_history,reward_history 三个列表
                # controller每次训练100轮,一轮选一个GNN结构

                """"
def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]
    
    x[::-1] : 将x序列翻转
                """

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
            """
 def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value
            
            """

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

        print("*" * 35, "training controller over", "*" * 35)

    def evaluate(self, gnn):
        """
        Evaluate a structure on the validation set.
        """
        # 将controller转化为验证模式
        # 是不是因为所有模型的参数都加载到了一个torch管理器中,
        # 所以训练GNN会对controller模型产生影响?所以在验证gnn时要固定controller参数??
        self.controller.eval()

        gnn = self.form_gnn_info(gnn)

        results = self.submodel_manager.retrain(gnn, format=self.args.format)

        if results:
            reward, scores = results
        else:
            return

        logger.info(f'eval | {gnn} | reward: {reward:8.2f} | scores: {scores:8.2f}')

    def derive_from_history(self):
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "r") as f:

            print("read_path:", self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file)

            lines = f.readlines()

        results = []
        best_val_score = "0"
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
            #for i in range(1):
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
        #for i in range(1):
            # manager.shuffle_data()
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print(f"best results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")
        return best_structure

    def derive(self, sample_num=None):
        # controller_train 类训练好了controller,使用训练好的controller来进行sample,采样GNN结构
        """
        sample a serial of structures, and return the best structure.
        """
        """
        # sample_num = 100
        
        """
        if sample_num is None and self.args.derive_from_history:
            # 当执行 best_actions = self.derive() 时调用函数derive_from_history()选取最佳action_best
            return self.derive_from_history()
        else:
            if sample_num is None:
                sample_num = self.args.derive_num_sample

            gnn_list, _, entropies = self.controller.sample(sample_num, with_details=True)
            # 默认使用训练好的controller采样100个child GNN
            max_R = 0
            best_actions = None
            filename = self.model_info_filename

            #对采样的child GNN进行验证
            for action in gnn_list:
                gnn = self.form_gnn_info(action)

                """
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
                """
                # 测试采样的GNN效果,使用val_score值来评估
                reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                               with_retrain=self.with_retrain)

                if reward is None:  # cuda error hanppened
                    continue
                else:
                    # 获取val_score
                    results = reward[1]

                #选择val_score最大的GNN结构
                if results > max_R:
                    max_R = results
                    best_actions = action
            # 记录最佳GNN结构,最佳val_score值
            logger.info(f'derive |action:{best_actions} |max_R: {max_R:8.6f}')

            # 验证最佳GNN结构,重新使用数据集训练GNN并得到其val_score与test_score
            self.evaluate(best_actions)
            # 返回最佳GNN结构
            return best_actions

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
