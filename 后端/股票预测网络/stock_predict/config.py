from data_utils import *


# -------------------参数配置----------------- #
class Arg:
    def __init__(self):
        # 模型存放路径
        self.train_model_dir = '.\logfile\\new_logfile\\'
        # 训练图存放路径
        self.train_graph_dir = '.\logfile\\new_logfile\graph\\train_270\\'
        # 验证loss存放路径
        self.val_graph_dir = '.\logfile\\new_logfile\graph\\val_270\\'
        # 模型名称
        self.model_name = 'model-270-17-19'
        self.rnn_unit = 128     # 隐层节点数
        self.input_size = 6     # 输入维度（既用几个特征）
        self.output_size = 6    # 输出维度（既使用分类类数预测）
        self.layer_num = 3      # 隐藏层层数
        self.lr = 0.0006         # 学习率
        self.time_step = 50     # 时间步长
        self.epoch = 50         # 训练次数
        self.epoch_fining = 30  # 微调的迭代次数
        self.batch_size = 128  # batch_size
        self.ratio = 0.8        # 训练集验证集比例
