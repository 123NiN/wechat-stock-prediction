from data_utils import *
import numpy as np
import pandas as pd
from config import Arg
import os


args = Arg()

# 该系列代码所要求的股票文件名称必须是股票代码+csv的格式，如000001.csv
# --------------------------训练集数据的处理--------------------- #
def get_train_data(train_dir, batch_size=args.batch_size, time_step=args.time_step):
    ratio = args.ratio
    len_index = []
    batch_index = []
    val_index = []
    df = open(train_dir)
    data_otrain = pd.read_csv(df)
    stock_len = len(data_otrain)
    data_train = data_otrain.iloc[:, 1:].values
    print(len(data_train))
    label_train = data_otrain.iloc[:, -1].values
    normalized_train_data = (data_train-np.mean(data_train, axis=0))/np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []   # 训练集x和y定义
    for i in range(len(normalized_train_data) + 1):
        if i % stock_len == 0:
            len_index.append(i)
    for i in range(len(len_index) - 1):
        for k in range(len_index[i], len_index[i + 1] - time_step - 1):
            x = normalized_train_data[k:k + time_step, :6]
            y = label_train[k + time_step, np.newaxis]
            temp_data = []
            # onehot编码
            for j in y:
                if j > 2:
                    temp_data.append([0, 0, 0, 0, 0, 1])
                elif 1 < j <= 2:
                    temp_data.append([0, 0, 0, 0, 1, 0])
                elif 0 < j <= 1:
                    temp_data.append([0, 0, 0, 1, 0, 0])
                elif -1 < j <= 0:
                    temp_data.append([0, 0, 1, 0, 0, 0])
                elif -2 < j <= -1:
                    temp_data.append([0, 1, 0, 0, 0, 0])
                else:
                    temp_data.append([1, 0, 0, 0, 0, 0])
            train_x.append(x.tolist())
            train_y.append(temp_data)
    train_len = int(len(train_x) * ratio)  # 按照8：2划分训练集和验证集
    train_x_1, train_y_1 = train_x[:train_len], train_y[:train_len]  # 训练集的x和标签
    val_x, val_y = train_x[train_len:], train_y[train_len:]  # 验证集的x和标签
    # 添加标签
    for i in range(len(train_x_1)):
        if i % batch_size == 0:
            batch_index.append(i)
    for i in range(len(val_x)):
        if i % batch_size == 0:
            val_index.append(i)
    batch_index.append(len(train_x_1))
    val_index.append(len(val_x))
    print(batch_index)
    print(val_index)
    print(np.shape(train_x))
    return batch_index, val_index, train_x_1, train_y_1, val_x, val_y


# --------------------------测试集数据的处理--------------------- #
# 测试集数据长度不能小于time_step
def get_test_data(test_dir, time_step=args.time_step):
    
    f = open(test_dir)
    df = pd.read_csv(f)
    stock_len = len(df)
    data_test = df.iloc[:, 1:].values
    label_test = df.iloc[:, -1].values
    batch_index = []
    normalized_test_data = (data_test-np.mean(data_test, axis=0))/np.std(data_test, axis=0)  # 标准化
    test_x, test_y = [], []
    for i in range(len(normalized_test_data) + 1):
        if i % stock_len == 0:
            batch_index.append(i)
    for i in range(len(batch_index)-1):
        if stock_len > time_step+1:
            for j in range(batch_index[i], batch_index[i + 1] - time_step - 1):
                x = normalized_test_data[j:j + time_step, :]
                y = label_test[j + time_step, np.newaxis]
                temp_data = []
                # 标签编码
                for k in y:
                    if k > 2:
                        temp_data.append([0, 0, 0, 0, 0, 1])
                    elif 1 < k <= 2:
                        temp_data.append([0, 0, 0, 0, 1, 0])
                    elif 0 < k <= 1:
                        temp_data.append([0, 0, 0, 1, 0, 0])
                    elif -1 < k <= 0:
                        temp_data.append([0, 0, 1, 0, 0, 0])
                    elif -2 < k <= -1:
                        temp_data.append([0, 1, 0, 0, 0, 0])
                    else:
                        temp_data.append([1, 0, 0, 0, 0, 0])
                test_x.append(x.tolist())
                test_y.extend(temp_data)
        else:
            for j in range(batch_index[i], batch_index[i]+1):
                x = normalized_test_data[j:j + time_step, :]
                y = label_test[j + time_step, np.newaxis]
                temp_data = []
                # 标签编码
                for k in y:
                    if k > 2:
                        temp_data.append([0, 0, 0, 0, 0, 1])
                    elif 1 < k <= 2:
                        temp_data.append([0, 0, 0, 0, 1, 0])
                    elif 0 < k <= 1:
                        temp_data.append([0, 0, 0, 1, 0, 0])
                    elif -1 < k <= 0:
                        temp_data.append([0, 0, 1, 0, 0, 0])
                    elif -2 < k <= -1:
                        temp_data.append([0, 1, 0, 0, 0, 0])
                    else:
                        temp_data.append([1, 0, 0, 0, 0, 0])
                test_x.append(x.tolist())
                test_y.extend(temp_data)

    print(batch_index)
    print(np.shape(test_x))
    return test_x, test_y

# --------------------------当天股票数据更新---------------------- #
# 该函数完成下载实时股票数据，与之前的数据拼接后拼接的x
# 只能用于获取一天的更新数据，不会对源文件进行更新，如果有断层（不只一天），请先下载整批数据，然后使用get_update_data来更新数据
# file_name是要用于预测的股票地址如'D:\data\\201904\\000001.csv'
def get_predict_data(file_name):
    f = open(file_name)
    f = pd.read_csv(f)
    hist_data = f[-args.time_step:]
    pre_data = hist_data.iloc[:, 1:].values
    x = (pre_data - np.mean(pre_data, axis=0)) / np.std(pre_data, axis=0)  # 标准化
    x = [x.tolist()]
    print(np.shape(x))
    return x


if __name__ == '__main__':
    #get_train_data()
    x,c = get_predict_data('.\data\\600050.csv')
    
