from dataprocess import *
import tensorflow.compat.v1 as tf
from sklearn.metrics import classification_report

tf.compat.v1.disable_eager_execution()
# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

tf.reset_default_graph()

weights = {
         'in': tf.Variable(tf.random_normal([args.input_size, args.rnn_unit])),
         'out': tf.Variable(tf.random_normal([args.rnn_unit, args.output_size]))
          }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[args.rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
         }


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, args.input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in)+b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, args.rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_unit)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell)  # 设置dropout
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * args.layer_num, state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(mlstm_cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = output_rnn[:, -1, :]
    output = tf.reshape(output, [-1, args.rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out)+b_out
    return pred, final_states


# -----------------------训练模型------------------------------------ #
# 用于训练模型，val（验证集）默认为开启
def train_lstm(file_path, time_step=args.time_step, val=True):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    Y = tf.placeholder(tf.float32, shape=[None, 1, args.output_size])
    batch_index, val_index, train_x, train_y, val_x, val_y = get_train_data(file_path)
    print('trian_y:{}, val_y:{}'.format(np.shape(train_y), np.shape(val_y)))
    pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.scalar('loss', loss)
        merged_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(args.train_graph_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(args.val_graph_dir)
        min_loss = 50
        # 训练.次
        for i in range(args.epoch):
            for j in range(len(batch_index)-1):
                summary_str, _, loss_ = sess.run([merged_summary_op, train_op, loss],
                                                 feed_dict={X: train_x[batch_index[j]:batch_index[j+1]],
                                                            Y: train_y[batch_index[j]:batch_index[j+1]]})
            if val:
                for j in range(len(val_index) - 1):
                    valid_str, loss_val = sess.run([merged_summary_op, loss],
                                                   feed_dict={X: val_x[val_index[j]:val_index[j+1]],
                                                              Y: val_y[val_index[j]:val_index[j+1]]})
            if i % 10 == 0:
                print("------------------------------------------------------")
                print('epoch: {}, train_loss: {:.4f}, Val_Loss: {:.4f}'.format(i+1, loss_, loss_val))
            train_writer.add_summary(summary_str, i)
            valid_writer.add_summary(valid_str, i)
            if loss_val < min_loss:
                min_loss = loss_val
                print("保存模型：", saver.save(sess, args.train_model_dir+args.model_name))

# -----------------------------测试模型------------------------------ #
# 用于测试
def test(file_path, time_step=args.time_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    test_x, test_y = get_test_data(file_path)
    print("---------数据加载完毕--------")
    pred, _ = lstm(X)
    pre_dict = []
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        saver.restore(sess, args.train_model_dir+args.model_name)
        print("----------模型加载完毕----------")
        if len(test_x) < 15000:
            prob = sess.run(pred, feed_dict={X: test_x})
            pre_dict.extend(prob)
        else:
            for i in range(len(test_x)//args.batch_size+1):
                prob = sess.run(pred, feed_dict={X: test_x[args.batch_size*i:args.batch_size*(i+1)]})
                pre_dict.extend(prob)
        pre_dict = np.array(pre_dict)
        test_label = np.array(test_y)
        a1 = list(np.argmax(pre_dict, 1))
        return a1


# -----------------------------预测模型------------------------------ #
# 用于预测第二天的收盘价
def predict(filepath, time_step=args.time_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    pre_x = get_predict_data(filepath)
    print("---------数据加载完毕--------")
    pred, _ = lstm(X)
    pre_y = []
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        saver.restore(sess, args.train_model_dir + args.model_name)
        print("----------模型加载完毕----------")
        if len(pre_x) < 15000:
            prob = sess.run(pred, feed_dict={X: pre_x})
            pre_y.extend(prob)
        else:
            for i in range(len(pre_x) // args.batch_size + 1):
                prob = sess.run(pred, feed_dict={X: pre_x[args.batch_size * i:args.batch_size * (i + 1)]})
                pre_y.extend(prob)
        pre_y = np.array(pre_y)
        a = list(np.argmax(pre_y, 1))
        return a
        '''
        print(a)
        if a[0] == 0:
            print("预测股票明天的收盘价跌2%及以上")
        elif a[0] == 1:
            print("预测股票明天的收盘价跌1%-2%")
        elif a[0] == 2:
            print("预测股票明天的收盘价跌1%以内")
        elif a[0] == 3:
            print("预测股票明天的收盘价涨1%以内")
        elif a[0] == 4:
            print("预测股票明天的收盘价涨1%-2%")
        else:
            print("预测股票明天的收盘价涨2%以上")
        '''
