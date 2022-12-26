from lstm_model import *

#get_stock_data('600050', '2019-01-1', '2021-08-23', '.\data\\') #获取股票数据
#train_lstm('.\data\\600050.csv') # 训练模型，参数在arg里调
#predict('.\data\\600050.csv') #预测

a1 = test('.\data\\600050.csv') #测试
f = pd.read_csv('.\data\\600050.csv')
for i in range(len(a1)):
    f.loc[609-i, 'pred'] = a1[-i-1]
f.to_csv('.\data\\pd600050.csv',index = False)

'''
def predict_a(pre_x, time_step=args.time_step):
    X = tf.placeholder(tf.float32, shape=[None, time_step, args.input_size])
    pred, _ = lstm(X)
    pre_y = []
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        saver.restore(sess, args.train_model_dir + args.model_name)
        if len(pre_x) < 15000:
            prob = sess.run(pred, feed_dict={X: pre_x})
            pre_y.extend(prob)
        else:
            for i in range(len(pre_x) // args.batch_size + 1):
                prob = sess.run(pred, feed_dict={X: pre_x[args.batch_size * i:args.batch_size * (i + 1)]})
                pre_y.extend(prob)
        pre_y = np.array(pre_y)
        a = list(np.argmax(pre_y, 1))
        return a[0]
for i in range(60):
    hist_data = f[-args.time_step-1:-i-1]
    pre_data = hist_data.iloc[:, 1:].values
    x = (pre_data - np.mean(pre_data, axis=0)) / np.std(pre_data, axis=0)  # 标准化
    x = [x.tolist()]
    f.loc[609-i, 'pred'] = predict_a(x)
f = f.loc[550:, :]
'''


