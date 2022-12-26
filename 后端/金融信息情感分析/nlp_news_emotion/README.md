### 1.文件结构
```python
data/ # 存放训练集和词典
    emdict/ # 存放词典
    trainset/ # 存放训练集
        ...
model/ # 我们训练好的model模型
    wordfreq_logistic.ml
operate_data.py # 将文本处理成词向量,并且保存了logfile.plk
demo.py # 使用者(非开发者)调用框架的样例
```