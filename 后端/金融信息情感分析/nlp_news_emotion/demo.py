# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import operate_data as od
from Stock_news_collection import StockNewsCollection
from pandas import DataFrame

VECTOR_MODE = {'onehot': 0, 'wordfreq': 1, 'tfidf': 2}

class Predictor(object):
    def __init__(self):
        self._model = None
        self.news = None
        self.__tag = None
        self._vec = None
        self.mode = None

    def load_model(self,path=None):
        if not path:
            path = os.path.join('model','wordfreq_logistic.ml')

        with open(path,'rb') as f:
            self._model = pickle.load(f)

    def set_mode(self,mode):
        if isinstance(mode,int):
            assert mode in VECTOR_MODE.values(), "没有这种vector方式"
        if isinstance(mode,str):
            assert mode in VECTOR_MODE.keys(), "没有这种vector方式"
            mode = VECTOR_MODE[mode]
        self.mode = mode

    def set_news(self,news):
        if not len(news):
            print("请输入有效的新闻文本,谢谢")
            return
        self.news = news

    def trans_vec(self):
        vec_list = od.words2Vec(self.news,od.emotionList,od.stopList,od.posList,od.negList,mode=self.mode)
        self._vec = np.array(vec_list).reshape(1,-1)

    # 调用的时候计算函数
    def __call__(self, *args, **kwargs):
        self.__tag = self._model.predict(self._vec)
        return self.__tag

    def get_tag(self):
        return self.__tag

od.loadStopwords()
od.loadEmotionwords()
od.loadDocument(od.stopList)
predictor = Predictor()
predictor.load_model()
predictor.set_mode(mode="wordfreq")#设置模型{'onehot': 0, 'wordfreq': 1, 'tfidf': 2}
#以上代码是初始化配置，只需要调用一次
def pre_single(news=''):
    predictor.set_news(news=news)
    predictor.trans_vec()
    tag = predictor() # 分类结果
    return [tag, sum(predictor._vec[0])]
def get_news_emotion(code, nums):
    #收集nums条股票code的资讯
    snc = StockNewsCollection()
    err, maxn, res = snc.crawling(code, nums)
    #print(res)
    #对资讯进行情感分析
    out = []
    v=-1
    for i in res:
        v += 1
        if v%100 == 99:
            print('n->', v)
        try:
            out.append([code]+i+pre_single(i[3]))
        except:
            print(i[1])
    df = DataFrame(out)
    df.to_csv('st%d.csv' % (code))
    #index_col = [row, code, time, url, title, content, emotion, sum]
    

#收集500条股票600050的资讯，将情感分析结果保存到csv文件中
get_news_emotion(600050, 500)


