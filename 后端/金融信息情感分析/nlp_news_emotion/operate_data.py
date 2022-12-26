#encoding:utf-8
import os
import pickle
import math
import jieba
import jieba.posseg as pseg

dictPath = os.path.join('data','emdict','userdict') # 针对linux兼容
jieba.load_userdict(dictPath) # 加载个人词典
stopwordsPath = os.path.join('data','emdict','stopword.plk') # 停用词
negwordsPath = os.path.join('data','emdict','negword.plk') # 消极词
poswordsPath = os.path.join('data','emdict','posword.plk') # 积极词
documentPath = os.path.join('data','trainset') # 训练样本的目录

stopList = []
emotionList = []
posList = []
negList = []
docList = [] # 所有文档的词语组成的2维词列表（tf-idf所需要的词列表）

# 加载停用词
def loadStopwords(path = stopwordsPath):
    global stopList
    with open(stopwordsPath,'rb') as f:
        stopList = pickle.load(f)

# 加载感情词
def loadEmotionwords(*paths):
    global posList,emotionList,negList
    if not len(paths):
        with open(negwordsPath,'rb') as f:
            negList = pickle.load(f)
            # t.extend(pickle.load(f))
        with open(poswordsPath,'rb') as f:
            posList = pickle.load(f)
            # t.extend(pickle.load(f))
        emotionList = posList+negList
    else:
        for path in paths:
            with open (path,'rb') as f:
                emotionList = pickle.load(f)

# 针对TF-IDF
# 读取所有数据集的词，是的全局变量变成二维的List
def loadDocument(stopList,path=documentPath):
    global docList
    docList = []
    for file in os.listdir(path):
        news = None
        with open(os.path.join(path,file),'r',encoding='utf-8') as f:
            news = f.read()
            noun = [word for word, flag in pseg.lcut(news) if flag.startswith('n')]
            news = list(jieba.cut(news))
            news = [word for word in news if (word not in stopList) and (word not in noun)]  # 过滤停用词和名词
        docList.append(news)
    return None


def words2Vec(news,emotionList,stopList,posList,negList,mode=0):
    """
    新闻文本翻译成词向量
    :param news: 新闻文本
    :param emotionList: 情感词列表
    :param stopList: 停用词列表
    :param posList: 积极词列表
    :param negList: 消极词列表
    :param mode: int and [0,5)。对应不同的翻译文本的方法
    :return: list类型（方便之后的操作，例如，numpy.array()）
    """
    # 参数类型检查
    assert isinstance(stopList,list) and isinstance(emotionList,list),"类型不对。Function 'word2vec' at OperateDat.py"

    noun = [word for word,flag in pseg.lcut(news) if flag.startswith('n')] # 名词列表

    # 过滤停用词和名词
    newswords = list(jieba.cut(news))
    newswords = [word for word in newswords if (word not in stopList) and (word not in noun)]

    wordsVec = []
    # one-hot
    # time:O(n)
    if mode==0:
        for word in emotionList:
            if word in newswords:  wordsVec.append(1)
            else:  wordsVec.append(0)
    # frequency
    # time:O(n)
    elif mode==1:
        for word in emotionList:
            wordsVec.append(newswords.count(word))
    # tf-idf
    # time:O(2*n*n)
    elif mode==2:
        global docList # 引用加载后的全局变量
        docSum = len(docList) # 第一维len代表了文件数
        for word in emotionList:
            TF = 0
            IDF= 0
            times = 0
            for doc in docList:
                if word in doc: times+=1
            IDF = math.log10(docSum/abs(times+1))
            times = 0
            for doc in docList:
                times+=doc.count(word)
            TF = newswords.count(word)/(times+1)
            wordsVec.append(TF*IDF)

    return wordsVec

