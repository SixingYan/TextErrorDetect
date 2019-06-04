'''
在这里训练embedding和word2vec
'''

from typing import List
import os
import time
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from tqdm import tqdm
tqdm.pandas(desc='Progress')

import const


def getData(path, source)->List:

    df = pd.read_csv(os.path.join(path, source))  # [:1000]
    df['sent'] = df['sent'].progress_apply(lambda x: str(x))
    df['words'] = df['sent'].progress_apply(lambda x: x.split())
    # print(df.head())
    return df


def getVec(words: List, w2v):
    ''' 去重复，去oov '''
    vlen = len(w2v.wv.vocab.keys())
    words = [w for w in list(set(words)) if w in w2v.wv.vocab.keys()]
    if words == []:
        return [0] * vlen
    res = []
    for w in list(w2v.wv.vocab.keys()):
        res.append(max(w2v.wv.similarity(w, x) for x in words))
    return res


def getW2V(path: str, target: str, df):
    """ [['a','asdf', 'sdf',...],...] 分好词的句子列表 """
    sents = df['words'].values.tolist()
    # print(sents)
    w2v = Word2Vec(sents, min_count=1, size=100, window=3)
    w2v.save(os.path.join(path, target)) 

def getWord2List(sourcepath: str, targetpath: str, df):
    """  """
    w2v = Word2Vec.load(sourcepath)
    # 建立对应向量
    print('Size is :', len(list(w2v.wv.vocab.keys())))
    df['vec'] = df['words'].progress_apply(lambda x: getVec(x, w2v))
    df[['vec']].to_csv(targetpath, index=None)


def getMax(words: List, w2v):
    """  """
    words = [w for w in list(set(words)) if w in w2v.wv.vocab.keys()]
    arrs = np.stack([w2v.wv[w] for w in words], axis=0)
    return np.max(arrs, axis=0)


def getMin(words: List, w2v):
    """  """
    words = [w for w in list(set(words)) if w in w2v.wv.vocab.keys()]
    arrs = np.stack([w2v.wv[w] for w in words], axis=0)
    return np.min(arrs, axis=0)


def getMean(words: List, w2v):
    """  """
    words = [w for w in list(set(words)) if w in w2v.wv.vocab.keys()]
    if words == []:
        return np.zeros(w2v.vector_size)
    arrs = np.stack([w2v.wv[w] for w in words], axis=0)
    return np.mean(arrs, axis=0)


def getWord2Vector(sourcepath, targetpath, df):
    pass
    w2v = Word2Vec.load(sourcepath)
    df['vec'] = df['words'].progress_apply(lambda x: getMean(x, w2v))
    df[['vec']].to_csv(targetpath, index=None)


'''
def getW2V_add():
    """ [['a','asdf', 'sdf',...],...] 分好词的句子列表 """
    sents = df['words'].values.tolist()
    # print(sents)
    w2v = Word2Vec(sents, min_count=1, size=200, window=3)
    w2v.wv.save_word2vec_format(os.path.join(path1, target1))
'''


def main():
    """  """
    df = getData(const.DATAPATH, 'kenlm_paopao_jieba_v2.csv')
    print('starting...')
    #getW2V(const.MODELPATH, 'klmpao_jieba_300_v2.w2v', df)
    #getWord2List(os.path.join(const.MODELPATH, 'klmpao_jieba_v2.w2v'), os.path.join(const.DATAPATH, 'klmpao_jieba_w2list_v2.vec'), df)
    #getWord2Vector(os.path.join(const.MODELPATH, 'klmpao_jieba_v2.w2v'), os.path.join(const.DATAPATH, 'klmpao_jieba_w2vmean_v2.vec'), df)
    #getWord2Vector(const.DATAPATH, 'klmpao_jieba_w2vmax_v2.vec', df)

if __name__ == '__main__':
    main()
