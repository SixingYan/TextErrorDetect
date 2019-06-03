'''
在这里训练embedding和word2vec
'''

from typing import List
import os
import time
import pandas as pd

from gensim.models import Word2Vec
from tqdm import tqdm
tqdm.pandas(desc='Progress')

import const


def getData(path, source)->List:

    df = pd.read_csv(os.path.join(path, source))  # [:100]
    df['sent'] = df['sent'].progress_apply(lambda x: str(x))
    df['words'] = df['sent'].progress_apply(lambda x: x.split())
    print(df.head())
    return df


def getW2V(path1: str, path2: str, target1: str, target2: str, df):
    """ [['a','asdf', 'sdf',...],...] 分好词的句子列表 """
    sents = df['words'].values.tolist()
    # print(sents)
    w2v = Word2Vec(sents, min_count=3)
    w2v.wv.save_word2vec_format(os.path.join(path1, target1))

    # 建立对应向量
    print('Size is :', len(list(w2v.wv.vocab.keys())))

    df['sent2vec'] = df['words'].progress_apply(lambda words: [max([w2v.wv.similarity(w, ws) for ws in list(set(words)) if ws in w2v.wv.vocab.keys()]) for w in w2v.wv.vocab.keys()])
    #sent2wordlist = [[max([w2v.similarity(w, ws) for ws in sent]) for w in wordlist] for sent in sents]

    df[['sent2vec']].to_csv(os.path.join(path2, target2), index=None)

def getW2V_add():
    """ [['a','asdf', 'sdf',...],...] 分好词的句子列表 """
    sents = df['words'].values.tolist()
    # print(sents)
    w2v = Word2Vec(sents, min_count=1, size=200, window=3)
    w2v.wv.save_word2vec_format(os.path.join(path1, target1))



def main():
    """  """
    df = getData(const.DATAPATH, 'kenlm_paopao_jieba_v2.csv')

    getW2V(const.MODELPATH, const.DATAPATH, 'klmpao_jieba_v2.w2v', 'klmpao_jieba_v2_w2v.csv', df)

if __name__ == '__main__':
    main()
