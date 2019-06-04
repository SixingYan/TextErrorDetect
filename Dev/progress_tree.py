# encoding = utf-8
import csv
import re
import random
from typing import List, Tuple
import pandas as pd
import os
import pickle
from pandas.core.frame import DataFrame

import demjson
import jieba
from nltk.util import ngrams

import const



import gc
from collections import Counter

from tqdm import tqdm
tqdm.pandas(desc='Progress')

from lm import LanModel
import const


def load_data():
    pass
    
def getPikcle(path: str):
    """  """
    with open(path, 'rb') as f:
        v = pickle.load(f)
    return v


def countReptMx(words: List):
    """  """
    mx = -1
    c = 1
    pre = None
    for i, w in enumerate(words):
        if pre is None:
            pre = w
            continue
        if w == pre:
            c += 1
        else:
            if c > mx:
                mx = c
            pre = w
            c = 1
        if i == len(words) - 1:
            if c > mx:
                mx = c
    return mx


def feature_engineering(X):
    """  """
    # 这里还是空格分词的
    X['sent'] = X['sent'].progress_apply(lambda x: str(x))

    # 空格分字
    X['sent_chars'] = X['sent'].progress_apply(lambda x: str(x))

    # 分字列表
    X['chars'] = X['sent'].progress_apply(lambda x: x.split())

    # 这里做合并,因为原本用了分字的句子，但是这里计算有要用原句的，也有要用结巴分词的，所以只能先合并
    X['sent'] = X['chars'].progress_apply(lambda x: ''.join(x))

    # 这里使用jieba分词列表
    X['jieba'] = X['sent'].progress_apply(lambda x: [w for w in jieba.cut(x)])

    # 空格分词
    X['sent_jieba'] = X['jieba'].progress_apply(lambda x: ' '.join(x))

    # len------------------------------
    X['len'] = X['sent'].progress_apply(lambda x: len(x))
    X.drop(['sent'], axis=1, inplace=True)

    # chinese char------------------------------
    #X['char_rate'] = X['chars'].progress_apply(lambda x: len(set(x)))
    #X['char_mx_rate'] = X['chars'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)

    X['pairs'] = X['chars'].progress_apply(lambda x: list(ngrams(x, 2)))
    #X['pair_rate'] = X['pairs'].progress_apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)
    #X['pair_mx_rate'] = X['pairs'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)

    # 找出重复片段
    #X['rept_mx'] = X['chars'].progress_apply(lambda x: countReptMx(x))
    #X['rept_mx_rate'] = X['rept_mx'] / X['len']

    # X['rept_mx_2'] = X['pairs'].progress_apply(lambda x: countReptMx(x))

    X.drop(['pairs', 'len'], axis=1, inplace=True)

    # language model------------------------------
    model_p2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_kenlm_chars_v2.pk'))
    model_n2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_paopao_chars_v2.pk'))
    model_p3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_kenlm_chars_v2.pk'))
    model_n3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_paopao_chars_v2.pk'))
    l2p = LanModel(model_p2, 2)
    l3p = LanModel(model_p3, 3)
    l2n = LanModel(model_n2, 2)
    l3n = LanModel(model_n3, 3)
    X['2n_etp_p'] = X['chars'].progress_apply(lambda x: l2p.entropy(x) if l2p.entropy(x) != float('inf') else -1)
    #X['2n_ppl_p'] = X['chars'].progress_apply(lambda x: l2p.perplexity(x) if l2p.perplexity(x) != float('inf') else -1)
    X['3n_etp_p'] = X['chars'].progress_apply(lambda x: l3p.entropy(x) if l3p.entropy(x) != float('inf') else -1)
    X['3n_ppl_p'] = X['chars'].progress_apply(lambda x: l3p.perplexity(x) if l3p.perplexity(x) != float('inf') else -1)
    X['2n_etp_n'] = X['chars'].progress_apply(lambda x: l2n.entropy(x) if l2n.entropy(x) != float('inf') else -1)
    X['2n_ppl_n'] = X['chars'].progress_apply(lambda x: l2n.perplexity(x) if l2n.perplexity(x) != float('inf') else -1)
    X['3n_etp_n'] = X['chars'].progress_apply(lambda x: l3n.entropy(x) if l3n.entropy(x) != float('inf') else -1)
    X['3n_ppl_n'] = X['chars'].progress_apply(lambda x: l3n.perplexity(x) if l3n.perplexity(x) != float('inf') else -1)
    del l2p, l3p, l2n, l3n, model_p2, model_p3, model_n2, model_n3
    X.drop(['chars', 'sent_chars'], axis=1, inplace=True)
    gc.collect()

    # 基于语言模型的计量
    model_p1 = getPikcle(os.path.join(const.PKPATH, 'lm_1_kenlm_jieba_v2.pk'))
    model_p2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_kenlm_jieba_v2.pk'))
    model_n2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_paopao_jieba_v2.pk'))
    model_p3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_kenlm_jieba_v2.pk'))
    model_n3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_paopao_jieba_v2.pk'))
    l1p = LanModel(model_p1, 1)
    l2p = LanModel(model_p2, 2)
    l3p = LanModel(model_p3, 3)
    l2n = LanModel(model_n2, 2)
    l3n = LanModel(model_n3, 3)
    #X['1n_ppl_p_jieba'] = X['jieba'].progress_apply(lambda x: l1p.perplexity(x) if l1p.perplexity(x) != float('inf') else -1)
    X['1n_etp_p_jieba'] = X['jieba'].progress_apply(lambda x: l1p.entropy(x) if l1p.entropy(x) != float('inf') else -1)
    X['2n_etp_p_jieba'] = X['jieba'].progress_apply(lambda x: l2p.entropy(x) if l2p.entropy(x) != float('inf') else -1)
    X['2n_ppl_p_jieba'] = X['jieba'].progress_apply(lambda x: l2p.perplexity(x) if l2p.perplexity(x) != float('inf') else -1)
    X['3n_etp_p_jieba'] = X['jieba'].progress_apply(lambda x: l3p.entropy(x) if l3p.entropy(x) != float('inf') else -1)
    X['3n_ppl_p_jieba'] = X['jieba'].progress_apply(lambda x: l3p.perplexity(x) if l3p.perplexity(x) != float('inf') else -1)
    X['2n_etp_n_jieba'] = X['jieba'].progress_apply(lambda x: l2n.entropy(x) if l2n.entropy(x) != float('inf') else -1)
    X['2n_ppl_n_jieba'] = X['jieba'].progress_apply(lambda x: l2n.perplexity(x) if l2n.perplexity(x) != float('inf') else -1)
    X['3n_etp_n_jieba'] = X['jieba'].progress_apply(lambda x: l3n.entropy(x) if l3n.entropy(x) != float('inf') else -1)
    X['3n_ppl_n_jieba'] = X['jieba'].progress_apply(lambda x: l3n.perplexity(x) if l3n.perplexity(x) != float('inf') else -1)
    del l1p, l2p, l3p, l2n, l3n, model_p1, model_p2, model_p3, model_n2, model_n3
    X.drop(['jieba', 'sent_jieba'], axis=1, inplace=True)
    gc.collect()

    return X


def extract(path: str, source: str, target: str):
    """  """
    # load data
    X = pd.read_csv(os.path.join(path, source))  # [:100]
    print('Data shape : ', X.shape)

    # get features
    X_f = feature_engineering(X)
    # X_f = X.drop(['sent', 'id'], axis=1)  # 把sentence, id 列删除

    print(X_f[:5])
    print(X_f.columns.values.tolist())

    # save data
    X_f.to_csv(os.path.join(path, target), index=None)


def dropcol(path, source, target):
    """  """
    X = pd.read_csv(os.path.join(path, source))  # [:100]
    print('Data shape : ', X.shape)

    X.drop(['len', 'char_rate', 'char_mx_rate', 'pair_rate', 'pair_mx_rate', 'rept_mx', 'rept_mx_rate', '2n_ppl_p', '1n_ppl_p_jieba'], axis=1, inplace=True)

    print(X[:5])
    print(X.columns.values.tolist())

    X.to_csv(os.path.join(path, target), index=None)


# --------------
# 通用工具

def parse_single(sent: str)->List:
    """ 把中文句子转化成字的列表 """
    return [w.strip() for w in sent if len(w.strip()) > 0]



def keepcn(s: str):
    """ 只保留中文 """
    return ''.join(w.strip() for w in re.findall(r'[\u4e00-\u9fa5]', s) if len(w.strip()) > 0)


def getPikcle(path: str):
    """  """
    with open(path, 'rb') as f:
        v = pickle.load(f)
    return v


def main():
    pass

if __name__ == '__main__':
    main()
