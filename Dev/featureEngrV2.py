'''
    对每一条sent，生成相应的特征，特征使用.csv存储
'''
from typing import List, Dict
import os
import pickle
import gc
from collections import Counter
import jieba
from nltk.util import ngrams
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='Progress')

from lm import LanModel
import const


def getPikcle(path: str):
    """  """
    with open(path, 'rb') as f:
        v = pickle.load(f)
    return v


def countReptMx(words: List):
    """  """
    mx = 1
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


def langFeatures(X, params: Dict, template: str):
    """  """
    name = params['name']
    for args in params['args']:
        n, cut = args
        m = getPikcle(os.path.join(const.PKPATH, template.format(n, name, cut)))
        lm = LanModel(m, n)
        X['{}n_etp_{}_{}'.format(n, name, cut)] = X[cut].progress_apply(lambda x: lm.entropy(x) if lm.entropy(x) != float('inf') else -1)
        X['{}n_ppl_{}_{}'.format(n, name, cut)] = X[cut].progress_apply(lambda x: lm.perplexity(x) if lm.perplexity(x) != float('inf') else -1)
        del m, lm
        gc.collect()
    return X


def feature_engineering(X):
    """  """
    # 这里还是空格分词的
    X['sent'] = X['sent'].progress_apply(lambda x: str(x))
    # 分字列表
    X['chars'] = X['sent'].progress_apply(lambda x: x.split())
    # 这里做合并,因为原本用了分字的句子，但是这里计算有要用原句的，也有要用结巴分词的，所以只能先合并
    X['sent'] = X['chars'].progress_apply(lambda x: ''.join(x))
    # 这里使用jieba分词列表
    X['jieba'] = X['sent'].progress_apply(lambda x: [w for w in jieba.cut(x)])
    # len------------------------------
    X['len'] = X['sent'].progress_apply(lambda x: len(x))
    X.drop(['sent'], axis=1, inplace=True)
    # 字组成的
    X['char_rate'] = X['chars'].progress_apply(lambda x: len(set(x)))
    X['char_mx_rate'] = X['chars'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)
    # 词组成的
    X['pairs'] = X['chars'].progress_apply(lambda x: list(ngrams(x, 2)))
    X['pair_rate'] = X['pairs'].progress_apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)
    X['pair_mx_rate'] = X['pairs'].progress_apply(lambda x: Counter(x).most_common(1)[0][1] / len(x) if len(x) > 0 else 0)
    # 找出重复片段
    X['rept_mx'] = X['chars'].progress_apply(lambda x: countReptMx(x))
    X['rept_mx_rate'] = X['rept_mx'] / X['len']
    X['rept_mx_2'] = X['pairs'].progress_apply(lambda x: countReptMx(x))
    X['pairs_count'] = X['pairs'].progress_apply(lambda x: len(x))
    X['rept_mx_2_rate'] = X['rept_mx_2'] / X['pairs_count']

    X.drop(['pairs', 'rept_mx', 'rept_mx_2', 'pairs_count'], axis=1, inplace=True)

    # language model------------------------------
    params = {
        'pos': {'name': 'kenlm', 'args': [(2, 'chars'), (3, 'chars'),
                                          (2, 'jieba'), (3, 'jieba'), (1, 'jieba')]},
        'neg': {'name': 'paopao', 'args': [(2, 'chars'), (3, 'chars'),
                                           (2, 'jieba'), (3, 'jieba'), (1, 'jieba')]},
        's1': {'name': 'weibo', 'args': [(2, 'chars'), (3, 'chars'),
                                         (2, 'jieba'), (3, 'jieba'), (1, 'jieba')]},
        's2': {'name': 'sms', 'args': [(2, 'chars'), (3, 'chars'),
                                       (2, 'jieba'), (3, 'jieba'), (1, 'jieba')]}}
    template = 'lm_{}_{}_{}.pk'
    X = langFeatures(X, params['pos'], template)
    X = langFeatures(X, params['neg'], template)
    X = langFeatures(X, params['s1'], template)
    X = langFeatures(X, params['s2'], template)
    X.drop(['jieba', 'chars'], axis=1, inplace=True)

    return X


def extract(path: str, source: str, target: str):
    """  """
    # load data
    X = pd.read_csv(os.path.join(path, source))#[:100]
    print('Data shape : ', X.shape)
    print(X.head())

    # get features
    X_f = feature_engineering(X)
    X_f = X.drop(['id'], axis=1)  # 把sentence, id 列删除

    print(X_f[:5])
    print(X_f.columns.values.tolist())

    # save data
    X_f.to_csv(os.path.join(path, target), index=None)


def dropcol(path, source, target):
    """  """
    X = pd.read_csv(os.path.join(path, source))  # [:100]
    print('Data shape : ', X.shape)

    X.drop([], axis=1, inplace=True)

    print(X[:5])
    print(X.columns.values.tolist())

    X.to_csv(os.path.join(path, target), index=None)


def main():
    extract(const.DATAPATH, 'kenlm_paopao_chars_train_v3.csv', 'data_kenlm_paopao_train_v4.csv')
    #extract(const.DATAPATH, 'kenlm_paopao_chars_test_v3.csv', 'data_kenlm_paopao_test_v4.csv')


if __name__ == '__main__':
    main()
