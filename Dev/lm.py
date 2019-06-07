from typing import List
import time
import pickle
import const
import pandas as pd
import os

# http://www.nltk.org/api/nltk.lm.html?highlight=lm#module-nltk.lm
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams


class LanModel(object):
    """
    """

    def __init__(self, lm, n):
        self.lm = lm
        self.n = n

    def entropy(self, chars: List):
        return self.lm.entropy(ngrams(chars, self.n,
                                      True, True, '<s>', '</s>'))

    def perplexity(self, chars: List)->float:
        return self.lm.perplexity(ngrams(chars, self.n,
                                         True, True, '<s>', '</s>'))


def getModel(n: int, text: List):
    """ 在这里训练模型 """
    lm = MLE(n)

    # get train, vocab
    train, vocab = [], set([])
    for t in text:
        g = ngrams(t, n,
                   pad_left=True, pad_right=True,
                   left_pad_symbol='<s>', right_pad_symbol='</s>')
        g = list(g)
        vocab = vocab | set(t)
        train.append(g)

    lm.fit(train, vocabulary_text=list(vocab))

    return lm


def getEveryModel(n: int, text: List, ngrams):
    """ get mixed-n model """
    lm = MLE(n)

    train, vocab = padded_everygram_pipeline(n, text)

    lm.fit(train, vocab)

    return lm


def getData(path: str, source: str, encode: str='UTF-8', col='sent')->List:
    """ """
    df = pd.read_csv(os.path.join(path, source), encoding=encode)

    df[col] = df[col].apply(lambda x: str(x))
    df[col] = df[col].apply(lambda x: x.split())

    return df[col].values.tolist()


def train(fname: str, n=2, cut='chars'):
    """"""
    print('dealing n = ', n)

    name = '{}_{}.csv'.format(fname, cut)
    print('loading : ', name)
    chars = getData(const.DATAPATH, name)

    # train
    print('start......')
    stime = time.time()
    m = getModel(n, chars)
    print('Time cost : ', (time.time() - stime) / 60)
    print('Vocab size : ', len(m.vocab))

    # save
    mname = 'lm_{}_{}_{}.pk'.format(n, fname, cut)
    with open(os.path.join(const.PKPATH, mname), 'wb') as f:
        pickle.dump(m, f)


def main():
    train('paopao', n=1, cut='jieba')

if __name__ == '__main__':
    main()
