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


'''
import nltk
brown = nltk.corpus.brown
text = brown.sents()

from nltk.lm import MLE
lm = MLE(1)

from nltk.lm.preprocessing import padded_everygram_pipeline
train, vocab = padded_everygram_pipeline(1, text)

lm.fit(train, vocab)

brown.sents()
[['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.'], ['The', 'jury', 'further', 'said', 'in', 'term-end', 'presentments', 'that', 'the', 'City', 'Executive', 'Committee', ',', 'which', 'had', 'over-all', 'charge', 'of', 'the', 'election', ',', '``', 'deserves', 'the', 'praise', 'and', 'thanks', 'of', 'the', 'City', 'of', 'Atlanta', "''", 'for', 'the', 'manner', 'in', 'which', 'the', 'election', 'was', 'conducted', '.'], ...]

# from nltk.util import bigrams
# 输入是ngram后的分词列表
# lm.entropy(bigrams(text[0]))
>>> list(bigrams(text[0]))
[('The', 'Fulton'), ('Fulton', 'County'), ('County', 'Grand'), ('Grand', 'Jury'), ('Jury', 'said'), ('said', 'Friday'), ('Friday', 'an'), ('an', 'investigation'), ('investigation', 'of'), ('of', "Atlanta's"), ("Atlanta's", 'recent'), ('recent', 'primary'), ('primary', 'election'), ('election', 'produced'), ('produced', '``'), ('``', 'no'), ('no', 'evidence'), ('evidence', "''"), ("''", 'that'), ('that', 'any'), ('any', 'irregularities'), ('irregularities', 'took'), ('took', 'place'), ('place', '.')]
'''


class LanModel(object):
    """
        用法是，再外部指定ngram
        m = MLE(2)
        lm = LanModel(m, n)
        lm
    """

    def __init__(self, lm, n):
        self.lm = lm
        self.n = n

    def entropy(self, chars: List):

        # def unigram(chars):
        #    return [(w,) for w in chars]
        #g = unigram(chars)
        #print('Get : ', g)
        # return self.lm.entropy(g)
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
    """"""
    df = pd.read_csv(os.path.join(path, source), encoding=encode)  # [:100]

    # print(df.head())
    df[col] = df[col].apply(lambda x: str(x))
    df[col] = df[col].apply(lambda x: x.split())
    # print(df['sent'])

    return df[col].values.tolist()

'''
from itertools import chain
def test0():
    n = 1n
    name = 'kenlm_chars.csv'
    chars = getData(const.DATAPATH, name)
    
    c = list(chain(*chars))
    #chars = []
    pass
'''


def train(n=None):
    """"""
    n = 1

    print('dealing n = ', n)

    name = 'paopao_pos.csv'
    print('loading : ', name)

    chars = getData(const.DATAPATH, name, col='pos')  # <<< 列表，列表里面是用分词（字）列表

    # train
    print('start......')
    stime = time.time()
    m = getModel(n, chars)
    print('Time cost : ', (time.time() - stime) / 60)
    print('Vocab size : ', len(m.vocab))

    # save
    mname = 'lm_{0}_paopao_pos.pk'.format(n)
    with open(os.path.join(const.PKPATH, mname), 'wb') as f:
        pickle.dump(m, f)


def main():

    # train()

    n = 1
    name = 'kenlm_pos.csv'
    chars = getData(const.DATAPATH, name)  # <<< 列表，列表里面是用分词（字）列表

    mname = 'lm_{0}_kenlm_jieba.pk'.format(n)
    with open(os.path.join(const.PKPATH, mname), 'rb') as f:
        m = pickle.load(f)

    # test
    print('Test......')
    lm = LanModel(m, n)
    #lst = [w[0] for w in list(lm.lm.vocab)]
    #print('嗯 ? ', chars[0][1] in lst)
    #print('Look up : ', lm.lm.vocab.lookup([list(lm.lm.vocab)[0][0]]))
    #print(list(lm.lm.vocab)[0][0] == '烦')
    print('Input : ', chars[1])
    print('Look up : ', lm.lm.vocab.lookup(chars[1]))
    print('Count : ', lm.lm.counts[chars[1][1]])
    #print('Vocab : ', list(lm.lm.vocab))
    #print('Looking ', lm.lm.counts['嗯'])
    #print('The same? : ', '嗯'==chars[1][1])

    val = lm.entropy(chars[0])
    print('Test entropy : ', val)


def show_feature_to_label():
    """ 计算相关度？还需要思考 """
    pass


if __name__ == '__main__':
    train()
