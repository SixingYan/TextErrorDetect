# encoding = utf-8

from typing import List
import time
import os

import const

from sklearn.model_selection import train_test_split
import fastText.FastText as FT


target1 = 'train_fasttext.tmp'
target2 = 'test_fasttext.tmp'


def cleantmp(paths):
    """  """
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)


def getData(path, source, testratio=0.2, reuse=False):
    """  """
    tpath1, tpath2 = os.path.join(path, target1), os.path.join(path, target2)
    if reuse is True:
        return tpath1, tpath2

    train, test = [], []
    with open(os.path.join(path, source), 'r', encoding='utf-8', errors='ignore') as f:
        train, test = train_test_split(list(f.readlines()), test_size=testratio, random_state=10)

    with open(tpath1, 'w', encoding='utf-8', errors='ignore') as f:
        f.writelines(train)

    with open(tpath2, 'w', encoding='utf-8', errors='ignore') as f:
        f.writelines(test)

    return tpath1, tpath2


def trainFT(path: str, n=1):
    """  """
    clf = FT.train_supervised(path, epoch=100, dim=100, wordNgrams=n, label='__label__', loss='softmax')
    return clf


def eval(clf, path: str):
    """ 
    classifier.predict('文本') #输出改文本的预测结
    """
    size, precision, recall = clf.test(path)
    # print(res)
    return precision, recall


def train():
    # info
    n = 1
    source = 'kenlm_paopao_jieba_fasttext.txt'
    target = 'klmpao_jieba_ft_model_{}.ft'.format(n)
    reuse = False
    delete = False
    print('Train detail: n={} || source={} || target={} || reuse={} || delete={}'.format(n, source, target, reuse, delete))

    # loading
    X_train, X_test = getData(const.DATAPATH, source, reuse=reuse)

    # train
    print('start...')
    stime = time.time()
    clf = trainFT(X_train, n=n)
    print('train time : ', (time.time() - stime) / 60)

    # Eval
    precision, recall = eval(clf, X_test)
    print('precision={:.4f}, recall={:.4f}'.format(precision, recall))

    # test
    #texts = ['不 乱 来 ， 这 个 可 能 进 来 我 没 哦 玩 游 戏 。', '不 乱 来 ， 这 个 可 能 进 来 我 没 哦 玩 游 戏 。']
    #print('predict labels: ', str([clf.predict(t) for t in texts]))
    #print('accuracy score : ', eval(clf, X_test))

    # clean tmp
    if delete:
        cleantmp([X_train, X_test])

    # save 保存模型
    clf.save_model(os.path.join(const.MODELPATH, target))
    # FT.load_model(path)


def data_explore():
    path = const.DATAPATH
    tpath1, tpath2 = os.path.join(path, target1), os.path.join(path, target2)

    print('Train Data : ')
    with open(tpath1, 'r', encoding='utf-8', errors='ignore') as f:
        text = str(f.read())
        print('__label__0 # :', text.count('__label__0'))
        print('__label__1 # :', text.count('__label__1'))

    print('Test Data : ')
    with open(tpath2, 'r', encoding='utf-8', errors='ignore') as f:
        text = str(f.read())
        print('__label__0 # :', text.count('__label__0'))
        print('__label__1 # :', text.count('__label__1'))


def trainUnSpv(path: str, n=1, model='skipgram'):
    """  """
    m = FT.train_unsupervised(path, epoch=100, dim=100, wordNgrams=n, model=model)
    return m


def train_unsupervised():
    """  """
    pass
    n = 1
    way = 'skipgram'
    source = 'kenlm_paopao_chars_fasttext_unspv.txt'
    target = 'klmpao_chars_ftembd_{}_{}.vec'.format(way, n)
    reuse = False
    delete = False
    print('Train detail: n={} || way={} || source={} || target={} || reuse={} || delete={}'.format(n, way, source, target, reuse, delete))

    # loading
    #X_train, X_test = getData(const.DATAPATH, source, reuse=reuse)

    # train
    print('start...')
    stime = time.time()
    model = trainUnSpv(os.path.join(const.DATAPATH, source), n=n, model=way)
    print('train time : ', (time.time() - stime) / 60)

    # test
    # texts = ['不','乱']
    # print(model.get_word_vector('不'))

    # clean tmp
    if delete:
        cleantmp([X_train, X_test])

    # save 保存模型
    model.save_model(os.path.join(const.MODELPATH, target))
    # FT.load_model(path)


'''
import fasttext

def trainFT(X_path: str, M_path: str, n: int=1, dim:int=200, epoch:int=10):
    """  """
    clf = fasttext.supervised(X_path, M_path, word_ngrams=n, epoch=5, dim=10)
    return clf

def eval(clf, X_path: str):
    """  """
    return clf.test(X_path).precision

def train():
    # info
    n = 1
    source = 'kenlm_paopao_chars_fasttext.txt'
    target = 'klmpao_chars_ft_model_{}'.format(n)
    reuse = False
    delete = False
    print('Train detail: n={} || source={} || target={} || reuse={} || delete={}'.format(n, source, target, reuse, delete))

    # loading
    X_train, X_test = getData(const.DATAPATH, source, reuse=reuse)

    # train
    stime = time.time()
    clf = trainFT(X_train, os.path.join(const.MODELPATH, target), n=n)
    print('train time : ', (time.time() - stime) / 60)

    texts = ['不 乱 来 ， 这 个 可 能 进 来 我 没 哦 玩 游 戏 。', '不 乱 来 ， 这 个 可 能 进 来 我 没 哦 玩 游 戏 。']
    labels = clf.predict(texts)
    print('predict : ', str(labels))
    print('accuracy score : ', eval(clf, X_test))

    if delete:
        cleantmp([X_train, X_test])
'''


def main():

    train_unsupervised()
    # data_explore()

if __name__ == '__main__':
    main()

'''
    classifier = fasttext.supervised(train, 'classifier.model', lable_prefix='__lable__')
    result = classifier.test(test)
    print("P@1:", result.precision)  # 准确率
    print("R@2:", result.recall)  # 召回率
    print("Number of examples:", result.nexamples)  # 预测错的例子
'''
