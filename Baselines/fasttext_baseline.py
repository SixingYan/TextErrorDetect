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
    return precision, recall


def train():
    # info
    n = 2
    source = 'kenlm_paopao_jieba_fasttext_v2.txt'
    target = 'klmpao_jieba_v2_ft_model_{}.ft'.format(n)
    reuse = False
    delete = True
    save = True
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
    if delete is True:
        cleantmp([X_train, X_test])

    # save 保存模型
    if save is True:
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
    n = 2
    way = 'skipgram'
    source = 'kenlm_paopao_jieba_fasttext_unspv.txt'
    target = 'klmpao_jieba_ftembd_{}_{}.vec'.format(way, n)
    reuse = False
    delete = False
    print('Train detail: n={} || way={} || source={} || target={} || reuse={} || delete={}'.format(n, way, source, target, reuse, delete))

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


def test():
    '''  '''
    source = 'klmpao_jieba_ft_model_3.ft'
    target = 'kenlm_paopao_jieba_v2_fasttext.txt'
    clf = FT.load_model(os.path.join(const.MODELPATH, source))
    size, precision, recall = clf.test(os.path.join(const.DATAPATH, target))
    print('accuracy score : ', precision)


def main():
    pass
    # test()
    train()
    # data_explore()

if __name__ == '__main__':
    main()
