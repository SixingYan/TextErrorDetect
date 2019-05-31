# encoding = utf-8

import random
from typing import List
import time
import os

import fasttext
#import fastText.FastText as ff
import const

target1 = 'train_fasttext.tmp'
target2 = 'test_fasttext.tmp'


def cleantmp(paths):
    """  """
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)


def getData(path, source, ratio=0.8):
    """  """
    train, test = [], []
    with open(os.path.join(path, source), 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if random.random() < 0.5:
                
                line = line.replace('__label0', '__label__1')
                #print('replace ： ', line)

            if random.random() < 0.8:
                train.append(line)
            else:
                test.append(line)

    with open(os.path.join(path, target1), 'w', encoding='utf-8', errors='ignore') as f:
        f.writelines(train)

    with open(os.path.join(path, target2), 'w', encoding='utf-8', errors='ignore') as f:
        f.writelines(test)

    return os.path.join(path, target1), os.path.join(path, target2)


def trainFT(X_path: str):
    """  """
    clf = fasttext.supervised(X_path, 'fasttext_model', label_prefix='__label__')
    return clf


def eval(clf, X_path: str):
    """  """
    return clf.test(X_path).precision


def main():

    X_train, X_test = getData(const.DATAPATH, 'kenlm_paopao_fasttext.txt')

    stime = time.time()
    clf = trainFT(X_train)
    print('train time : ', (time.time() - stime) / 60)

    texts = ['不 乱 来 ， 这 个 可 能 进 来 我 没 哦 玩 游 戏 。', '不 乱 来 ， 这 个 可 能 进 来 我 没 哦 玩 游 戏 。']
    labels = clf.predict(texts)
    print('predict : ', str(labels))
    print('accuracy score : ', eval(clf, X_test))

    #cleantmp([X_train, X_test])

if __name__ == '__main__':
    main()

'''
    classifier = fasttext.supervised(train, 'classifier.model', lable_prefix='__lable__')
    result = classifier.test(test)
    print("P@1:", result.precision)  # 准确率
    print("R@2:", result.recall)  # 召回率
    print("Number of examples:", result.nexamples)  # 预测错的例子
'''
