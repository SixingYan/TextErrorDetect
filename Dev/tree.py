from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.tree.export import export_text
from sklearn.utils import shuffle

import pandas as pd
import os
import time

import const  # 用于记录常量，如文件夹路径


def valid(clf, path, source):
    X = pd.read_csv(os.path.join(path, source))
    X = shuffle(X)

    print('DATA Explore -----------')
    print(X['target'].value_counts())
    y = X['target'].values
    X = X.drop(['target'], axis=1)

    X.describe().to_csv(os.path.join(path, 'describe_{}.csv'.format(source)))

    print('DATA -------------------')
    print(X.columns.values.tolist())

    print('Score={:.4f}'.format(clf.score(X, y)))


def train_valid_dt(source1, source2):
    """ 决策树，就是使用这里的代码 """
    X_train, X_test, y_train, y_test = getData(const.DATAPATH, source1)
    print('starting...')
    stime = time.time()
    clf = DT(random_state=10)
    clf.fit(X_train, y_train)

    tree_text = export_text(clf, feature_names=X_train.columns.values.tolist(), max_depth=20)
    print('Tree Structure : ')
    print(tree_text)

    with open(os.path.join(const.DATAPATH, 'dt_structure_{}.txt'.format(source)), 'w', encoding='utf-8', errors='ignore') as f:
        f.write(tree_text)

    print('Feature importance : ')
    print(clf.feature_importances_)
    print('Time cost {:.2f} ||| Score={:.4f}'.format((time.time() - stime) / 60, clf.score(X_test, y_test)))

    valid(clf, const.DATAPATH, source2)

    return clf


def getData(path, source):
    """ 数据导入 """
    X = pd.read_csv(os.path.join(path, source))
    print(X.describe())

    #print('DATA Explore -----------')
    print(X['target'].value_counts())
    y = X['target'].values
    #X = X.drop(['target', 'id'], axis=1)
    X = X.drop(['target'], axis=1)
    X.describe().to_csv(os.path.join(path, 'describe_{}.csv'.format(source)))

    print('DATA -------------------')
    print(X.columns.values.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    return X_train, X_test, y_train, y_test


def train_valid_rf(source1, source2):
    """ 随机森林 """
    X_train, X_test, y_train, y_test = getData(const.DATAPATH, source1)  # <<< 训练/测试 的 分割
    print('starting...')
    stime = time.time()
    clf = RF(random_state=10)
    clf.fit(X_train, y_train)

    print('Time cost {:.2f} ||| Score={:.4f}'.format((time.time() - stime) / 60, clf.score(X_test, y_test)))

    valid(clf, const.DATAPATH, source2)


def main():
    pass
    source1 = 'data_kenlm_paopao_train_v3.csv'
    source2 = 'data_kenlm_paopao_test_v3.csv'
    # train_valid_dt(source1, source2)
    train_valid_rf(source1, source2)


if __name__ == '__main__':
    main()
    # test('data_kenlm_paopao_train_v3.csv')
