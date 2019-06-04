from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
from sklearn.tree.export import export_text, export_graphviz

from sklearn.utils import shuffle


from sklearn.ensemble import RandomForestClassifier as RF

import pandas as pd
import os
import time

import const  # 用于记录常量，如文件夹路径
'''
['2n_etp_p', '3n_etp_p', '3n_ppl_p', '2n_etp_n', '2n_ppl_n', '3n_etp_n', '3n_ppl_n', '1n_etp_p_jieba', '2n_etp_p_jieba', '2n_ppl_p_jieba', '3n_etp_p_jieba', '3n_ppl_p_jieba', '2n_etp_n_jieba', '2n_ppl_n_jieba', '3n_etp_n_jieba', '3n_ppl_n_jieba']

'''


def valid(clf, path, source):
    X = pd.read_csv(os.path.join(path, source))
    X = shuffle(X)

    print('DATA Explore -----------')
    print(X['target'].value_counts())
    y = X['target'].values
    X = X.drop(['target', 'rept_mx_2'], axis=1)

    X.describe().to_csv(os.path.join(path, 'describe_{}.csv'.format(source)))

    print('DATA -------------------')
    print(X.columns.values.tolist())

    print('Score={:.4f}'.format(clf.score(X, y)))


def test(source):
    """ 决策树，就是使用这里的代码 """
    X_train, X_test, y_train, y_test = getData(const.DATAPATH, source)
    print('starting...')
    stime = time.time()
    clf = DT(random_state=10)
    clf.fit(X_train, y_train)

    tree_text = export_text(clf, feature_names=X_train.columns.values.tolist(), max_depth=100)
    print('Tree Structure : ')
    print(tree_text)

    with open(os.path.join(const.DATAPATH, 'dt_structure_{}.txt'.format(source)), 'w', encoding='utf-8', errors='ignore') as f:
        f.write(tree_text)

    print('Feature importance : ')
    print(clf.feature_importances_)
    print('Time cost {:.2f} ||| Score={:.4f}'.format((time.time() - stime) / 60, clf.score(X_test, y_test)))

    # 输出树结构的图像文件
    # with open(os.path.join(const.DATAPATH, 'DecisionTree.dot'), "w") as f:
    #    res = export_graphviz(clf, feature_names=X_train.columns.values.tolist(), out_file=f)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test


def main():
    """ 随机森林 """
    X_train, X_test, y_train, y_test = getData(const.DATAPATH, 'data_kenlm_paopao_v21.csv')  # <<< 训练/测试 的 分割
    print('starting...')
    stime = time.time()
    clf = RF(random_state=10)
    clf.fit(X_train, y_train)

    print('Time cost {:.2f} ||| Score={:.4f}'.format((time.time() - stime) / 60, clf.score(X_test, y_test)))


def getPNG():
    pass
    import pydot
    (graph,) = pydot.graph_from_dot_file(os.path.join(const.DATAPATH, 'DecisionTree.dot'))
    graph.write_png(os.path.join(const.DATAPATH, 'DecisionTree.png'))

if __name__ == '__main__':
    #clf = test()
    #valid(clf, const.DATAPATH, 'data_un_normal.csv')

    test('data_un_normal.csv')
