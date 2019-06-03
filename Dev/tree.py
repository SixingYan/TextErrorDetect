'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression()


rnd_clf = RandomForestClassifier()


svm_clf = SVC()
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

rnd_clf.fit()

'''

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree.export import export_text

from sklearn.ensemble import RandomForestClassifier as RF

import pandas as pd
import os
import time

import const


def trainDT(X, y):
    """ decision tree """
    tree_clf = DT(max_depth=5, splitter="random", max_leaf_nodes=50)
    tree_clf.fit(X, y)
    return tree_clf


def eval(clf, X, y):
    """ """
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)


def test():
    """  """
    X_train, X_test, y_train, y_test = getData(const.DATAPATH, 'data_kenlm_paopao.csv')  # <<< 训练/测试 的 分割

    print('start ......')

    stime = time.time()
    tree_clf = trainDT(X_train, y_train)
    print('train time : ', (time.time() - stime) / 60)

    print('accuracy score : ', eval(tree_clf, X_test, y_test))

    tree_text = export_text(tree_clf, feature_names=X_train.columns.values.tolist())
    print('Tree Structure : ')
    print(tree_text)

# 数据导入


def getData(path, source):
    """  """
    X = pd.read_csv(os.path.join(path, source))
    y = X['target'].values
    X = X.drop(['target'], axis=1)
    X = X.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test


def main():
    path = 'D:/yansixing/ErrorDetection/Datasets/Data/'
    source = 'data_kenlm_paopao_v2.2.csv'
    X_train, X_test, y_train, y_test = getData(path, source)

    print('starting...')
    stime = time.time()
    # rf = RF(n_estimators=28, max_depth=47, min_samples_split=10,
    #        min_samples_leaf=5, max_features=29, oob_score=True, random_state=10)
    #clf = RF(random_state=10)
    clf = DT(random_state=10)
    clf.fit(X_train, y_train)

    print('Time cost {:.2f} ||| Score={:.4f}'.format((time.time() - stime) / 60, clf.score(X_test, y_test)))


if __name__ == '__main__':
    main()
