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

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree.export import export_text


import pandas as pd
import os
import time

import const


def getData(path, source):
    """  """
    X = pd.read_csv(os.path.join(path, source))
    
    #print(X['target'].value_counts())
    #raise
    
    y = X['target'].values
    X = X.drop(['target'], axis=1)
    X = X.sample(frac=1)
    
    
    #print(X[:3])
    #print(X.columns.values.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    #print(X_train[:3])
    #print(X_test[:3])
    return X_train, X_test, y_train, y_test
    # splits = list(StratifiedKFold(n_splits=split_num, shuffle=True, random_state=SEED).split(train_X, train_y))


def trainDT(X, y):
    """ decision tree """
    tree_clf = DecisionTreeClassifier(max_depth=5, splitter="random", max_leaf_nodes=50)
    tree_clf.fit(X, y)
    return tree_clf



def trainRF(X, y):
    from sklearn.ensemble import RandomForestClassifier
    rnd_clf = RandomForestClassifier()
    rnd_clf.fit(X, y)
    return rnd_clf



def eval(clf, X, y):
    """ """
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)


def main():
    
    X_train, X_test, y_train, y_test = getData(const.DATAPATH, 'data_kenlm_paopao.csv')  # <<< 训练/测试 的 分割
    
    print('start ......')
    
    stime = time.time()
    tree_clf = trainDT(X_train, y_train)
    print('train time : ', (time.time() - stime) / 60)
    
    print('accuracy score : ', eval(tree_clf, X_test, y_test))
    
    tree_text = export_text(tree_clf, feature_names=X_train.columns.values.tolist())
    print('Tree Structure : ')
    print(tree_text)

if __name__ == '__main__':
    main()
