from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time

import const


def getData(path, source):
    """  """
    X = pd.read_csv(os.path.join(path, source))
    y = X['target'].values
    X = X[['3n_etp_n_jieba']]
    #print(X.head())
    #X = X.drop(['target'], axis=1)
    X = X.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test


def main():
    path = 'D:/yansixing/ErrorDetection/Datasets/Data/'
    source = 'data_kenlm_paopao_v2.2.csv'
    X_train, X_test, y_train, y_test = getData(path, source)
    print('starting...')
    stime = time.time()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print('Time cost {:.2f} ||| Score={:.4f}'.format((time.time() - stime) / 60, clf.score(X_test, y_test)))

if __name__ == '__main__':
    main()
