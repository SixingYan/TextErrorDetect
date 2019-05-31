from typing import Tuple
import pandas as pd
import os
import numpy as np
import time

# https://www.cnblogs.com/jin-liang/p/9638197.html
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

path = 'D:/yansixing/ErrorDetection/Datasets/Data/'
source = 'data_kenlm_paopao_v3.csv'
# 数据导入


def getData(path: str, source: str):
    """  """
    X = pd.read_csv(os.path.join(path, source))
    y = X['target'].values
    X = X.drop(['target'], axis=1)
    X = X.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = getData(path, source)


def roc_auc(dt, X, y):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y, dt.predict(X))
    return auc(false_positive_rate, true_positive_rate)


def get_roc_auc(dt)->Tuple:
    """  """
    dt.fit(X_train, y_train)
    return roc_auc(dt, X_train, y_train), roc_auc(dt, X_test, y_test)  # roc_auc1, roc_auc2


stime = time.time()
dt = DT()
dt.fit(X_train, y_train)
auc_val = roc_auc(dt, X_test, y_test)
del dt
print('Time cost : ', (time.time() - stime) / 60)
print('baseline, AUC : ', auc_val)


stime = time.time()
dt = DT(class_weight='balanced')
dt.fit(X_train, y_train)
auc_val = roc_auc(dt, X_test, y_test)
del dt
print('Time cost : ', (time.time() - stime) / 60)
print('weight balanced, AUC : ', auc_val)


stime = time.time()
dt = DT(splitter='best')
dt.fit(X_train, y_train)
auc_val = roc_auc(dt, X_test, y_test)
del dt
print('Time cost : ', (time.time() - stime) / 60)
print("splitter='best', AUC : ", auc_val)


stime = time.time()
dt = DT(splitter='random')
dt.fit(X_train, y_train)
auc_val = roc_auc(dt, X_test, y_test)
del dt
print('Time cost : ', (time.time() - stime) / 60)
print("splitter='random', AUC : ", auc_val)


max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results, test_results = [], []

for max_depth in max_depths:
    print('_______________________________')
    print('Train max_depth : ', max_depth)
    stime = time.time()
    dt = DT(max_depth=max_depth)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)
    del dt
    print('Time cost : ', (time.time() - stime) / 60)
    print("max_depth={0}, AUC : ".format(max_depth), roc_auc2)


line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


min_samples_splits = np.linspace(0.005, 0.5, 100, endpoint=True)
train_results, test_results = [], []

for min_samples_split in min_samples_splits:
    print('_______________________________')
    print('Train min_samples_split : ', min_samples_split)
    stime = time.time()
    dt = DT(min_samples_split=min_samples_split)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)
    del dt
    print('Time cost : ', (time.time() - stime) / 60)
    print("min_samples_split={0}, AUC : ".format(min_samples_split), roc_auc2)


line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


min_samples_leafs = np.linspace(0.0001, 0.01, 100, endpoint=True)
train_results, test_results = [], []

for min_samples_leaf in min_samples_leafs:
    print('_______________________________')
    print('Train min_samples_leaf : ', min_samples_leaf)
    stime = time.time()
    dt = DT(min_samples_leaf=min_samples_leaf)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)
    del dt
    print('Time cost : ', (time.time() - stime) / 60)
    print("min_samples_leaf={0}, AUC : ".format(min_samples_leaf), roc_auc2)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()

maxfeatures = list(range(1, int(0.5 * X_train.shape[1])))
train_results, test_results = [], []
for fnum in maxfeatures:
    print('_______________________________')
    print('Train fnum : ', fnum)
    stime = time.time()
    dt = DT(max_features=fnum)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)
    print('Time cost : ', (time.time() - stime) / 60)
    print("fnum={0}, AUC : ".format(fnum), roc_auc2)

line1, = plt.plot(maxfeatures, train_results, 'b', label='Train AUC')
line2, = plt.plot(maxfeatures, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.show()