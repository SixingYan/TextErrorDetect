import pandas as pd
import os

import jieba

import const


# 用于训练语言模型

def cut_to_csv(path: str, source: str, target: str):
    """  """
    X = pd.read_csv(os.path.join(path, source))

    X['sent'] = X['sent'].progress_apply(
        lambda x: jieba.cut(''.join(x.split())))

    X.to_csv(os.path.join(path, target), index=None)


def cut_word():
    """  """
    cut_to_csv(const.DATAPATH, 'kenlm_chars.csv', 'kenlm_jieba.csv')
    cut_to_csv(const.DATAPATH, 'paopao_chars.csv', 'paopao_jieba.csv')
    cut_to_csv(const.DATAPATH, 'sougou_chars.csv', 'sougou_jieba.csv')


# 用于生成句法特征
import jieba.posseg as pseg


def pos_to_csv(path: str, source: str, target: str):
    """  """
    X = pd.read_csv(os.path.join(path, source))

    X['pos'] = X['sent'].progress_apply(
        lambda x: [t for w, t in pseg.cut(''.join(x.split()))])

    X.drop(['sent'], axis=1).to_csv(os.path.join(path, target), index=None)


def get_pos():
    """  """
    pos_to_csv(const.DATAPATH, 'kenlm_chars.csv', 'kenlm_pos.csv')
    pos_to_csv(const.DATAPATH, 'paopao_chars.csv', 'paopao_pos.csv')
    pos_to_csv(const.DATAPATH, 'sougou_chars.csv', 'sougou_pos.csv')


# 实现句法的语言模型
def function():
    pass





# -----------------------
# https://www.cnblogs.com/jin-liang/p/9638197.html
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from sklearn.metrics import roc_curve, auc


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)


def get_roc_auc(dt):
    """  """
    dt.fit(x_train, y_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_train, dt.predict(x_train))
    roc_auc1 = auc(false_positive_rate, true_positive_rate)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_test, dt.predict(x_test))
    roc_auc2 = auc(false_positive_rate, true_positive_rate)

    return roc_auc1, roc_auc2


# 先测试
DecisionTreeClassifier(class_weight='balanced')

# 先测试
DecisionTreeClassifier(splitter='best')  # 'random'


max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results, test_results = [], []

for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)


line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


min_samples_splits = np.linspace(0.005, 0.5, 100, endpoint=True)
train_results, test_results = [], []

for min_samples_split in min_samples_splits:
    dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)


line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


min_samples_leafs = np.linspace(0.0001, 0.01, 100, endpoint=True)
train_results, test_results = [], []

for min_samples_leaf in min_samples_leafs:
    dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)


line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()


train_results, test_results = [], []
for fnum in range(1, int(0.5 * data.shape[1])):
    dt = DecisionTreeClassifier(max_features=fnum)
    roc_auc1, roc_auc2 = get_roc_auc(dt)
    train_results.append(roc_auc1)
    test_results.append(roc_auc2)

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.show()
