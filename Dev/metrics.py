import math
from typing import List, Tuple

import numpy as np

'''
均使用加一平滑
'''


def statistic(vals: List)->Tuple:
    names = ['平均值', '极大', '极小', '25%', '75%']
    perc25, perc50, perc75 = np.percentile(
        vals, (25, 50, 75), interpolation='midpoint')
    mean, mx, mn = sum(vals) / len(vals), max(vals), min(vals)
    return [perc25, perc50, perc75, mean, mx, mn], names

# 内在指标


def prepelxity(word_freq: List, word_count: int)->float:
    return math.exp(-math.log(word_freq) / word_count)


def explore_task_1():
    """ ppl：基于正例语料/负例语料/标准语料的计算 """
    G:
        Class, word_count:
            int, word_dist_count:
                int, sents:
                    List

    # get model

    # get
    pvals = []
    for s in sents:
        word_freq = [(G.search(w) + 1) / (word_count + word_dist_count)
                     for w in sent]
        pvals.append(prepelxity(word_freq, word_count))

    info, info_name = statistic(pvals)

    # ([[],[],[]],info_name,['','',''])


# 外在指标，和一个baseline做对比
def to_corpus(G: Class, word_count: int, word_dist_count: int, sent: str)->float:
    """ 基于语料的ngram距离 """
    res = 1
    for w in sent.split():
        res *= (G.search(w) + 1) / (word_count + word_dist_count)
    return res


def explore_task_2():
    """ 
        ngram：测试集在正例语料/负例语料/标准语料上表现
    """
    sent  # 候选集
    g, word_count, word_dist_count = load('(ngram,wcount,wdistcount).pickle')

    freq = [to_corpus(g, word_count, word_dist_count, sent) for s in sents]

    # 计算平均值，中位数，极大，极小，25%，75% 众数


xdf_tag_fragment_feature
