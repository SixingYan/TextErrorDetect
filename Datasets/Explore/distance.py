import pandas as pd

from typing import List, Tuple
# 抽出正例
# 抽出反例

# 句子的长度也可以作为一个feature

# 训练人名数据集。


import Levenshtein  # https://www.jb51.net/article/98449.htm

# 两者在一些指数上的表现

对比正例和反例之间的距离


对比正 / 反例和语料的ngram指标 / ppl指标

是否能通过计算候选和完整语料之间的距离，来判断（这相当于基本验证指标的有效性）


# 计算正例的ngram
#


def parse_phrase(sent: str):
    pass


def pos_tagging(sent: str, tool: str='jieba')->Tuple:
    """
        return tagging_list, word_list
    """
    if tool == 'jieba':
        l = jieba.posseg.cut(sent)
        return [w for w, t in l], [t for w, t in l]


def dist_jaro()->float:
    """ """
    pass


def dist_jaro_winkler()->float:
    """ """
    pass


def dist_editedist()->float:
    """ """
    pass


def dist_ratio()->float:
    """"""
    pass


def dist_ngram(n)->float:
    """ """
    pass


'''


同一个来源
正例语料
    - pickle结果
    - csv结果
    - 字符切分结果

反例语料
    - 字符切分结果
'''


import ngram  # https://pythonhosted.org/ngram/tutorial.html#comparing-and-searching-strings
import pickle


def get_ngram(sents):
    g = ngram.NGram(sents)


def get_ngram_res(g, word):
    return g.search(word) + 1  # add one 平滑


def test_ngram(sents, s):
    """

    """
    g = get_ngram(sents)

    res = 1
    for w in s:
        res *= get_ngram_res(g, w)

    return res


'''


def 


    from sklearn.feature_extraction.text import CountVectorizer

    # 将字用空格拼成一句话
    data = [' '.join(chars) for chars in chars_list]

    vec = CountVectorizer(min_df=1, ngram_range=(1,1))

    X = vec.fit_transform(data)

    vec.get_feature_names() # -> List[str] 可以取出所有字的表


def test_ngram_1():
'''


if __name__ == '__main__':
    pass
