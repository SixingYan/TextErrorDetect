'''
    对每一条sent，生成相应的特征，特征使用.csv存储
'''
import re
import os
import pickle

from nltk.util import ngrams
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='Progress')


from lm import LanModel
import const

# 计数使用-正则匹配
#en_regex = re.compile(r'[a-zA-z]')
#num_regex = re.compile(r'[0-9]')
cn_regex = re.compile(r'[\u4E00-\u9FA5]')


def getPikcle(path: str):
    """  """
    with open(path, 'rb') as f:
        v = pickle.load(f)
    return v

# 填充语料模型
'''
model_p1 = getPikcle(os.path.join(const.PKPATH, 'lm_1_kenlm_p.pk'))
model_n1 = getPikcle(os.path.join(const.PKPATH, 'lm_1_paopao_n.pk'))
model_s11 = getPikcle(os.path.join(const.PKPATH, 'lm_1_sougou.pk'))
model_p2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_kenlm_p.pk'))
model_n2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_paopao_n.pk'))
model_s12 = getPikcle(os.path.join(const.PKPATH, 'lm_2_sougou.pk'))
model_p3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_kenlm_p.pk'))
model_n3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_paopao_n.pk'))
model_s13 = getPikcle(os.path.join(const.PKPATH, 'lm_3_sougou.pk'))

model_p1 = getPikcle(os.path.join(const.PKPATH, 'lm_1_kenlm_jieba.pk'))

model_n1 = getPikcle(os.path.join(const.PKPATH, 'lm_1_paopao_jieba.pk'))
model_s11 = getPikcle(os.path.join(const.PKPATH, 'lm_1_sougou_jieba.pk'))
model_p2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_kenlm_jieba.pk'))
model_n2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_paopao_jieba.pk'))
model_s12 = getPikcle(os.path.join(const.PKPATH, 'lm_2_sougou_jieba.pk'))
model_p3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_kenlm_jieba.pk'))
model_n3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_paopao_jieba.pk'))
model_s13 = getPikcle(os.path.join(const.PKPATH, 'lm_3_sougou_jieba.pk'))

model_p1 = getPikcle(os.path.join(const.PKPATH, 'lm_1_kenlm_pos.pk'))
model_n1 = getPikcle(os.path.join(const.PKPATH, 'lm_1_paopao_pos.pk'))
model_s11 = getPikcle(os.path.join(const.PKPATH, 'lm_1_sougou_pos.pk'))
model_p2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_kenlm_pos.pk'))
model_n2 = getPikcle(os.path.join(const.PKPATH, 'lm_2_paopao_pos.pk'))
model_s12 = getPikcle(os.path.join(const.PKPATH, 'lm_2_sougou_pos.pk'))
model_p3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_kenlm_pos.pk'))
model_n3 = getPikcle(os.path.join(const.PKPATH, 'lm_3_paopao_pos.pk'))
model_s13 = getPikcle(os.path.join(const.PKPATH, 'lm_3_sougou_pos.pk'))


l1p = LanModel(model_p1, 1)

l2p = LanModel(model_p2, 2)
l3p = LanModel(model_p3, 3)

l1n = LanModel(model_n1, 1)
l2n = LanModel(model_n2, 2)
l3n = LanModel(model_n3, 3)

l1s1 = LanModel(model_s11, 1)
l2s1 = LanModel(model_s12, 2)
l3s1 = LanModel(model_s13, 3)
'''

# 停用词表（集合类型）
# < 0.1
# stopword_1 = getPikcle(os.path.join(const.PKPATH, 'stopword_baidu.pickle'))

# 效果不佳 < 0.01
# stopword_2 = getPikcle(os.path.join(const.PKPATH, 'stopword_mixed.pickle'))


def count_wdict(x: str, n: int, d: int)->int:
    """  """
    res = 0
    for terms in ngrams(x, n):  # no padding
        res += 1 if ''.join(terms) in d else 0
    return res


def count_conf():
    # lack of data since it need to visit site.google
    pass


def feature_engineering(X):
    """  """

    X['sent'] = X['sent'].progress_apply(lambda x: str(x))

    # len------------------------------
    # < 0.1
    # X['len'] = X['sent'].progress_apply(lambda x: len(x))

    # stopword------------------------------
    # stopword_baidu 1 < 0.01
    # X['1n_stp_bd'] = X['sent'].progress_apply(lambda x: count_wdict(x, 1, stopword_1))

    # stopword_baidu 2
    # < 0.1
    # X['2n_stp_bd'] = X['sent'].progress_apply(lambda x: count_wdict(x, 2, stopword_1))

    # stopword_mixed 1  < 0.01
    # X['1n_stp_mx'] = X['sent'].progress_apply(lambda x: count_wdict(x, 1, stopword_2))

    # stopword_mixed 2 效果不佳，直接取消
    # X['2n_stp_mx'] = X['sent'].progress_apply(lambda x: count_wdict(x, 2, stopword_2))

    # number count------------------------------
    # < 0.1
    # X['num_rate'] = X['sent'].progress_apply(lambda x: len(num_regex.findall(x)) / len(x))

    # chinese char------------------------------
    X['cn_rate'] = X['sent'].progress_apply(lambda x: len(cn_regex.findall(x)) / len(x))

    # English word rate------------------------------
    # < 0.1
    # X['en_rate'] = X['sent'].progress_apply(lambda x: len(en_regex.findall(x)) / len(x))

    # language model------------------------------
    X['sent'] = X['sent'].progress_apply(lambda x: x.split())

    # < 0.01
    # X['1n_ppl_p'] = X['sent'].progress_apply(lambda x: l1p.perplexity(x) if l1p.perplexity(x) != float('inf') else -1)
    # < 0.1
    # X['1n_etp_p'] = X['sent'].progress_apply(lambda x: l1p.entropy(x) if l1p.entropy(x) != float('inf') else -1)

    X['2n_etp_p'] = X['sent'].progress_apply(lambda x: l2p.entropy(x) if l2p.entropy(x) != float('inf') else -1)
    X['2n_ppl_p'] = X['sent'].progress_apply(lambda x: l2p.perplexity(x) if l2p.perplexity(x) != float('inf') else -1)

    X['3n_etp_p'] = X['sent'].progress_apply(lambda x: l3p.entropy(x) if l3p.entropy(x) != float('inf') else -1)
    X['3n_ppl_p'] = X['sent'].progress_apply(lambda x: l3p.perplexity(x) if l3p.perplexity(x) != float('inf') else -1)

    # < 0.1
    # X['1n_ppl_n'] = X['sent'].progress_apply(lambda x: l1n.perplexity(x) if l1n.perplexity(x) != float('inf') else -1)
    X['1n_etp_n'] = X['sent'].progress_apply(lambda x: l1n.entropy(x) if l1n.entropy(x) != float('inf') else -1)

    X['2n_etp_n'] = X['sent'].progress_apply(lambda x: l2n.entropy(x) if l2n.entropy(x) != float('inf') else -1)
    X['2n_ppl_n'] = X['sent'].progress_apply(lambda x: l2n.perplexity(x) if l2n.perplexity(x) != float('inf') else -1)

    X['3n_etp_n'] = X['sent'].progress_apply(lambda x: l3n.entropy(x) if l3n.entropy(x) != float('inf') else -1)
    X['3n_ppl_n'] = X['sent'].progress_apply(lambda x: l3n.perplexity(x) if l3n.perplexity(x) != float('inf') else -1)

    # < 0.1
    # X['1n_ppl_s1'] = X['sent'].progress_apply(lambda x: l1s1.perplexity(x) if l1s1.perplexity(x) != float('inf') else -1)
    # X['1n_etp_s1'] = X['sent'].progress_apply(lambda x: l1s1.entropy(x) if l1s1.entropy(x) != float('inf') else -1)

    X['2n_etp_s1'] = X['sent'].progress_apply(lambda x: l2s1.entropy(x) if l2s1.entropy(x) != float('inf') else -1)
    X['2n_ppl_s1'] = X['sent'].progress_apply(lambda x: l2s1.perplexity(x) if l2s1.perplexity(x) != float('inf') else -1)

    # < 0.01
    #X['3n_etp_s1'] = X['sent'].progress_apply(lambda x: l3s1.entropy(x) if l3s1.entropy(x) != float('inf') else -1)
    # < 0.1
    #X['3n_ppl_s1'] = X['sent'].progress_apply(lambda x: l3s1.perplexity(x) if l3s1.perplexity(x) != float('inf') else -1)

    # confusion set count------------------------------
    # X['conf'] = X['sent'].progress_apply(lambda x: count_conf(x))

    # 基于分词结果的指标，目前暂不明确
    # pass

    return X


def extract(path: str, source: str, target: str):
    """  """
    # load data
    X = pd.read_csv(os.path.join(path, source))  # [:100]
    print('Data shape : ', X.shape)

    # get features
    X = feature_engineering(X)
    X_f = X.drop(['sent', 'id'], axis=1)  # 把sentence, id 列删除

    # print(X_f[:5])
    print(X_f.columns.values.tolist())

    # save data
    X_f.to_csv(os.path.join(path, target), index=None)


def add_feature(S, X):
    """  """

    S['sent'] = S['sent'].progress_apply(lambda x: str(x))
    S['sent'] = S['sent'].progress_apply(lambda x: x.split())

    X['1n_ppl_p_jieba'] = S['sent'].progress_apply(lambda x: l1p.perplexity(x) if l1p.perplexity(x) != float('inf') else -1)
    X['1n_etp_p_jieba'] = S['sent'].progress_apply(lambda x: l1p.entropy(x) if l1p.entropy(x) != float('inf') else -1)

    X['2n_etp_p_jieba'] = S['sent'].progress_apply(lambda x: l2p.entropy(x) if l2p.entropy(x) != float('inf') else -1)
    X['2n_ppl_p_jieba'] = S['sent'].progress_apply(lambda x: l2p.perplexity(x) if l2p.perplexity(x) != float('inf') else -1)

    X['3n_etp_p_jieba'] = S['sent'].progress_apply(lambda x: l3p.entropy(x) if l3p.entropy(x) != float('inf') else -1)
    X['3n_ppl_p_jieba'] = S['sent'].progress_apply(lambda x: l3p.perplexity(x) if l3p.perplexity(x) != float('inf') else -1)

    #
    X['1n_ppl_n_jieba'] = S['sent'].progress_apply(lambda x: l1n.perplexity(x) if l1n.perplexity(x) != float('inf') else -1)
    X['1n_etp_n_jieba'] = S['sent'].progress_apply(lambda x: l1n.entropy(x) if l1n.entropy(x) != float('inf') else -1)

    X['2n_etp_n_jieba'] = S['sent'].progress_apply(lambda x: l2n.entropy(x) if l2n.entropy(x) != float('inf') else -1)
    X['2n_ppl_n_jieba'] = S['sent'].progress_apply(lambda x: l2n.perplexity(x) if l2n.perplexity(x) != float('inf') else -1)

    X['3n_etp_n_jieba'] = S['sent'].progress_apply(lambda x: l3n.entropy(x) if l3n.entropy(x) != float('inf') else -1)
    X['3n_ppl_n_jieba'] = S['sent'].progress_apply(lambda x: l3n.perplexity(x) if l3n.perplexity(x) != float('inf') else -1)

    #
    X['1n_ppl_s1_jieba'] = S['sent'].progress_apply(lambda x: l1s1.perplexity(x) if l1s1.perplexity(x) != float('inf') else -1)
    X['1n_etp_s1_jieba'] = S['sent'].progress_apply(lambda x: l1s1.entropy(x) if l1s1.entropy(x) != float('inf') else -1)

    X['2n_etp_s1_jieba'] = S['sent'].progress_apply(lambda x: l2s1.entropy(x) if l2s1.entropy(x) != float('inf') else -1)
    X['2n_ppl_s1_jieba'] = S['sent'].progress_apply(lambda x: l2s1.perplexity(x) if l2s1.perplexity(x) != float('inf') else -1)

    X['3n_etp_s1_jieba'] = S['sent'].progress_apply(lambda x: l3s1.entropy(x) if l3s1.entropy(x) != float('inf') else -1)
    X['3n_ppl_s1_jieba'] = S['sent'].progress_apply(lambda x: l3s1.perplexity(x) if l3s1.perplexity(x) != float('inf') else -1)

    '''
    S['pos'] = S['pos'].progress_apply(lambda x: str(x))
    S['pos'] = S['pos'].progress_apply(lambda x: x.split())

    X['1n_ppl_p_pos'] = S['pos'].progress_apply(lambda x: l1p.perplexity(x) if l1p.perplexity(x) != float('inf') else -1)
    X['1n_etp_p_pos'] = S['pos'].progress_apply(lambda x: l1p.entropy(x) if l1p.entropy(x) != float('inf') else -1)

    X['2n_etp_p_pos'] = S['pos'].progress_apply(lambda x: l2p.entropy(x) if l2p.entropy(x) != float('inf') else -1)
    X['2n_ppl_p_pos'] = S['pos'].progress_apply(lambda x: l2p.perplexity(x) if l2p.perplexity(x) != float('inf') else -1)

    X['3n_etp_p_pos'] = S['pos'].progress_apply(lambda x: l3p.entropy(x) if l3p.entropy(x) != float('inf') else -1)
    X['3n_ppl_p_pos'] = S['pos'].progress_apply(lambda x: l3p.perplexity(x) if l3p.perplexity(x) != float('inf') else -1)

    # < 0.1
    # X['1n_ppl_n_pos'] = S['pos'].progress_apply(lambda x: l1n.perplexity(x) if l1n.perplexity(x) != float('inf') else -1)
    X['1n_etp_n_pos'] = S['pos'].progress_apply(lambda x: l1n.entropy(x) if l1n.entropy(x) != float('inf') else -1)

    X['2n_etp_n_pos'] = S['pos'].progress_apply(lambda x: l2n.entropy(x) if l2n.entropy(x) != float('inf') else -1)
    X['2n_ppl_n_pos'] = S['pos'].progress_apply(lambda x: l2n.perplexity(x) if l2n.perplexity(x) != float('inf') else -1)

    X['3n_etp_n_pos'] = S['pos'].progress_apply(lambda x: l3n.entropy(x) if l3n.entropy(x) != float('inf') else -1)
    X['3n_ppl_n_pos'] = S['pos'].progress_apply(lambda x: l3n.perplexity(x) if l3n.perplexity(x) != float('inf') else -1)

    #
    X['1n_ppl_s1_pos'] = S['pos'].progress_apply(lambda x: l1s1.perplexity(x) if l1s1.perplexity(x) != float('inf') else -1)
    X['1n_etp_s1_pos'] = S['pos'].progress_apply(lambda x: l1s1.entropy(x) if l1s1.entropy(x) != float('inf') else -1)

    X['2n_etp_s1_pos'] = S['pos'].progress_apply(lambda x: l2s1.entropy(x) if l2s1.entropy(x) != float('inf') else -1)
    X['2n_ppl_s1_pos'] = S['pos'].progress_apply(lambda x: l2s1.perplexity(x) if l2s1.perplexity(x) != float('inf') else -1)

    X['3n_etp_s1_pos'] = S['pos'].progress_apply(lambda x: l3s1.entropy(x) if l3s1.entropy(x) != float('inf') else -1)
    X['3n_ppl_s1_pos'] = S['pos'].progress_apply(lambda x: l3s1.perplexity(x) if l3s1.perplexity(x) != float('inf') else -1)
    
    '''
    return X


def reextract(path, source1, source2, target):
    """  """
    # load data
    S = pd.read_csv(os.path.join(path, source1))  # [:100]
    X = pd.read_csv(os.path.join(path, source2))  # [:100]
    print('Data shape : ', X.shape)

    # drop
    #X.drop(['3n_etp_s1', '2n_stp_mx', '1n_stp_mx', '1n_stp_bd', '1n_ppl_p'], axis=1, inplace=True)
    X.drop(['len', 'num_rate', 'en_rate', '1n_ppl_n', '1n_etp_p', '1n_ppl_s1', '1n_etp_s1', '3n_ppl_s1', '3n_ppl_s1', '1n_ppl_n_pos'], axis=1, inplace=True)

    X_f = add_feature(S, X)
    print(X_f[:5])
    print(X_f.columns.values.tolist())

    # save data
    X_f.to_csv(os.path.join(path, target), index=None)


def get_sent_embd(words, m1):

    sum(for w in words if w in )/len(words)

def _sim(sent:str, m1, m2):
    """ sent 空格分隔的句子 """
    get_sentence_vector()

def extract_embd():
    # load data
    S = pd.read_csv(os.path.join(path, source1))  # [:100]
    X = pd.read_csv(os.path.join(path, source2))  # [:100]
    print('Data shape : ', X.shape)

    # drop
    X.drop(['len', 'num_rate', 'en_rate', '1n_ppl_n', '1n_etp_p', '1n_ppl_s1', '1n_etp_s1', '3n_ppl_s1', '3n_ppl_s1', '1n_ppl_n_pos'], axis=1, inplace=True)
    
    ['1n_p2n_cbow'] 
    ['1n_p2s1_cbow']
    ['1n_n2s1_cbow']


def dropcol(path, source, target):
    """  """
    X = pd.read_csv(os.path.join(path, source))  # [:100]
    print('Data shape : ', X.shape)

    X.drop(['2n_stp_bd', '1n_ppl_p_pos', '1n_ppl_p_jieba', '1n_ppl_n_jieba', '2n_ppl_n_jieba',
            '1n_ppl_s1_jieba', '2n_etp_s1_jieba', '3n_etp_s1_jieba', '3n_ppl_s1_jieba'], axis=1, inplace=True)
    
    print(X[:5])
    print(X.columns.values.tolist())

    X.to_csv(os.path.join(path, target), index=None)


def main():
    # extract(const.DATAPATH, 'kenlm_paopao_chars.csv', 'data_kenlm_paopao.csv')
    # 二次抽取
    # reextract(const.DATAPATH, 'kenlm_paopao_pos.csv', 'data_kenlm_paopao.csv', 'data_kenlm_paopao_v2.0.csv')
    # reextract(const.DATAPATH, 'kenlm_paopao_jieba.csv', 'data_kenlm_paopao_v2.0.csv', 'data_kenlm_paopao_v2.1.csv')

    dropcol(const.DATAPATH,'data_kenlm_paopao_v2.1.csv','data_kenlm_paopao_v2.2.csv')

    #X = pd.read_csv(os.path.join(const.DATAPATH, 'data_paopao.csv'))[:100]
    # print(X)

if __name__ == '__main__':
    main()
