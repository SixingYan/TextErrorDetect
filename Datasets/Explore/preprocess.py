# encoding = utf-8
import csv
import re
import random
from typing import List, Tuple
import pandas as pd
import os
import pickle
from pandas.core.frame import DataFrame

import demjson
import jieba
import jieba.posseg as pseg
from nltk.util import ngrams

import const


# --------------
# 抽取sougou的txt文件的整个字符串作为负语料


def extract_txt_sougou(path, source, hid):
    """  """
    # print(os.path.join(path, source))
    sents, iD = [], 0
    with open(os.path.join(path, source), 'r', encoding='gbk', errors='ignore') as f:
        for s in parse_sent(f.read()):
            sent = ' '.join(parse_single(s)).encode('gbk').decode('gbk').encode('utf-8').decode('utf-8')
            if len(sent.strip()) == 0:
                continue
            ID = '{0}{1}'.format(hid, iD)
            sents.append([ID, keepcn(sent), const.POS])
            iD += 1
    # print(sents)
    return sents


def extract_sent_sougou(path, sdir, target):
    """ 文件夹下的子文件夹 再读子文件夹下的文件，"""
    with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore', newline='') as f:
        csv.writer(f).writerow(['id', 'sent', 'target'])

    for dname in os.listdir(path + sdir):
        spath = os.path.join(path + sdir, dname)
        for fname in os.listdir(spath):
            try:
                hid = fname.split('.txt')[0]

                sents = extract_txt_sougou(path, os.path.join(
                    spath, fname), hid)

                with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore', newline='') as f:
                    csv.writer(f).writerows(sents)
                print('complete : ', fname)

            except Exception as e:
                print('file {} error '.format(fname))
                print(e)
                print('_____________________________')


# --------------


# --------------
# 抽取paopao的txt文件的整个字符串作为负语料


def extract_txt_paopao(path, source, hid):
    """  """
    #print(os.path.join(path, source))
    sents, iD = [], 0
    with open(os.path.join(path, source), 'r', encoding='utf_8_sig', errors='ignore') as f:
        for s in parse_sent(f.read()):
            ID = '{0}{1}'.format(hid, iD)
            sents.append([ID, ' '.join(parse_single(keepcn(s))), const.NEG])
            iD += 1
    return sents


def extract_sent_paopao(path, sdir, target):
    """ 文件夹下的子文件夹 再读子文件夹下的文件，"""
    with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore', newline='') as f:
        csv.writer(f).writerow(['id', 'sent', 'target'])

    for dname in os.listdir(path + sdir):
        spath = os.path.join(path + sdir, dname)
        for fname in os.listdir(spath):
            try:
                name = fname.split('.txt')[0]
                hashID = hash(name)

                sents = extract_txt_paopao(path, os.path.join(
                    spath, fname), hashID)

                with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore', newline='') as f:
                    csv.writer(f).writerows(sents)
                print('complete : ', fname)
                # break

            except Exception as e:
                print('file {} error '.format(fname))
                print(e)
                print('_____________________________')

# --------------


# --------------
# 抽取kenlm的json文件的每段话作为正语料
def extract_txt_kenlm(path: str, source: str, hid: str)->Tuple:
    """ 处理单文（加入HashID 作为唯一标识） """
    sents_ts, sents_space = [], []

    with open(os.path.join(path, source), 'r', encoding='utf_8_sig', errors='ignore') as f:
        iD = 0
        s = f.read()
        dicts = demjson.decode(s)
        #dicts = json.loads(f.read())
        for d in dicts:
            ID = '{0}{1}'.format(hid, iD)
            #.csv - sents, timestamp
            sents_ts.append([ID, keepcn(d['onebest']), d['bg'], d['ed'], const.POS])
            # .csv - only sentence
            sents_space.append(
                [ID, ' '.join(parse_single(keepcn(d['onebest']))), const.POS])

            iD += 1
            #print('now is :', iD)
    return sents_ts, sents_space


def extract_sent_kenlm(path, sdir, target1, target2):
    """ 把文件夹里的文件的文本提取出来 """

    with open(os.path.join(path, target1), 'a', encoding='utf-8', errors='ignore', newline='') as f:
        csv.writer(f).writerow(['id', 'sentence', 'begin', 'end', 'target'])

    with open(os.path.join(path, target2), 'a', encoding='utf-8', errors='ignore', newline='') as f:
        csv.writer(f).writerow(['id', 'sent', 'target'])

    # 读取文件夹里的所有文件名称，遍历抽取
    spath = os.path.join(path, sdir)
    for fname in os.listdir(spath):
        fpath = os.path.join(spath, fname)
        if not os.path.isdir(fpath):
            try:
                name = fname.split('.txt')[0]
                hashID = hash(name)
                sents_ts, sents_space = extract_txt_kenlm(path, fpath, hashID)

                with open(os.path.join(path, target1), 'a', encoding='utf-8', errors='ignore', newline='') as f:
                    csv.writer(f).writerows(sents_ts)

                with open(os.path.join(path, target2), 'a', encoding='utf-8', errors='ignore', newline='') as f:
                    csv.writer(f).writerows(sents_space)

                print('complete : ', fname)

            except Exception as e:
                print('file {} error '.format(fname))
                print(e)
                print('_____________________________')

# --------------

# --------------
# 构造符合fasttext要求的文本


def build_sents_fasttext_unspv(path: str, source: str, target: str):
    df = pd.read_csv(os.path.join(path, source))
    print('Data shape : ', df.shape)
    with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore') as f:
        for x in df['sent'].values.tolist():
            line = '{0}\n'.format(x)
            f.write(line)


def build_sents_fasttext(path: str, source: str, target: str):
    """  """
    df = pd.read_csv(os.path.join(path, source))  # [:100]

    print('Data shape : ', df.shape)
    with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore') as f:
        for x, y in zip(df['target'].values.tolist(), df['sent'].values.tolist()):
            #y = ''.join(w.strip() for w in re.findall(r'[\u4e00-\u9fa5]', str(y)) if len(w.strip()) > 0)
            line = '{0}\t__label__{1}\n'.format(y, x)
            f.write(line)

# --------------
# --------------
# 抽取政治知识树


def extract_zhengzhi(path, source, target):
    df = pd.read_csv(os.path.join(path, source), encoding='gb2312')
    df['sentence'] = df['sentence'].apply(lambda x: str(x))
    df['sentence'] = df['sentence'].apply(lambda x: x.encode('utf-8').decode('utf-8'))
    df = df[['sentence', 'target']]
    df['sent_chars'] = df['sentence'].apply(lambda x: ' '.join(parse_single(x)))
    df['sent_jieba'] = df['sentence'].apply(lambda x: ' '.join(w for w in jieba.cut(x)))
    df = df[['sent_chars', 'sent_jieba', 'target']]
    print(df.head())
    df.to_csv(os.path.join(const.DATAPATH, target), index=None)

# --------------
# --------------
# 抽取句法特征


# 用于训练语言模型

def cut_to_csv(path: str, source: str, target: str):
    """  """
    print('now start : ', source)

    X = pd.read_csv(os.path.join(path, source))
    X['sent'] = X['sent'].apply(
        lambda x: str(x))
    X['sent'] = X['sent'].apply(
        lambda x: ' '.join(w for w in jieba.cut(''.join(x.split()))))

    X.to_csv(os.path.join(path, target), index=None)


def cut_word():
    """  """
    cut_to_csv(const.DATAPATH, 'kenlm_chars_v2.csv', 'kenlm_jieba_v2.csv')
    cut_to_csv(const.DATAPATH, 'paopao_chars_v2.csv', 'paopao_jieba_v2.csv')
    cut_to_csv(const.DATAPATH, 'sougou_chars_v2.csv', 'sougou_jieba_v2.csv')


# 用于生成句法特征
def pos_to_csv(path: str, source: str, target: str):
    """  """
    print('now start : ', source)

    X = pd.read_csv(os.path.join(path, source))  # [:100]

    X['sent'] = X['sent'].apply(lambda x: str(x))
    X['pos'] = X['sent'].apply(
        lambda x: ' '.join(t for w, t in pseg.cut(''.join(x.split()))))

    X = X.drop(['sent'], axis=1)
    # print(X.head())

    X.to_csv(os.path.join(path, target), index=None)


def get_pos():
    """  """
    pos_to_csv(const.DATAPATH, 'kenlm_chars.csv', 'kenlm_pos.csv')
    pos_to_csv(const.DATAPATH, 'paopao_chars.csv', 'paopao_pos.csv')
    pos_to_csv(const.DATAPATH, 'sougou_chars.csv', 'sougou_pos.csv')

# --------------
# --------------
# 抽取自标注数据集，用于测试


def extract_parnoise():
    # 抽取数据
    path = 'D:/yansixing/tmp'
    source = 'parnoise_data.csv'
    poslist, neglist = [], []
    with open(os.path.join(path, source), 'r', encoding='gbk', errors='ignore', newline='') as f:
        for line in f.readlines():
            pos = None
            if ',' in line:
                neg, pos = tuple(line.split(','))
            else:
                neg = line
            if pos is not None and len(pos.strip()) > 1:
                poslist.append(pos.strip())
            if len(neg.strip()) > 1:
                neglist.append(neg.strip())

    targets = [1] * len(poslist) + [0] * len(neglist)
    sent = poslist + neglist
    df = DataFrame({'target': targets, 'sent': sent})

    print('Data shape : ', df.shape)
    print(df.head())

    #pattern = re.compile(r'([\u4e00-\u9fa5])')
    df['sent'] = df['sent'].apply(lambda x: ''.join(w.strip() for w in re.findall(r'[\u4e00-\u9fa5]', x) if len(w.strip()) > 0))

    # 使用特征抽取模式，只使用一个特征
    print('Get features')

    def getFeature(X):
        m = getPikcle(os.path.join(const.PKPATH, 'lm_3_paopao_jieba.pk'))
        X['sent'] = X['sent'].apply(lambda x: str(x))
        X['sent'] = X['sent'].apply(lambda x: ' '.join(w for w in jieba.cut(x)))
        X['3n_etp_n_jieba'] = X['sent'].apply(lambda x: m.entropy(ngrams(x, 3, True, True, '<s>', '</s>')) if m.entropy(ngrams(x, 3, True, True, '<s>', '</s>')) != float('inf') else -1)
        return X

    df = getFeature(df)
    print(df.head())
    df.drop(['sent'], axis=1).to_csv(os.path.join(const.DATAPATH, 'parnoise_feats.csv'))

    # 使用fasttext格式
    print('Get fasttext')
    with open(os.path.join(const.DATAPATH, 'parnoise_fasttext.txt'), 'a', encoding='utf-8', errors='ignore') as f:
        for x, y in zip(df['target'].values.tolist(), df['sent'].values.tolist()):
            line = '{0}\t__label__{1}\n'.format(y, x)
            f.write(line)


# --------------
# --------------
# 抽取normal/unormal

def extract_normal():
    path = 'D:/yansixing/'
    source1 = 'normal_20190531.txt'
    poslist, neglist = [], []
    with open(os.path.join(path, source1), 'r', encoding='utf-8', errors='ignore', newline='') as f:
        for line in f.readlines():
            poslist.append(line.split()[-1])

    source2 = 'unnormal_20190531.txt'
    with open(os.path.join(path, source2), 'r', encoding='utf-8', errors='ignore', newline='') as f:
        for line in f.readlines():
            neglist.append(line.split()[-1])

    targets = [const.POS] * len(poslist) + [const.NEG] * len(neglist)
    sent = poslist + neglist
    df = DataFrame({'target': targets, 'sent': sent})

    df['sent'] = df['sent'].apply(lambda x: str(x))
    df['sent'] = df['sent'].apply(lambda x: ''.join(w.strip() for w in re.findall(r'[\u4e00-\u9fa5]', x) if len(w.strip()) > 0))
    df['sent'] = df['sent'].apply(lambda x: ' '.join(parse_single(x)))

    df.to_csv(os.path.join(const.DATAPATH, 'un_normal_chars.csv'), index=None)

    print('Data shape : ', df.shape)
    print(df.head())
    '''
    # 使用fasttext格式
    print('Get fasttext')
    with open(os.path.join(const.DATAPATH, 'normal_chars_fasttext.txt'), 'a', encoding='utf-8', errors='ignore') as f:
        for x, y in zip(df['target'].values.tolist(), df['sent'].values.tolist()):
            line = '{0}\t__label__{1}\n'.format(y, x)
            f.write(line)
    '''

# --------------
# --------------
# 通用工具


def parse_sent(para: str)->List:
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def parse_single(sent: str)->List:
    """ 把中文句子转化成字的列表 """
    sent = keepcn(sent)
    return [w.strip() for w in sent if len(w.strip()) > 0]


def mergeDf(path: str, sources: List, target: str):
    """ 合并多个dataframe """

    df = pd.read_csv(os.path.join(path, sources[0]))

    if len(sources) > 1:
        for s in sources[1:]:
            df = pd.concat([df, pd.read_csv(os.path.join(path, s))], ignore_index=True)

    print(df.head())

    df.to_csv(os.path.join(path, target), index=None)


def keepcn(s: str):
    """ 只保留中文 """
    #print(s)
    return ''.join(w.strip() for w in re.findall(r'[\u4e00-\u9fa5]', s) if len(w.strip()) > 0)


def getPikcle(path: str):
    """  """
    with open(path, 'rb') as f:
        v = pickle.load(f)
    return v


def main():
    pass
    """ 运行处理的任务，写明任务内容 """
    # 完成
    # extract_sent_kenlm(const.DATAPATH, 'kenlm_corpus/',
    #                 'kenlm_sentences.csv', 'kenlm_chars.csv')
    # 完成
    # extract_sent_paopao(const.DATAPATH, 'paopao_coupus/', 'paopao_chars.csv')
    # 完成
    # extract_sent_sougou(const.DATAPATH, 'sougou/', 'sougou_chars.csv')
    # 完成
    # mergeDf(const.DATAPATH, ['paopao_chars.csv', 'kenlm_chars.csv'], 'kenlm_paopao_chars.csv')

    # 完成
    # build_sents_fasttext(const.DATAPATH, 'kenlm_paopao_chars.csv', 'kenlm_paopao_chars_fasttext.txt')
    # build_sents_fasttext(const.DATAPATH, 'kenlm_paopao_jieba.csv', 'kenlm_paopao_jieba_fasttext.txt')
    # build_sents_fasttext_unspv(const.DATAPATH, 'kenlm_paopao_chars.csv', 'kenlm_paopao_chars_fasttext_unspv.txt')
    # build_sents_fasttext_v2(const.DATAPATH, 'kenlm_paopao_jieba.csv', 'kenlm_paopao_jieba_v2_fasttext.txt')
    # 完成
    # cut_word()
    # 完成
    # get_pos()
    # 完成
    # mergeDf(const.DATAPATH, ['paopao_jieba.csv', 'kenlm_jieba.csv'], 'kenlm_paopao_jieba.csv')
    # 完成
    # mergeDf(const.DATAPATH, ['paopao_pos.csv', 'kenlm_pos.csv'], 'kenlm_paopao_pos.csv')
    # 完成
    # extract_parnoise()
    # extract_normal()

    #
    extract_zhengzhi(const.DATAPATH, 'zhengzhi_file.csv', 'zhengzhi.csv')



if __name__ == '__main__':
    main()
