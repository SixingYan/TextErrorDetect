import csv
import re
import random
from typing import List, Tuple
import pandas as pd
import os

import demjson
import jieba
import jieba.posseg as pseg

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
            sents.append([ID, sent, const.POS])
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
            sents.append([ID, ' '.join(parse_single(s)), const.NEG])
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
            sents_ts.append([ID, d['onebest'], d['bg'], d['ed'], const.POS])
            # .csv - only sentence
            sents_space.append(
                [ID, ' '.join(parse_single(d['onebest'])), const.POS])

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


def build_sents_fasttext(path: str, source: str, target: str):
    """  """
    df = pd.read_csv(os.path.join(path, source))[:100]

    data = ['__label__{0} {1} \n'.format(x, y) for x in df[
        'target'].values.tolist() for y in df['sent'].values.tolist()]

    with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore') as f:
        f.writelines(data)

    return len(data)

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
    cut_to_csv(const.DATAPATH, 'kenlm_chars.csv', 'kenlm_jieba.csv')
    cut_to_csv(const.DATAPATH, 'paopao_chars.csv', 'paopao_jieba.csv')
    cut_to_csv(const.DATAPATH, 'sougou_chars.csv', 'sougou_jieba.csv')


# 用于生成句法特征
def pos_to_csv(path: str, source: str, target: str):
    """  """
    print('now start : ', source)

    X = pd.read_csv(os.path.join(path, source))#[:100]

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
    '''
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    return [w.strip() for w in pattern.split(sent) if len(w.strip()) > 0]
    '''
    return [w.strip() for w in sent if len(w.strip()) > 0]


def mergeDf(path: str, sources: List, target: str):
    """ 合并多个dataframe """

    df = pd.read_csv(os.path.join(path, sources[0]))

    if len(sources) > 1:
        for s in sources[1:]:
            df = pd.concat([df, pd.read_csv(os.path.join(path, s))], ignore_index=True)

    print(df.head())
    
    df.to_csv(os.path.join(path, target), index=None)


def sample_subset_csv(path, source, target, ratio):
    """  """
    scount = 0
    tcount = 0
    with open(path + source, 'r') as f:
        for line in f:
            scount += 1
            if random.random() < ratio:
                with open(path + target, 'a') as fp:
                    fp.write(line)
                tcount += 1

    return scount, tcount


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

    #
    # build_sents_fasttext(const.DATAPATH, 'kenlm_paopao_chars.csv', 'kenlm_paopao_fasttext.txt',)

    # 完成
    # cut_word()

    # 完成
    # get_pos()

    # 完成
    # mergeDf(const.DATAPATH, ['paopao_jieba.csv', 'kenlm_jieba.csv'], 'kenlm_paopao_jieba.csv')

    # 完成
    # mergeDf(const.DATAPATH, ['paopao_pos.csv', 'kenlm_pos.csv'], 'kenlm_paopao_pos.csv')

if __name__ == '__main__':
    main()
