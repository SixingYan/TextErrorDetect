import os
import pandas as pd
path = ''
source = ''
target = ''


def extract():
    # 抽取数据
    poslist, neglist = [], []
    with open(os.path.join(path, source), 'r', encoding='utf-8', errors='ignore', newline='') as f:
        for line in f.readlines():
            pos = None
            if ',' in line:
                neg, pos = tuple(line.split(','))
            else:
                neg = line
            if pos is not None:
                poslist.append(pos)
            neglist.append(neg)

    target = [1] * len(poslist) + [0] * len(neglist)
    sent = poslist + neglist

    

    pd.to_csv


def main():
    extract()
if __name__ == '__main__':
    main()
