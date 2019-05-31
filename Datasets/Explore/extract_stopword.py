import pickle
import os

import const


def ext_baidu():
    """  """
    words = []
    with open(os.path.join(const.DATAPATH, 'stopwords_baidu.txt'), 'r', encoding='utf_8_sig', errors='ignore') as f:
        words = set(f.read().strip().split(','))

    with open(os.path.join(const.PKPATH, 'stopword_baidu.pickle'), 'wb') as f:
        pickle.dump(words, f)

    return len(words)


def ext_mixed():
    """  """
    words = []
    with open(os.path.join(const.DATAPATH, 'stopwords_mixed.txt'), 'r', encoding='utf_8_sig', errors='ignore') as f:
        for w in f.readlines():
            if len(w.strip()) > 0:
                words.append(w.strip())

    with open(os.path.join(const.PKPATH, 'stopword_mixed.pickle'), 'wb') as f:
        pickle.dump(words, f)

    return len(words)


def main():
    ext_baidu()
    ext_mixed()

if __name__ == '__main__':
    main()
