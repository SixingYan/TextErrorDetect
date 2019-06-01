'''
在这里完成模型的训练
'''

from typing import List
import os
import time
import pandas as pd

from gensim.models import Word2Vec


def getData()->List:

    return


def getW2V(path: str, target: str, chars: List):
	""" [['a','asdf', 'sdf',...],...] 分好词的句子列表 """

    print('start...')

    vec = Word2Vec(chars)

    vec.save_word2vec_format(os.path.join(path, target))


def getFtCBOW():
	"""  """
    pass


def getFtSG():
	"""  """
    pass


def main():
	"""  """
    #
    #
    pass


if __name__ == '__main__':
    main()
