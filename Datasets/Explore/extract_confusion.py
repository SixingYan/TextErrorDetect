from conf import langconv


def Tra2Simp(sent):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    return langconv.Converter('zh-hans').convert(sent)


def main():
    pass


def test():
    traditional_sentence = '憂郁的臺灣烏龜'
    simplified_sentence = Traditional2Simplified(traditional_sentence)
    print(simplified_sentence)

    '''
    输出结果：
        忧郁的台湾乌龟
    '''


if __name__ == "__main__":
    main()
