import platform

DATAPATH = 'D:/yansixing/ErrorDetection/Datasets/Data/' if platform.platform().startswith(
    'Windows') else '/Users/alfonso/workplace/xdf/TextErrorDetect/Datasets/Data/'

MODELPATH = 'D:/yansixing/ErrorDetection/Datasets/Model/' if platform.platform().startswith('Windows') else ''

PKPATH = 'D:/yansixing/ErrorDetection/Datasets/Pickle/' if platform.platform().startswith('Windows') else ''


POS = 1
NEG = 0
