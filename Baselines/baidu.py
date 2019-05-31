import os
import toml
from typing import List
from aip import AipNlp


def _getConfig():
    """  """
    cur_dir_path = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(cur_dir_path, 'config.toml')

    config = toml.load(filepath)["baidu"]

    app_id = config.get('app_id', '')
    api_key = config.get('api_key', '')
    secret_key = config.get('secret_key', '')

    return app_id, api_key, secret_key


def do(data, labels)->float:
    """
        返回正确率
    """
    app_id, api_key, secret_key = _getConfig()
    client = AipNlp(app_id, api_key, secret_key)

    #data, labels = _getTestData()
    corr, count = 0, len(data)

    for i, sent in enumerate(data):
        try:
            res = client.ecnet(sent)  # <- dict
            r = 1 if res['item']['vec_fragment'] == [] else 0
            corr += (1 if labels[i] == r else 0)
        except Exception as e:
            print(e)
            print('_____________________')
            break

    return float(corr) / count


def main():

    pass

if __name__ == '__main__':
    main()
