import os
path = 'D:/yansixing/weibo_content_corpus0'
file = 'weibo.xml'
target = 'weibo.txt'
with open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore') as f:
    i = 0
    for line in f:
        if '<article>' in line or '</article>' in line:
            line = line.replace('<article>', '').replace('amp;', ' ').replace('</article>', '').strip()
            if len(line) < 3:
                continue
            i += 1
            with open(os.path.join(path, target), 'a', encoding='utf-8', errors='ignore') as ft:
                ft.write(line + '\n')

    print('i : ', i)
