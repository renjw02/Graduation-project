"""
目标生成：readlang中的pairs，格式为[['en', 'sql'], ['en', 'sql'], ...]
'en'和'sql'是字符串形式，token之间以空格隔开
eg. ['i am', 'select max ( name ) from table where id = 1']
"""
import json


def read_data(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as lines:
        for line in lines:
            info = json.loads(line)
            en = info['question']
            query = info['query']
            pairs.append((en, query))
    return pairs

            

if __name__ == '__main__':
    pairs = read_data('data/wikisql.json')
    with open('data/new_wikisql.json', 'w', encoding='utf-8') as f:
        for question, query in pairs:
            f.write(json.dumps({'question': question, 'query': query}) + '\n')