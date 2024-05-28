# import sqlite3

# conn = sqlite3.connect('./data/wikisql/train.db')
# cur = conn.cursor()

# q = 'SELECT sql from sqlite_master WHERE tbl_name = 1-10007452-3'
# cur.execute(q)

# rows = cur.fetchall()
# print(rows)
# for row in rows:
#     print(row)


import json

with open('data/wikisql_dev_new.json', 'r') as file:
    data = file.readlines()

# data = data[:56355]
keywords = ['SELECT', 'FROM', 'WHERE', 'SUM', 'MIN', 'MAX', 'AVG', 'AND', 'COUNT']

with open('data/wikisql_dev_new.json', 'w') as file:
    for line in data:
        line = json.loads(line)
        table_id = line['table_id'].replace('-', '_')
        line['question'] = line['question'].lower()
        line['query'] = line['query'].lower()
        new_line = {'question': line['question'], 'query': line['query'], 'table_id': table_id}
        file.write(json.dumps(new_line) + '\n')



