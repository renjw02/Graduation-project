import sqlite3
import json

conn = sqlite3.connect('dataset/imdb/imdb.sqlite')
c = conn.cursor()

# with open('dataset/imdb/imdb_pql_sql.json', 'r') as f:
#     data = json.load(f)

# for d in data:
#     c.execute(d['sql'])
#     rows = c.fetchall()
#     print(rows)

q = 'select actoralias0.birth_year from actor as actoralias0 where actoralias0.name = \"Ellen Page\" ;'
c.execute(q)
print(c.fetchall())

c.close()

