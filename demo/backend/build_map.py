file1 = 'dataset/imdb/imdb_preprocessed.json'
file2 = 'dataset/imdb/imdb_original.json'

import json
with open(file1, 'r') as f:
    data1 = json.load(f)

with open(file2, 'r') as f:
    data2 = f.readlines()


for i in range(len(data2)):
    data2[i] = json.loads(data2[i])


data = []
for i in range(len(data1)):
    for j in range(len(data2)):
        if data1[i]['input_sequence'].split("|")[0].strip() == data2[j]['question'].strip():
            tmp = {}
            tmp['pql'] = data1[i]['output_sequence'].split('|')[1].strip()
            tmp['sql'] = data2[j]['query']
            data.append(tmp)
            break

with open('dataset/imdb/imdb_pql_sql.json', 'w') as f:
    json.dump(data, f, indent=4)
            
