# -- coding: utf-8 --**

import json
import re
import records
from py2neo import *
from babel.numbers import parse_decimal, NumberFormatError

def get_neo4j_table(file_path):
    tables = []
    with open(file_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            info = json.loads(line.strip())
            if info['scope'] == 'cypher':
                tables.append(info['table_id'])
    return tables

def get_table_details(table_ids, file_path):
    tables = []
    with open(file_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            info = json.loads(line.strip())
            if info['id'] in table_ids:
                tables.append(info)
    return tables

def connect_neo4j(tables):
    graph = Graph("http://166.111.80.64:27474", auth=("neo4j", "thss11-421"))
    
    for table in tables:
        for row in table['rows']:
            node = Node(table['id'], **dict(zip(table['header'], row)))
            # print(node)
            graph.create(node)

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

# 存储进neo4j的时候，由于没有列的概念，sql中的列的顺序被打乱了，因此通过列的索引无法获取列名
# 需要先查询sql获取列名，然后再查询neo4j
def execute_cypher(table_id, select_index, aggregation_index, conditions, header):
    graph = Graph("http://166.111.80.64:27474", auth=("neo4j", "thss11-421"))
    select = 'n.`' + header[select_index] + '`'
    agg = agg_ops[aggregation_index]
    if agg:
        select = '{}({})'.format(agg, select)
    where_clause = []
    where_map = {}
    for col_index, op, val in conditions:
        if type(val) == str:
            if '"' in val:
                where_clause.append('`{}` {} \'{}\''.format(header[col_index], cond_ops[op], val))
            else:
                where_clause.append('`{}` {} \"{}\"'.format(header[col_index], cond_ops[op], val))
        else:
            where_clause.append('`{}` {} {}'.format(header[col_index], cond_ops[op], val))
    where_str = ''
    if where_clause:
        where_str = 'WHERE n.' + ' AND n.'.join(where_clause)
    query = 'MATCH (n:`{table_id}`) {where_str} RETURN {select} AS result'.format(table_id=table_id, select=select, where_str=where_str)
    print(query)
    out = graph.run(query).data()
    return [o['result'] for o in out]
    


schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')


def execute_sql(table_id, select_index, aggregation_index, conditions, lower=True):
        db = records.Database('sqlite:///{}'.format('data/test.db'))
        conn = db.get_connection()

        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[
            0].sql.replace('\n', '')
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, bytes)):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    # print (val)
                    val = float(parse_decimal(val, locale='en_US'))
                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(col_index, cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        out = conn.query(query, **where_map)
        return [o.result for o in out]

if __name__ == "__main__":
    # table_ids = get_neo4j_table('data/tokenized_test.jsonl')
    # tables = get_table_details(table_ids, 'data/test.tables.jsonl')
    # print(len(tables))
    # connect_neo4j(tables)
    # print(execute_sql('1-10015132-16', 2, 0, [[0, 0, "Terrence Ross"]]))
    # print(execute_cypher('1-10015132-16', 2, 0, [[0, 0, "Terrence Ross"]], ["Player","No.","Nationality","Position","Years in Toronto","School\/Club Team"]))


"""
TEST code

"""
   # MATCH (n:`1-10015132-16`) WHERE n.player = "terrence ross" RETURN n.nationality
    # cur_query = ["MATCH", "(", "n", ":", "table_", ")", "WHERE", "n.winning", "constructor", "EQL", "benetton", "-", "ford", "AND", "n.pole", "position", "EQL", "damon", "hill", "RETURN", "none", "(", "n.round", ")"]
    # cur_where_query = cur_query[cur_query.index('WHERE'):cur_query.index('RETURN')]
    # print(cur_where_query)
    # cur_where_query = [item.replace('n.', '') for item in cur_where_query]
    # print(cur_where_query)