import records
import re
from babel.numbers import parse_decimal, NumberFormatError
from py2neo import Graph

schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']


class NewDBEngine:

    def __init__(self, fdb):
        self.sql = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.sql.get_connection()
        self.neo4j = Graph("http://166.111.80.64:27474", auth=("neo4j", "thss11-421"))

    def execute_query_sql(self, table_id, query, *args, **kwargs):
        return self.execute_sql(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute_sql(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[
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
        out = self.conn.query(query, **where_map)
        return [o.result for o in out]

    def execute_cypher(self, table_id, select_index, aggregation_index, conditions, header):
        try:
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
            out = graph.run(query).data()
            return [o['result'] for o in out]
        except Exception as e:
            # with open('error.log', 'a') as f:
            #     f.write(str(e) + '\n')
            #     f.write('header: {}\n'.format(header))
            #     f.write('select_index: {}\n'.format(select_index))
            #     f.write('table_id: {}\n'.format(table_id))
            return None