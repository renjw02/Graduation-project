import re
import sqlite3
import datetime
import json
import pandas as pd
from py2neo import Graph

class Evaluator():
    def __init__(self, table_path):
        self.table_path = table_path
        self.conn = sqlite3.connect(table_path)
        self.cur = self.conn.cursor()

    def execute_sql(self, sql):
        self.cur.execute(sql)
        rows = self.cur.fetchall()
        return rows
    
class WikiSQLEvaluator(Evaluator):
    def __init__(self, table_path):
        super().__init__(table_path)
        self.neo4j = Graph("http://166.111.80.64:27474", auth=("neo4j", "thss11-421"))
        self.table_map = {}
    
    def evaluate_execution(self, ground_truth, predictions, file):
        assert len(ground_truth) == len(predictions)
        self.build_table_map()
        n = len(ground_truth)
        ex_num = 0
        for gt, pred in zip(ground_truth, predictions):
            if 'match' in gt:
                flag = 1
            else:
                flag = 0

            try:
                gt_results = self.execute_query(gt, flag)
            except:
                continue
            try:
                pred_results = self.execute_query(pred, flag)
            except:
                pred_results = None

            if gt_results == pred_results:
                ex_num += 1
    
    def execute_cypher(self, cypher, table_id):
        # match (n:table_) where n.years in toronto = 1995 -96 return n.school/club team
        pattern = r"match \(n:(\w+)\) where (.*) return n\.(.+)"
        match = re.search(pattern, cypher)
        if match:
            table = match.group(1)
            conditions = match.group(2)
            conditions = conditions.replace('n.', '').replace(' -', '-')
            column = match.group(3)
            # print('table:', table)
            # print('conditions:', conditions)
            # print('column:', column)
        else:
            return None
        
        sql = f"select {column} from {table} where {conditions}"
        # print(sql)
        
        return self.execute_sql(sql, table_id)

    def execute_sql(self, sql, table_id):
        sql = sql.replace('table_', 'table_' + table_id)
        # 先获取表结构
        self.cur.execute(f"PRAGMA  table_info(table_{table_id})")
        table_info = self.cur.fetchall()
        column_num = len(table_info)
        columns = [info[1] for info in table_info]
        types = [info[2] for info in table_info]
        # print(columns)
        # print(types)

        self.build_table_map()
        table = self.table_map[table_id]
        header = table['header']
        column_map = dict(zip(header, columns))
        for k, v in column_map.items():
            sql = sql.replace(k.lower(), v)

        # print(sql)
        if 'where' not in sql:
            self.cur.execute(sql)
            rows = self.cur.fetchall()
            print(rows)
            return rows

        # 获取where子句
        sql = sql.split(' ')
        before_where = ' '.join(sql[:sql.index('where') + 1])
        where_clause = sql[sql.index('where') + 1:]
        where_clause = ' '.join(where_clause)
        conditions = where_clause.split(' and ')
        for condition in conditions:
            condition = condition.split(' ')
            column = condition[0]
            ops = condition[1]
            value = ' '.join(condition[2:])  # str

            # if types[columns.index(column)] == 'TEXT':
            #     value = f"'{value}'"
            value = f"'{value}'"

            where_clause = where_clause.replace(' '.join(condition), f'{column} {ops} {value}')
        
        
        # print(where_clause)
        sql = before_where + ' ' + where_clause
        # print(sql)

        self.cur.execute(sql)
        rows = self.cur.fetchall()
        print(rows)
        return rows


    def execute_query(self, query, flag, table_id):
        if flag == 1:
            return self.execute_cypher(query, table_id)
        else:
            return self.execute_sql(query, table_id)
        
    def build_table_map(self):
        tables = pd.read_json('data/wikisql/test.tables.jsonl', lines=True)
        data = pd.DataFrame()
        for index, line in tables.iterrows():
            self.table_map[line["id"].replace('-', '_')] = line

class AdvisingEvaluator(Evaluator):
    def __init__(self, table_path):
        super().__init__(table_path)

    def evaluate_execution(self, ground_truth, predictions, file):
        assert len(ground_truth) == len(predictions)
        n = len(ground_truth)
        ex_num = 0
        tmp = 0
        for _gt, _pred in zip(ground_truth, predictions):
            gt = self.convert2sql(_gt)
            pred = self.convert2sql(_pred)
            tmp += 1
            print(tmp)
            gt_results = self.execute_sql(gt)
            if gt_results == []:
                if _gt == _pred:
                    ex_num += 1
                continue
            try:
                pred_results = self.execute_sql(pred)
                print("pred")
            except:
                pred_results = None

            if gt_results == pred_results:
                ex_num += 1
        
        with open(file + '_acc.txt', 'w') as f:
            f.write(f'Execution: {ex_num/n}')
            f.write('\n')

    def convert2sql(self, pql):
        nested = '{ match (a:area) return a as areaalias0 }'
        case1 = 'select count( * ) = 0 from course as coursealias0 , course_offering as course_offeringalias0 , instructor as instructoralias0 , offering_instructor as offering_instructoralias0 where coursealias0.course_id = course_offeringalias0.course_id and coursealias0.number = 611 and instructoralias0.name like "%Zuzana Srostlik%" or coursealias0.name like "%Alphonse Burdi%"'
        if nested in pql:
            pql = pql.replace(nested, '( select * from area ) as areaalias0')
        return pql



class RestaurantsEvaluator(Evaluator):
    def __init__(self, table_path):
        super.__init__(table_path)

    def evaluate_execution(self, ground_truth, predictions, file):
        # evaluate the user based on the restaurants
        assert len(ground_truth) == len(predictions)
        n = len(ground_truth)
        ex_num = 0
        for gt, pred in zip(ground_truth, predictions):
            gt = self.convert2sql(gt)
            pred = self.convert2sql(pred)
            gt_results = self.execute_sql(gt)
            try:
                pred_results = self.execute_sql(pred)
            except:
                pred_results = None
            

            if gt_results == pred_results:
                ex_num += 1
        
        with open(file + '_acc.txt', 'w') as f:
            f.write(f'Execution: {ex_num/n}')
            f.write('\n')


    def convert2sql(self, pql):
        nested = '{ match (g:geographic) where g.region = \"bay area\" return g.city_name }'
        alias0 = 'from { match (geographicalias1:geographic) where geographicalias1.region = \"bay area\" return geographicalias1 }'
        alias1 = 'from { match (geographicalias0:geographic) where geographicalias0.county = \"yolo county\" return geographicalias0 }'  # 3
        alias2 = 'from { match (geographicalias0:geographic) where geographicalias0.county = \"santa cruz county\" return geographicalias0 }' # 3
        alias3 = 'from { match (geographicalias0:geographic) where geographicalias0.region = \"bay area\" return geographicalias0 }'    # 60
        alias4 = 'from { match (geographicalias0:geographic) where geographicalias0.region = \"yosemite and mono lake area\" return geographicalias0 }'    # 33

        if nested in pql:
            pql = pql.replace(nested, '( select geographic.city_name from geographic where geographic.region = "bay area" )')
        if alias0 in pql:
            pql = pql.replace(alias0, 'from ( select * from geographic where geographic.region = \"bay area\" ) as geographicalias1')
        if alias1 in pql:
            pql = pql.replace(alias1, 'from ( select * from geographic where geographic.county = \"yolo county\" ) as geographicalias0')
        if alias2 in pql:
            pql = pql.replace(alias2, 'from ( select * from geographic where geographic.county = \"santa cruz county\" ) as geographicalias0')
        if alias3 in pql:
            pql = pql.replace(alias3, 'from ( select * from geographic where geographic.region = \"bay area\" ) as geographicalias0')
        if alias4 in pql:
            pql = pql.replace(alias4, 'from ( select * from geographic where geographic.region = \"yosemite and mono lake area\" ) as geographicalias0')
        
        return pql
    
if __name__ == '__main__':
    evaluator = WikiSQLEvaluator('./data/wikisql/test.db')
    # c = 'match (n:table_) where n.party = democratic and n.state = new york and n.representative = terence j. quinn return n.lifespan'
    # evaluator.execute_cypher(c, '2_12601141_1')
    # evaluator.execute_sql("select sum(2007) from table_ where 2009 < 7,006 and 2008 > 2,226", '2_1597347_1')
    # evaluator.execute_sql('select * from table_', '2_1597347_1')

    # # print(out)
    # with open('./data/wikisql_test.json', 'r') as f:
    #     data = f.readlines()

    # error_case = []
    # for d in data:
    #     d = json.loads(d)
    #     if 'select' in d['query']:
    #         flag = 0
    #     else:
    #         flag = 1
    #     try:
    #         r = evaluator.execute_query(d['query'], flag, d['table_id'])
    #         # error_case.append(r)
    #     except Exception as e:
    #         print(e)
    #         error_case.append(d['query'])

    # with open('error_case.txt', 'w') as f:
    #     for e in error_case:
    #         for i in e:
    #             f.write(str(i))
    #         f.write('\n')