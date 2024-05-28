import re
import sqlite3
import datetime
import json
import pandas as pd
from py2neo import Graph
import datetime

class Evaluator():
    def __init__(self, table_path):
        self.table_path = table_path
        self.conn = sqlite3.connect(table_path)
        self.cur = self.conn.cursor()

    def execute_sql(self, sql):
        self.cur.execute(sql)
        rows = self.cur.fetchall()
        return rows

    def evaluate_logic_form(self, ground_truth, predictions):
        assert len(ground_truth) == len(predictions)
        n = len(ground_truth)
        ex_num = 0
        for _gt, _pred in zip(ground_truth, predictions):
            if _gt == _pred:
                ex_num += 1

        return ex_num / n

class WikiSQLEvaluator(Evaluator):
    def __init__(self, table_path, table_file_path):
        super().__init__(table_path)
        # self.neo4j = Graph("http://166.111.80.64:27474", auth=("neo4j", "thss11-421"))
        self.table_map = {}
        self.table_file_path = table_file_path

    def replace_whole_word(self, text, word_to_replace, replacement):
        pattern = r'\s' + re.escape(word_to_replace) + r'\s'
        replacement1 = ' ' + replacement + ' '
        result = re.sub(pattern, replacement1, text)

        pattern = r'\(' + re.escape(word_to_replace) + r'\)'
        replacement2 = '(' + replacement + ')'
        result = re.sub(pattern, replacement2, result)

        return result
    
    def evaluate_execution(self, ground_truth, predictions, table_ids):
        assert len(ground_truth) == len(predictions)
        n = len(ground_truth)
        ex_num = 0
        error_case = []
        for gt, pred, table_id in zip(ground_truth, predictions, table_ids):
            if 'match' in gt:
                flag = 1
            else:
                flag = 0
            try:
                gt_results = self.execute_query(gt, flag, table_id)
            except:
                error_case.append((gt, table_id))
            try:
                pred_results = self.execute_query(pred, flag, table_id)
            except:
                pred_results = None

            if gt_results == pred_results:
                ex_num += 1

        with open(f'error_case_{datetime.datetime.now().strftime("%H%M%S")}.txt', 'w') as f:
            for q, table_id in error_case:
                f.write(q)
                f.write('\t')
                f.write(table_id)
                f.write('\n')
        
        return ex_num / n
    
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
        columns = [info[1] for info in table_info]
        # print(columns)
        # print(types)

        self.build_table_map()
        table = self.table_map[table_id]
        header = table['header']
        # print(header)
        column_map = dict(zip(header, columns))
        for k, v in column_map.items():
            # sql = sql.replace(k.lower(), v)
            sql = self.replace_whole_word(sql, k.lower(), v)

        # print(sql)
        if 'where' not in sql:
            self.cur.execute(sql)
            rows = self.cur.fetchall()
            # print(rows)
            return rows

        # 获取where子句
        sql = sql.strip().split(' ')
        
        # print(sql)
        # 没有condition
        if sql[-1] == 'where':
            sql = ' '.join(sql[:-1])
            self.cur.execute(sql)
            rows = self.cur.fetchall()
            return rows

        before_where = ' '.join(sql[:sql.index('where') + 1])
        where_clause = sql[sql.index('where') + 1:]
        where_clause = ' '.join(where_clause)
        conditions = where_clause.split(' and ')
        # print(conditions)
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
        # print(rows)
        return rows


    def execute_query(self, query, flag, table_id):
        if flag == 1:
            return self.execute_cypher(query, table_id)
        else:
            return self.execute_sql(query, table_id)
        
    def build_table_map(self):
        tables = pd.read_json(self.table_file_path, lines=True)
        tables.drop(columns=['page_title', 'rows', 'section_title', 'page_id', 'types'], inplace=True)
        data = pd.DataFrame()
        for index, line in tables.iterrows():
            self.table_map[line["id"].replace('-', '_')] = line

    def execution(self, pql, table_id):
        if 'select' in pql:
            return self.execute_sql(pql, table_id)
        else:
            return self.execute_cypher(pql, table_id)

class AdvisingEvaluator(Evaluator):
    def __init__(self, table_path):
        super().__init__(table_path)

    def evaluate_execution(self, ground_truth, predictions, file=None):
        assert len(ground_truth) == len(predictions)
        n = len(ground_truth)
        ex_num = 0
        tmp = 0
        for _gt, _pred in zip(ground_truth, predictions):
            gt = self.convert2sql(_gt)
            pred = self.convert2sql(_pred)
            # tmp += 1
            # print(tmp)
            gt_results = self.execute_sql(gt)
            if gt_results == []:
                if _gt == _pred:
                    ex_num += 1
                continue
            try:
                pred_results = self.execute_sql(pred)
                # print("pred")
            except:
                pred_results = None

            if gt_results == pred_results:
                ex_num += 1
        
        # with open(file + '_acc.txt', 'w') as f:
        #     f.write(f'Execution: {ex_num/n}')
        #     f.write('\n')
        return ex_num / n

    def convert2sql(self, pql):
        nested = '{ match (area:area) return area }'
        nested_1 = '{ match (area:area) return area as areaalias0 }'
        case1 = 'select count( * ) = 0 from course as coursealias0 , course_offering as course_offeringalias0 , instructor as instructoralias0 , offering_instructor as offering_instructoralias0 where coursealias0.course_id = course_offeringalias0.course_id and coursealias0.number = 611 and instructoralias0.name like "%Zuzana Srostlik%" or coursealias0.name like "%Alphonse Burdi%"'
        if nested in pql:
            pql = pql.replace(nested, '( select * from area ) as area')
        elif nested_1 in pql:
            pql = pql.replace(nested_1, '( select * from area ) as areaalias0')
        return pql
    
    def execution(self, pql):
        # return self.execute_sql(self.convert2sql(pql))
        return None

class RestaurantsEvaluator(Evaluator):
    def __init__(self, table_path):
        super().__init__(table_path)

    def evaluate_execution(self, ground_truth, predictions, file=None):
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
        
        # with open(file + '_acc.txt', 'w') as f:
        #     f.write(f'Execution: {ex_num/n}')
        #     f.write('\n')
        return ex_num / n


    def convert2sql(self, pql):
        nested = '{ match (g:geographic) where g.region = \"bay area\" return g.city_name }'
        alias0 = 'from { match (geographicalias1:geographic) where geographicalias1.region = \"bay area\" return geographicalias1 }'
        alias1 = 'from { match (geographicalias0:geographic) where geographicalias0.county = \"yolo county\" return geographicalias0 }'  # 3
        alias2 = 'from { match (geographicalias0:geographic) where geographicalias0.county = \"santa cruz county\" return geographicalias0 }' # 3
        alias3 = 'from { match (geographicalias0:geographic) where geographicalias0.region = \"bay area\" return geographicalias0 }'    # 60
        alias4 = 'from { match (geographicalias0:geographic) where geographicalias0.region = \"yosemite and mono lake area\" return geographicalias0 }'    # 33
        alias5 = 'from { match (geographic:geographic) where geographic.region = \"bay area\" return geographic }'
        alias6 = 'from { match (geographic:geographic) where geographic.county = \"yolo county\" return geographic }'  # 3
        alias7 = 'from { match (geographic:geographic) where geographic.county = \"santa cruz county\" return geographic }' # 3
        alias8 = 'from { match (geographic:geographic) where geographic.region = \"bay area\" return geographic }'    # 60
        alias9 = 'from { match (geographic:geographic) where geographic.region = \"yosemite and mono lake area\" return geographic }'    # 33

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
        if alias5 in pql:
            pql = pql.replace(alias5, 'from ( select * from geographic where geographic.region = \"bay area\" ) as geographic')
        if alias6 in pql:
            pql = pql.replace(alias6, 'from ( select * from geographic where geographic.county = \"yolo county\" ) as geographic')
        if alias7 in pql:
            pql = pql.replace(alias7, 'from ( select * from geographic where geographic.county = \"santa cruz county\" ) as geographic')
        if alias8 in pql:
            pql = pql.replace(alias8, 'from ( select * from geographic where geographic.region = \"bay area\" ) as geographic')
        if alias9 in pql:
            pql = pql.replace(alias9, 'from ( select * from geographic where geographic.region = \"yosemite and mono lake area\" ) as geographic')
        
        return pql
    
    def execution(self, pql):
        return self.execute_sql(self.convert2sql(pql))
    
class IMDBEvaluator(Evaluator):
    def __init__(self, table_path):
        self.table_path = table_path
    
    def evaluate_execution(self, ground_truth, predictions, file=None):
        return self.evaluate_logic_form(ground_truth, predictions)
    
    def execution(self, pql):
        return None

if __name__ == '__main__':
    evaluator = WikiSQLEvaluator('data/wikisql/dev.db', 'data/wikisql/dev.tables.jsonl')
    # evaluator = RestaurantsEvaluator('./restaurants/restaurants-db.added-in-2020.sqlite')
    # evaluator = AdvisingEvaluator('./advising/advising-db.added-in-2020.sqlite')

    # # print(out)
    with open('data/wikisql/wikisql_dev.json', 'r') as f:
        data = f.readlines()

    error_case = []
    num = 0
    for d in data:
        d = json.loads(d)
        if d['scope'] == 'sql':
            flag = 0
        else:
            flag = 1
        # num += 1
        # print(num)
        try:
            r = evaluator.execute_query(d['query'], flag, d['table_id'])
            # error_case.append(r)
        except Exception as e:
            # print(d['query'])
            print(e)
            error_case.append((d['query'], e))
        # try:
        #     gt = evaluator.convert2sql(d['query'])
        #     r = evaluator.execute_sql(gt)
        #     # print(r)
        # except Exception as e:
        #     print(e)
        #     error_case.append(d['query'])

    print(len(error_case))
    with open('error_case.txt', 'w') as f:
        for q, e in error_case:
            f.write(q)
            f.write('\t')
            f.write(e)
            f.write('\n')

    # q = 'select min(# of reigns) from table_ where '
    # r = evaluator.execute_sql(q, '1_10182508_5')
    # print(r)