from library.query import Query
import sqlparse
import json

def sql_to_cypher(sql_statement):
    # 使用sqlparse解析SQL语句
    parsed = sqlparse.parse(sql_statement)[0]
    # print(parsed.tokens)

    columns = []
    table = None
    agg = None
    conditions = []
    from_flag = False
    for token in parsed.tokens[1:]:      
        if token.value.upper() == 'FROM':
            from_flag = True
        # 获取select部分的列名
        if not from_flag and not token.is_whitespace:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    columns.append(identifier.value)
            elif isinstance(token, sqlparse.sql.Function):
                agg = token.tokens[0].value
                columns.append(token.tokens[1].value[1:-1])
            else:
                columns.append(token.value)
        
        # 获取from部分的表名 wikisql只有单表
        if from_flag:
            if isinstance(token, sqlparse.sql.Identifier):
                table = token.get_real_name()
        
        # 获取where部分的条件
        if isinstance(token, sqlparse.sql.Where):
            # print(token.tokens)
            condition = []
            for _token in token.tokens[1:]:
                if _token.value.upper() == 'AND':
                    conditions.append(' '.join(condition))
                    condition.clear()
                elif not _token.is_whitespace:
                    condition.append(_token.value)
            if condition:
                conditions.append(' '.join(condition))

    if agg is None:
        return f"MATCH (n:{table}) WHERE n.{' AND n.'.join(conditions)} RETURN n.{' '.join(columns)}"

    return f"MATCH (n:{table}) WHERE n.{' AND n.'.join(conditions)} RETURN {agg}(n.{' '.join(columns)})"


def get_neo4j_query(data):
    print('convert {data} dataset'.format(data=data))
    data_path = 'data/tokenized_' + data + '.jsonl'
    new_data = []

    with open(data_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            info = json.loads(line.strip())
            if info['tokenized_query'][0] == 'MATCH':
                info['scope'] = 'cypher'
            else:
                info['scope'] = 'sql'
            new_data.append(json.dumps(info) + '\n')

    new_data_path = 'data/tokenized_' + data + '_new.jsonl'
    with open(new_data_path, 'w', encoding='utf-8') as lines:
        lines.writelines(new_data)

# 找到tokeknized_test_new.jsonl中的cypher语句,并找出对应的table文件中的表