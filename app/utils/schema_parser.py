def parse_schema(schema_file):
    """
    解析表结构文件
    Args:
        schema_file (str): 表结构文件路径
    Returns:
        dict: 解析后的表结构数据
    """
    import json
    with open(schema_file, 'r', encoding='utf-8') as file:
        return json.load(file)

if __name__ == "__main__":
    schema = parse_schema('data/tables.json')
    print("table:", schema)
