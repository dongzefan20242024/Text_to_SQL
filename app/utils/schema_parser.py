import json

def parse_schema(schema_path):
    """
    Parses schema JSON file and returns table information.
    """
    with open(schema_path, 'r') as file:
        schema = json.load(file)
    return schema
