def validate_sql(sql_query, schema):

    table_name = sql_query.split('FROM')[1].split()[0]
    return table_name in [table['table_name'] for table in schema]
