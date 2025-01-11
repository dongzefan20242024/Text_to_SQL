def generate_sql(table_name, columns, conditions=None):

    base_query = f"SELECT {', '.join(columns)} FROM {table_name}"
    if conditions:
        where_clause = " AND ".join([f"{col} {cond}" for col, cond in conditions.items()])
        return f"{base_query} WHERE {where_clause};"
    return f"{base_query};"
