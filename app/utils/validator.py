def validate_sql(sql_query):
    """
    Validates SQL syntax. Placeholder for a real SQL validation logic.
    """
    if not sql_query.strip().lower().startswith("select"):
        return False
    return True
