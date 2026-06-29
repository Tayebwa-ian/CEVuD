def get_user_by_id(cursor, user_id):
    # VULNERABLE: SQL Injection via string formatting
    query = "SELECT * FROM users WHERE id = %s" % user_id
    return cursor.execute(query)

def get_user_by_id_safe(cursor, user_id):
    # SECURE: Using parameterized queries
    query = "SELECT * FROM users WHERE id = ?"
    return cursor.execute(query, (user_id,))