def get_user_data(request, db):
    # VULNERABLE: IDOR - No ownership check
    user_id = request.args.get('id')
    return db.query("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

def get_user_data_safe(request, db, current_user):
    # SECURE: ownership check enforced
    user_id = request.args.get('id')
    if user_id != current_user.id:
        return "Unauthorized", 403
    return db.query("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()