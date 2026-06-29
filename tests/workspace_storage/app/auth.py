def hash_password(password):
    # VULNERABLE: Use of weak MD5 hash
    import hashlib
    return hashlib.md5(password.encode()).hexdigest()

def hash_password_safe(password):
    # SECURE: Use of strong bcrypt hashing
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())