def read_user_file(request):
    # VULNERABLE: Path Traversal
    import os
    filename = request.args.get('file')
    filepath = os.path.join("/var/www/uploads", filename)
    with open(filepath, 'r') as f:
        return f.read()

def read_user_file_safe(request):
    # SECURE: Use abspath and check prefix
    import os
    base_dir = "/var/www/uploads"
    filename = request.args.get('file')
    filepath = os.path.abspath(os.path.join(base_dir, filename))
    if not filepath.startswith(os.path.abspath(base_dir)):
        return "Access Denied", 403
    with open(filepath, 'r') as f:
        return f.read()