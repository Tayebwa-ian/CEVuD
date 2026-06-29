def extract_zip(zip_file, dest_dir):
    # VULNERABLE: Zip Slip path traversal
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(dest_dir)

def extract_zip_safe(zip_file, dest_dir):
    # SECURE: Validating member paths to prevent traversal
    import zipfile
    import os
    with zipfile.ZipFile(zip_file, 'r') as zf:
        for member in zf.infolist():
            target_path = os.path.abspath(os.path.join(dest_dir, member.filename))
            if not target_path.startswith(os.path.abspath(dest_dir)):
                raise Exception("Path Traversal Attempt Detected")
            zf.extract(member, dest_dir)