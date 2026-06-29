def run_ping(request):
    # VULNERABLE: Command injection via shell=True
    import subprocess
    hostname = request.args.get('host')
    return subprocess.check_output(f"ping -c 1 {hostname}", shell=True)

def run_ping_safe(request):
    # SECURE: Avoiding shell=True and using argument list
    import subprocess
    hostname = request.args.get('host')
    return subprocess.check_output(["ping", "-c", "1", hostname])