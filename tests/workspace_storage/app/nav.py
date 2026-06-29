def redirect_to_next(request):
    # VULNERABLE: Open Redirect
    from flask import redirect
    target = request.args.get('next')
    return redirect(target)

def redirect_to_next_safe(request):
    # SECURE: Validate redirect target domain
    from flask import redirect, url_for
    target = request.args.get('next')
    if target.startswith('/'):
        return redirect(target)
    return redirect(url_for('index'))