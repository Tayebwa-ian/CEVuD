def render_user_profile(request):
    # VULNERABLE: Reflected XSS
    from flask import render_template_string
    name = request.args.get('name', 'Guest')
    return render_template_string(f"<h1>Welcome {name}</h1>")

def render_user_profile_safe(request):
    # SECURE: Using standard render_template with auto-escaping
    from flask import render_template
    name = request.args.get('name', 'Guest')
    return render_template("profile.html", name=name)