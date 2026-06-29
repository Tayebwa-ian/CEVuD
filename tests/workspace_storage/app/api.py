def proxy_request(request):
    # VULNERABLE: SSRF - direct use of user-supplied URL
    import requests
    target_url = request.args.get('url')
    return requests.get(target_url).text

def proxy_request_safe(request):
    # SECURE: Validate URL against allowlist
    import requests
    ALLOWED_DOMAINS = ['api.trusted.com']
    target_url = request.args.get('url')
    domain = target_url.split('/')[2]
    if domain in ALLOWED_DOMAINS:
        return requests.get(target_url).text
    return "Unauthorized", 403