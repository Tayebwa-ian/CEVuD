def log_event(request):
    # VULNERABLE: Log Injection
    import logging
    user_input = request.args.get('data')
    logging.info(f"User input received: {user_input}")

def log_event_safe(request):
    # SECURE: Sanitize input for logs to prevent CRLF injection
    import logging
    user_input = request.args.get('data').replace('\n', '').replace('\r', '')
    logging.info(f"User input received: {user_input}")