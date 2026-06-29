def load_config(request):
    # VULNERABLE: Unsafe YAML loading
    import yaml
    data = request.get_data()
    return yaml.load(data)

def load_config_safe(request):
    # SECURE: Using safe_load
    import yaml
    data = request.get_data()
    return yaml.safe_load(data)