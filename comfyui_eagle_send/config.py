import os


def get_eagle_host() -> str:
    host = os.environ.get("EAGLE_API_HOST")
    return host.strip() if isinstance(host, str) and host.strip() else "http://127.0.0.1:41595"

