from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

try:
    import urllib.request as _urlreq  # type: ignore
except Exception:  # pragma: no cover
    _urlreq = None  # type: ignore


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, str]:
    if _urlreq is None:
        return 0, "urllib not available in this Python environment"
    data = json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(url, data=data, headers=headers, method="POST")
    try:
        with _urlreq.urlopen(req, timeout=30) as resp:
            code = getattr(resp, "status", resp.getcode())
            text = resp.read().decode("utf-8", errors="replace")
            return code, text
    except Exception as exc:  # include HTTPError, URLError
        if hasattr(exc, "code") and hasattr(exc, "read"):
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                text = str(exc)
            return int(getattr(exc, "code", 0) or 0), text
        return 0, str(exc)


def send_to_eagle(host: str, paths: List[str], tags: List[str]) -> Tuple[int, str]:
    base = host.strip().rstrip("/")
    url = base + "/api/item/addFromPaths"
    items: List[Dict[str, Any]] = []
    for p in paths:
        item: Dict[str, Any] = {"path": p}
        if tags:
            item["tags"] = tags
        items.append(item)
    payload: Dict[str, Any] = {"items": items}
    headers = {"Content-Type": "application/json"}
    return _post_json(url, payload, headers)

