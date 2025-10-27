from __future__ import annotations
import re
from typing import List


def normalize_prompt(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = r"(?:\r?\n|,|;|\||/|\uFF0C|\u3001|\uFF1B|\uFF5C|\uFF0F|(?i:BREAK))"
    t = re.sub(pattern, "\n", t)
    t = re.sub(r"\n+", "\n", t)
    return t.strip("\n")


def _clean_tag(token: str) -> str:
    token_str = (token or "").strip()
    if not token_str:
        return ""
    if token_str and token_str[0] in "([{":
        token_str = token_str[1:].strip()
    if token_str and token_str[-1] in ")]}":
        token_str = token_str[:-1].strip()
    match = re.match(r"^([^:(){}\[\]]+):\d+(?:\.\d+)?$", token_str)
    if match:
        token_str = match.group(1).strip()
    token_str = re.sub(r"\s+", " ", token_str)
    return token_str


def prompt_to_tags(text: str, max_tags: int = 128) -> List[str]:
    normalized = normalize_prompt(text)
    tags: List[str] = []
    seen: set[str] = set()
    for raw in normalized.split("\n"):
        cleaned = _clean_tag(raw)
        if cleaned and cleaned not in seen:
            tags.append(cleaned)
            seen.add(cleaned)
        if len(tags) >= max_tags:
            break
    return tags

