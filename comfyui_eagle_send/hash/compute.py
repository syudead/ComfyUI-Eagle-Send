from __future__ import annotations
import hashlib
import os
from typing import Dict, Optional

import folder_paths


def calculate_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def short10(sha256_hex: str) -> str:
    return (sha256_hex or "")[:10]


def _basename_no_ext(name: str) -> str:
    base = os.path.basename(name)
    if "." in base:
        base = ".".join(base.split(".")[:-1])
    return base


def resolve_checkpoint_by_basename(model_basename: str) -> Optional[str]:
    try:
        names = folder_paths.get_filename_list("checkpoints")
        target = None
        b = model_basename.lower()
        for n in names:
            if _basename_no_ext(n).lower() == b:
                target = n
                break
        if target:
            return folder_paths.get_full_path("checkpoints", target)
    except Exception:
        pass
    return None


def resolve_loras_by_basenames(lora_basenames: list[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        all_loras = folder_paths.get_filename_list("loras")
        lower_map = { _basename_no_ext(n).lower(): n for n in all_loras }
        for name in lora_basenames:
            key = (name or "").lower()
            if key in lower_map:
                out[name] = folder_paths.get_full_path("loras", lower_map[key])
    except Exception:
        pass
    return out

