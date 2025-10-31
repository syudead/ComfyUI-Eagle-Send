from __future__ import annotations
import hashlib
import json
import os
from typing import Dict, Optional

import folder_paths

# Minimal persistent cache for file hashes
# Key: normalized absolute path (normcase(realpath(abspath(path))))
# Val: [size:int, mtime_ns:int, sha256:str]
_CACHE: Dict[str, list] = {}
_CACHE_MTIME: float | None = None


def _norm_abs_path(path: str) -> str:
    try:
        # Normalize to a stable absolute path for cache keying
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))
    except Exception:
        return path


def _cache_file_path() -> str:
    # Store alongside the package root (comfyui_eagle_send/hash_cache.json)
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "hash_cache.json")


def _load_cache_if_changed() -> None:
    global _CACHE, _CACHE_MTIME
    try:
        p = _cache_file_path()
        st = os.stat(p)
        mtime = st.st_mtime
        if _CACHE_MTIME is not None and _CACHE_MTIME == mtime:
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # ensure only expected structures are kept
                new_cache: Dict[str, list] = {}
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, list) and len(v) == 3:
                        size, mtime_ns, sha = v
                        if isinstance(size, int) and isinstance(mtime_ns, int) and isinstance(sha, str):
                            new_cache[k] = [size, mtime_ns, sha]
                _CACHE = new_cache
            else:
                _CACHE = {}
        _CACHE_MTIME = mtime
    except FileNotFoundError:
        _CACHE = {}
        _CACHE_MTIME = None
    except Exception:
        # On any error, fall back to empty cache
        _CACHE = {}
        _CACHE_MTIME = None


def _save_cache() -> None:
    global _CACHE, _CACHE_MTIME
    try:
        p = _cache_file_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f, ensure_ascii=False, separators=(",", ":"))
        # Refresh observed mtime after write
        try:
            _CACHE_MTIME = os.stat(p).st_mtime
        except Exception:
            _CACHE_MTIME = None
    except Exception:
        # Ignore persistence failures to keep hashing functional
        pass


def calculate_sha256(file_path: str) -> str:
    # Try to use persistent+in-memory cache keyed by absolute normalized path
    key = _norm_abs_path(file_path)
    file_size: Optional[int] = None
    file_mtime_ns: Optional[int] = None
    try:
        st = os.stat(file_path)
        file_size = int(st.st_size)
        # Prefer nanosecond precision when available
        file_mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except Exception:
        # If stat fails, skip cache and compute directly
        pass

    if file_size is not None and file_mtime_ns is not None:
        _load_cache_if_changed()
        try:
            entry = _CACHE.get(key)
            if (
                isinstance(entry, list)
                and len(entry) == 3
                and isinstance(entry[0], int)
                and isinstance(entry[1], int)
                and isinstance(entry[2], str)
                and entry[0] == file_size
                and entry[1] == file_mtime_ns
            ):
                return entry[2]
        except Exception:
            pass

    # Cache miss or stat unavailable: compute hash
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    digest = sha256_hash.hexdigest()

    if file_size is not None and file_mtime_ns is not None:
        try:
            _CACHE[key] = [file_size, file_mtime_ns, digest]
            _save_cache()
        except Exception:
            pass
    return digest


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
