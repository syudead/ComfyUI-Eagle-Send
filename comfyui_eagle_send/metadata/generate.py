from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple

from ..parsing.workflow import parse_workflow_resources
from ..hash.compute import (
    calculate_sha256,
    short10,
    resolve_checkpoint_by_basename,
    resolve_loras_by_basenames,
)
from ..image.a1111 import build_parameters


def build_a1111_with_hashes(
    positive: str,
    width: int,
    height: int,
    extra_pnginfo: Any,
) -> Tuple[str, str, List[str]]:
    """Construct A1111 parameters string with model/LoRA short hashes.

    Returns (a1111_params, model_name, loras)
    """
    resources = parse_workflow_resources(extra_pnginfo)
    model_name = resources.get("model_name") or ""
    loras = resources.get("loras") or []
    lora_weights = resources.get("lora_weights") or {}

    model_hash_short = ""
    if model_name:
        ckpt_path = resolve_checkpoint_by_basename(model_name)
        if ckpt_path:
            try:
                model_hash_short = short10(calculate_sha256(ckpt_path))
            except Exception:
                model_hash_short = ""

    lora_paths = resolve_loras_by_basenames(loras)
    hashes_dict: Dict[str, str] = {}
    if model_hash_short:
        hashes_dict["model"] = model_hash_short
    for ln, lp in lora_paths.items():
        try:
            h = short10(calculate_sha256(lp))
            if h:
                hashes_dict[f"LORA:{ln}"] = h
        except Exception:
            pass

    def _fmt_weight(v: float) -> str:
        try:
            # Trim trailing zeros
            s = ("%g" % float(v))
            return s
        except Exception:
            return "1"

    # Do not modify positive prompt with <lora:...> tokens; keep as-is
    new_positive = (positive or "").strip()

    a1111_params = build_parameters(
        positive=new_positive,
        width=width,
        height=height,
        model_basename=model_name or "",
        model_hash10=model_hash_short,
        hashes=hashes_dict or None,
    )
    return a1111_params, model_name, loras
