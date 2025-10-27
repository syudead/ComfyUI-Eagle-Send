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


def _fmt_weight(v: float) -> str:
    try:
        return ("%g" % float(v))
    except Exception:
        return "1"


def build_a1111_with_hashes(
    positive: str,
    negative: str | None,
    width: int,
    height: int,
    extra_pnginfo: Any,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[str, str, List[str], Dict[str, float]]:
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


    # Do not modify positive prompt with <lora:...> tokens; keep as-is
    new_positive = (positive or "").strip()

    ov = overrides or {}
    # Map sampler/scheduler to Civitai-pretty name (based on d2 send eagle)
    def _sampler_for_civitai(sampler: str | None, scheduler: str | None) -> str:
        s = (sampler or "").strip()
        sch = (scheduler or "").strip()
        def with_karras(name: str) -> str:
            return f"{name} Karras" if sch == "karras" else name
        def with_karras_exp(name: str) -> str:
            if sch == "karras":
                return f"{name} Karras"
            if sch == "exponential":
                return f"{name} Exponential"
            return name
        match s:
            case "euler" | "euler_cfg_pp":
                return "Euler"
            case "euler_ancestral" | "euler_ancestral_cfg_pp":
                return "Euler a"
            case "heun" | "heunpp2":
                return "Heun"
            case "dpm_2":
                return with_karras("DPM2")
            case "dpm_2_ancestral":
                return with_karras("DPM2 a")
            case "lms":
                return with_karras("LMS")
            case "dpm_fast":
                return "DPM fast"
            case "dpm_adaptive":
                return "DPM adaptive"
            case "dpmpp_2s_ancestral":
                return with_karras("DPM++ 2S a")
            case "dpmpp_sde" | "dpmpp_sde_gpu":
                return with_karras("DPM++ SDE")
            case "dpmpp_2m":
                return with_karras("DPM++ 2M")
            case "dpmpp_2m_sde" | "dpmpp_2m_sde_gpu":
                return with_karras("DPM++ 2M SDE")
            case "dpmpp_3m_sde" | "dpmpp_3m_sde_gpu":
                return with_karras_exp("DPM++ 3M SDE")
            case "lcm":
                return "LCM"
            case "ddim":
                return "DDIM"
            case "uni_pc" | "uni_pc_bh2":
                return "UniPC"
        if sch == "normal" or not sch:
            return s
        return f"{s}_{sch}" if s else sch

    sampler_pretty = None
    if ov.get("sampler_name") is not None:
        sampler_pretty = _sampler_for_civitai(ov.get("sampler_name"), ov.get("scheduler"))
    a1111_params = build_parameters(
        positive=new_positive,
        negative=negative or "",
        width=width,
        height=height,
        model_basename=model_name or "",
        model_hash10=model_hash_short,
        hashes=hashes_dict or None,
        steps=ov.get("steps"),
        cfg_scale=ov.get("cfg_scale"),
        seed=ov.get("seed"),
        sampler_name=sampler_pretty if sampler_pretty is not None else ov.get("sampler_name"),
        scheduler=None,  # already encoded in sampler_pretty if applicable
        clip_skip=ov.get("clip_skip"),
    )
    return a1111_params, model_name, loras, lora_weights


def build_eagle_annotation(
    positive: str,
    negative: str | None,
    width: int,
    height: int,
    model_name: str | None,
    loras: List[str] | None = None,
    lora_weights: Dict[str, float] | None = None,
    overrides: Dict[str, Any] | None = None,
    memo_text: str | None = None,
) -> str:
    ov = overrides or {}
    pos = (positive or "").strip()
    neg = (negative or "").strip()
    steps = ov.get("steps") if isinstance(ov.get("steps"), int) else 0
    cfg = ov.get("cfg_scale") if isinstance(ov.get("cfg_scale"), (int, float)) else 0
    seed = ov.get("seed") if isinstance(ov.get("seed"), int) else 0
    sampler = str(ov.get("sampler_name") or "").strip()
    scheduler = str(ov.get("scheduler") or "").strip()
    clip_skip = ov.get("clip_skip") if isinstance(ov.get("clip_skip"), int) else None
    model = model_name or ""
    lines: list[str] = []
    # Positive
    lines.append(pos)
    lines.append("")
    # Negative (ensure trailing newline after the line)
    lines.append(f"Negative prompt:{neg}")
    # Model immediately after Negative
    lines.append(f"Model: {model}")
    # LoRA weighted list
    if loras:
        lw = lora_weights or {}
        for ln in loras:
            if ln in lw:
                lines.append(f"LoRA: {ln} ({_fmt_weight(lw[ln])})")
            else:
                lines.append(f"LoRA: {ln}")
    # Empty line separator
    lines.append("")
    # Remaining parameters
    tail = f"Steps: {steps}, Sampler: {sampler} {scheduler}, CFG scale: {cfg}, Seed: {seed}, Size: {width}x{height}"
    if clip_skip is not None:
        tail += f", Clip skip: {abs(clip_skip)}"
    lines.append(tail)
    if memo_text and memo_text.strip():
        lines.append("")
        lines.append(f"Memo: {memo_text.strip()}")
    return "\n".join(lines)
