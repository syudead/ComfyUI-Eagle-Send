from __future__ import annotations
import json


def build_parameters(
    positive: str,
    negative: str | None,
    width: int,
    height: int,
    model_basename: str | None,
    model_hash10: str | None,
    hashes: dict[str, str] | None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    clip_skip: int | None = None,
) -> str:
    pos = (positive or "").strip()
    line1 = pos
    line2 = f"Negative prompt: {(negative or '').strip()}"
    # A1111 keys; fill with provided values or placeholders
    s_steps = steps if isinstance(steps, int) else 0
    s_cfg = cfg_scale if isinstance(cfg_scale, (int, float)) else 0
    s_seed = seed if isinstance(seed, int) else 0
    s_sampler = (sampler_name or "").strip()
    s_sched = (scheduler or "").strip()
    sampler_field = s_sampler
    if s_sched and s_sched != "normal":
        sampler_field = f"{s_sampler}_{s_sched}" if s_sampler else s_sched
    segs: list[str] = [
        f"Steps: {s_steps}",
        f"Sampler: {sampler_field}",
        f"CFG scale: {s_cfg}",
        f"Seed: {s_seed}",
        f"Size: {width}x{height}",
    ]
    if isinstance(clip_skip, int):
        segs.append(f"Clip skip: {abs(clip_skip)}")
    if model_hash10:
        segs.append(f"Model hash: {model_hash10}")
    if model_basename:
        segs.append(f"Model: {model_basename}")
    segs.append("Version: ComfyUI")
    if hashes:
        try:
            hashes_str = json.dumps(hashes, separators=(",", ":"))
            segs.append(f"Hashes: {hashes_str}")
        except Exception:
            pass
    line3 = ", ".join(segs)
    return f"{line1}\n{line2}\n{line3}"
