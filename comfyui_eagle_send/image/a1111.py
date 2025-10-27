from __future__ import annotations
import json


def build_parameters(
    positive: str,
    width: int,
    height: int,
    model_basename: str | None,
    model_hash10: str | None,
    hashes: dict[str, str] | None,
) -> str:
    pos = (positive or "").strip()
    line1 = pos
    line2 = "Negative prompt: "
    # Include minimal A1111 keys so PNGInfo parsers recognize the block
    segs: list[str] = [
        "Steps: 0",
        "Sampler: ",
        "CFG scale: 0",
        "Seed: 0",
        f"Size: {width}x{height}",
    ]
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
