from __future__ import annotations
import os
import json
from typing import Any, Dict, List

import folder_paths  # ComfyUI helper


def save_images_output(
    pil_images: List[Any],
    filename_prefix: str,
    prompt: str | None,
    extra_pnginfo: Dict[str, Any] | None,
    a1111_params: str | None = None,
) -> List[str]:
    paths: List[str] = []
    if not pil_images:
        return paths
    output_dir = folder_paths.get_output_directory()
    width, height = pil_images[0].size
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, output_dir, width, height
    )

    pnginfo = None
    try:
        from PIL.PngImagePlugin import PngInfo  # type: ignore

        pnginfo = PngInfo()
        # parameters: prefer A1111 string if provided; fallback to raw prompt
        params_text = a1111_params if (isinstance(a1111_params, str) and a1111_params.strip()) else (prompt or "")
        if params_text:
            pnginfo.add_text("parameters", params_text)
        if isinstance(extra_pnginfo, dict):
            for key, val in extra_pnginfo.items():
                try:
                    pnginfo.add_text(key, json.dumps(val))
                except Exception:
                    pass
        # Also keep original prompt JSON for tools that read it
        try:
            if isinstance(prompt, str):
                pnginfo.add_text("prompt", json.dumps(prompt))
        except Exception:
            pass
    except Exception:
        pnginfo = None

    has_batch_token = "%batch_num%" in filename
    for batch_number, pil_image in enumerate(pil_images):
        if has_batch_token:
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            cur_counter = counter
        else:
            filename_with_batch_num = filename
            cur_counter = counter + batch_number
        file_name = f"{filename_with_batch_num}_{cur_counter:05}_.png"
        save_path = os.path.join(full_output_folder, file_name)
        if pnginfo is not None:
            pil_image.save(save_path, format="PNG", pnginfo=pnginfo)
        else:
            pil_image.save(save_path, format="PNG")
        paths.append(save_path)
    return paths
