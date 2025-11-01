from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

from ..config import get_eagle_host
from ..image.tensor_convert import tensor_to_pil_list
from ..image.save import save_images_output
from ..metadata.generate import build_a1111_with_hashes
from ..parsing.tags import prompt_to_tags
from ..parsing.workflow import parse_workflow_resources
from ..eagle.api import send_to_eagle


class EagleSend:
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
            "optional": {
                "negative": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "d2_pipe": ("D2_TD2Pipe",),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "response")
    FUNCTION = "send"
    CATEGORY = "integration/Eagle"

    def send(
        self,
        images,
        filename_prefix: str,
        prompt: str,
        negative: str = "",
        d2_pipe=None,
        extra_pnginfo=None,
    ):
        pil_images = tensor_to_pil_list(images)
        saved_paths: List[str] = []
        if pil_images:
            width, height = pil_images[0].size
            # Extract overrides from d2_pipe if provided
            ov = {}
            try:
                if d2_pipe is not None:
                    # getattr-safe extraction; tolerate dict-like
                    getter = (lambda k: getattr(d2_pipe, k, None))
                    if isinstance(d2_pipe, dict):
                        getter = (lambda k: d2_pipe.get(k))
                    steps = getter("steps")
                    cfg = getter("cfg")
                    seed = getter("seed") or getter("noise_seed")
                    sampler_name = getter("sampler_name") or getter("sampler")
                    scheduler = getter("scheduler")
                    clip_skip = getter("clip_skip")
                    if steps is not None:
                        ov["steps"] = int(steps)
                    if cfg is not None:
                        ov["cfg_scale"] = float(cfg)
                    if seed is not None:
                        ov["seed"] = int(seed)
                    if sampler_name:
                        ov["sampler_name"] = str(sampler_name)
                    if scheduler:
                        ov["scheduler"] = str(scheduler)
                    if clip_skip is not None:
                        ov["clip_skip"] = int(clip_skip)
            except Exception:
                ov = {}

            a1111_params, model_name, loras, lora_weights = build_a1111_with_hashes(
                positive=prompt or "",
                negative=negative or "",
                width=width,
                height=height,
                extra_pnginfo=extra_pnginfo,
                overrides=ov or None,
            )
            saved_paths = save_images_output(
                pil_images,
                filename_prefix,
                prompt,
                extra_pnginfo,
                a1111_params=a1111_params,
            )

        host = get_eagle_host()
        tags = prompt_to_tags(prompt)

        # add model/lora from workflow (EXTRA_PNGINFO)
        if model_name:
            tag_model = f"model:{model_name}"
            if tag_model not in tags:
                tags.append(tag_model)
        for ln in loras:
            tag_lora = f"lora:{ln}"
            if tag_lora not in tags:
                tags.append(tag_lora)

        # Build Eagle memo (annotation) for Eagle
        try:
            from ..metadata.generate import build_eagle_annotation
            annotation_text = build_eagle_annotation(
                positive=prompt or "",
                negative=negative or "",
                width=width,
                height=height,
                model_name=model_name,
                loras=loras,
                lora_weights=lora_weights,
                overrides=ov or None,
                memo_text=None,
            )
        except Exception:
            annotation_text = a1111_params

        code, resp_text = send_to_eagle(host, saved_paths, tags, annotation=annotation_text)
        ok = 200 <= int(code or 0) < 300
        resp = {
            "http": code,
            "paths": len(saved_paths),
            "tags_count": len(tags),
            "tags": tags,
            "model_name": model_name,
            "loras": loras,
            "host": host,
            "parameters": a1111_params,
            "annotation": annotation_text,
            "success": ok,
            "body": resp_text,
        }
        return (images, json.dumps(resp, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "EagleSend": EagleSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleSend": "Eagle: Send Images",
}
