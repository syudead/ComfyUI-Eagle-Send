from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

from ..config import get_eagle_host
from ..image.tensor_convert import tensor_to_pil_list
from ..image.save import save_images_output
from ..parsing.tags import prompt_to_tags
from ..parsing.workflow import parse_workflow_resources
from ..eagle.api import send_to_eagle


class EagleSend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI/EagleSend"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
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
        extra_pnginfo=None,
    ):
        pil_images = tensor_to_pil_list(images)
        saved_paths: List[str] = []
        if pil_images:
            saved_paths = save_images_output(pil_images, filename_prefix, prompt, extra_pnginfo)

        host = get_eagle_host()
        tags = prompt_to_tags(prompt)

        # add model/lora from workflow (EXTRA_PNGINFO)
        resources = parse_workflow_resources(extra_pnginfo)
        model_name = resources.get("model_name") or ""
        if model_name:
            tag_model = f"model:{model_name}"
            if tag_model not in tags:
                tags.append(tag_model)
        loras = resources.get("loras") or []
        for ln in loras:
            tag_lora = f"lora:{ln}"
            if tag_lora not in tags:
                tags.append(tag_lora)

        code, resp_text = send_to_eagle(host, saved_paths, tags)
        resp = {
            "http": code,
            "paths": len(saved_paths),
            "tags_count": len(tags),
            "tags": tags,
            "model_name": model_name,
            "loras": loras,
            "body": resp_text,
        }
        return (images, json.dumps(resp, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "EagleSend": EagleSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleSend": "Eagle: Send Images",
}

