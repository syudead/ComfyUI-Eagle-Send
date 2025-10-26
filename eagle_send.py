from __future__ import annotations
import os
import json
from typing import List, Any, Dict
import re
import folder_paths

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    # Prefer stdlib to avoid extra dependencies
    import urllib.request as _urlreq
except Exception:  # pragma: no cover
    _urlreq = None


class EagleSend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI/EagleSend"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "response")
    FUNCTION = "send"
    CATEGORY = "integration/Eagle"

    def _ensure_deps(self):
        if torch is None:
            raise RuntimeError("torch not available; required by ComfyUI IMAGE tensors")
        if Image is None:
            raise RuntimeError("PIL (Pillow) not available; install pillow")
        if _urlreq is None:
            raise RuntimeError("urllib not available in this Python environment")

    def _tensor_to_pil_list(self, images_tensor) -> List[Any]:
        self._ensure_deps()
        pil_images: List[Any] = []
        if images_tensor is None:
            return pil_images

        if not isinstance(images_tensor, torch.Tensor):
            raise TypeError("images must be a torch.Tensor from ComfyUI")

        tensor = images_tensor.detach().cpu().clamp(0.0, 1.0)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4 or tensor.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unexpected IMAGE tensor shape: {tuple(tensor.shape)}")

        tensor = (tensor * 255.0).round().to(torch.uint8)

        for frame_index in range(tensor.shape[0]):
            image_tensor_slice = tensor[frame_index]
            if image_tensor_slice.shape[-1] == 1:
                np_array = image_tensor_slice[:, :, 0].numpy()
                pil_image = Image.fromarray(np_array, mode="L")
            elif image_tensor_slice.shape[-1] == 3:
                np_array = image_tensor_slice.numpy()
                pil_image = Image.fromarray(np_array, mode="RGB")
            else:  # 4
                np_array = image_tensor_slice.numpy()
                pil_image = Image.fromarray(np_array, mode="RGBA")
            pil_images.append(pil_image)
        return pil_images

    def _save_images_output(self, pil_images: List[Any], filename_prefix: str) -> List[str]:
        paths: List[str] = []
        if not pil_images:
            return paths
        output_dir = folder_paths.get_output_directory()
        width, height = pil_images[0].size
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, width, height
        )
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
            pil_image.save(save_path, format="PNG")
            paths.append(save_path)
        return paths

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, str]:
        data = json.dumps(payload).encode("utf-8")
        req = _urlreq.Request(url, data=data, headers=headers, method="POST")
        try:
            with _urlreq.urlopen(req, timeout=30) as resp:
                code = getattr(resp, "status", resp.getcode())
                text = resp.read().decode("utf-8", errors="replace")
                return code, text
        except Exception as exc:  # include HTTPError, URLError
            if hasattr(exc, "code") and hasattr(exc, "read"):
                try:
                    text = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    text = str(exc)
                return int(getattr(exc, "code", 0) or 0), text
            return 0, str(exc)

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize separators to newlines and replace literal BREAK with newlines."""
        if not isinstance(prompt, str):
            return ""
        text = prompt.replace("\r\n", "\n").replace("\r", "\n")
        for sep in [",", "，", "、", ";", "；", "|", "｜", "/", "／"]:
            text = text.replace(sep, "\n")
        text = re.sub(r"(?i)\bBREAK\b", "\n", text)
        return token_strext

    def _clean_tag(self, token: str) -> str:
        """Trim, strip simple wrappers/weights, and collapse spaces."""
        token_str = (token or "").strip()
        if not token_str:
            return ""
        # Strip one layer of wrapping brackets/parens
        if token_str and token_str[0] in "([{":
            token_str = token_str[1:].strip()
        if token_str and token_str[-1] in ")]}":
            token_str = token_str[:-1].strip()
        # Convert "tag:1.2" -> "tag"
        match = re.match(r"^([^:(){}\\[\\]]+):\\d+(?:\\.\\d+)?$", token_str)
        if match:
            token_str = match.group(1).strip()
        token_str = re.sub(r"\\s+", " ", token_str)
        return token_str

    def _prompt_to_tags(self, prompt: str) -> List[str]:
        text = self._normalize_prompt(prompt)
        tags: List[str] = []
        seen: set[str] = set()
        for raw in text.split("\n"):
            tag = self._clean_tag(raw)
            if tag and tag not in seen:
                tags.append(tag)
                seen.add(tag)
            if len(tags) >= 128:
                break
        return token_strags

    def _send_to_eagle(self, host: str, endpoint_path: str, paths: List[str], tags: List[str]) -> Tuple[int, str]:
        base = host.strip().rstrip("/")
        path = "/" + endpoint_path.strip().lstrip("/")
        url = base + path

        payload: Dict[str, Any] = {"paths": paths}
        if isinstance(tags, list) and tags:
            payload["tags"] = tags

        headers = {"Content-Type": "application/json"}
        code, text = self._post_json(url, payload, headers)

        # Fallback: try singular key if server rejects plural
        if code == 404 or (code >= 400 and "paths" in text.lower()):
            results = []
            for p in paths:
                one_payload = dict(payload)
                one_payload.pop("paths", None)
                one_payload["path"] = p
                c, t = self._post_json(url, one_payload, headers)
                results.append({"path": p, "status": c, "response": t})
            return 207, json.dumps(results, ensure_ascii=False)

        return code, text

    def send(
        self,
        images,
        filename_prefix: str,
        prompt: str,
    ):
        pil_images = self._tensor_to_pil_list(images)
        saved_paths: List[str] = []
        if pil_images:
            saved_paths = self._save_images_output(pil_images, filename_prefix)

        # Hardcoded Eagle API config
        host = os.environ.get("EAGLE_API_HOST", "http://127.0.0.1:41595")
        endpoint_path = "/api/item/addFromPaths"

        tags = self._prompt_to_tags(prompt)
        code, resp_text = self._send_to_eagle(host, endpoint_path, saved_paths, tags)
        resp_text = f"HTTP {code}: {resp_text}"

        # No cleanup: images saved to output directory like SaveImage node

        return (images, resp_text)


NODE_CLASS_MAPPINGS = {
    "EagleSend": EagleSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleSend": "Eagle: Send Images",
}
