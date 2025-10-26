from __future__ import annotations
import os
import json
from typing import List, Tuple, Any, Dict
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
    import urllib.error as _urlerr
except Exception:  # pragma: no cover
    _urlreq = None
    _urlerr = None


class EagleSend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI/EagleSend"}),
                "folder_id": ("STRING", {"default": ""}),
                "tags": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": ""}),
                "annotation": ("STRING", {"default": "", "multiline": True}),
                "dry_run": ("BOOLEAN", {"default": False}),
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
        imgs: List[Any] = []
        if images_tensor is None:
            return imgs

        if not isinstance(images_tensor, torch.Tensor):
            raise TypeError("images must be a torch.Tensor from ComfyUI")

        t = images_tensor.detach().cpu().clamp(0.0, 1.0)

        if t.ndim == 3:
            t = t.unsqueeze(0)

        if t.ndim != 4 or t.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unexpected IMAGE tensor shape: {tuple(t.shape)}")

        t = (t * 255.0).round().to(torch.uint8)

        for i in range(t.shape[0]):
            img = t[i]
            if img.shape[-1] == 1:
                arr = img[:, :, 0].numpy()
                pil = Image.fromarray(arr, mode="L")
            elif img.shape[-1] == 3:
                arr = img.numpy()
                pil = Image.fromarray(arr, mode="RGB")
            else:  # 4
                arr = img.numpy()
                pil = Image.fromarray(arr, mode="RGBA")
            imgs.append(pil)
        return imgs

    def _save_images_output(self, pil_images: List[Any], filename_prefix: str) -> List[str]:
        paths: List[str] = []
        if not pil_images:
            return paths
        output_dir = folder_paths.get_output_directory()
        w, h = pil_images[0].size
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, w, h
        )
        has_batch_token = "%batch_num%" in filename
        for batch_number, im in enumerate(pil_images):
            if has_batch_token:
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                cur_counter = counter
            else:
                filename_with_batch_num = filename
                cur_counter = counter + batch_number
            file = f"{filename_with_batch_num}_{cur_counter:05}_.png"
            save_path = os.path.join(full_output_folder, file)
            im.save(save_path, format="PNG")
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
        except Exception as e:  # include HTTPError, URLError
            if hasattr(e, "code") and hasattr(e, "read"):
                try:
                    text = e.read().decode("utf-8", errors="replace")
                except Exception:
                    text = str(e)
                return int(getattr(e, "code", 0) or 0), text
            return 0, str(e)

    def _send_to_eagle(self, host: str, endpoint_path: str, paths: List[str], meta: Dict[str, Any]) -> Tuple[int, str]:
        base = host.strip().rstrip("/")
        path = "/" + endpoint_path.strip().lstrip("/")
        url = base + path

        payload: Dict[str, Any] = {"paths": paths}

        # Attach optional metadata if present
        folder_id = (meta.get("folderId") or meta.get("folder_id") or "").strip()
        if folder_id:
            payload["folderId"] = folder_id

        tags = meta.get("tags")
        if isinstance(tags, list) and tags:
            payload["tags"] = tags

        name = (meta.get("name") or "").strip()
        if name:
            payload["name"] = name

        annotation = meta.get("annotation")
        if isinstance(annotation, str) and annotation:
            payload["annotation"] = annotation

        # rating/star not used

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
        folder_id: str,
        tags: str,
        name: str,
        annotation: str,
        dry_run: bool,
    ):
        pil_images = self._tensor_to_pil_list(images)
        saved_paths: List[str] = []
        if pil_images:
            saved_paths = self._save_images_output(pil_images, filename_prefix)

        tag_list: List[str] = []
        if isinstance(tags, str) and tags.strip():
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        meta = {
            "folderId": folder_id.strip(),
            "tags": tag_list,
            "name": name.strip(),
            "annotation": annotation or "",
        }

        # Hardcoded Eagle API config
        host = os.environ.get("EAGLE_API_HOST", "http://127.0.0.1:41595")
        endpoint_path = "/api/item/addFromPaths"

        if dry_run:
            msg = {
                "host": host,
                "endpoint_path": endpoint_path,
                "paths": saved_paths,
                "meta": meta,
                "note": "dry_run=true; not sent to Eagle",
            }
            resp_text = json.dumps(msg, ensure_ascii=False)
        else:
            code, resp_text = self._send_to_eagle(host, endpoint_path, saved_paths, meta)
            resp_text = f"HTTP {code}: {resp_text}"

        # No cleanup: images saved to output directory like SaveImage node

        return (images, resp_text)


NODE_CLASS_MAPPINGS = {
    "EagleSend": EagleSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleSend": "Eagle: Send Images",
}
