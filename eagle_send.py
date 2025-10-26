from __future__ import annotations
import os
import json
import tempfile
from typing import List, Tuple, Any, Dict

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
                "host": ("STRING", {"default": os.environ.get("EAGLE_API_HOST", "http://127.0.0.1:41595")}),
                "folder_id": ("STRING", {"default": ""}),
                "tags": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": ""}),
                "annotation": ("STRING", {"default": "", "multiline": True}),
                "rating": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "endpoint_path": ("STRING", {"default": "/api/item/addFromPaths"}),
                "api_token": ("STRING", {"default": ""}),
                "keep_temp": ("BOOLEAN", {"default": False}),
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

    def _save_images_temp(self, pil_images: List[Any]) -> Tuple[str, List[str]]:
        temp_dir = tempfile.mkdtemp(prefix="comfyui_eagle_")
        paths: List[str] = []
        for idx, im in enumerate(pil_images):
            p = os.path.join(temp_dir, f"image_{idx+1:03d}.png")
            im.save(p, format="PNG")
            paths.append(p)
        return temp_dir, paths

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

    def _build_headers(self, api_token: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = api_token.strip()
        if token:
            # Common Eagle pattern uses Authorization: Bearer <token>
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _send_to_eagle(self, host: str, endpoint_path: str, paths: List[str], meta: Dict[str, Any], api_token: str) -> Tuple[int, str]:
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

        rating = meta.get("rating")
        if isinstance(rating, int) and rating > 0:
            payload["star"] = rating

        headers = self._build_headers(api_token)
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
        host: str,
        folder_id: str,
        tags: str,
        name: str,
        annotation: str,
        rating: int,
        endpoint_path: str,
        api_token: str,
        keep_temp: bool,
        dry_run: bool,
    ):
        pil_images = self._tensor_to_pil_list(images)
        temp_dir = ""
        saved_paths: List[str] = []
        if pil_images:
            temp_dir, saved_paths = self._save_images_temp(pil_images)

        tag_list: List[str] = []
        if isinstance(tags, str) and tags.strip():
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        meta = {
            "folderId": folder_id.strip(),
            "tags": tag_list,
            "name": name.strip(),
            "annotation": annotation or "",
            "rating": int(rating) if isinstance(rating, int) else 0,
        }

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
            code, resp_text = self._send_to_eagle(host, endpoint_path, saved_paths, meta, api_token)
            resp_text = f"HTTP {code}: {resp_text}"

        # Cleanup temp files if requested
        if not keep_temp and temp_dir and os.path.isdir(temp_dir):
            try:
                for p in saved_paths:
                    if os.path.exists(p):
                        os.remove(p)
                os.rmdir(temp_dir)
            except Exception:
                # Best-effort cleanup
                pass

        return (images, resp_text)


NODE_CLASS_MAPPINGS = {
    "EagleSend": EagleSend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleSend": "Eagle: Send Images",
}
