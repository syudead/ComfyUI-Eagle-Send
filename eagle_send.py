from __future__ import annotations
import os
import json
import re
from typing import List, Any, Dict

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
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
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

    def _normalize_prompt(self, text: str) -> str:
        """Normalize common separators to newlines (ASCII + fullwidth + BREAK)."""
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        pattern = r"(?:\r?\n|,|;|\||/|\uFF0C|\u3001|\uFF1B|\uFF5C|\uFF0F|(?i:BREAK))"
        t = re.sub(pattern, "\n", t)
        t = re.sub(r"\n+", "\n", t)
        return t.strip("\n")

    def _clean_tag(self, token: str) -> str:
        """Trim, strip simple wrappers/weights, and collapse spaces."""
        token_str = (token or "").strip()
        if not token_str:
            return ""
        if token_str and token_str[0] in "([{":
            token_str = token_str[1:].strip()
        if token_str and token_str[-1] in ")]}":
            token_str = token_str[:-1].strip()
        match = re.match(r"^([^:(){}\[\]]+):\d+(?:\.\d+)?$", token_str)
        if match:
            token_str = match.group(1).strip()
        token_str = re.sub(r"\s+", " ", token_str)
        return token_str

    def _prompt_to_tags(self, text: str) -> List[str]:
        normalized = self._normalize_prompt(text)
        tags: List[str] = []
        seen: set[str] = set()
        for raw in normalized.split("\n"):
            cleaned = self._clean_tag(raw)
            if cleaned and cleaned not in seen:
                tags.append(cleaned)
                seen.add(cleaned)
            if len(tags) >= 128:
                break
        return tags

    # --- Workflow parsing (fixed node definitions) ---
    MODEL_NODE_TYPES = {"CheckpointLoaderSimple", "CheckpointLoader"}
    LORA_NODE_TYPES = {
        "LoraLoader",
        "LoraLoaderModelOnly",
        "Power Lora Loader (rgthree)",
    }

    def _as_nodes_list(self, workflow_obj: Any) -> List[Dict[str, Any]]:
        if isinstance(workflow_obj, list):
            return [x for x in workflow_obj if isinstance(x, dict)]
        if isinstance(workflow_obj, dict):
            nodes = workflow_obj.get("nodes")
            if isinstance(nodes, list):
                return [x for x in nodes if isinstance(x, dict)]
        return []

    def _find_str_in_widgets(self, widgets_values: Any, exts: List[str]) -> str:
        try:
            if isinstance(widgets_values, list):
                for v in widgets_values:
                    if isinstance(v, str):
                        lv = v.lower()
                        if any(lv.endswith(e) for e in exts):
                            return v
        except Exception:
            pass
        return ""

    def _value_from_inputs(self, node: Dict[str, Any], key: str) -> Any:
        try:
            inputs = node.get("inputs", {}) or {}
            if key in inputs:
                return inputs[key]
        except Exception:
            pass
        return None

    def _strings_from_structure(self, obj: Any) -> List[str]:
        out: List[str] = []
        try:
            if isinstance(obj, str):
                out.append(obj)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    out.extend(self._strings_from_structure(v))
            elif isinstance(obj, dict):
                for v in obj.values():
                    out.extend(self._strings_from_structure(v))
        except Exception:
            pass
        return out

    def _normalize_name_drop_ext(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        base = name.strip()
        # use last path component only
        base = base.replace("\\", "/").split("/")[-1]
        for ext in (".safetensors", ".ckpt", ".pth", ".pt"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
                break
        return base.strip()

    def _parse_workflow_resources(self, extra_pnginfo: Any) -> Dict[str, Any]:
        result = {"model_name": "", "loras": []}
        if not isinstance(extra_pnginfo, dict):
            return result
        wf = extra_pnginfo.get("workflow")
        if isinstance(wf, str):
            try:
                wf = json.loads(wf)
            except Exception:
                wf = None
        nodes = self._as_nodes_list(wf)
        if not nodes:
            return result

        exts = [".safetensors", ".ckpt", ".pth", ".pt"]
        model_found = False
        lora_names: List[str] = []

        for node in nodes:
            # Respect node-level disabled/bypass flags
            if bool(node.get("disabled")) or bool(node.get("bypass")):
                continue
            t = node.get("type")
            if t in self.MODEL_NODE_TYPES and not model_found:
                # Prefer explicit input
                v = self._value_from_inputs(node, "ckpt_name")
                if isinstance(v, str) and v.strip():
                    result["model_name"] = self._normalize_name_drop_ext(v)
                    model_found = True
                else:
                    # Fallback within this node: official widgets_values[0] carries ckpt_name
                    wv = node.get("widgets_values")
                    if isinstance(wv, list) and wv and isinstance(wv[0], str) and wv[0].strip():
                        result["model_name"] = self._normalize_name_drop_ext(wv[0])
                        model_found = True
            elif t in self.LORA_NODE_TYPES:
                inp = node.get("inputs", None)
                if t == "Power Lora Loader (rgthree)":
                    # Inputs may be dict of lora_* entries or entries may be present in widgets_values
                    handled = False
                    if isinstance(inp, dict):
                        for key, val in inp.items():
                            if isinstance(key, str) and key.startswith("lora_") and isinstance(val, dict):
                                if val.get("on") is True and isinstance(val.get("lora"), str):
                                    nm = val.get("lora").strip()
                                    if nm and ("/" not in nm and "\\" not in nm):
                                        lora_names.append(self._normalize_name_drop_ext(nm))
                                        handled = True
                    if not handled:
                        wv = node.get("widgets_values")
                        if isinstance(wv, list):
                            for entry in wv:
                                if isinstance(entry, dict) and entry.get("on") is True and isinstance(entry.get("lora"), str):
                                    nm = entry.get("lora").strip()
                                    if nm and ("/" not in nm and "\\" not in nm):
                                        lora_names.append(self._normalize_name_drop_ext(nm))
                else:
                    # LoraLoader / LoraLoaderModelOnly: read lora_name input only
                    v = inp.get("lora_name") if isinstance(inp, dict) else None
                    if isinstance(v, str):
                        nm = v.strip()
                        if nm and ("/" not in nm and "\\" not in nm):
                            lora_names.append(self._normalize_name_drop_ext(nm))

        seen = set()
        uniq_loras = []
        for n in lora_names:
            if n not in seen:
                uniq_loras.append(n)
                seen.add(n)
        result["loras"] = uniq_loras
        return result


    def _save_images_output(self, pil_images: List[Any], filename_prefix: str, prompt: str | None, extra_pnginfo: Dict[str, Any] | None) -> List[str]:
        paths: List[str] = []
        if not pil_images:
            return paths
        output_dir = folder_paths.get_output_directory()
        width, height = pil_images[0].size
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, width, height
        )
        # Prepare PNG metadata: embed workflow and human prompt
        pnginfo = None
        try:
            from PIL.PngImagePlugin import PngInfo  # type: ignore
            pnginfo = PngInfo()
            if isinstance(prompt, str) and prompt.strip():
                pnginfo.add_text("parameters", prompt)
            if isinstance(extra_pnginfo, dict):
                for key, val in extra_pnginfo.items():
                    try:
                        pnginfo.add_text(key, json.dumps(val))
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

    def _send_to_eagle(self, host: str, paths: List[str], tags: List[str]) -> Tuple[int, str]:
        base = host.strip().rstrip("/")
        url = base + "/api/item/addFromPaths"
        items: List[Dict[str, Any]] = []
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            item: Dict[str, Any] = {"path": p, "name": name}
            if tags:
                item["tags"] = tags
            items.append(item)
        payload: Dict[str, Any] = {"items": items}
        headers = {"Content-Type": "application/json"}
        code, text = self._post_json(url, payload, headers)
        return code, text

    def send(
        self,
        images,
        filename_prefix: str,
        prompt: str,
        extra_pnginfo=None,
    ):
        pil_images = self._tensor_to_pil_list(images)
        saved_paths: List[str] = []
        if pil_images:
            saved_paths = self._save_images_output(pil_images, filename_prefix, prompt, extra_pnginfo)

        host = os.environ.get("EAGLE_API_HOST", "http://127.0.0.1:41595")
        tags = self._prompt_to_tags(prompt)
        # add model/lora from workflow (EXTRA_PNGINFO)
        resources = self._parse_workflow_resources(extra_pnginfo)
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

        code, resp_text = self._send_to_eagle(host, saved_paths, tags)
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



