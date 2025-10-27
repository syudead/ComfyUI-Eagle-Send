from __future__ import annotations
import json
from typing import Any, Dict, List


MODEL_NODE_TYPES = {"CheckpointLoaderSimple", "CheckpointLoader"}
LORA_NODE_TYPES = {
    "LoraLoader",
    "LoraLoaderModelOnly",
    "Power Lora Loader (rgthree)",
}


def _as_nodes_list(workflow_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(workflow_obj, list):
        return [x for x in workflow_obj if isinstance(x, dict)]
    if isinstance(workflow_obj, dict):
        nodes = workflow_obj.get("nodes")
        if isinstance(nodes, list):
            return [x for x in nodes if isinstance(x, dict)]
    return []


def _value_from_inputs(node: Dict[str, Any], key: str) -> Any:
    try:
        inputs = node.get("inputs", {}) or {}
        if key in inputs:
            return inputs[key]
    except Exception:
        pass
    return None


def _normalize_name_drop_ext(name: str) -> str:
    if not isinstance(name, str):
        return ""
    base = name.strip()
    base = base.replace("\\", "/").split("/")[-1]
    for ext in (".safetensors", ".ckpt", ".pth", ".pt"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base.strip()


def parse_workflow_resources(extra_pnginfo: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"model_name": "", "loras": [], "lora_weights": {}}
    if not isinstance(extra_pnginfo, dict):
        return result
    wf = extra_pnginfo.get("workflow")
    if isinstance(wf, str):
        try:
            wf = json.loads(wf)
        except Exception:
            wf = None
    nodes = _as_nodes_list(wf)
    if not nodes:
        return result

    model_found = False
    lora_names: List[str] = []
    lora_weights: Dict[str, float] = {}

    for node in nodes:
        if bool(node.get("disabled")) or bool(node.get("bypass")):
            continue
        t = node.get("type")
        if t in MODEL_NODE_TYPES and not model_found:
            v = _value_from_inputs(node, "ckpt_name")
            if isinstance(v, str) and v.strip():
                result["model_name"] = _normalize_name_drop_ext(v)
                model_found = True
            else:
                wv = node.get("widgets_values")
                if isinstance(wv, list) and wv and isinstance(wv[0], str) and wv[0].strip():
                    result["model_name"] = _normalize_name_drop_ext(wv[0])
                    model_found = True
        elif t in LORA_NODE_TYPES:
            inp = node.get("inputs", {}) or {}
            if t == "Power Lora Loader (rgthree)":
                handled = False
                if isinstance(inp, dict):
                    for key, val in inp.items():
                        if isinstance(key, str) and key.startswith("lora_") and isinstance(val, dict):
                            if val.get("on") is True and isinstance(val.get("lora"), str):
                                nm = val.get("lora").strip()
                                if nm:
                                    base = _normalize_name_drop_ext(nm)
                                    lora_names.append(base)
                                    try:
                                        if isinstance(val.get("strength"), (int, float)):
                                            lora_weights[base] = float(val.get("strength"))
                                    except Exception:
                                        pass
                                    handled = True
                if not handled:
                    wv = node.get("widgets_values")
                    if isinstance(wv, list):
                        for entry in wv:
                            if isinstance(entry, dict) and entry.get("on") is True and isinstance(entry.get("lora"), str):
                                nm = entry.get("lora").strip()
                                if nm:
                                    base = _normalize_name_drop_ext(nm)
                                    lora_names.append(base)
                                    try:
                                        if isinstance(entry.get("strength"), (int, float)):
                                            lora_weights[base] = float(entry.get("strength"))
                                    except Exception:
                                        pass
            else:
                v = inp.get("lora_name") if isinstance(inp, dict) else None
                if isinstance(v, str):
                    nm = v.strip()
                    if nm:
                        base = _normalize_name_drop_ext(nm)
                        lora_names.append(base)
                        # Try common strength keys for standard loaders
                        try:
                            strength = None
                            for k in ("strength", "strength_model", "strength_clip"):
                                if isinstance(inp.get(k), (int, float)):
                                    strength = float(inp.get(k))
                                    break
                            if strength is not None:
                                lora_weights[base] = strength
                        except Exception:
                            pass

    seen = set()
    uniq_loras = []
    for n in lora_names:
        if n not in seen:
            uniq_loras.append(n)
            seen.add(n)
    result["loras"] = uniq_loras
    # Keep only weights for detected unique names
    result["lora_weights"] = {n: lora_weights[n] for n in uniq_loras if n in lora_weights}
    return result
