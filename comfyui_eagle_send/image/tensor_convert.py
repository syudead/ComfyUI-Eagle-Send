from __future__ import annotations
from typing import Any, List

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def ensure_deps():
    if torch is None:
        raise RuntimeError("torch not available; required by ComfyUI IMAGE tensors")
    if Image is None:
        raise RuntimeError("PIL (Pillow) not available; install pillow")


def tensor_to_pil_list(images_tensor) -> List[Any]:
    ensure_deps()
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

