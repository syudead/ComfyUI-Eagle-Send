ComfyUI-Eagle-Send
===================

ComfyUI custom node to send generated images to Eagle (https://eagle.cool/) via the local Eagle API.

Features
- Saves images to ComfyUI `output/` using the same naming as Save Image
- Sends one or more saved image paths to Eagle via `addFromPaths`
- Parses the prompt (from a connected TEXT socket) into tags and attaches them to each item
- Extracts model and LoRA names from workflow metadata and adds tags like `model:<name>`, `lora:<name>`

Installation
- Place this folder under your ComfyUI `custom_nodes` directory as `ComfyUI-Eagle-Send`.
- Ensure Pillow (PIL) is available (ComfyUI environments typically include it).
- Restart ComfyUI.

Node: Eagle: Send Images
- `images` (IMAGE)
- `prompt` (STRING) forceInput only. Provide via a node connection; the widget is not used.
- `filename_prefix` (STRING) default `ComfyUI/EagleSend`
  - Saves to `output/<prefix>_xxxxx_.png` (Save Image style). Use subfolders like `ComfyUI/Eagle` if desired.

Notes
- Images are saved to ComfyUI `output/` and those paths are sent to Eagle.
- Payload strictly follows Eagle API JSON for `addFromPaths`: `{ "items": [{ "path": "...", "tags": ["..."] }] }`.
- Eagle host is read from `EAGLE_API_HOST` environment variable if set; otherwise defaults to `http://127.0.0.1:41595`.

Project layout (refactor)
- `comfyui_eagle_send/nodes/eagle_send.py` – Node definition (thin)
- `comfyui_eagle_send/image/tensor_convert.py` – IMAGE tensor to PIL
- `comfyui_eagle_send/image/save.py` – Save PNG and embed metadata
- `comfyui_eagle_send/parsing/tags.py` – Prompt-to-tags
- `comfyui_eagle_send/parsing/workflow.py` – Model/LoRA extraction
- `comfyui_eagle_send/eagle/api.py` – Eagle `addFromPaths` client
- `comfyui_eagle_send/config.py` – Host configuration

Troubleshooting
- Verify Eagle is running and the API is enabled (default port 41595).
- Check ComfyUI console for the `HTTP <code>` and response body returned by Eagle.

License
- Same terms as the upstream repository; no additional license header added here.
