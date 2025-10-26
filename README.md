ComfyUI-Eagle-Send
===================

ComfyUI custom node to send generated images to Eagle (https://eagle.cool/) via the local Eagle API.

Features
- Saves images to ComfyUI `output/` using the same naming as Save Image
- Sends one or more saved image paths to Eagle via `addFromPaths`
- Ensures unique filenames for batched images (even without `%batch_num%`)

Installation
- Place this folder under your ComfyUI `custom_nodes` directory as `ComfyUI-Eagle-Send`.
- Ensure Pillow (PIL) is available (ComfyUI environments typically include it).
- Restart ComfyUI.

Node: Eagle: Send Images
- Input: `images` (IMAGE)
- `filename_prefix` (STRING) default `ComfyUI/EagleSend`
  - Saves to `output/<prefix>_xxxxx_.png` (Save Image style). Use subfolders like `ComfyUI/Eagle` if desired.

Notes
- Images are saved to ComfyUI `output/` and those paths are sent to Eagle.
- If the server rejects `paths`, the node retries per-file with `path`.
- Eagle host is read from `EAGLE_API_HOST` environment variable if set; otherwise defaults to `http://127.0.0.1:41595`.

Troubleshooting
- Verify Eagle is running and the API is enabled (default port 41595).
- Check ComfyUI console for the `HTTP <code>` and response body returned by Eagle.

License
- Same terms as the upstream repository; no additional license header added here.
