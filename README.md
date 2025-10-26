ComfyUI-Eagle-Send
===================

ComfyUI custom node to send generated images to Eagle (https://eagle.cool/) via the local Eagle API.

Features
- Sends one or more images to Eagle by posting file paths to the API
- Optional metadata: folderId, tags, name, annotation, rating (stars)
- Configurable host and endpoint path, optional API token header
- Dry-run mode to inspect payload without sending

Installation
- Place this folder under your ComfyUI `custom_nodes` directory as `ComfyUI-Eagle-Send`.
- Ensure Python can import Pillow (PIL) – ComfyUI usually includes it.
- Restart ComfyUI.

Node: Eagle: Send Images
- Input: `images` (IMAGE) – connect from your workflow
- Host: Eagle API base (default `http://127.0.0.1:41595`)
- endpoint_path: Default `/api/item/addFromPaths` (adjust if your API differs)
- folder_id: Target Eagle folderId (optional)
- tags: Comma-separated tags (optional)
- name: Item name/title (optional)
- annotation: Notes (optional)
- rating: 0–5 star rating (optional)
- api_token: Optional token; sent as `Authorization: Bearer <token>`
- keep_temp: Keep temporary PNG files (for debugging)
- dry_run: Do not send; only return the would-be payload

Notes
- The node saves images to a temporary directory and posts their absolute paths to Eagle.
- If the server rejects `paths`, the node retries per-file with `path`.
- If your Eagle requires a different header or endpoint, adjust `endpoint_path` and use a reverse proxy or modify the node as needed.

Troubleshooting
- Verify Eagle is running and the API is enabled (default port 41595).
- Check ComfyUI console for the `HTTP <code>` and response body returned by Eagle.
- Use `dry_run=true` to validate payload and paths.

License
- Same terms as the upstream repository; no additional license header added here.

