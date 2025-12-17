# ComfyUI Eagle Send

Custom node for ComfyUI that saves generated images and sends them to the Eagle image manager. While sending, it embeds rich metadata (including an A1111-compatible parameters string and workflow details) and automatically creates Eagle tags from your prompt and workflow.

---

## Main Features

- Send images to Eagle from a ComfyUI workflow.
- Save PNGs locally with embedded metadata.
  - Writes an A1111-style "parameters" text chunk (Steps, Sampler, CFG, Seed, Size, Model, Hashes, etc.).
  - Stores the original ComfyUI `extra_pnginfo` as separate JSON text chunks when available.
- Auto-generate Eagle tags from the positive prompt and add `model:<name>` / `lora:<name>` from the workflow.
- Build Eagle memo (annotation) including positive/negative prompts and key generation settings.
- Optionally read generation settings from a `d2_pipe` object (steps, cfg, seed, sampler, scheduler, clip_skip).
- Return both the original `IMAGE` tensor and a JSON string summarizing the Eagle request/metadata.

---

## Installation

### Manual
1. Go to your ComfyUI installation folder.
2. Navigate to `custom_nodes`.
3. Clone this repository into that folder.
4. Restart ComfyUI.

---

## Inputs / Options

Required
- `images: IMAGE`
  - ComfyUI image tensor. Passed through unchanged on output and converted to PNG when saving.
- `filename_prefix: STRING` (default: `ComfyUI`)
  - Prefix used with ComfyUI's output naming (via `folder_paths.get_save_image_path`).
  - Supports a single datetime token: `%datetime%` → expands to `yyyymmdd_HHmmss`.
    - Example: `ComfyUI_%datetime%` → `ComfyUI_20251213_024501`
- `prompt: STRING`
  - Positive prompt text. Used for tag generation, parameters text, and Eagle memo.

Optional
- `negative: STRING` (default: empty)
  - Negative prompt text. Included in parameters text and Eagle memo.
- `d2_pipe: D2_TD2Pipe`
  - Optional container for generation settings. If present, the node attempts to read:
    - `steps`, `cfg`, `seed` (or `noise_seed`), `sampler_name` (or `sampler`), `scheduler`, `clip_skip`.
  - Sampler/scheduler are normalized to familiar names for readability in the parameters string.

Hidden
- `extra_pnginfo: EXTRA_PNGINFO`
  - ComfyUI provides this dict automatically. When it contains a `workflow`, the node extracts the checkpoint name, active LoRAs and their strengths. This information is used to:
    - Append `model:<name>` and `lora:<name>` Eagle tags.
    - Compute and embed short hashes for model/LoRAs when their files are found via `folder_paths`.

Outputs
- `(IMAGE, STRING)`
  - The original image tensor and a JSON string describing the operation, e.g. HTTP status, saved paths, tags, model/LoRA list, parameters text, memo text, and raw body.

---

## Other Features

Local saving
- Files are written under ComfyUI's output directory using its standard naming rules.
- A `parameters` text chunk is always written to PNG. The node also adds `prompt` and each key of `extra_pnginfo` as JSON strings when available.

Tag generation
- Prompts are split on common delimiters (commas, line breaks, semicolons, pipes, slashes, full-width punctuation, and the token "BREAK").
- Basic cleanup removes surrounding brackets and numeric weights like `token:1.2`.
- Duplicate tags are removed while preserving order. Up to 128 tags are kept.

Workflow parsing and hashes
- Detects model from common checkpoint loader nodes and LoRAs from standard loaders (including "Power Lora Loader (rgthree)").
- Resolves files via ComfyUI `folder_paths` and computes SHA256 short hashes to include in the A1111 parameters string.

Eagle memo (annotation)
- Multi-line text composed of positive prompt, a line for negative prompt, model name, LoRAs with optional weights, and a compact settings line (Steps, Sampler, CFG, Seed, Size, Clip skip when present).

Eagle connectivity and limits
- Default host is `http://127.0.0.1:41595`.
- Uses Eagle's `POST /api/item/addFromPaths` endpoint; the Eagle app must be able to access the saved image paths.
- This node saves PNG files; other formats are not emitted by this implementation.
