<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ComfyUI Integration

A [ComfyUI](https://github.com/comfy-org/ComfyUI) custom node pack that drives
Dynamo's OpenAI-compatible `/v1/images/generations` and `/v1/videos` endpoints.
Output comes back as a native `IMAGE` / `VIDEO` so the rest of the graph
(`SaveImage`, `SaveVideo`, downstream nodes) consumes it unchanged. No
checkpoint loads inside ComfyUI.

## Install

```bash
# 1. Backend
bash examples/backends/vllm/launch/agg_omni_flux2_klein.sh   # image
bash examples/backends/vllm/launch/agg_omni_video.sh          # video

# 2. Node pack into ComfyUI
cp -r examples/comfyui/node_pack ~/ComfyUI/custom_nodes/comfyui-dynamo

# 3. Workflow library into ComfyUI
mkdir -p ~/ComfyUI/user/default/workflows
cp examples/comfyui/workflows/registered/*.json \
   ~/ComfyUI/user/default/workflows/

# 4. ComfyUI
cd ~/ComfyUI && python main.py --listen 127.0.0.1 --port 8188
```

Open `http://127.0.0.1:8188`, left sidebar &rarr; Workflows &rarr; double-click
any `Dynamo_*` &rarr; **Queue**.

The node pack uses only stdlib + torch/numpy/PIL (already shipped by ComfyUI),
so no extra `pip install` is needed.

## Workflows

Four UI-format workflows in [`workflows/registered/`](./workflows/registered/),
one per I/O combination. Each starts with a `DynamoEndpointConfig` node so
swapping the Dynamo URL is one widget edit per workflow.

| File | I/O | Default model |
|---|---|---|
| `Dynamo_TextToImage.json`  | text &rarr; image           | `black-forest-labs/FLUX.2-klein-4B` |
| `Dynamo_ImageEdit.json`    | image + text &rarr; image   | `black-forest-labs/FLUX.2-klein-4B` |
| `Dynamo_TextToVideo.json`  | text &rarr; video           | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`   |
| `Dynamo_ImageToVideo.json` | image + text &rarr; video   | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`   |

Three matching API-format workflows in [`workflows/api/`](./workflows/api/) for
headless `POST /prompt` automation.

ComfyUI reads the registered workflows live from `user/default/workflows/`; no
restart needed after dropping a new file in.

## Nodes

Available under the **Dynamo** category in the right-click Add Node menu.

| Node | Output |
|---|---|
| `Dynamo Endpoint Config` | `DYNAMO_ENDPOINT` (`base_url`, `api_key`, `timeout_s`) |
| `Dynamo Text-to-Image`   | `IMAGE` |
| `Dynamo Image Edit`      | `IMAGE` |
| `Dynamo Text-to-Video`   | `VIDEO` |
| `Dynamo Image-to-Video`  | `VIDEO` |
| `Dynamo List Models`     | JSON string |

Every generation node has its own `base_url` / `api_key` widgets, but wiring an
`endpoint` socket from `Dynamo Endpoint Config` overrides them &mdash; one source
of truth per workflow.

## Pointing at a different Dynamo

Edit the `base_url` widget on the `Dynamo Endpoint Config` node and save the
workflow (Ctrl+S). For Dynamo behind a reverse proxy with auth, set `api_key`
on the same node; the node pack sends `Authorization: Bearer <key>` on every
request.

For headless automation the same applies on the API-format workflows in
[`workflows/api/`](./workflows/api/).

## Defaults

- `response_format` defaults to `"b64_json"` in every shipped workflow. Dynamo's
  protocol default is `"url"`, but unless the worker was launched with
  `--media-output-http-url` and an HTTP file server is in front of the storage
  path, the URL is `file://` and unreachable from a remote ComfyUI.
- Float widgets (`guidance_scale`, `boundary_ratio`, `guidance_scale_2`) use
  `-1.0` as the "use model default" sentinel so a literal `0.0`
  (CFG-distilled FLUX-klein, DMD2) is forwarded verbatim.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `comfyui-dynamo` missing from ComfyUI startup log | Path must be `<ComfyUI>/custom_nodes/comfyui-dynamo/__init__.py`, not nested. |
| `Dynamo returned a file:// URL` | Set the node's `response_format` widget to `b64_json`. |
| Connection refused on port 8000 | Dynamo isn't listening, or it bound to `127.0.0.1` and ComfyUI is remote. Use `DYN_HTTP_HOST=0.0.0.0`. |
| `value_not_in_list` on `size` widget | The widget is a fixed COMBO. Image: `512x512` / `768x768` / `1024x1024` / `1024x1792` / `1792x1024`. Video: `832x480` / `480x832` / `1280x720` / `720x1280`. |
| Long video hangs to socket timeout | Bump `timeout_s` on the video node and the endpoint config (default 1800 s). |
| `model` not found | Hit `GET /v1/models` (or use the `Dynamo List Models` node) and copy the exact ID into the `model` widget. |
