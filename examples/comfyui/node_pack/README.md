# comfyui-dynamo

ComfyUI custom nodes for [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)'s
OpenAI-compatible image and video endpoints.

## Install

```bash
cp -r comfyui-dynamo /path/to/ComfyUI/custom_nodes/
# restart ComfyUI
```

No extra Python deps &mdash; uses only stdlib + `torch` / `numpy` / `Pillow`, all
of which ship with ComfyUI.

## Nodes

| Node | Inputs | Outputs |
|---|---|---|
| `Dynamo Endpoint Config` | `base_url`, `api_key`, `timeout_s` | `endpoint` (DYNAMO_ENDPOINT) |
| `Dynamo Text-to-Image` | prompt, model, size, n, steps, guidance, seed, negative_prompt | `IMAGE` |
| `Dynamo Image Edit` | `IMAGE`, prompt, model, size, steps, guidance, seed | `IMAGE` |
| `Dynamo Text-to-Video` | prompt, model, size, num_frames, fps, steps, guidance, seed | `VIDEO`, video path |
| `Dynamo Image-to-Video` | `IMAGE`, prompt, model, size, num_frames, fps, steps, guidance, boundary_ratio, guidance_scale_2 | `VIDEO`, video path |
| `Dynamo List Models` | `base_url` | JSON string |

Each generation node accepts an optional `endpoint` socket (from
`Dynamo Endpoint Config`) which overrides the inline `base_url`/`api_key`/`timeout_s`
widgets. Wire one config node into a graph to keep many generation nodes consistent.

## Defaults

- `response_format` defaults to `b64_json`. **Don't change this** unless your Dynamo
  worker is launched with `--media-output-http-url` and an HTTP file server backs the
  configured filesystem path. Otherwise Dynamo's default `url` returns `file://` URLs
  unreachable from a remote ComfyUI.
- `base_url` defaults to `http://localhost:8000` (Dynamo's `DYN_HTTP_PORT`).
- `seed = -1` and any zero-valued `nvext` field are omitted from the request so the
  worker uses its model defaults.

## Mapping to Dynamo schemas

The widgets translate to fields on `NvCreateImageRequest` / `NvCreateVideoRequest`:

| Widget | Goes to | Dynamo path |
|---|---|---|
| `model`, `prompt`, `size`, `n`, `response_format` | top-level | `lib/llm/src/protocols/openai/images.rs:16` |
| `negative_prompt`, `steps`, `guidance_scale`, `seed` | `nvext` | `image_protocol.py:13` |
| `num_frames`, `fps`, `boundary_ratio`, `guidance_scale_2` | `nvext` (video only) | `video_protocol.py:16` |

## License

Apache-2.0 (matches Dynamo).
