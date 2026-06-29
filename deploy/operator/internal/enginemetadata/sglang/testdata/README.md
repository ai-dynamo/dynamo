<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

These fixtures are real SGLang HTTP metadata responses from `Qwen/Qwen3-0.6B`
servers with KV events enabled.

- `models.json`: `GET /v1/models` from `lmsysorg/sglang:v0.5.12.post1`.
- `model_info_generation.json`: `GET /model_info` from
  `lmsysorg/sglang:v0.5.12.post1`.
- `server_info_qwen3_huggingface_legacy_kv_events_config.json`:
  `GET /server_info` from `lmsysorg/sglang:v0.5.12.post1`, covering legacy
  `kv_events_config`.
- `server_info_qwen3_huggingface_structured_kv_events.json`:
  `GET /server_info` from the SGLang `v0.5.14` source tag, covering structured
  `kv_events`.

Failure cases are generated in tests by mutating the typed metadata parsed from
these real responses.

Baseline capture command:

```bash
docker run --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  lmsysorg/sglang:v0.5.12.post1 \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 30000 \
    --page-size 16 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'

curl -s http://127.0.0.1:30000/v1/models > models.json
curl -s http://127.0.0.1:30000/model_info > model_info_generation.json
curl -s http://127.0.0.1:30000/server_info > server_info_qwen3_huggingface_legacy_kv_events_config.json
```
