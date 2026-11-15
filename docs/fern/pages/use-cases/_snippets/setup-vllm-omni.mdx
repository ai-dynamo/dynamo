---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
subtitle: Reusable setup steps for the vLLM-Omni backend
---

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) generates images, video, and audio through `python -m dynamo.vllm.omni`.

**Prerequisites**

- A working vLLM backend setup.
- An `amd64` host. Dynamo container builds do not install vLLM-Omni on `arm64`.

Dynamo container images include vLLM-Omni. A PyPI installation of `ai-dynamo[vllm]` does not include it. For a source installation, install the vLLM-Omni release that matches your vLLM version:

```bash
pip install git+https://github.com/vllm-project/vllm-omni.git@<version>
```

The `agg_omni_*.sh` and `disagg_omni_glm_image.sh` launchers start the frontend and workers directly on one host. They are intended for local development and testing and do not create a `DynamoGraphDeployment`.

Generated media is written through [fsspec](https://filesystem-spec.readthedocs.io/) storage. The default is `file:///tmp/dynamo_media`. Set `--media-output-fs-url` for S3, GCS, or Azure Blob storage, and optionally `--media-output-http-url` to rewrite the URLs returned in responses.

vLLM-Omni ships built-in stage configurations for supported models. Set `--stage-configs-path` only to override them.
