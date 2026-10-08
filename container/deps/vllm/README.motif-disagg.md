<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 disaggregated runtime

This branch starts at `db6a3705c1d9a9df4f94324636c454e45dc000aa`.
The build applies two guarded vLLM NIXL fixes: validate cache rank using the
individual attention backend, and retain Motif's NHD cache layout. Motif
disaggregation is restricted to TP2 on both prefill and decode workers.
The patch fails on an unknown source version and runs only during image
construction. Serving uses `python3 -m dynamo.vllm` directly.

The pinned CI image reports source commit
`d11469cbc14d540498175fe9a5eb18e4b6adf1c2`. That commit and `db6a3705`
have the identical Git tree `c3a1bd9474d38139c76f9ceebab819906606085d`.
The incremental Dockerfile reuses these exact binaries and adds the two
patched Python files. It retains the parent image's dependencies, user,
entrypoint, and license attribution. OCI labels record the new source revision
and immutable parent digest.

Build from a clean checkout of the branch, using a new image tag:

```bash
docker build --platform linux/amd64 \
  --build-arg DYNAMO_COMMIT_SHA="$(git rev-parse HEAD)" \
  -f container/deps/vllm/Dockerfile.motif-disagg \
  -t "${IMAGE}" .
docker push "${IMAGE}"
```

The regular full-build template applies the same patch when its CUDA base is
`ghcr.io/motiftechnologies/vllm`:

```bash
python3 container/render.py --framework vllm --target runtime --output-short-filename
docker build --platform linux/amd64 \
  --build-arg DYNAMO_COMMIT_SHA="$(git rev-parse HEAD)" \
  -f container/rendered.Dockerfile -t "${IMAGE}" .
```
