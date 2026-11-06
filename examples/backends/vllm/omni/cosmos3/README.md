<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 with vLLM-Omni

This directory holds the client for experimental Cosmos3 image, video, and
image-to-video generation through vLLM-Omni. The launch scripts and sample JSON
requests are in [`examples/backends/vllm/launch/`](../../launch/).

For prerequisites, deployment steps, and configuration, see
[Cosmos 3 with vLLM-Omni](../../../../../docs/fern/pages/use-cases/diffusion/cosmos3.md).

## Contents

| Path | Description |
|---|---|
| `generate.py` | Requests a Cosmos3 image or short video and verifies the returned media |
