<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2

> [!IMPORTANT]
> These recipes run only on the model-specific image
> `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.1-a.x-k2-post.1`. Stock Dynamo vLLM
> runtimes, including 1.6.0, do not register the checkpoint's `axk2` model type, and
> the worker fails while it builds the engine config. See
> [Recipes on Model-Specific Images](../README.md#recipes-on-model-specific-images).

See the [Dynamo recipe documentation](https://docs.nvidia.com/dynamo/dev/recipes/).
