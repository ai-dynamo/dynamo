# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code vendored from vLLM (Apache-2.0); each file names its source and commit.

Block hashing (``kv_cache_utils``, ``hashing``) and the KV event wire format
plus ZMQ publisher (``kv_events``). Rank workers import these: no ``vllm``,
``dynamo._core``, torch or DeepInfer.
"""
