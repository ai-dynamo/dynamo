# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepInfer (github.com/deepseek-ai/DeepInfer) in-process on Dynamo's unified backend.

One Dynamo worker per DeepInfer engine drives all of its DP ranks in lockstep;
see ``DESIGN.md``. Run with ``python -m dynamo.deepinfer``; on an engine that
spans nodes, ``python -m dynamo.deepinfer.node`` runs the other nodes' ranks.
This package must stay importable without torch or DeepInfer: rank processes
and the node launcher import submodules of it, and DeepInfer itself is imported
lazily by ``di_engine``.
"""
