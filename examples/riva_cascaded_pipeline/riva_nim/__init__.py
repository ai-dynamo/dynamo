# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA NIM worker building blocks for a Dynamo cascaded voice pipeline.

This package wraps RIVA NIM speech connectors (ASR, TTS) as Dynamo workers and
provides a realtime orchestrator that chains ASR -> LLM -> TTS. It is kept
self-contained (relative imports only, no dependency on the surrounding
example) so it can be moved to ``components/src/dynamo/riva/`` by relocating the
directory without changing import statements.

The package is named ``riva_nim`` rather than ``riva`` so it does not shadow the
``nvidia-riva-client`` package, which is imported as top-level ``riva``.
"""
