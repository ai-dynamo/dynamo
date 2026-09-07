# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible realtime serving for the vLLM backend."""

from .handler import RealtimeHandler, RealtimeTranscriptionHandler

__all__ = ["RealtimeHandler", "RealtimeTranscriptionHandler"]
