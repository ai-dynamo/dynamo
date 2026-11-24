# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat helpers across vLLM's entrypoints-to-renderers module migration."""

# Supported vLLM images expose different module layouts. Only fall back when
# the migrated module itself is absent; a broken transitive import must surface.
try:
    from vllm.renderers.chat_utils import load_chat_template, make_tool_call_id
except ModuleNotFoundError as exc:
    if exc.name != "vllm.renderers.chat_utils":
        raise
    from vllm.entrypoints.chat_utils import load_chat_template, make_tool_call_id

__all__ = ["load_chat_template", "make_tool_call_id"]
