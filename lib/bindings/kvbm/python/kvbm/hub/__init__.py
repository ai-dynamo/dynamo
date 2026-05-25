# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side facade for the KVBM prefill router.

`PrefillRouterHandler` brings up its own velo participant, registers a
velo dispatch handler that drives the captured vLLM `AsyncLLM`, and
registers itself with a remote `kvbm_hub` via HTTP. `make_dispatch_lambda`
builds the `(req_dict, event) -> None` closure the handler invokes for
each request. `try_wrap_engine` is the auto-detect entry point both
``dynamo.vllm`` and the standalone prefill entrypoint use.
"""

# kvbm._core is a Rust extension module, so `hub` lives on it as an attribute
# rather than a true submodule path; mirror the v2 import pattern.
from kvbm._core import hub as _hub
from kvbm.hub.detect import try_wrap_engine
from kvbm.hub.dispatch_lambda import make_dispatch_lambda

CompletionEvent = _hub.CompletionEvent
PrefillRouterHandler = _hub.PrefillRouterHandler

__all__ = [
    "CompletionEvent",
    "PrefillRouterHandler",
    "make_dispatch_lambda",
    "try_wrap_engine",
]
