---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Custom Chat Processors
subtitle: Serve chat requests with your own Python chat engine inside the Dynamo Frontend.
---

A custom chat processor replaces Dynamo's chat preprocessing and response building with your own
Python code, as the built-in `vllm` and `sglang` processors do. Dynamo still serves HTTP, routes each
request to a worker, and streams the result back. Use one when your workers need a request shape or
a rendering step that no built-in processor provides.

The Frontend loads the processor named by `--dyn-chat-processor` (env var `DYN_CHAT_PROCESSOR`).
Any value other than `dynamo`, `vllm` or `sglang` names a custom processor:

- A name registered under the `dynamo.frontend.chat_processors` entry-point group, recommended for
  packaged processors.
- A direct `module:function` path for development.

## Write a processor

A processor is a provider function. The Frontend calls it once at startup, inside its event loop,
with the parsed Frontend configuration and the `--dyn-chat-processor-config` JSON object (empty when
unset). It returns a chat engine factory.

The Frontend calls the factory for each token-input model when its first worker registers, and
again if the model's card changes. The factory returns a `PythonAsyncEngine` that serves
`/v1/chat/completions`, `/v1/responses` and `/v1/messages` for that model. `/v1/completions` keeps
Dynamo's own preprocessing. If the factory raises, the Frontend retries with backoff and serves the
model once it succeeds. A model whose workers register with text input preprocesses requests
itself, so its chat requests go straight to a worker and skip the processor.

```python
# my_processor.py
import asyncio

from dynamo.llm import PythonAsyncEngine


class MyChatEngine:
    def __init__(self, routed_engine, options):
        self._routed_engine = routed_engine
        self._options = options

    async def generate(self, request, context=None):
        preprocessed = ...  # render and tokenize `request`
        stream = await self._routed_engine.generate(preprocessed, context=context)
        async for item in stream:
            ...  # build and yield OpenAI chat completion chunks


def provider(config, options):
    async def chat_engine_factory(instance_id, mdc, routed_engine):
        engine = MyChatEngine(routed_engine, options)
        return PythonAsyncEngine(engine.generate, asyncio.get_running_loop())

    return chat_engine_factory
```

The factory receives:

- `mdc`, the model's deployment card. `mdc.runtime_config()` returns what the worker published when
  it registered, its `runtime_data` included, and `mdc.local_dir()` the model's downloaded files.
- `routed_engine`, which sends a preprocessed request to a worker through Dynamo's router:
  `await routed_engine.generate(request, context=context)`.

## Register and run it

For an installable package, register the provider under the `dynamo.frontend.chat_processors`
entry-point group. The entry-point name is the value passed to the Frontend.

```toml
# pyproject.toml
[project.entry-points."dynamo.frontend.chat_processors"]
my-processor = "my_processor:provider"
```

Start the Frontend with the processor and its settings:

```bash
python -m dynamo.frontend --dyn-chat-processor my-processor \
  --dyn-chat-processor-config '{"render_workers": 8}'
```

During development, pass the module path instead of a registered name:

```bash
python -m dynamo.frontend --dyn-chat-processor my_processor:provider
```

The Frontend rejects a name that resolves to nothing when it parses its arguments, and lists the
registered processors. A custom processor runs only with the HTTP Frontend, not with
`--interactive` or `--kserve-grpc-server`.

## Errors

An exception your engine raises becomes the HTTP error response:

- Raise `dynamo.llm.exceptions.InvalidArgument(message)` to reject a request with HTTP 400 and
  show `message` to the client.
- Raise `dynamo.llm.HttpError(code, message)` for another status. The client gets that status for
  a 4xx, 503 or 529, and 500 for any other 5xx, with the status's standard text; `message` is
  logged, not returned.
- A worker's error reaches your engine from `routed_engine` as `InvalidArgument` for a rejected
  request, and as `dynamo.llm.exceptions.RoutedEngineError`, a `ValueError`, for any other
  classified error. Re-raise it, or let it escape, and the client gets the status the worker's
  error maps to.

## Response headers

To set an HTTP response header, call `context.set_response_header(name, value)` before your
engine's first yield, or before it raises:

```python
from dynamo.llm.exceptions import InvalidArgument


async def generate(request, context):
    context.set_response_header("x-my-header", "value")
    raise InvalidArgument("refused")
    yield  # unreachable; makes `generate` an async generator
```

The header is on the response, error responses included, when the Frontend reads your engine's
stream before it commits the response. That is every non-streaming `/v1/chat/completions` request,
and a streaming one only when `DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS` is set.

Otherwise the call returns `False` and sets nothing. That covers a streaming request without that
setting, a call made after a streaming response has started, and every `/v1/responses` and
`/v1/messages` request.

When the response can carry the header, the call raises `ValueError` for a name or value that is
not a valid header, and for a header that the HTTP layer owns (`content-type`, `content-length`,
`content-encoding`, `x-request-id` and hop-by-hop headers).
