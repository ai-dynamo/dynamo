# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone vLLM OpenAI-API server with the KVBM prefill-router auto-attach.

Same arg surface as `python -m vllm.entrypoints.openai.api_server`. The
only behavioural delta vs. plain vLLM is that, after the AsyncLLM engine
is built and before the FastAPI app starts serving, we call
:func:`kvbm.hub.try_wrap_engine` to register this worker as a velo
prefill target with whichever hub the rendered ``kv_transfer_config``
points at. Aggregated or decode workers (no hub URL / non-prefill role)
get a graceful no-op.

Mirrors the auto-attach hook merged into ``dynamo.vllm``'s worker
factory (``components/src/dynamo/vllm/worker_factory.py``) — running
this entrypoint produces the same hub-side wiring as running
``dynamo.vllm --disaggregation-mode prefill``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level strong references for PrefillRouterHandler instances so they
# outlive the build_and_serve coroutine frame. PrefillRouterHandler holds an
# RAII guard that issues a hub DELETE + closes the velo transport on drop;
# without this anchor, the handler would be GC'd as soon as build_and_serve
# returned, breaking the velo channel between hub and worker.
_LIVE_HANDLERS: list = []


def _install_kvbm_hook() -> None:
    """Monkey-patch ``vllm.entrypoints.openai.api_server.build_and_serve`` so it
    runs ``kvbm.hub.try_wrap_engine`` after engine construction and before
    ``init_app_state`` / serve.

    The handler is parked on the FastAPI ``app.state`` so it survives for
    the life of the server and can be shut down cleanly on exit.
    """
    from vllm.entrypoints.openai import api_server

    original = api_server.build_and_serve

    async def patched_build_and_serve(
        engine_client, listen_address, sock, args, **uvicorn_kwargs
    ):
        vllm_config = getattr(engine_client, "vllm_config", None)
        handler: Any = None
        if vllm_config is None:
            logger.debug(
                "kvbm.vllm.prefill: engine_client has no vllm_config attr; "
                "skipping prefill-router auto-wire"
            )
        else:
            try:
                from kvbm.hub import try_wrap_engine

                handler = try_wrap_engine(vllm_config, engine_client)
                if handler is not None:
                    # Print (not log) so the marker is visible regardless of
                    # vllm's logging config — the smoke harness greps for it.
                    print(
                        f"kvbm prefill router auto-wired "
                        f"(worker velo id={handler.worker_velo_id()})",
                        flush=True,
                    )
            except Exception as e:
                logger.warning("kvbm prefill router auto-wire failed: %s", e)
                handler = None

        if handler is not None:
            # Anchor BEFORE awaiting serve so the handler can never be GC'd
            # mid-flight if the build_and_serve frame gets unwound.
            _LIVE_HANDLERS.append(handler)

        return await original(
            engine_client, listen_address, sock, args, **uvicorn_kwargs
        )

    api_server.build_and_serve = patched_build_and_serve


def main() -> None:
    """Drop-in replacement for ``python -m vllm.entrypoints.openai.api_server``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _install_kvbm_hook()

    import uvloop
    from vllm.entrypoints.openai.api_server import (
        cli_env_setup,
        make_arg_parser,
        run_server,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="KVBM-aware vLLM OpenAI-compatible API server (prefill auto-wire)."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
