#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone GMS + SGLang smoke harness (no Dynamo required).

Dynamo's launcher is normally what makes ``--load-format gms`` legal: it
patches the argparse choices list and then swaps the string for the loader
class. This script does those two steps programmatically so GMS can be
exercised against a bare SGLang install.

Start the GMS server first:

    python -m gpu_memory_service --device 0 --device-type xpu

Omit --tag so the server serves every production tag: the integration opens a
client for both "weights" and "kv_cache", so restricting it to one leaves the
other socket missing.

Then WRITE mode, which populates the server and holds the weights open:

    python sglang_gms_smoke.py --keep-alive

Then RO mode in a second process, which adopts what the writer published:

    python sglang_gms_smoke.py --read-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

logger = logging.getLogger("gms.smoke")


def _resolve_declare():
    """Return SGLang's launcher-stage config override helper.

    Mirrors gpu_memory_service.integrations.sglang, which requires 0.5.21+:
    that release folded the published-config guard into declare_resolution()
    and dropped the declare_late_resolution() variant that carried it in
    0.5.18-0.5.20. Imported lazily so --help works without SGLang installed.
    """
    try:
        from sglang.srt.arg_groups.overrides import declare_resolution
    except ImportError as exc:
        raise RuntimeError(
            "This SGLang build does not expose declare_resolution(); GMS "
            "cannot inject its loader. Upgrade to SGLang 0.5.21 or newer."
        ) from exc

    return declare_resolution


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument(
        "--read-only",
        action="store_true",
        help="Request RO mode (gms_read_only). Needs a writer to have published "
        "weights to the same GMS server first.",
    )
    p.add_argument(
        "--ro-connect-timeout-ms",
        type=int,
        default=None,
        help="Give up waiting for the writer after this long. Default waits "
        "indefinitely.",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Launch the SGLang HTTP server instead of running one offline "
        "generation. This is the GMS equivalent of `sglang serve`.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--keep-alive",
        action="store_true",
        help="Block after generating so a second process can attach in RO mode.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import sglang as sgl
    from gpu_memory_service.integrations.sglang import setup_gms
    from sglang.srt.server_args import ServerArgs

    extra: dict = {}
    if args.read_only:
        extra["gms_read_only"] = True
    if args.ro_connect_timeout_ms is not None:
        extra["gms_ro_connect_timeout_ms"] = args.ro_connect_timeout_ms

    # Build the raw args, then push everything else through the official
    # resolution API. Field names moved into arg_groups in 0.5.21, so setting
    # them as constructor kwargs is not portable across versions.
    #
    # host/port are the exception and must be passed here. In 0.5.21 they live
    # in the "serving" config bag, which __post_init__ publishes from these
    # fields. A later declare() override reaches the bag (uvicorn binds it) but
    # not ServerArgs.port, which is what ServerArgs.url() reads -- so warmup
    # would probe the default 30000 while the server listened elsewhere.
    server_args = ServerArgs(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
    )

    declare = _resolve_declare()

    declare(
        server_args,
        "gms.smoke",
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        model_loader_extra_config=json.dumps(extra) if extra else None,
        # GMS has no pauseable graph allocator hook, so capture must be off.
        # See integrations/sglang/memory_saver.py::cuda_graph.
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
    )

    # Flips enable_memory_saver and returns the loader class. Must run before
    # the engine is constructed, while the config is still unpublished.
    loader_cls = setup_gms(server_args)
    declare(server_args, "gms.smoke.loader", load_format=loader_cls)

    # With the default RW_OR_RO lock the server itself decides: the first
    # process to connect gets RW and loads from disk, later processes get RO
    # from the COMMITTED layout and import instead. --read-only only forces
    # the RO request, which fails if nothing has been published yet.
    mode = "RO (forced)" if args.read_only else "RW_OR_RO (auto)"
    logger.info("Starting SGLang with GMS, lock=%s, model=%s", mode, args.model_path)

    started = time.perf_counter()

    if args.serve:
        from sglang.srt.entrypoints.http_server import launch_server

        def _ready():
            logger.info(
                "[GMS-TIMING] server ready in %.2f s (lock=%s)",
                time.perf_counter() - started,
                mode,
            )

        logger.info("Serving on http://%s:%d", args.host, args.port)
        launch_server(server_args, launch_callback=_ready)
        return 0

    engine = sgl.Engine(server_args=server_args)
    logger.info(
        "[GMS-TIMING] engine ready in %.2f s (lock=%s)",
        time.perf_counter() - started,
        mode,
    )
    try:
        out = engine.generate(
            args.prompt,
            {"max_new_tokens": args.max_new_tokens, "temperature": 0.0},
        )
        logger.info("Generation result: %s", out)

        if args.keep_alive:
            logger.info("Holding the engine open. Press Enter or Ctrl-C to exit.")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
    finally:
        engine.shutdown()

    logger.info("GMS %s smoke test completed", mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
