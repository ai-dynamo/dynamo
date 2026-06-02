#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal prompt-enhance visual generation demo.

Assumes ``launch.sh`` is already running: enhancer LLM on :8000 and
diffusion backend on :8001. The bundled launcher starts a video backend by
default; use ``--mode image`` against an image diffusion frontend.

Usage::

    python client.py "a bear by a campfire under starlight"
    python client.py --enhancer off "an astronaut on a black sand beach"
    python client.py --mode image --t2i-url http://127.0.0.1:8001 "a glass teapot"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dynamo.common.multimodal import EnhancerMode, T2IEnhancedClient, T2VEnhancedClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", help="User-facing prompt for the generated media.")
    p.add_argument(
        "--mode",
        choices=["video", "image"],
        default="video",
        help="Generation backend to call.",
    )
    p.add_argument(
        "--llm-url", default="http://127.0.0.1:8000", help="Enhancer frontend URL."
    )
    p.add_argument(
        "--t2v-url", default="http://127.0.0.1:8001", help="T2V frontend URL."
    )
    p.add_argument(
        "--t2i-url", default="http://127.0.0.1:8001", help="T2I frontend URL."
    )
    p.add_argument(
        "--llm-model",
        default="Qwen/Qwen3-0.6B",
        help="Model name registered against the enhancer frontend.",
    )
    p.add_argument(
        "--t2v-model",
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Model name registered against the T2V frontend.",
    )
    p.add_argument(
        "--t2i-model",
        default="black-forest-labs/FLUX.1-dev",
        help="Model name registered against the T2I frontend.",
    )
    p.add_argument(
        "--enhancer",
        choices=[m.value for m in EnhancerMode],
        default=EnhancerMode.AUTO.value,
        help="Bypass the LLM enhancer when set to 'off'.",
    )
    p.add_argument("--steps", type=int, default=25, help="num_inference_steps.")
    p.add_argument("--size", default=None, help="Output resolution, e.g. 832x480.")
    p.add_argument("--frames", type=int, default=33, help="Video num_frames.")
    p.add_argument(
        "--response-format",
        choices=["url", "b64_json"],
        default="url",
        help="OpenAI-compatible media response format.",
    )
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    size = args.size or ("832x480" if args.mode == "video" else "1024x1024")
    if args.mode == "video":
        async with T2VEnhancedClient(
            llm_url=args.llm_url,
            t2v_url=args.t2v_url,
            llm_model=args.llm_model,
            t2v_model=args.t2v_model,
        ) as client:
            result = await client.generate(
                args.prompt,
                enhancer=args.enhancer,
                size=size,
                response_format=args.response_format,
                num_inference_steps=args.steps,
                num_frames=args.frames,
            )
    else:
        async with T2IEnhancedClient(
            llm_url=args.llm_url,
            t2i_url=args.t2i_url,
            llm_model=args.llm_model,
            t2i_model=args.t2i_model,
        ) as client:
            result = await client.generate(
                args.prompt,
                enhancer=args.enhancer,
                size=size,
                response_format=args.response_format,
                nvext={"num_inference_steps": args.steps},
            )

    if result.url is not None:
        print(f"{args.mode}_url      : {result.url}")
    else:
        b64_len = len(result.b64_json or "")
        print(f"{args.mode}_b64_json : {b64_len} base64 chars")
    print(f"rewritten      : {result.rewritten_prompt!r}")
    t = result.timings
    print(
        f"timings_ms     : enhance={t.enhance_ms:.1f}  "
        f"generation={t.generation_ms:.1f}  e2e={t.e2e_ms:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
