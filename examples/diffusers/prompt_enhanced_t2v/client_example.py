#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal demo of T2VEnhancedClient against the launched topology.

Assumes ``container.sh`` is already running: enhancer LLM on
:8000 and diffusion backend on :8001.

Usage::

    python client_example.py "a bear by a campfire under starlight"
    python client_example.py --enhancer off "an astronaut on a black sand beach"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dynamo.common.clients import EnhancerMode, T2VEnhancedClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", help="User-facing prompt for the video.")
    p.add_argument(
        "--llm-url", default="http://127.0.0.1:8000", help="Enhancer frontend URL."
    )
    p.add_argument(
        "--t2v-url", default="http://127.0.0.1:8001", help="T2V frontend URL."
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
        "--enhancer",
        choices=[m.value for m in EnhancerMode],
        default=EnhancerMode.AUTO.value,
        help="Bypass the LLM enhancer when set to 'off'.",
    )
    p.add_argument("--steps", type=int, default=25, help="num_inference_steps.")
    p.add_argument(
        "--size", default="832x480", help="Output resolution, e.g. 832x480."
    )
    p.add_argument("--frames", type=int, default=33, help="num_frames.")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    async with T2VEnhancedClient(
        llm_url=args.llm_url,
        t2v_url=args.t2v_url,
        llm_model=args.llm_model,
        t2v_model=args.t2v_model,
    ) as client:
        result = await client.generate(
            args.prompt,
            enhancer=args.enhancer,
            size=args.size,
            num_inference_steps=args.steps,
            num_frames=args.frames,
        )
    print(f"video_url      : {result.url}")
    print(f"rewritten      : {result.rewritten_prompt!r}")
    t = result.timings
    print(
        f"timings_ms     : enhance={t.enhance_ms:.1f}  "
        f"t2v={t.t2v_ms:.1f}  e2e={t.e2e_ms:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
