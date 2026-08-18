#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create the deterministic AIPerf raw-payload input used by the EPD sweep."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

INSTRUCTION = (
    "\nUse the first attached page to answer this question: "
    "What is the actual value per 1000 during 1975? "
    "Then transcribe the visible content verbatim from each subsequent page."
)
ROLE_CONTEXT = {
    "system": (
        " Document pages may combine headings, tables, dates, quantities, labels,"
        " and annotations. Nearby rows and columns can clarify abbreviated or"
        " ambiguous entries, while units and punctuation distinguish similar values."
    ),
    "user": (
        " The attached material may contain headings, tables, dates, numerical values,"
        " labels, and short annotations. Consider each section in context, distinguish"
        " printed entries from surrounding notes, and report only information visible"
        " in the documents."
    ),
}


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


def load_tokenizer(path: str) -> Tokenizer:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )


def _count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def exact_text(
    tokenizer: Tokenizer, target: int, *, role: str, suffix: str = ""
) -> str:
    """Build stable text containing exactly ``target`` tokenizer tokens."""

    suffix_tokens = _count(tokenizer, suffix)
    if target < suffix_tokens:
        raise ValueError(
            f"ISL segment {target} cannot fit {suffix_tokens}-token suffix"
        )
    if target == 0:
        return ""
    unit = ROLE_CONTEXT[role]
    repeats = max(2, target // _count(tokenizer, unit) + 2)
    keep = target - suffix_tokens
    for _ in range(64):
        ids = tokenizer.encode(unit * repeats, add_special_tokens=False)
        while len(ids) < keep:
            repeats *= 2
            ids = tokenizer.encode(unit * repeats, add_special_tokens=False)
        prefix = tokenizer.decode(
            ids[:keep],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        value = prefix + suffix
        observed = _count(tokenizer, value)
        if observed == target:
            return value
        keep += target - observed
    raise RuntimeError(f"could not construct exact {target}-token text")


def text_blocks(tokenizer: Tokenizer, isl: int) -> tuple[str, str, str]:
    """Use the historical 2:7 system/user split (2000+7000 at ISL 9000)."""

    if isl <= 0:
        raise ValueError("ISL must be positive")
    system_tokens = (isl * 2) // 9
    system = exact_text(tokenizer, system_tokens, role="system")
    user = exact_text(tokenizer, isl - system_tokens, role="user", suffix=INSTRUCTION)
    if _count(tokenizer, system) + _count(tokenizer, user) != isl:
        raise AssertionError("generated text does not match ISL")
    return system, user[: -len(INSTRUCTION)], INSTRUCTION


def select_images(image_dir: Path, count: int) -> list[Path]:
    from PIL import Image

    paths = sorted(image_dir.glob("*.png"))
    if count <= 0 or len(paths) < count:
        raise ValueError(f"requested {count} images; found {len(paths)} in {image_dir}")
    for path in paths[:count]:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"expected normalized downloaded PNG: {path}")
    return paths[:count]


def build_payload(
    *,
    tokenizer: Tokenizer,
    backend: str,
    model: str,
    image_dir: Path,
    image_url_root: str,
    image_count: int,
    image_token_budget: int,
    isl: int,
    osl: int,
) -> dict[str, Any]:
    system, prefix, instruction = text_blocks(tokenizer, isl)
    content: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
    for image in select_images(image_dir, image_count):
        url = f"{image_url_root.rstrip('/')}/{urllib.parse.quote(image.name)}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instruction})
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "min_tokens": osl,
        "max_tokens": osl,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if backend == "vllm":
        payload["mm_processor_kwargs"] = {
            "min_pixels": 65_536,
            "max_pixels": image_token_budget * 1_024,
        }
    elif backend != "sglang":
        raise ValueError(f"unsupported backend: {backend}")
    return payload


def write_dataset(path: Path, payload: dict[str, Any], records: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    path.write_text((line + "\n") * records, encoding="utf-8")
