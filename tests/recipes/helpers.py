# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request helpers and output-quality assertions for endpoint-attached tests.

The assertions here exist because of a real miss. On a 3-replica Kimi-K2.5
deployment one replica emitted the ``!`` character (token 0) for every generated
token -- 3000 tokens with no recovery -- while Kubernetes reported the pod
``1/1 Running`` with 0 restarts and the logs held no error. 22 of 25 tests in
this suite passed against it, because every one of them asserted that a field
was *present* or *truthy* rather than that it was *language*. ``"!!!!"`` is a
non-empty string.

:func:`assert_natural_language` closes that gap, and
:func:`worker_id_of` plus a unique prompt prefix let a test attribute output to
the replica that produced it -- necessary because KV-aware routing pins a given
prefix to one worker, so a fixed prompt samples one replica rather than the
deployment.
"""

import json
import urllib.request
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

import pytest

# Below this length a response carries too little signal to judge statistically.
# "4" is a correct and complete answer to "What is 2+2?" and must not be called
# degenerate for being short or low-diversity.
_MIN_LENGTH_FOR_STATISTICS = 24

# A single character owning more than this share of the non-whitespace body is
# the signature of token-repetition. English prose peaks around 12% ('e'), so
# there is a wide margin between healthy text and a stuck decoder.
_MAX_SINGLE_CHAR_SHARE = 0.5

# Distinct characters expected in any real sentence of appreciable length.
_MIN_DISTINCT_CHARS = 8


def post(endpoint, path: str, body: Dict[str, Any], timeout: int = 300) -> Dict:
    """POST JSON to the attached endpoint and return the decoded response."""
    data = json.dumps({"model": endpoint.model, **body}).encode()
    headers = {"Content-Type": "application/json", **dict(endpoint.headers or {})}
    request = urllib.request.Request(
        f"{endpoint.base_url}{path}", data=data, headers=headers
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def stream(endpoint, body: Dict[str, Any], timeout: int = 300) -> List[Dict]:
    """POST a streaming chat request and return the parsed SSE chunks."""
    data = json.dumps({"model": endpoint.model, **body}).encode()
    headers = {"Content-Type": "application/json", **dict(endpoint.headers or {})}
    request = urllib.request.Request(
        f"{endpoint.base_url}/v1/chat/completions", data=data, headers=headers
    )
    chunks: List[Dict] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            try:
                chunks.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pytest.fail(f"malformed SSE chunk: {line[:200]}")
    return chunks


def message_text(body: Dict) -> str:
    """All generated text from a chat response: reasoning and answer combined.

    Both are joined deliberately. A reasoning deployment may spend its whole
    token budget inside the think block and return ``content=""`` legitimately,
    so asserting on ``content`` alone is flaky; but a stuck decoder fills
    ``reasoning_content`` with garbage and leaves ``content`` empty, which
    asserting on ``content`` alone would silently accept. The concatenation is
    non-empty in both cases and degenerate in only one.
    """
    message = (body.get("choices") or [{}])[0].get("message") or {}
    return (message.get("reasoning_content") or "") + (message.get("content") or "")


def degeneracy_reason(text: str) -> Optional[str]:
    """Return why ``text`` is not natural language, or None if it looks fine.

    Deliberately conservative: it answers "is this a stuck decoder?", not "is
    this a good answer?". Short replies are exempt from the statistical checks
    so that terse-but-correct answers never fail.
    """
    stripped = text.strip()
    if not stripped:
        return "empty"
    if not any(char.isalnum() for char in stripped):
        return f"no alphanumeric character in {len(stripped)} chars: {stripped[:40]!r}"
    if len(stripped) < _MIN_LENGTH_FOR_STATISTICS:
        return None

    dense = [char for char in stripped if not char.isspace()]
    counts = Counter(dense)
    char, occurrences = counts.most_common(1)[0]
    share = occurrences / len(dense)
    if share > _MAX_SINGLE_CHAR_SHARE:
        return (
            f"character {char!r} is {share:.0%} of {len(dense)} non-whitespace "
            f"characters (limit {_MAX_SINGLE_CHAR_SHARE:.0%}) -- the decoder is "
            f"repeating one token"
        )
    if len(counts) < _MIN_DISTINCT_CHARS:
        return (
            f"only {len(counts)} distinct characters across {len(dense)} "
            f"non-whitespace characters"
        )
    return None


def assert_natural_language(text: str, label: str = "response") -> None:
    """Assert the model produced language rather than repeated-token output."""
    reason = degeneracy_reason(text)
    assert reason is None, (
        f"{label} is not natural language: {reason}\n"
        f"  first 120 chars: {text[:120]!r}\n"
        f"  length: {len(text)}"
    )


def worker_id_of(body: Dict) -> Optional[str]:
    """The worker that served this response, when nvext surfaced it.

    Looks for any ``*worker_id`` key anywhere in the payload: aggregated
    deployments report ``worker_id`` while disaggregated ones report
    ``prefill_worker_id`` and ``decode_worker_id``.
    """
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("worker_id") and isinstance(value, (int, str)):
                    found.append(str(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body)
    return found[0] if found else None


def unique_prompt(text: str) -> str:
    """Prefix a prompt so it cannot reuse another request's KV cache entry.

    KV-aware routing sends identical prefixes to the same worker, so repeating a
    fixed prompt measures one replica however many times it is sent. A unique
    prefix redistributes requests across the fleet.
    """
    return f"[{uuid.uuid4().hex[:8]}] {text}"
