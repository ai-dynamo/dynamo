# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public APIs of an already running deployment."""

import json
import logging
import re
from functools import partial
from pathlib import Path

import pytest

from tests.deploy.dgd_utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    MIN_RESPONSE_CONTENT_LENGTH,
    TEST_PROMPT,
    validate_chat_response,
)
from tests.deploy.response_checks import (
    validate_embedding,
    validate_stop_response,
    validate_stream,
)
from tests.utils.client import send_request

logger = logging.getLogger(__name__)


def check_embedding_api(
    url: str, model: str, output: Path, *, request_sender=None
) -> None:
    """Check embedding formats, dimensions, usage, and batch/unary agreement."""
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, inputs, options in (
        ("default", "Hello", {}),
        ("float", "Hello", {"encoding_format": "float"}),
        ("world", "World", {"encoding_format": "float"}),
        ("batch", ["Hello", "World"], {"encoding_format": "float"}),
        ("dimensions", "Hello", {"dimensions": 128, "encoding_format": "float"}),
        ("base64", "Hello", {"dimensions": 128, "encoding_format": "base64"}),
    ):
        payload = {"model": model, "input": inputs, **options}
        data = _request(
            url, payload, output / f"{name}.json", request_sender=request_sender
        )
        # Qwen3-Embedding-0.6B's native dimension is 1024.
        results[name] = validate_embedding(
            data,
            2 if name == "batch" else 1,
            options.get("dimensions", 1024),
            model,
            options.get("encoding_format", "float"),
        )
    for actual, expected in (
        (results["default"].data[0], results["float"].data[0]),
        (results["batch"].data[0], results["float"].data[0]),
        (results["batch"].data[1], results["world"].data[0]),
        (results["base64"].data[0], results["dimensions"].data[0]),
    ):
        # Batch kernels can differ slightly from unary inference in reduced precision.
        assert actual.embedding == pytest.approx(
            expected.embedding, rel=1e-2, abs=1e-3
        ), "Embedding differs from corresponding unary result"
    assert results["batch"].usage.prompt_tokens == (
        results["float"].usage.prompt_tokens + results["world"].usage.prompt_tokens
    ), "Batch usage differs from unary inputs"


def check_chat_api(url: str, model: str, output: Path, *, request_sender=None) -> None:
    """Check unary, streaming, token-limit, and stop behavior on one endpoint."""
    output.mkdir(parents=True, exist_ok=True)
    request = partial(_request, request_sender=request_sender)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "temperature": 0.0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    unary = request(
        url,
        payload,
        output / "unary.json",
        min_content_length=MIN_RESPONSE_CONTENT_LENGTH,
    )
    request(url, {**payload, "stream": True}, output / "stream.json")
    limited = request(url, {**payload, "max_tokens": 1}, output / "limited.json")
    assert limited.usage.completion_tokens == 1, limited.usage
    assert limited.choices[0].finish_reason == "length", limited.choices[0]
    content = unary.choices[0].message.content
    # Use a later word, avoiding a whitespace boundary at the start of output.
    match = next(
        (
            m
            for m in re.finditer(r"\S+", content)
            if 0 < m.start() < len(content) // 2
            and content.find(m.group()) == m.start()
        ),
        None,
    )
    assert match is not None, "Cannot derive an interior stop sequence"
    stop = match.group()
    stopped = request(url, {**payload, "stop": stop}, output / "stop.json")
    validate_stop_response(stopped, unary, stop)


def _request(
    url: str,
    payload: dict,
    artifact: Path,
    min_content_length: int = 0,
    *,
    request_sender=None,
):
    record = {"request": payload}
    logger.info("Checking deployment API case %s", artifact.stem)
    try:
        sender = send_request if request_sender is None else request_sender
        with sender(
            url,
            payload,
            timeout=float(DEFAULT_REQUEST_TIMEOUT),
            method="POST",
            stream=payload.get("stream", False),
        ) as response:
            record["http_status"] = response.status_code
            if payload.get("stream"):
                lines = []
                record["response"] = lines

                def capture():
                    for line in response.iter_lines():
                        text = line.decode("utf-8")
                        lines.append(text)
                        yield text

                response.raise_for_status()
                validate_stream(capture(), payload["model"])
                return None
            record["response"] = response.text
            response.raise_for_status()
            if "messages" in payload:
                return validate_chat_response(
                    response,
                    payload["model"],
                    min_content_length=min_content_length,
                    max_tokens=payload["max_tokens"],
                    stop=payload.get("stop"),
                )
            return response.json()
    finally:
        artifact.write_text(json.dumps(record, indent=2))
