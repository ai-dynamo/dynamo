# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public APIs of an already running deployment."""

import json
import logging
import re
from pathlib import Path

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


def check_deployment_api(
    base_url: str,
    model: str,
    scenario: str,
    output: Path,
    *,
    endpoint: str | None = None,
) -> None:
    """Run shared API cases and retain raw responses even if a contract fails."""
    output.mkdir(parents=True, exist_ok=True)
    if endpoint is None:
        endpoint = (
            "/v1/embeddings" if scenario == "embedding" else "/v1/chat/completions"
        )
    url = base_url + endpoint
    if scenario == "embedding":
        for name, inputs, encoding in (
            ("default", "Hello", None),
            ("float", "Hello", "float"),
            ("batch", ["Hello", "World"], "float"),
        ):
            payload = {"model": model, "input": inputs}
            if encoding:
                payload["encoding_format"] = encoding
            data = _request(url, payload, output / f"{name}.json")
            assert data["model"] == model, data
            # The example uses Qwen3-Embedding-0.6B's native 1024 dimensions.
            validate_embedding(data, 2 if name == "batch" else 1, 1024)
        return

    assert scenario == "chat", scenario
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "temperature": 0.0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    unary = _request(
        url,
        payload,
        output / "unary.json",
        min_content_length=MIN_RESPONSE_CONTENT_LENGTH,
    )
    _request(url, {**payload, "stream": True}, output / "stream.json")
    _request(url, {**payload, "max_tokens": 1}, output / "limited.json")
    content = unary["choices"][0]["message"]["content"]
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
    stopped = _request(url, {**payload, "stop": stop}, output / "stop.json")
    validate_stop_response(stopped, unary, stop)


def _request(url: str, payload: dict, artifact: Path, min_content_length: int = 0):
    record = {"request": payload}
    logger.info("Checking deployment API case %s", artifact.stem)
    try:
        with send_request(
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
                validate_stream(capture())
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
