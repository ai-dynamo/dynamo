#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the llm-d Batch Gateway to Dynamo example."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
OUTPUT_EXCERPT_LENGTH = 160


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--tenant", default="dynamo-batch-example")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


class BatchClient:
    """Minimal OpenAI Batch client for the validation workflow."""

    def __init__(self, base_url: str, tenant: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer unused",
            "X-MaaS-Username": self.tenant,
        }

    def _send(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> bytes:
        headers = self._headers()
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{method} {path} returned HTTP {error.code}: {response_body}"
            ) from error

    def _send_json(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        raw_response = self._send(method, path, body, "application/json")
        response = json.loads(raw_response)
        if not isinstance(response, dict):
            raise RuntimeError(f"{method} {path} returned a non-object JSON response")
        return response

    def upload(self, filename: str, contents: bytes) -> str:
        """Upload a JSONL file and return its file identifier."""
        boundary = f"dynamo-batch-{uuid.uuid4().hex}"
        parts = [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            b"batch\r\n",
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            b"Content-Type: application/jsonl\r\n\r\n",
            contents,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        response = self._send_json_bytes(
            "POST",
            "/v1/files",
            b"".join(parts),
            f"multipart/form-data; boundary={boundary}",
        )
        file_identifier = response.get("id")
        if not isinstance(file_identifier, str) or not file_identifier:
            raise RuntimeError(f"file upload did not return an id: {response}")
        return file_identifier

    def _send_json_bytes(
        self, method: str, path: str, body: bytes, content_type: str
    ) -> dict[str, object]:
        raw_response = self._send(method, path, body, content_type)
        response = json.loads(raw_response)
        if not isinstance(response, dict):
            raise RuntimeError(f"{method} {path} returned a non-object JSON response")
        return response

    def create_batch(self, file_identifier: str) -> str:
        """Create a chat-completions batch and return its identifier."""
        response = self._send_json(
            "POST",
            "/v1/batches",
            {
                "input_file_id": file_identifier,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        batch_identifier = response.get("id")
        if not isinstance(batch_identifier, str) or not batch_identifier:
            raise RuntimeError(f"batch creation did not return an id: {response}")
        return batch_identifier

    def cancel(self, batch_identifier: str) -> dict[str, object]:
        """Request cancellation of a batch."""
        return self._send_json("POST", f"/v1/batches/{batch_identifier}/cancel")

    def wait_for_terminal(self, batch_identifier: str) -> dict[str, object]:
        """Poll a batch until it reaches a terminal state."""
        deadline = time.monotonic() + self.timeout_seconds
        last_status = ""
        while time.monotonic() < deadline:
            batch = self._send_json("GET", f"/v1/batches/{batch_identifier}")
            status = batch.get("status")
            if not isinstance(status, str):
                raise RuntimeError(f"batch has no string status: {batch}")
            if status != last_status:
                print(f"batch {batch_identifier}: {status}", flush=True)
                last_status = status
            if status in TERMINAL_STATUSES:
                return batch
            time.sleep(1)
        raise TimeoutError(
            f"batch {batch_identifier} did not finish in {self.timeout_seconds} seconds"
        )

    def file_content(self, file_identifier: str) -> bytes:
        """Retrieve file content from the Batch API."""
        return self._send("GET", f"/v1/files/{file_identifier}/content")


def jsonl_lines(contents: bytes) -> list[dict[str, object]]:
    """Decode non-empty JSONL lines into JSON objects."""
    lines = []
    for raw_line in contents.decode("utf-8").splitlines():
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise RuntimeError("JSONL output contains a non-object line")
        lines.append(value)
    return lines


def require_file_identifier(batch: dict[str, object], field: str) -> str:
    """Return a required output or error file identifier."""
    file_identifier = batch.get(field)
    if not isinstance(file_identifier, str) or not file_identifier:
        raise RuntimeError(f"terminal batch did not set {field}: {batch}")
    return file_identifier


def require_request_counts(
    batch: dict[str, object], expected_completed: int, expected_failed: int
) -> None:
    """Validate the terminal request counts returned by the Batch API."""
    counts = batch.get("request_counts")
    if not isinstance(counts, dict):
        raise RuntimeError(f"terminal batch did not return request_counts: {batch}")
    completed = counts.get("completed")
    failed = counts.get("failed")
    if completed != expected_completed or failed != expected_failed:
        raise RuntimeError(
            "unexpected request counts: "
            f"completed={completed}, failed={failed}, "
            f"expected completed={expected_completed}, failed={expected_failed}"
        )


def completion_excerpt(output_line: dict[str, object]) -> str:
    """Return a one-line excerpt from a successful Batch response."""
    custom_identifier = output_line.get("custom_id")
    response = output_line.get("response")
    if not isinstance(response, dict):
        raise RuntimeError(f"{custom_identifier} output has no response object")
    body = response.get("body")
    if not isinstance(body, dict):
        raise RuntimeError(f"{custom_identifier} output has no response body")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{custom_identifier} output has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"{custom_identifier} output has an invalid first choice")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"{custom_identifier} output has no response message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{custom_identifier} output has no response content")

    excerpt = " ".join(content.split())
    if len(excerpt) > OUTPUT_EXCERPT_LENGTH:
        return excerpt[: OUTPUT_EXCERPT_LENGTH - 3] + "..."
    return excerpt


def make_request(custom_identifier: str, model: str, max_tokens: int) -> str:
    """Build one OpenAI Batch JSONL request line."""
    return json.dumps(
        {
            "custom_id": custom_identifier,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Count slowly for request {custom_identifier}",
                    }
                ],
                "max_tokens": max_tokens,
            },
        },
        separators=(",", ":"),
    )


def input_with_model(input_path: Path, model: str) -> bytes:
    """Return the checked-in success requests with the selected model."""
    requests = jsonl_lines(input_path.read_bytes())
    for request in requests:
        custom_identifier = request.get("custom_id")
        body = request.get("body")
        if not isinstance(body, dict):
            raise RuntimeError(f"{custom_identifier} input has no request body")
        body["model"] = model
    return (
        "\n".join(json.dumps(request, separators=(",", ":")) for request in requests)
        + "\n"
    ).encode()


def validate_success(client: BatchClient, input_path: Path, model: str) -> None:
    """Validate upload, completion, and output retrieval."""
    input_file = client.upload(input_path.name, input_with_model(input_path, model))
    batch_identifier = client.create_batch(input_file)
    batch = client.wait_for_terminal(batch_identifier)
    if batch["status"] != "completed":
        raise RuntimeError(f"success batch reached {batch['status']}: {batch}")
    require_request_counts(batch, expected_completed=2, expected_failed=0)

    output_file = require_file_identifier(batch, "output_file_id")
    output_lines = jsonl_lines(client.file_content(output_file))
    custom_identifiers = {line.get("custom_id") for line in output_lines}
    expected = {"dynamo-batch-1", "dynamo-batch-2"}
    if len(output_lines) != len(expected) or custom_identifiers != expected:
        raise RuntimeError(
            "unexpected successful output records: "
            f"count={len(output_lines)}, ids={custom_identifiers}, "
            f"expected count={len(expected)}, ids={expected}"
        )
    print(f"retrieved {len(output_lines)} successful output lines", flush=True)
    for output_line in sorted(output_lines, key=lambda line: str(line["custom_id"])):
        print(
            f"  {output_line['custom_id']}: {completion_excerpt(output_line)}",
            flush=True,
        )


def validate_error_file(client: BatchClient) -> None:
    """Validate request-level error file assembly and retrieval."""
    contents = (
        make_request("dynamo-batch-unmapped", "unmapped/model", 8) + "\n"
    ).encode()
    input_file = client.upload("batch-error-input.jsonl", contents)
    batch_identifier = client.create_batch(input_file)
    batch = client.wait_for_terminal(batch_identifier)
    if batch["status"] != "completed":
        raise RuntimeError(f"error batch reached {batch['status']}: {batch}")
    require_request_counts(batch, expected_completed=0, expected_failed=1)

    error_file = require_file_identifier(batch, "error_file_id")
    error_lines = jsonl_lines(client.file_content(error_file))
    if len(error_lines) != 1:
        raise RuntimeError(f"expected one error line, received {len(error_lines)}")
    if error_lines[0].get("custom_id") != "dynamo-batch-unmapped":
        raise RuntimeError(f"unexpected error output: {error_lines[0]}")
    print("retrieved the unmapped-model error file", flush=True)


def validate_cancellation(client: BatchClient, model: str) -> None:
    """Validate that queued work can be cancelled."""
    lines = [make_request(f"dynamo-cancel-{index}", model, 256) for index in range(64)]
    contents = ("\n".join(lines) + "\n").encode()
    input_file = client.upload("batch-cancel-input.jsonl", contents)
    batch_identifier = client.create_batch(input_file)
    cancellation = client.cancel(batch_identifier)
    if cancellation.get("status") not in {"cancelling", "cancelled"}:
        raise RuntimeError(
            f"cancel request returned an unexpected state: {cancellation}"
        )
    batch = client.wait_for_terminal(batch_identifier)
    if batch["status"] != "cancelled":
        raise RuntimeError(f"cancelled batch reached {batch['status']}: {batch}")
    print("batch reached cancelled", flush=True)


def main() -> None:
    """Run the example workflow."""
    args = parse_args()
    input_path = Path(__file__).with_name("batch-input.jsonl")
    client = BatchClient(args.base_url, args.tenant, args.timeout_seconds)

    validate_success(client, input_path, args.model)
    validate_error_file(client)
    validate_cancellation(client, args.model)
    print("llm-d Batch Gateway -> dedicated Dynamo example passed", flush=True)


if __name__ == "__main__":
    main()
