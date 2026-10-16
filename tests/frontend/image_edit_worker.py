# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU image worker for upload transport and rolling-upgrade tests."""

import asyncio
import hashlib
import json
import os

import uvloop
from pydantic import BaseModel, ValidationError

from dynamo.common.protocols.image_protocol import NvCreateImageRequest
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.llm.exceptions import InvalidArgument
from dynamo.runtime import DistributedRuntime

MODEL_NAME = "image-edit-contract"
ENDPOINT_PATH = "image_edit_contract.backend.generate"


class LegacyRequest(BaseModel):
    input_reference: str | None = None


async def generate(request, context):
    """Report the received wire shape and ordered hashes, or await cancellation.

    The legacy mode rejects array references even if discovery advertised support.
    """
    if os.environ.get("IMAGE_EDIT_LEGACY_READER") == "1":
        try:
            LegacyRequest.model_validate(request)
        except ValidationError:
            raise InvalidArgument(
                "Legacy image worker requires a scalar reference"
            ) from None
    parsed = NvCreateImageRequest.model_validate(request)
    references = parsed.input_reference or []
    report = {
        "wire_type": type(request.get("input_reference")).__name__,
        "digests": [hashlib.sha256(value.encode()).hexdigest() for value in references],
    }
    if parsed.prompt == "wait-for-cancel":
        path = os.environ["IMAGE_EDIT_CANCEL_FILE"]
        with open(path + ".started", "w") as output:
            output.write("started")
        try:
            await context.async_killed_or_stopped()
        finally:
            with open(path, "w") as output:
                output.write("cancelled")
        return
    yield {
        "created": 1,
        "data": [{"b64_json": "AA==", "revised_prompt": json.dumps(report)}],
    }


async def main():
    """Serve a CPU protocol worker with optional image-edit capability metadata."""
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    limit = os.environ.get("IMAGE_EDIT_CAPABILITY")
    capability = (
        None
        if limit is None
        else {"image_edits": {"schema_version": 1, "max_input_images": int(limit)}}
    )
    await register_model(
        ModelInput.Text,
        ModelType.Images,
        endpoint,
        MODEL_NAME,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
        user_data=capability,
    )
    await endpoint.serve_endpoint(generate)


if __name__ == "__main__":
    uvloop.run(main())
