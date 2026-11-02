# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and replay helpers for native read contracts."""

from contextlib import asynccontextmanager

from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_reads as reads
from dynamo.vllm.tests.decision_head_test_support import DecisionContext


@asynccontextmanager
async def monitor(context, request_id):
    context.append(request_id)
    yield


def question(ordinal=0, kind="Choice", values=("red", "blue")):
    return {
        "ordinal": ordinal,
        "kind": kind,
        "instructions": "Choose.",
        "candidates": [{"value": value} for value in values],
    }


def wire(profile=native_registry.LIQUID_PROFILE, questions=None):
    return {
        "decision_read": {
            "protocol_version": 1,
            "profile": profile,
            "model": "test-model",
            "input": "state",
            "questions": questions or [question()],
            "temperature": 1,
            "cache_salt": "request-isolation",
            "max_input_tokens": 16384,
            "max_tokenization_work": 32768,
            "max_encoding_bytes": 65536,
        }
    }


async def adapter(engine, profile=native_registry.LIQUID_PROFILE, **kwargs):
    return await reads.ReadExecutionAdapter.create(
        engine,
        native_registry.PROFILES[profile],
        abort_monitor=monitor,
        max_concurrency=2,
        **kwargs,
    )


async def execute(instance, request):
    context = DecisionContext()
    response = await instance.execute(
        request,
        context,
        request_id="native-read",
        cancelled=lambda: False,
        trace_headers={"traceparent": "test"},
    )
    return response, context
