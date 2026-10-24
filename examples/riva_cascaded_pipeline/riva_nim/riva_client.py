# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA gRPC connection helpers shared by the ASR and TTS workers."""

from __future__ import annotations

import asyncio

import grpc
from riva.client import ASRService, Auth, SpeechSynthesisService

from .config import RivaConnectionConfig


def build_auth(config: RivaConnectionConfig) -> Auth:
    """Create a RIVA ``Auth`` for the given connection config.

    A local RIVA server uses an insecure channel with no metadata. The NVCF
    endpoint uses a TLS channel (``use_ssl``) plus per-call gRPC metadata: the
    ``function-id`` and an ``authorization`` Bearer token, attached only when
    the corresponding config fields are set.

    Args:
        config: The RIVA connection settings.

    Returns:
        A ``riva.client.Auth`` configured for the target server.
    """
    metadata = []
    if config.function_id:
        metadata.append(["function-id", config.function_id])
    if config.api_key:
        metadata.append(["authorization", f"Bearer {config.api_key}"])
    return Auth(
        ssl_root_cert=config.ssl_root_cert,
        use_ssl=config.use_ssl,
        uri=config.server,
        metadata_args=metadata or None,
    )


def build_tts_service(config: RivaConnectionConfig) -> SpeechSynthesisService:
    """Create a RIVA ``SpeechSynthesisService`` (TTS) for the given connection config."""
    return SpeechSynthesisService(build_auth(config))


def build_asr_service(config: RivaConnectionConfig) -> ASRService:
    """Create a RIVA ``ASRService`` for the given connection config."""
    return ASRService(build_auth(config))


async def await_riva_call(rpc_future: grpc.Future, timeout_s: float):
    """Await a RIVA gRPC future with a client-side deadline, off the event loop.

    RIVA's ``ASRService`` / ``SpeechSynthesisService`` methods accept
    ``future=True`` to return a gRPC future. Waiting via ``future.result(timeout)``
    bounds the wait and — unlike ``asyncio.wait_for`` around a blocking call —
    ``future.cancel()`` on timeout actually terminates the in-flight RPC and
    frees the worker thread, so an unresponsive backend can't tie up capacity.

    Args:
        rpc_future: A gRPC future from a ``future=True`` RIVA call.
        timeout_s: Client-side deadline in seconds.

    Returns:
        The RPC response.

    Raises:
        grpc.FutureTimeoutError: If the deadline elapses (the RPC is cancelled).
    """
    try:
        return await asyncio.to_thread(rpc_future.result, timeout_s)
    except grpc.FutureTimeoutError:
        rpc_future.cancel()
        raise
