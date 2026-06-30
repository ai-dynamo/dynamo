# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RIVA gRPC connection helpers shared by the ASR and TTS workers."""

from __future__ import annotations

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
