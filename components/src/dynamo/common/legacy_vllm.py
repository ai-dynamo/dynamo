# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-owned declarations for exact pre-capability vLLM deployments.

Keep this contract aligned with protocols/common/legacy_vllm.rs. It is startup
configuration, never a public request parameter or synthetic worker metadata.
It identifies legacy lowering, not full N-2 native-serving conformance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate legacy vLLM target key")
        result[key] = value
    return result


class LegacyVllmRelease(Enum):
    # TODO(1.8): remove with the corresponding N-2 adapters.
    DYNAMO_14 = "1.4.0"
    DYNAMO_15 = "1.5.0"

    @property
    def engine_version(self) -> str:
        return "0.26.0" if self is LegacyVllmRelease.DYNAMO_14 else "0.28.0"

    def supports_field(self, name: str) -> bool:
        return name in {"allowed_token_ids", "bad_words_token_ids"} or (
            name == "logprob_token_ids" and self is LegacyVllmRelease.DYNAMO_15
        )


@dataclass(frozen=True)
class _Declaration:
    namespace: str
    component: str
    endpoint: str
    model: str
    worker_type: str
    release: LegacyVllmRelease


@dataclass(frozen=True)
class LegacyVllmTargets:
    declarations: tuple[_Declaration, ...] = ()

    @classmethod
    def from_json(cls, source: str) -> LegacyVllmTargets:
        if len(source.encode()) > 65536:
            raise ValueError("legacy vLLM target config is too large")
        entries = json.loads(source, object_pairs_hook=_unique_object)
        if not isinstance(entries, list) or len(entries) > 128:
            raise ValueError(
                "legacy vLLM targets must be an array of at most 128 entries"
            )
        declarations = []
        scopes = set()
        scope_fields = ("namespace", "component", "endpoint", "model", "worker_type")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                *scope_fields,
                "dynamo_release",
            }:
                raise ValueError("legacy vLLM target has missing or unknown fields")
            scope = tuple(entry[name] for name in scope_fields)
            if any(
                not isinstance(value, str)
                or not value
                or len(value.encode()) > 512
                or value.strip() != value
                or "*" in value
                or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)
                for value in scope
            ):
                raise ValueError(
                    "legacy vLLM targets require nonempty exact scope strings"
                )
            if entry["worker_type"] not in {"aggregated", "prefill", "decode"}:
                raise ValueError(
                    "legacy vLLM sampling target has unsupported worker role"
                )
            if scope in scopes:
                raise ValueError("duplicate legacy vLLM target scope")
            scopes.add(scope)
            try:
                release = LegacyVllmRelease(entry["dynamo_release"])
            except ValueError as error:
                raise ValueError("unsupported legacy vLLM release") from error
            declarations.append(_Declaration(*scope, release))
        return cls(tuple(declarations))

    def resolve(
        self,
        namespace: str,
        component: str,
        endpoint: str,
        model: str,
        worker_type: str | None,
        model_input: str,
    ) -> LegacyVllmRelease | None:
        if model_input != "tokens":
            return None
        for declaration in self.declarations:
            if (
                declaration.namespace == namespace
                and declaration.component == component
                and declaration.endpoint == endpoint
                and declaration.model == model
                and declaration.worker_type == worker_type
            ):
                return declaration.release
        return None
