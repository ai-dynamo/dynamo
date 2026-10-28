# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline resolver and immutable references for real-engine serving tests."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from dynamo.common.lora.runtime import (
    ResolveContext,
    ResolvedLoRA,
    RuntimeLoRANotFoundError,
    RuntimeLoRAResolverUnavailableError,
)

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_REPO = "codelion/Qwen3-0.6B-accuracy-recovery-lora"
CHANGED_SOURCE = "wandb-artifact:///test/contract-violation/revision-changes"
ERROR_SOURCE_PREFIX = "wandb-artifact:///test/errors/"


def adapter_source(revision: str, replica: str) -> str:
    return f"wandb-artifact:///test/{ADAPTER_REPO}/{revision}/{replica}"


class OfflineResolver:
    protocol_version = 2
    schemes = frozenset({"wandb-artifact"})

    def __init__(self) -> None:
        self._calls: dict[str, int] = {}
        self._revision = os.environ["DYNAMO_TEST_LORA_REVISION"]
        self._sources = frozenset(
            adapter_source(self._revision, replica)
            for replica in ("replica-a", "replica-b")
        )

    async def resolve(self, *, source_uri: str, context: ResolveContext):
        calls = self._calls.get(source_uri, 0) + 1
        self._calls[source_uri] = calls
        with Path(os.environ["DYNAMO_TEST_LORA_CALLS"]).open("a") as stream:
            stream.write(
                json.dumps({"source": source_uri, "key": context.adapter_key}) + "\n"
            )
        failure = source_uri.removeprefix(ERROR_SOURCE_PREFIX)
        if failure == "missing":
            raise RuntimeLoRANotFoundError("test provider private reference")
        if failure == "unavailable":
            raise RuntimeLoRAResolverUnavailableError("test provider private reference")
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "invalid-result":
            return object()
        if (
            source_uri not in self._sources
            and source_uri != CHANGED_SOURCE
            and failure != "malformed"
        ):
            return None

        destination = context.cache_root / context.adapter_key
        source = Path(os.environ["DYNAMO_TEST_LORA_ADAPTER"])

        def materialize() -> None:
            if destination.exists():
                return
            paths = [
                source / "adapter_config.json",
                source / "adapter_model.safetensors",
            ]
            if sum(path.stat().st_size for path in paths) > context.max_download_bytes:
                raise ValueError("test snapshot exceeds reserved download budget")
            with tempfile.TemporaryDirectory(dir=context.cache_root) as temporary:
                stage = Path(temporary) / "snapshot"
                stage.mkdir()
                for path in paths:
                    shutil.copyfile(path, stage / path.name)
                if failure == "malformed":
                    (stage / "adapter_config.json").write_text('{"r": []}')
                stage.rename(destination)

        await asyncio.to_thread(materialize)
        revision = self._revision
        if source_uri == CHANGED_SOURCE and calls > 1:
            revision = "deliberately-changed-provider-revision"
        return ResolvedLoRA(local_path=destination, source_revision=revision)
