# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin token worker. Only this module imports the target Dynamo bindings."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import re
import struct
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import dynamo._core
import dynamo.frontend
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime

from .contract import (
    ReplayError,
    digest,
    dumps,
    file_digest,
    load_corpus,
    loads,
    scheduled_events,
    verify_assets,
    write_json,
)

ANNOTATION_PREFIX = "frontend_replay:"


class Context(Protocol):
    def is_stopped(self) -> bool:
        ...

    def is_killed(self) -> bool:
        ...


async def serve(
    corpus_path: Path,
    model_dir: Path,
    output_dir: Path,
    endpoint_path: str,
    limit: int | None,
) -> None:
    corpus = load_corpus(corpus_path, limit)
    assets = verify_assets(corpus, model_dir)
    cases = {case.identifier: case for case in corpus.cases}
    config = ModelRuntimeConfig()
    for key, value in corpus.profile["runtime_config"].items():
        if key == "structural_tag":
            config.set_structural_tag(value)
        elif key == "engine_specific":
            from_value = value
            if not isinstance(from_value, dict):
                raise ReplayError("engine_specific must be an object")
            for name, item in from_value.items():
                config.set_engine_specific(name, dumps(item))
        else:
            setattr(config, key, value)
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(endpoint_path)
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        str(model_dir.resolve()),
        model_name=corpus.profile["model_name"],
        worker_type=WorkerType.Aggregated,
        runtime_config=config,
        ignore_weights=True,
    )
    core_path = Path(dynamo._core.__file__).resolve()
    frontend_path = Path(dynamo.frontend.__file__).resolve()
    if "components/src" in str(frontend_path):
        raise ReplayError(
            "frontend imports a source checkout; use the installed target package"
        )
    write_json(
        output_dir / "worker-ready.json",
        {
            "schema": "dynamo-frontend-replay-worker/v1",
            "manifest_sha256": corpus.manifest_sha256,
            "selected_ids_sha256": digest(list(cases)),
            "assets": assets,
            "python": sys.executable,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("ai-dynamo", "ai-dynamo-runtime")
            },
            "module_paths": {"core": str(core_path), "frontend": str(frontend_path)},
            "core_sha256": file_digest(core_path),
            "dynamo_commit_sha": os.environ.get("DYNAMO_COMMIT_SHA"),
        },
    )
    active: set[str] = set()

    async def generate(
        request: dict[str, Any], context: Context
    ) -> AsyncIterator[dict[str, Any]]:
        identifiers = [
            item[len(ANNOTATION_PREFIX) :]
            for item in request.get("annotations", [])
            if isinstance(item, str) and item.startswith(ANNOTATION_PREFIX)
        ]
        if len(identifiers) != 1:
            raise ReplayError("request requires exactly one replay annotation")
        plan = loads(identifiers[0])
        execution = plan["execution_id"]
        if (
            not isinstance(execution, str)
            or not re.fullmatch(r"[0-9a-f]{32}", execution)
            or execution in active
        ):
            raise ReplayError("invalid or reused execution ID")
        active.add(execution)
        case = cases[plan["case_id"]]
        frames = scheduled_events(case, plan["schedule"])
        observed = dict(request)
        packed = observed.get("token_ids")
        if isinstance(packed, bytes):
            if len(packed) % 4:
                raise ReplayError("invalid packed prompt IDs")
            observed["token_ids"] = list(
                struct.unpack("<" + "I" * (len(packed) // 4), packed)
            )
        write_json(output_dir / "worker" / f"{execution}.request.json", observed)
        emitted: list[int] = []
        completed = False
        try:
            for index, frame in enumerate(frames):
                if context.is_stopped() or context.is_killed():
                    return
                emitted.extend(frame["token_ids"])
                if index == len(frames) - 1:
                    completed = True
                    write_json(
                        output_dir / "worker" / f"{execution}.emission.json",
                        {
                            "state": "completed",
                            "token_ids": emitted,
                            "terminal_count": 1,
                        },
                    )
                yield frame
                await asyncio.sleep(0)
        finally:
            if not completed:
                write_json(
                    output_dir / "worker" / f"{execution}.emission.json",
                    {"state": "cancelled", "token_ids": emitted, "terminal_count": 0},
                )

    await endpoint.serve_endpoint(generate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "model-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(
        serve(args.corpus, args.model_dir, args.output_dir, args.endpoint, args.limit)
    )


if __name__ == "__main__":
    main()
