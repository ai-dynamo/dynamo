# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One strict, hash-verified source reader for native CPU reference fixtures."""

import ast
import os
import subprocess
from pathlib import Path
from types import ModuleType

from dynamo.vllm import decision_native_loader as native_contract
from dynamo.vllm.decision_cache_provenance import git_blob_id

BUNDLED_SOURCE_ROOT = Path(__file__).parent / "fixtures/vllm_0_33_0"


def load_pinned_engine_errors():
    """Execute only hash-verified upstream exception definitions without vLLM."""
    repository = source_repository()
    module = ModuleType("_pinned_native_engine_exceptions")
    exec(  # noqa: S102 -- Hash-verified stdlib-only upstream exception module.
        compile(
            read_pinned_source(repository, "vllm/exceptions.py"),
            module.__name__,
            "exec",
        ),
        module.__dict__,
    )
    tree = ast.parse(read_pinned_source(repository, "vllm/v1/engine/exceptions.py"))
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    exec(  # noqa: S102 -- Exact upstream classes reuse the verified base above.
        compile(tree, module.__name__, "exec"), module.__dict__
    )
    return module


def source_repository():
    """Use bundled CPU fixtures unless an explicit Git source override is set."""
    repository = os.environ.get("DYNAMO_VLLM_SOURCE_REPO")
    if repository == "":
        raise ValueError("DYNAMO_VLLM_SOURCE_REPO override must be nonempty")
    return repository


def read_pinned_source(repository, path):
    """Verify either source against its exact upstream Git blob before execution."""
    expected = dict(native_contract.PINNED_SOURCE_BLOBS)[path]
    if repository is None:
        data = (BUNDLED_SOURCE_ROOT / f"{path}.txt").read_bytes()
    else:
        result = subprocess.run(
            [
                "git",
                "-c",
                "gc.auto=0",
                "-C",
                str(repository),
                "show",
                f"{native_contract.PINNED_VLLM_COMMIT}:{path}",
            ],
            check=True,
            capture_output=True,
            timeout=10,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        data = result.stdout
    assert (
        git_blob_id(data) == expected
    ), f"Native fixture {path} differs from its pinned Git blob"
    return data.decode()
