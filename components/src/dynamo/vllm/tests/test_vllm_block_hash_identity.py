# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do two vLLM workers give one content the same KV event block hash?

The router's ownership delegate reports a block by the engine hash of its KV events, so a
fleet-wide history of blocks (first copy appears, last copy goes) holds only if every worker
hashes one content alike. vLLM chains each block hash from ``NONE_HASH``, which it derives from
``PYTHONHASHSEED`` when that is set; otherwise from a fixed seed for sha256 and from random bytes
for xxhash. ``dynamo.vllm`` sets ``PYTHONHASHSEED=0`` when it is unset.

Each simulated worker runs ``init_none_hash`` afresh, as a new worker process does once at
startup: ``NONE_HASH`` depends only on the seed it resolves (or on fresh random bytes), so a
fresh call in one process derives what a separate process would. The Rust side, that equal
engine hashes from two workers give one delegate key, is
``lib/kv-router/src/indexer/ledger_key_tests.rs``.
"""

import os
import subprocess
import sys
from importlib.metadata import version

import pytest
from packaging.version import Version

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.core.kv_cache_utils")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

BLOCK_SIZE = 16
TOKENS = list(range(1, 4 * BLOCK_SIZE + 1))


def worker_block_hashes(monkeypatch, seed, algorithm):
    """The KV event block hashes one fresh worker emits for ``TOKENS``."""
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.request import Request

    if seed is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", seed)
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    hash_fn = get_hash_fn_by_name(algorithm)
    kv_cache_utils.init_none_hash(hash_fn)
    request = Request(
        request_id="block-hash-identity",
        prompt_token_ids=TOKENS,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=kv_cache_utils.get_request_block_hasher(BLOCK_SIZE, hash_fn),
    )
    hashes = [kv_cache_utils.maybe_convert_block_hash(h) for h in request.block_hashes]
    assert len(hashes) == len(TOKENS) // BLOCK_SIZE
    return hashes


@pytest.fixture
def restore_none_hash(monkeypatch):
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setattr(
        kv_cache_utils,
        "NONE_HASH",
        getattr(kv_cache_utils, "NONE_HASH", None),
        raising=False,
    )


@pytest.mark.parametrize(
    ("seeds", "algorithm", "alike"),
    [
        pytest.param(("0", "0"), "sha256", True, id="same-seed-sha256"),
        pytest.param(("0", "0"), "xxhash", True, id="same-seed-xxhash"),
        pytest.param(("0", "1"), "sha256", False, id="different-seeds"),
        pytest.param((None, None), "xxhash", False, id="no-seed-xxhash"),
    ],
)
def test_two_workers_hash_one_content_alike_only_with_one_seed(
    monkeypatch, restore_none_hash, seeds, algorithm, alike
):
    if algorithm == "xxhash":
        pytest.importorskip(
            "xxhash", reason="vLLM's xxhash algorithms need the xxhash package"
        )
    first = worker_block_hashes(monkeypatch, seeds[0], algorithm)
    second = worker_block_hashes(monkeypatch, seeds[1], algorithm)
    if alike:
        assert first == second
    else:
        # The chain starts at NONE_HASH, so no block of the content shares a hash.
        assert not set(first) & set(second)


def test_with_no_seed_sha256_workers_hash_alike(monkeypatch, restore_none_hash):
    """vLLM 0.30 derives NONE_HASH from a fixed seed for cryptographic hashes."""
    if Version(version("vllm")) < Version("0.30.0"):
        pytest.skip("vLLM before 0.30 seeds sha256 NONE_HASH with random bytes")
    first = worker_block_hashes(monkeypatch, None, "sha256")
    second = worker_block_hashes(monkeypatch, None, "sha256")
    assert first == second


@pytest.mark.parametrize(("given", "expected"), [(None, "0"), ("7", "7")])
def test_the_dynamo_vllm_entry_point_pins_the_seed(given, expected):
    """``python -m dynamo.vllm`` sets PYTHONHASHSEED=0 when unset and keeps a given value."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    if given is not None:
        env["PYTHONHASHSEED"] = given
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, dynamo.vllm.__main__; print(os.environ['PYTHONHASHSEED'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == expected
