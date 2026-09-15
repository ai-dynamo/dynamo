# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.triton.backend_args: the custom argparse actions
(ThreeArgsDictAction, TwoArgsDictAction) that parse Triton's compound flags
(--backend-config, --cache-config, --host-policy, --cuda-memory-pool-byte-size)
into structured dicts."""

import argparse
from collections.abc import Iterable
from typing import Any

import pytest

from dynamo.triton.backend_args import ThreeArgsDictAction, TwoArgsDictAction

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _run_action(
    action_cls: type[argparse.Action], dest: str, values: Iterable[str]
) -> Any:
    """Apply ``action_cls`` to each value in turn against one namespace,
    returning the accumulated ``dest`` dict. Raises ``argparse.ArgumentError``
    on malformed input, exactly as it does mid-parse (argparse turns that into
    a usage error + exit on the real CLI)."""
    parser = argparse.ArgumentParser()
    action = action_cls(option_strings=["--x"], dest=dest)
    namespace = argparse.Namespace()
    for value in values:
        action(parser, namespace, value)
    return getattr(namespace, dest)


# --- ThreeArgsDictAction: '<name>,<setting>=<value>' -> {name: {setting: value}}


@pytest.mark.parametrize(
    "bad_value",
    [
        "tensorrt",  # no ',' and no '='
        "tensorrt,plugins",  # missing '='
        ",plugins=x",  # empty name
        "tensorrt,=x",  # empty setting
        "tensorrt,plugins=",  # empty value
    ],
)
def test_nested_dict_action_rejects_malformed(bad_value):
    """ThreeArgsDictAction rejects malformed '<name>,<setting>=<value>' tokens."""
    with pytest.raises(argparse.ArgumentError):
        _run_action(ThreeArgsDictAction, "backend_configuration", [bad_value])


def test_nested_dict_action_rejects_duplicate_setting():
    """The same (name, setting) twice is rejected, not silently overwritten."""
    with pytest.raises(argparse.ArgumentError):
        _run_action(
            ThreeArgsDictAction,
            "backend_configuration",
            ["tensorrt,plugins=/a.so", "tensorrt,plugins=/b.so"],
        )


def test_nested_dict_action_merges_distinct_settings():
    """Distinct settings merge per name; only the first ',' and first '=' split,
    so a value may itself contain ',' and '='."""
    result = _run_action(
        ThreeArgsDictAction,
        "backend_configuration",
        ["tensorrt,plugins=/a.so;/b.so", "tensorrt,coalesce=on", "python,shm=a=b,c"],
    )
    assert result == {
        "tensorrt": {"plugins": "/a.so;/b.so", "coalesce": "on"},
        "python": {"shm": "a=b,c"},
    }


# --- TwoArgsDictAction: '<key>:<value>' -> {int: int}


@pytest.mark.parametrize(
    "bad_value",
    [
        "1024",  # missing ':'
        "a:1024",  # non-integer key
        "0:abc",  # non-integer value
        "0:1:2",  # extra ':'
        ":1024",  # empty key
        "0:",  # empty value
        "-1:1024",  # negative key
        "0:-5",  # negative value
    ],
)
def test_two_args_dict_action_rejects_malformed(bad_value):
    """TwoArgsDictAction rejects malformed/negative '<key>:<value>' tokens."""
    with pytest.raises(argparse.ArgumentError):
        _run_action(TwoArgsDictAction, "cuda_memory_pool_sizes", [bad_value])


def test_two_args_dict_action_rejects_duplicate_key():
    """The same key twice is rejected, not silently overwritten."""
    with pytest.raises(argparse.ArgumentError):
        _run_action(
            TwoArgsDictAction,
            "cuda_memory_pool_sizes",
            ["0:1024", "0:2048"],
        )


def test_two_args_dict_action_merges_distinct_keys():
    """Distinct keys merge into a {int: int} dict."""
    result = _run_action(
        TwoArgsDictAction,
        "cuda_memory_pool_sizes",
        ["0:1024", "1:2048"],
    )
    assert result == {0: 1024, 1: 2048}
