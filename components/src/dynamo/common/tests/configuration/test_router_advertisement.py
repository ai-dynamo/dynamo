#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for the worker-side router advertisement group."""

import argparse

import pytest

from dynamo.common.configuration.groups.router_args import (
    RouterArgGroup,
    add_worker_router_arguments,
    build_router_config,
    parse_worker_router_config,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _router_flags() -> set[str]:
    parser = argparse.ArgumentParser(add_help=False)
    add_worker_router_arguments(parser)
    return {opt for action in parser._actions for opt in action.option_strings}


def test_omitting_router_mode_advertises_nothing():
    """The default must inherit the frontend's config.

    Every deployment that predates this feature omits --router-mode. If that
    produced a RouterConfig, each worker would override the frontend's global
    mode the moment it was upgraded.
    """
    config, _ = parse_worker_router_config([])
    assert config.router_mode is None
    assert build_router_config(config) is None


def test_non_router_arguments_pass_through_untouched():
    """The engine parser downstream must still see its own flags."""
    config, remainder = parse_worker_router_config(
        ["--model", "m", "--router-mode", "kv", "--tensor-parallel-size", "2"]
    )
    assert config.router_mode == "kv"
    assert remainder == ["--model", "m", "--tensor-parallel-size", "2"]


def test_kv_tuning_is_carried_not_reset():
    """A worker advertising KV must carry its tuning, not silently default it.

    The card replaces the frontend's config wholesale, so if the worker could
    name a mode but not the tuning that goes with it, advertising `kv` would
    quietly discard whatever KV tuning the frontend was configured with.
    """
    with pytest.warns(FutureWarning, match="--router-policy-config"):
        config, _ = parse_worker_router_config(
            ["--router-mode", "kv", "--router-kv-overlap-score-credit", "2.5"]
        )
    router_config = build_router_config(config)
    assert router_config is not None
    assert router_config.kv_router_config.overlap_score_credit == 2.5


def test_frontend_only_arguments_are_not_offered_to_workers():
    """Flags a worker card cannot carry must not appear on a worker."""
    flags = _router_flags()
    assert "--router-min-initial-workers" not in flags
    assert "--enforce-disagg" not in flags
    assert "--admission-control" not in flags
    # ... while the ones a worker set genuinely advertises are present.
    assert "--router-mode" in flags
    assert "--router-session-affinity-ttl-secs" in flags
    assert "--router-kv-events" in flags


def test_router_arg_group_refuses_to_guess_its_caller():
    """`RouterArgGroup()` must not fall back to frontend-shaped defaults.

    Copying the frontend's construction into a worker would default that worker
    to `--router-mode round-robin`. Because a worker's card replaces the
    frontend's configuration wholesale, the worker would then silently override
    a frontend running any other mode -- with no error, just routing that
    ignores the operator. Requiring both arguments makes that a startup
    TypeError instead.
    """
    with pytest.raises(TypeError):
        RouterArgGroup()  # type: ignore[call-arg]


_ABSOLUTE_PREFILL_DEPRECATION = "--active-prefill-tokens-threshold is deprecated"


@pytest.mark.parametrize("source", ["cli", "env"])
def test_deprecated_absolute_prefill_threshold_warns_when_advertised(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    source: str,
):
    """A worker forwarding the deprecated threshold must see the migration notice.

    Workers never run the frontend's validation, so the warning has to come
    from the shared path that builds the advertisement, for the CLI flag and
    the environment variable alike.
    """
    monkeypatch.delenv("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD", raising=False)
    argv = ["--router-mode", "kv"]
    if source == "cli":
        argv += ["--active-prefill-tokens-threshold", "123"]
    else:
        monkeypatch.setenv("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD", "123")
    caplog.set_level("WARNING")

    config, _ = parse_worker_router_config(argv)
    assert config.active_prefill_tokens_threshold == 123
    assert _ABSOLUTE_PREFILL_DEPRECATION not in caplog.text

    assert build_router_config(config) is not None
    assert caplog.text.count(_ABSOLUTE_PREFILL_DEPRECATION) == 1


def test_absolute_prefill_threshold_unset_or_unadvertised_stays_quiet(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """No notice without a value, and none when the value is never forwarded.

    A worker that omits --router-mode advertises nothing, so the threshold is
    dropped along with every other router flag rather than forwarded.
    """
    monkeypatch.delenv("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD", raising=False)
    caplog.set_level("WARNING")

    config, _ = parse_worker_router_config(["--router-mode", "kv"])
    assert build_router_config(config) is not None

    config, _ = parse_worker_router_config(["--active-prefill-tokens-threshold", "123"])
    assert build_router_config(config) is None

    assert _ABSOLUTE_PREFILL_DEPRECATION not in caplog.text


def test_unknown_mode_is_rejected():
    config, _ = parse_worker_router_config([])
    config.router_mode = "nonsense"
    with pytest.raises(ValueError, match="unknown router mode"):
        build_router_config(config)


@pytest.mark.parametrize("backend", ["vllm", "trtllm", "sglang"])
def test_router_flags_do_not_collide_with_backend_flags(backend):
    """Guard against a backend later adding a flag the router group owns.

    The router flags are parsed by their own parser off the backend parser's
    remainder. If a backend registered the same spelling, it would silently
    consume the flag first and the advertisement would be lost -- no argparse
    error, just routing that quietly ignores the operator.
    """
    pytest.importorskip(
        {"vllm": "vllm", "trtllm": "tensorrt_llm", "sglang": "sglang"}[backend],
        reason=f"{backend} is not installed in this environment",
    )
    if backend == "vllm":
        from dynamo.vllm.backend_args import DynamoVllmArgGroup as group
    elif backend == "trtllm":
        from dynamo.trtllm.backend_args import DynamoTrtllmArgGroup as group
    else:
        from dynamo.sglang.backend_args import DynamoSGLangArgGroup as group

    from dynamo.common.configuration.groups.runtime_args import DynamoRuntimeArgGroup

    parser = argparse.ArgumentParser(add_help=False)
    DynamoRuntimeArgGroup().add_arguments(parser)
    group().add_arguments(parser)
    backend_flags = {opt for action in parser._actions for opt in action.option_strings}

    collisions = sorted(_router_flags() & backend_flags)
    assert not collisions, (
        f"{backend} registers router flags {collisions}; the router parser would "
        f"never see them. Rename the backend flag or route it through "
        f"WorkerRouterConfig."
    )
