# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from dynamo.trtllm.utils import model_path as model_path_module
from dynamo.trtllm.utils.model_path import resolve_model_path

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

MODEL = "Qwen/Qwen3-0.6B"
COMMIT = "c1899de289a04d12100db370d81485cdf75e47ca"
OTHER_COMMIT = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"


class _ReachableHub:
    """A Hub that answers, without the tests reaching the network to find out."""

    def model_info(self, *args, **kwargs):
        return object()


class _UnreachableHub:
    def model_info(self, *args, **kwargs):
        raise OSError("no route to host")


def _hub_raising(error):
    class _Hub:
        def model_info(self, *args, **kwargs):
            raise error

    return _Hub


def _clear_cache_env(monkeypatch):
    for name in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "TRANSFORMERS_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    # Offline by default: the fallback is only for a Hub that cannot answer,
    # and no case here may depend on the runner having network.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _hub(tmp_path, monkeypatch):
    """Point Hugging Face's cache lookup at an empty directory under tmp_path."""
    _clear_cache_env(monkeypatch)
    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    hub = hf_home / "hub"
    hub.mkdir(parents=True)
    return hub


def _repo(hub):
    repo = hub / "models--Qwen--Qwen3-0.6B"
    (repo / "refs").mkdir(parents=True)
    (repo / "snapshots").mkdir(parents=True)
    return repo


def _snapshot(repo, commit, complete=True):
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir()
    if complete:
        (snapshot / "config.json").write_text('{"model_type": "qwen3"}')
        (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def test_empty_ref_falls_back_to_the_cached_snapshot(tmp_path, monkeypatch):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)

    resolved = resolve_model_path(MODEL)

    assert resolved == str(snapshot)


def test_valid_ref_keeps_the_repository_id(tmp_path, monkeypatch):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text(COMMIT)
    _snapshot(repo, COMMIT)

    assert resolve_model_path(MODEL) == MODEL


def test_no_local_cache_keeps_the_repository_id(tmp_path, monkeypatch):
    _hub(tmp_path, monkeypatch)

    assert resolve_model_path(MODEL) == MODEL


def test_unusable_ref_without_a_complete_snapshot_keeps_the_repository_id(
    tmp_path, monkeypatch
):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    _snapshot(repo, COMMIT, complete=False)

    assert resolve_model_path(MODEL) == MODEL


def test_ref_naming_an_incomplete_snapshot_keeps_the_repository_id(
    tmp_path, monkeypatch
):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text(OTHER_COMMIT)
    _snapshot(repo, OTHER_COMMIT, complete=False)
    _snapshot(repo, COMMIT)

    assert resolve_model_path(MODEL) == MODEL


@pytest.mark.parametrize("revision", ["release-x", COMMIT])
def test_explicit_revision_keeps_the_repository_id(tmp_path, monkeypatch, revision):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    _snapshot(repo, COMMIT)

    assert resolve_model_path(MODEL, revision) == MODEL


@pytest.mark.parametrize("ref", [None, "not-a-commit-hash"])
def test_a_ref_that_is_not_empty_keeps_the_repository_id(tmp_path, monkeypatch, ref):
    """Only an empty ref makes the engine's lookup return the hashless path."""
    repo = _repo(_hub(tmp_path, monkeypatch))
    if ref is not None:
        (repo / "refs" / "main").write_text(ref)
    _snapshot(repo, COMMIT)

    assert resolve_model_path(MODEL) == MODEL


def test_snapshot_with_an_unresolved_link_keeps_the_repository_id(
    tmp_path, monkeypatch
):
    """A link to a blob that has not landed marks a download still running."""
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(repo / "blobs" / "absent")

    assert resolve_model_path(MODEL) == MODEL


def test_config_without_weights_keeps_the_repository_id(tmp_path, monkeypatch):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    (snapshot / "model.safetensors").unlink()

    assert resolve_model_path(MODEL) == MODEL


@pytest.mark.parametrize("weights", ["model.safetensors", "pytorch_model.bin"])
def test_sharded_snapshot_requires_every_indexed_shard(tmp_path, monkeypatch, weights):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    (snapshot / "model.safetensors").unlink()
    (snapshot / (weights + ".index.json")).write_text(
        json.dumps({"weight_map": {"a": "shard-1", "b": "shard-2"}})
    )
    (snapshot / "shard-1").write_bytes(b"first")

    assert resolve_model_path(MODEL) == MODEL

    (snapshot / "shard-2").write_bytes(b"second")

    assert resolve_model_path(MODEL) == str(snapshot)


@pytest.mark.parametrize("index", ["{", "[]", "{}", '{"weight_map": {}}'])
def test_unusable_weight_index_keeps_the_repository_id(tmp_path, monkeypatch, index):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    (snapshot / "model.safetensors.index.json").write_text(index)

    assert resolve_model_path(MODEL) == MODEL


def test_a_reachable_hub_keeps_the_repository_id(tmp_path, monkeypatch):
    """The Hub resolves the revision itself; do not guess from a stale cache."""
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    _snapshot(repo, COMMIT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(model_path_module, "HfApi", _ReachableHub)

    assert resolve_model_path(MODEL) == MODEL


def test_a_hub_that_cannot_answer_falls_back_to_the_cached_snapshot(
    tmp_path, monkeypatch
):
    """Offline mode is not the only way the lookup loses the Hub."""
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(model_path_module, "HfApi", _UnreachableHub)

    assert resolve_model_path(MODEL) == str(snapshot)


def test_a_connection_failure_falls_back_to_the_cached_snapshot(tmp_path, monkeypatch):
    httpx = pytest.importorskip("httpx")
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        model_path_module, "HfApi", _hub_raising(httpx.ConnectError("no route"))
    )

    assert resolve_model_path(MODEL) == str(snapshot)


def test_a_hub_that_refuses_the_repository_keeps_the_repository_id(
    tmp_path, monkeypatch
):
    """A refusal naming the repository is an answer, so the Hub is reachable."""
    httpx = pytest.importorskip("httpx")
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    _snapshot(repo, COMMIT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    refusal = GatedRepoError(
        "gated",
        response=httpx.Response(
            403, request=httpx.Request("GET", "https://huggingface.co")
        ),
    )
    monkeypatch.setattr(model_path_module, "HfApi", _hub_raising(refusal))

    assert resolve_model_path(MODEL) == MODEL


def test_an_unexpected_probe_error_is_not_read_as_an_outage(tmp_path, monkeypatch):
    """A defect in this call must surface, not license a substitution."""
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    _snapshot(repo, COMMIT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        model_path_module, "HfApi", _hub_raising(TypeError("bad argument"))
    )

    with pytest.raises(TypeError):
        resolve_model_path(MODEL)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 429, 500, 503])
def test_generic_http_errors_distinguish_refusals_from_outages(
    tmp_path, monkeypatch, status
):
    httpx = pytest.importorskip("httpx")
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    error = HfHubHTTPError(
        "lookup failed",
        response=httpx.Response(
            status, request=httpx.Request("GET", "https://huggingface.co")
        ),
    )
    monkeypatch.setattr(model_path_module, "HfApi", _hub_raising(error))

    expected = str(snapshot) if status in (408, 429, 500, 503) else MODEL
    assert resolve_model_path(MODEL) == expected


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
@pytest.mark.parametrize("value", ["1", "on", "Yes", "TRUE"])
def test_the_offline_variables_are_read_as_huggingface_reads_them(
    tmp_path, monkeypatch, variable, value
):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv(variable, value)
    # A probe would be a bug here: the variable already says there is no Hub.
    monkeypatch.setattr(model_path_module, "HfApi", _ReachableHub)

    assert resolve_model_path(MODEL) == str(snapshot)


@pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
def test_a_non_true_offline_value_still_probes_the_hub(tmp_path, monkeypatch, value):
    """huggingface_hub treats only 1/ON/YES/TRUE as offline, so neither may this."""
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    _snapshot(repo, COMMIT)
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    monkeypatch.setattr(model_path_module, "HfApi", _ReachableHub)

    assert resolve_model_path(MODEL) == MODEL


def test_legacy_cache_variable_locates_the_snapshot(tmp_path, monkeypatch):
    """huggingface_hub still honours HUGGINGFACE_HUB_CACHE, so this must too."""
    _clear_cache_env(monkeypatch)
    hub = tmp_path / "legacy-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    repo = _repo(hub)
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)

    assert resolve_model_path(MODEL) == str(snapshot)
