# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest
import yaml

from dynamo.aisimulate.output.dgd import dgdradapter as adapter_module
from dynamo.aisimulate.output.dgd.dgdradapter import DGDRAdapter
from dynamo.aisimulate.output.dgd.kube_status import STATUS_FILE_NAME
from dynamo.aisimulate.output.dgd.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


class _Record:
    """Fake CandidateRecordLike, matching the real CandidateRecord's shape."""

    def __init__(
        self,
        candidate_id: str,
        *,
        status: str = "feasible",
        score: float | None = None,
        used_gpus: int | None = None,
    ):
        self.candidate_id = candidate_id
        self.status = status
        self.score = score
        self.used_gpus = used_gpus
        self.config = {
            "backend": "vllm",
            "backend_version": "0.20.1",
            "candidate_id": candidate_id,
        }


def _options() -> DGDGenerationOptions:
    return DGDGenerationOptions(
        runtime_image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0", num_gpus_per_node=8
    )


def _fake_render(record, workload, options, *, dgd_name, renderer="aic"):
    return f"kind: DynamoGraphDeployment\nmetadata:\n  name: {dgd_name}\n"


def _read_status(output_dir) -> dict:
    return yaml.safe_load((output_dir / STATUS_FILE_NAME).read_text())


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except FileNotFoundError:
            pass  # the background thread hasn't written the first snapshot yet
        time.sleep(0.01)
    raise AssertionError("condition was never satisfied")


def test_pareto_goal_is_rejected_explicitly(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="scalar goals"):
        DGDRAdapter(tmp_path, _options(), workload={}, is_pareto=True)


def test_top_n_above_the_configmap_size_ceiling_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="ConfigMap size budget"):
        DGDRAdapter(tmp_path, _options(), workload={}, top_n=21)


def test_top_n_at_the_configmap_size_ceiling_is_accepted(tmp_path) -> None:
    DGDRAdapter(tmp_path, _options(), workload={}, top_n=20)


def test_top_n_retention_evicts_the_worst_scoring_candidate(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(adapter_module, "render_dgd", _fake_render)
    adapter = DGDRAdapter(tmp_path, _options(), workload={}, top_n=2)
    with adapter:
        adapter.on_candidate(_Record("c-low", score=1.0, used_gpus=4))
        adapter.on_candidate(_Record("c-high", score=3.0, used_gpus=4))
        adapter.on_candidate(_Record("c-mid", score=2.0, used_gpus=4))
        _wait_until(lambda: len(_read_status(tmp_path).get("candidates", [])) == 2)

    status = _read_status(tmp_path)
    retained_ids = {entry["id"] for entry in status["candidates"]}
    assert retained_ids == {"c-high", "c-mid"}
    assert status["status"] == "success"
    # Every retained entry carries its manifest inline, not just a path --
    # the controller never reads the Sweeper pod's filesystem.
    for entry in status["candidates"]:
        assert "DynamoGraphDeployment" in entry["manifest"]


def test_retained_candidates_are_ordered_best_first_in_snapshot(
    monkeypatch, tmp_path
) -> None:
    # The DGDR(v2) controller derives each DynamoGraphDeploymentCandidate's
    # one-based Rank from this list's position, so it must be sorted by
    # score (descending, fewer GPUs as the tie-break) -- not by dict
    # insertion/update order, which is merely "most-recently-touched" and
    # has no relationship to rank.
    monkeypatch.setattr(adapter_module, "render_dgd", _fake_render)
    adapter = DGDRAdapter(tmp_path, _options(), workload={}, top_n=5)
    with adapter:
        adapter.on_candidate(_Record("c-mid", score=2.0, used_gpus=4))
        adapter.on_candidate(_Record("c-low", score=1.0, used_gpus=4))
        adapter.on_candidate(_Record("c-high", score=3.0, used_gpus=4))
        # Re-touch "c-low" last so it is most-recently-updated in the
        # dict, while still being the worst score -- this is what would
        # break a dict-insertion-order read.
        adapter.on_candidate(_Record("c-low", score=1.0, used_gpus=4))
        _wait_until(lambda: len(_read_status(tmp_path).get("candidates", [])) == 3)

    status = _read_status(tmp_path)
    assert [entry["id"] for entry in status["candidates"]] == [
        "c-high",
        "c-mid",
        "c-low",
    ]


def test_non_feasible_candidate_is_never_retained_or_rendered(
    monkeypatch, tmp_path
) -> None:
    rendered_ids = []

    def _tracking_render(record, workload, options, *, dgd_name, renderer="aic"):
        rendered_ids.append(record.candidate_id)
        return _fake_render(
            record, workload, options, dgd_name=dgd_name, renderer=renderer
        )

    monkeypatch.setattr(adapter_module, "render_dgd", _tracking_render)
    adapter = DGDRAdapter(tmp_path, _options(), workload={}, top_n=5)
    with adapter:
        adapter.on_candidate(_Record("c-infeasible", status="infeasible"))
        adapter.on_candidate(_Record("c-ok", score=1.0, used_gpus=1))
        _wait_until(lambda: len(_read_status(tmp_path).get("candidates", [])) == 1)

    # Every snapshot write re-renders the whole retained set (including the
    # close()-triggered terminal flush), so "c-ok" may appear more than once
    # -- what matters is "c-infeasible" never does.
    assert set(rendered_ids) == {"c-ok"}


def test_materialization_failure_is_reported_as_data_not_raised(
    monkeypatch, tmp_path
) -> None:
    def _failing_render(record, workload, options, *, dgd_name, renderer="aic"):
        raise CandidateMaterializationError(
            "renderer output must contain exactly one DynamoGraphDeployment"
        )

    monkeypatch.setattr(adapter_module, "render_dgd", _failing_render)
    adapter = DGDRAdapter(tmp_path, _options(), workload={}, top_n=5)
    with adapter:
        adapter.on_candidate(_Record("c-bad", score=1.0, used_gpus=1))
        _wait_until(lambda: _read_status(tmp_path).get("candidates"))

    status = _read_status(tmp_path)
    [entry] = status["candidates"]
    assert entry["outcome"] == "materialization_failed"
    assert "DynamoGraphDeployment" in entry["error"]
    assert not (tmp_path / "candidates" / "c-bad.yaml").exists()


def test_on_round_updates_counters_without_touching_candidates(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(adapter_module, "render_dgd", _fake_render)
    adapter = DGDRAdapter(tmp_path, _options(), workload={}, top_n=5)
    with adapter:
        adapter.on_round(1, [object(), object(), object()])
        _wait_until(lambda: _read_status(tmp_path).get("round_no") == 1)
        adapter.on_round(2, [object(), object()])
        _wait_until(lambda: _read_status(tmp_path).get("round_no") == 2)

    status = _read_status(tmp_path)
    assert status["round_no"] == 2
    assert status["cumulative_evaluated"] == 5  # 3 + 2, accumulated across rounds
    assert "candidates" not in status


def test_close_always_flushes_terminal_status_even_without_candidates(tmp_path) -> None:
    adapter = DGDRAdapter(tmp_path, _options(), workload={})
    adapter.start()
    adapter.close(
        status=adapter_module.SweepRunStatus.FAILED, error="search raised ValueError"
    )

    status = _read_status(tmp_path)
    assert status["status"] == "failed"
    assert status["error"] == "search raised ValueError"


def test_close_without_start_still_flushes_synchronously(tmp_path) -> None:
    # Covers setup failing before start() is ever reached (e.g. a bad
    # workload) -- close() must still leave a valid terminal snapshot, with
    # no background thread to join.
    adapter = DGDRAdapter(tmp_path, _options(), workload={})
    adapter.close(
        status=adapter_module.SweepRunStatus.FAILED, error="setup raised before start"
    )

    status = _read_status(tmp_path)
    assert status["status"] == "failed"
    assert status["error"] == "setup raised before start"
    assert not adapter._thread.is_alive()


def test_context_manager_marks_failed_status_on_exception(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        with DGDRAdapter(tmp_path, _options(), workload={}):
            raise RuntimeError("search blew up")

    status = _read_status(tmp_path)
    assert status["status"] == "failed"


def test_candidate_manifest_is_inlined_and_also_written_to_the_shared_output_dir(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(adapter_module, "render_dgd", _fake_render)
    adapter = DGDRAdapter(tmp_path, _options(), workload={}, top_n=5)
    with adapter:
        adapter.on_candidate(_Record("c-ok", score=1.0, used_gpus=1))
        _wait_until(lambda: _read_status(tmp_path).get("candidates"))

    # The manifest is inlined into the status snapshot -- this is what the
    # ConfigMap relay actually carries, and all the controller ever reads.
    status = _read_status(tmp_path)
    [entry] = status["candidates"]
    assert "DynamoGraphDeployment" in entry["manifest"]

    # It's also written to disk at `path`, purely as a human-readable
    # artifact alongside the Sweeper's own output -- not read by the
    # controller, which has no access to this filesystem.
    manifest_path = tmp_path / "candidates" / "c-ok.yaml"
    assert manifest_path.exists()
    assert "DynamoGraphDeployment" in manifest_path.read_text()
    assert entry["path"] == "candidates/c-ok.yaml"


def test_enqueue_never_blocks_the_caller_even_when_queue_is_full(
    monkeypatch, tmp_path
) -> None:
    # Hold the background thread off via a never-set stop event substitute:
    # simplest is a queue_size of 1 and a slow-rendering fake so the drain
    # thread can't keep up, then flood on_candidate from the "search loop".
    monkeypatch.setattr(
        adapter_module,
        "render_dgd",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("render should not be reached by this test")
        ),
    )
    adapter = DGDRAdapter(tmp_path, _options(), workload={}, top_n=5, queue_size=1)
    # Don't start() the background thread at all -- this isolates _enqueue's
    # own non-blocking behavior from however fast the drain loop happens to be.
    started = threading.Event()

    def _flood() -> None:
        started.set()
        for i in range(50):
            adapter.on_candidate(_Record(f"c-{i}", score=float(i), used_gpus=1))

    thread = threading.Thread(target=_flood)
    thread.start()
    started.wait()
    thread.join(timeout=2.0)
    assert (
        not thread.is_alive()
    ), "on_candidate blocked the caller instead of dropping overflow"
