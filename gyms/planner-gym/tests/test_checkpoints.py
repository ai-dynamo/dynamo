# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint reuse is tied to exact inputs and preserves prior attempts."""

from __future__ import annotations

import json
import shutil

import pytest
from autoscaling_arena.checkpoints import RunJournal

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _signature():
    return {
        "replay_config_sha256": "configuration-one",
        "input_sha256": "input-one",
        "software": {"commit": "revision-one"},
    }


def test_resume_only_reuses_success_and_preserves_retry_evidence(tmp_path):
    signature = _signature()
    with RunJournal(tmp_path, signature) as journal:
        success = {
            "run_id": "0001-sim-static",
            "status": "ok",
            "metrics": {"goodput": 3},
        }
        failure = {
            "run_id": "0002-sim-static",
            "status": "failed",
            "error": "interrupted",
        }
        journal.record(success["run_id"], success)
        journal.record(failure["run_id"], failure)
        run_dir = tmp_path / "runs" / failure["run_id"]
        run_dir.mkdir()
        (run_dir / "worker.log").write_text("first attempt evidence")

    with RunJournal(tmp_path, signature, resume=True) as journal:
        assert journal.load_success(success["run_id"]) == success
        assert journal.load_success(failure["run_id"]) is None
        assert journal.load_success("0003-sim-static") is None
        journal.prepare_run(failure["run_id"])
        assert not run_dir.exists()
        assert journal.load_success(failure["run_id"]) is None
        archived = list((tmp_path / "attempts").iterdir())
        assert len(archived) == 1
        assert (
            archived[0] / "run" / "worker.log"
        ).read_text() == "first attempt evidence"
        assert (
            json.loads((archived[0] / "checkpoint.json").read_text())["result"]
            == failure
        )
        replacement = {**failure, "status": "ok"}
        journal.record(failure["run_id"], replacement)
        assert journal.load_success(failure["run_id"]) == replacement


def test_incomplete_attempt_without_checkpoint_is_archived(tmp_path):
    with RunJournal(tmp_path, _signature()) as journal:
        run_dir = tmp_path / "runs" / "incomplete"
        run_dir.mkdir()
        (run_dir / "partial.log").write_text("still valuable")
        assert journal.load_success("incomplete") is None
        journal.prepare_run("incomplete")
        journal.prepare_run("incomplete")
        assert len(list((tmp_path / "attempts").iterdir())) == 1
        assert (
            next((tmp_path / "attempts").glob("*/run/partial.log")).read_text()
            == "still valuable"
        )


@pytest.mark.parametrize("field", ["replay_config_sha256", "input_sha256", "software"])
def test_stale_signature_cannot_reuse_checkpoints_and_releases_lock(tmp_path, field):
    signature = _signature()
    with RunJournal(tmp_path, signature) as journal:
        journal.record("run-1", {"run_id": "run-1", "status": "ok"})
    changed = {**signature, field: "changed"}
    with pytest.raises(ValueError, match="signature does not match"):
        with RunJournal(tmp_path, changed, resume=True):
            pass
    with RunJournal(tmp_path, signature, resume=True) as journal:
        assert journal.load_success("run-1")["status"] == "ok"


def test_live_writer_is_exclusive_and_exception_releases_lock(tmp_path):
    with pytest.raises(RuntimeError, match="caller failure"):
        with RunJournal(tmp_path, _signature()):
            with pytest.raises(ValueError, match="another process"):
                with RunJournal(tmp_path, _signature(), resume=True):
                    pass
            raise RuntimeError("caller failure")
    with RunJournal(tmp_path, _signature(), resume=True):
        pass


def test_unknown_or_nonempty_directories_are_not_adopted(tmp_path):
    marker = tmp_path / "unrelated.txt"
    marker.write_text("retain this file")
    with pytest.raises(ValueError, match="empty session"):
        with RunJournal(tmp_path, _signature()):
            pass
    with pytest.raises(ValueError, match="cannot read checkpoint"):
        with RunJournal(tmp_path, _signature(), resume=True):
            pass
    assert marker.read_text() == "retain this file"


@pytest.mark.parametrize(
    "content",
    [
        '{"schema_version":',
        "[]",
        '{"schema_version":2}',
        '{"schema_version":true}',
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"value":NaN}',
        '{"schema_version":1,"value":1e999}',
    ],
)
def test_partial_and_invalid_checkpoint_fail_clearly(tmp_path, content):
    with RunJournal(tmp_path, _signature()) as journal:
        path = tmp_path / "checkpoints" / "run-1.json"
        path.write_text(content)
        with pytest.raises(ValueError, match="checkpoint"):
            journal.load_success("run-1")
        assert path.read_text() == content


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "wrong-run"},
        {"signature_sha256": "wrong-signature"},
        {"result": {"run_id": "other-run", "status": "ok"}},
        {"result": {"run_id": "run-1", "status": "running"}},
        {"result": {"run_id": "run-1", "status": []}},
    ],
)
def test_checkpoint_identity_and_terminal_status_are_validated(tmp_path, change):
    with RunJournal(tmp_path, _signature()) as journal:
        journal.record("run-1", {"run_id": "run-1", "status": "ok"})
        path = tmp_path / "checkpoints" / "run-1.json"
        value = json.loads(path.read_text())
        path.write_text(json.dumps({**value, **change}))
        with pytest.raises(ValueError, match="checkpoint"):
            journal.load_success("run-1")


@pytest.mark.parametrize(
    "run_id", ["../outside", "/absolute", "..", "with/slash", "", "a" * 241]
)
def test_unsafe_run_ids_are_rejected(tmp_path, run_id):
    with RunJournal(tmp_path, _signature()) as journal:
        for operation in (journal.load_success, journal.prepare_run):
            with pytest.raises(ValueError, match="safe filename"):
                operation(run_id)
        with pytest.raises(ValueError, match="safe filename"):
            journal.record(run_id, {"run_id": run_id, "status": "ok"})


def test_invalid_record_does_not_replace_previous_checkpoint(tmp_path):
    with RunJournal(tmp_path, _signature()) as journal:
        original = {"run_id": "run-1", "status": "ok"}
        journal.record("run-1", original)
        with pytest.raises(ValueError, match="identity"):
            journal.record("run-1", {"run_id": "wrong", "status": "ok"})
        with pytest.raises(ValueError, match="JSON"):
            journal.record("run-1", {**original, "metric": float("nan")})
        assert journal.load_success("run-1") == original
        assert sorted(path.name for path in (tmp_path / "checkpoints").iterdir()) == [
            "run-1.json"
        ]


def test_journal_requires_active_context_and_copies_signature(tmp_path):
    signature = _signature()
    journal = RunJournal(tmp_path, signature)
    signature["input_sha256"] = "mutated"
    with pytest.raises(RuntimeError, match="context manager"):
        journal.load_success("run-1")
    with journal:
        journal.record("run-1", {"run_id": "run-1", "status": "ok"})
    with RunJournal(tmp_path, _signature(), resume=True) as resumed:
        assert resumed.load_success("run-1") is not None


def test_symlinked_artifact_directory_cannot_move_external_files(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("protected")
    root = tmp_path / "session"
    root.mkdir()
    with RunJournal(root, _signature()) as journal:
        (root / "runs" / "run-1").symlink_to(external, target_is_directory=True)
        with pytest.raises(ValueError, match="symlinks"):
            journal.prepare_run("run-1")
    assert marker.read_text() == "protected"


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_missing_or_changed_report_cannot_be_reused(tmp_path, mutation):
    run_id = "run-with-report"
    report = tmp_path / "runs" / run_id / "trace-report.json"
    with RunJournal(tmp_path, _signature()) as journal:
        report.parent.mkdir()
        report.write_text('{"complete":true}')
        result = {
            "run_id": run_id,
            "status": "ok",
            "artifacts": {"trace_report": str(report)},
        }
        journal.record(run_id, result)
        assert journal.load_success(run_id) == result
    if mutation == "delete":
        report.unlink()
    else:
        report.write_text('{"complete":false}')
    with RunJournal(tmp_path, _signature(), resume=True) as journal:
        assert journal.load_success(run_id) is None
        journal.prepare_run(run_id)
        checkpoint = next((tmp_path / "attempts").glob("*/checkpoint.json"))
        archived = json.loads(checkpoint.read_text())
        assert archived["artifact_manifest"][0]["kind"] == "file"
        assert (
            archived["artifact_manifest"][0]["path"]
            == "runs/run-with-report/trace-report.json"
        )


@pytest.mark.parametrize(
    "changed_name", ["summary.json", "requests.jsonl", "telemetry.jsonl", "worker.log"]
)
def test_directory_manifest_checks_every_persistent_output(tmp_path, changed_name):
    run_id = "run-with-directory"
    run_dir = tmp_path / "runs" / run_id
    with RunJournal(tmp_path, _signature()) as journal:
        run_dir.mkdir()
        nested = run_dir / "reports"
        nested.mkdir()
        for name in ("summary.json", "requests.jsonl", "telemetry.jsonl", "worker.log"):
            (nested / name).write_text("original")
        result = {
            "run_id": run_id,
            "status": "ok",
            "artifacts": {"directory": f"runs/{run_id}"},
        }
        journal.record(run_id, result)
        assert journal.load_success(run_id) == result
        (nested / changed_name).write_text("changed")
        assert journal.load_success(run_id) is None


def test_missing_output_at_record_is_never_a_reusable_success(tmp_path):
    with RunJournal(tmp_path, _signature()) as journal:
        path = tmp_path / "runs" / "run-1" / "trace-report.json"
        path.parent.mkdir()
        result = {
            "run_id": "run-1",
            "status": "ok",
            "artifacts": {"trace_report": str(path)},
        }
        journal.record("run-1", result)
        assert journal.load_success("run-1") is None
        path.write_text("arrived after completion")
        assert journal.load_success("run-1") is None


def test_empty_directories_metadata_and_external_inputs_are_handled(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    external = tmp_path / "input.jsonl"
    external.write_text("external input")
    with RunJournal(root, _signature()) as journal:
        run_dir = root / "runs" / "run-1"
        run_dir.mkdir()
        result = {
            "run_id": "run-1",
            "status": "ok",
            "artifacts": {
                "directory": str(run_dir),
                "source": str(external),
                "url": "https://example.invalid/report.json",
                "metadata": {
                    "contract": "example.contract.v1",
                    "sample_count": 0,
                    "sha256": "a" * 64,
                },
            },
        }
        journal.record("run-1", result)
        checkpoint = json.loads((root / "checkpoints" / "run-1.json").read_text())
        assert [entry["path"] for entry in checkpoint["artifact_manifest"]] == [
            "runs/run-1"
        ]
        external.write_text("changed external input")
        assert journal.load_success("run-1") == result


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        {},
        [{"path": "../outside", "kind": "file", "sha256": "a" * 64}],
        [{"path": "/absolute", "kind": "file", "sha256": "a" * 64}],
        [{"path": "runs/run-1/report.json", "kind": "file", "sha256": "invalid"}],
        [],
    ],
)
def test_malformed_or_missing_artifact_manifest_is_rejected(tmp_path, manifest):
    with RunJournal(tmp_path, _signature()) as journal:
        report = tmp_path / "runs" / "run-1" / "report.json"
        report.parent.mkdir()
        report.write_text("output")
        journal.record(
            "run-1",
            {
                "run_id": "run-1",
                "status": "ok",
                "artifacts": {"trace_report": str(report)},
            },
        )
        checkpoint = tmp_path / "checkpoints" / "run-1.json"
        value = json.loads(checkpoint.read_text())
        value["artifact_manifest"] = manifest
        checkpoint.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="artifact manifest"):
            journal.load_success("run-1")


def test_replacing_artifact_with_external_symlink_prevents_reuse(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    external = tmp_path / "protected.txt"
    external.write_text("output")
    with RunJournal(root, _signature()) as journal:
        report = root / "runs" / "run-1" / "report.json"
        report.parent.mkdir()
        report.write_text("output")
        journal.record(
            "run-1",
            {
                "run_id": "run-1",
                "status": "ok",
                "artifacts": {"trace_report": str(report)},
            },
        )
        report.unlink()
        report.symlink_to(external)
        assert journal.load_success("run-1") is None
    assert external.read_text() == "output"


@pytest.mark.parametrize("with_artifacts", [False, True])
def test_relocated_session_cannot_return_original_artifact_links(
    tmp_path, with_artifacts
):
    original_root = tmp_path / "original-session"
    original_root.mkdir()
    with RunJournal(original_root, _signature()) as journal:
        result = {"run_id": "run-1", "status": "ok"}
        if with_artifacts:
            report = original_root / "runs" / "run-1" / "report.json"
            report.parent.mkdir()
            report.write_text("completed output")
            result["artifacts"] = {"trace_report": str(report)}
        journal.record("run-1", result)
    relocated_root = tmp_path / "relocated-session"
    shutil.copytree(original_root, relocated_root)
    with pytest.raises(ValueError, match="relocated sessions cannot be resumed"):
        with RunJournal(relocated_root, _signature(), resume=True):
            pytest.fail("relocated session must be rejected before loading results")
    with RunJournal(original_root, _signature(), resume=True) as journal:
        assert journal.load_success("run-1") == result
