# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay :class:`ArgV` over every shipped deployment manifest.

Unit tests prove the constructs this module *intends* to handle. This one proves
the corpus contains nothing else. It is the test that would have caught the
defect it was written for: ``-lc`` containers were 100 of the 184 shell-invoked
containers in ``recipes/`` and ``examples/``, and every helper that scanned them
returned nothing while reporting the flag as absent.

Skipped when the manifests are not present, so the harness stays installable and
testable on its own.
"""

import pathlib
import subprocess

import pytest
from dynamo_test.argv import ArgForm, ArgV

yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOTS = [REPO / "recipes", REPO / "examples"]

pytestmark = pytest.mark.skipif(
    not all(r.is_dir() for r in ROOTS),
    reason="deployment manifests are not present next to the harness",
)


def _containers(path):
    """Every container in every DGD document, across both schema versions."""
    try:
        documents = list(yaml.safe_load_all(path.read_text()))
    except Exception:
        return
    for doc in documents:
        if not isinstance(doc, dict) or doc.get("kind") != "DynamoGraphDeployment":
            continue
        spec = doc.get("spec") or {}
        # v1alpha1
        for name, service in (spec.get("services") or {}).items():
            main = ((service or {}).get("extraPodSpec") or {}).get("mainContainer")
            if main:
                yield name, main
        # v1beta1
        for component in spec.get("components") or []:
            name = (component or {}).get("name", "<unnamed>")
            if component.get("container"):
                yield name, component["container"]
            pod_spec = (component.get("podTemplate") or {}).get("spec") or {}
            for container in pod_spec.get("containers") or []:
                yield name, container
            main = (component.get("extraPodSpec") or {}).get("mainContainer")
            if main:
                yield name, main


def _shell_containers():
    for root in ROOTS:
        for path in sorted(root.rglob("*.yaml")):
            for name, container in _containers(path):
                if not isinstance(container, dict):
                    continue
                argv = ArgV.from_container(
                    container, source=f"{path.relative_to(REPO)}[{name}]"
                )
                if argv.form is ArgForm.SHELL:
                    yield argv


@pytest.fixture(scope="module")
def shell_commands():
    found = list(_shell_containers())
    assert found, "expected the corpus to contain shell-invoked containers"
    return found


def test_every_shipped_shell_command_is_parseable(shell_commands):
    """``shlex`` failed on four of these; a parse failure must not be silent."""
    unparseable = [
        (a.source, a.parse_error) for a in shell_commands if not a.is_parseable
    ]
    assert unparseable == []


def test_the_login_shell_majority_is_detected(shell_commands):
    """``-lc`` outnumbers ``-c``; a predicate that matches only ``-c`` sees half."""
    login = [a for a in shell_commands if a.command[-1] != "-c"]
    assert len(login) > len(shell_commands) / 3, (
        f"only {len(login)} of {len(shell_commands)} use a non-bare-c shell flag; "
        "if the corpus really changed this much, re-measure before relaxing this"
    )


def test_editing_any_shipped_command_preserves_everything_else(shell_commands):
    """Two edits per manifest; every untouched line must be byte-identical.

    This is the property a token round-trip cannot hold: it flattens
    continuations, drops comments, and re-quotes operators into arguments.
    """
    damage = []
    for argv in shell_commands:
        before = argv.as_shell_string()

        edited = argv
        for flag in ("--model-path", "--model", "--served-model-name"):
            if edited.get(flag).is_known:
                edited = edited.set(flag, "sentinel/model")
                break
        edited = edited.set("--dyn-corpus-probe", "42")
        after = edited.as_shell_string()

        if len(after.splitlines()) != len(before.splitlines()):
            damage.append(f"{argv.source}: line count changed")
            continue
        for old, new in zip(before.splitlines(), after.splitlines()):
            if old == new:
                continue
            if "sentinel/model" not in new and "--dyn-corpus-probe" not in new:
                damage.append(f"{argv.source}: unrelated line changed: {old.strip()!r}")

        if len(edited.as_container_args()) != 1:
            damage.append(f"{argv.source}: shell args must stay a single string")

    assert damage == []


def test_comments_survive_an_edit(shell_commands):
    """The comments explain why each flag is set; losing them loses the reason."""
    damage = []
    for argv in shell_commands:
        before = argv.as_shell_string()
        comments = [ln for ln in before.splitlines() if ln.lstrip().startswith("#")]
        if not comments:
            continue
        after = argv.set("--dyn-corpus-probe", "42").as_shell_string()
        if [ln for ln in after.splitlines() if ln.lstrip().startswith("#")] != comments:
            damage.append(argv.source)
    assert damage == []


@pytest.mark.skipif(
    subprocess.run(["which", "bash"], capture_output=True).returncode != 0,
    reason="bash is required to syntax-check the rewritten commands",
)
def test_every_edited_command_still_parses_under_bash(shell_commands):
    """The independent check: bash's own parser, not ours."""
    rejected = []
    for argv in shell_commands:
        after = argv.set("--dyn-corpus-probe", "42").as_shell_string()
        result = subprocess.run(
            ["bash", "-n", "-c", after], capture_output=True, text=True
        )
        if result.returncode != 0:
            rejected.append(f"{argv.source}: {result.stderr.strip()[:120]}")
    assert rejected == []


# ------------------------------------------------------- the dialect, on the corpus


def _worker_argvs():
    """Every container that actually launches an inference engine."""
    from dynamo_test.dialect import detect

    for root in ROOTS:
        for path in sorted(root.rglob("*.yaml")):
            for name, container in _containers(path):
                if not isinstance(container, dict):
                    continue
                argv = ArgV.from_container(
                    container, source=f"{path.relative_to(REPO)}[{name}]"
                )
                backend = detect(argv)
                if backend.is_known:
                    yield backend.require(), argv


@pytest.fixture(scope="module")
def workers():
    found = list(_worker_argvs())
    assert found, "expected the corpus to contain engine workers"
    return found


def test_the_engine_is_detected_for_every_worker(workers):
    """Detection reads ``python -m dynamo.<engine>``, not a substring.

    Matching a backend name anywhere in the command picks up image names and
    environment variables; that mis-attributed 12 TensorRT-LLM containers when
    this was first measured the loose way.
    """
    from dynamo_test.dialect import DIALECTS

    assert {b for b, _ in workers} <= set(DIALECTS)
    assert len(workers) > 100


# Manifests whose worker declares no model at all, and does not defer to a
# config file either. Each is a real defect, not a gap in the reader.
#
# All three are the YAML folded-scalar bug: `>-` collapses the newline in
# `\<newline>` into a space, making it an escaped space, so bash delivers
# " --model" with a leading space and argparse rejects it. Fixed by OPS-8534
# (PR #14321); when that lands these entries should be deleted and this test
# will say so.
KNOWN_UNDECLARED_MODEL = {
    "recipes/qwen3-vl-32b-fp8/vllm/agg/deploy.yaml[VllmWorker]",
    "recipes/qwen3-vl-32b-fp8/vllm/hetero_hardware_disagg/deploy.yaml[EncodeWorker]",
    "recipes/qwen3-vl-32b-fp8/vllm/hetero_hardware_disagg/deploy.yaml[VllmDecodeWorker]",
}


def test_no_worker_reports_its_model_absent_without_reason(workers):
    """Every engine worker either declares a model or defers to a config file.

    ``ABSENT`` means the reader looked at the whole command line and the model
    genuinely is not there — which for a worker is a broken manifest. This is
    the check that found OPS-8534.
    """
    from dynamo_test.dialect import for_backend

    absent = {
        argv.source
        for backend, argv in workers
        if for_backend(backend).read(argv, "model").is_absent
    }
    unexpected = absent - KNOWN_UNDECLARED_MODEL
    assert unexpected == set(), f"workers declaring no model: {sorted(unexpected)}"

    fixed = KNOWN_UNDECLARED_MODEL - absent
    assert fixed == set(), (
        f"these now declare a model: {sorted(fixed)}. Remove them from "
        "KNOWN_UNDECLARED_MODEL."
    )


def test_a_config_file_makes_the_model_unknown_not_absent(workers):
    """Deferring to ``--config`` is not the same as declaring nothing.

    Four SGLang workers put every setting in ``/etc/sglang/*.yaml`` and pass only
    ``--config``. Reporting "no model" for those would be false; the reader
    cannot see the file, so the honest answer is UNKNOWN.
    """
    from dynamo_test.dialect import for_backend

    deferred = [
        argv
        for backend, argv in workers
        if for_backend(backend).read(argv, "model").is_unknown and argv.is_parseable
    ]
    assert deferred, "expected some workers to configure themselves from a file"
    for argv in deferred:
        fact = for_backend("sglang").read(argv, "model")
        assert "defers to" in fact.detail or "could not be tokenised" in fact.detail


def test_both_live_spellings_are_actually_exercised(workers):
    """Guards the multi-spelling tables against being quietly reduced to one.

    SGLang uses ``--tp`` and ``--tensor-parallel-size``; TensorRT-LLM uses
    ``--model-path`` and ``--model``. If the corpus stops using one, this fails
    and the table can be simplified deliberately rather than by accident.
    """
    sglang_tp = {
        f
        for backend, argv in workers
        if backend == "sglang"
        for f in ("--tp", "--tensor-parallel-size")
        if argv.get(f).is_known
    }
    trtllm_model = {
        f
        for backend, argv in workers
        if backend == "trtllm"
        for f in ("--model-path", "--model")
        if argv.get(f).is_known
    }
    assert sglang_tp == {"--tp", "--tensor-parallel-size"}, sglang_tp
    assert trtllm_model == {"--model-path", "--model"}, trtllm_model


def test_setting_a_semantic_round_trips_on_every_worker(workers):
    """Write through the dialect, read it back, on all of them."""
    from dynamo_test.dialect import for_backend

    damage = []
    for backend, argv in workers:
        dialect = for_backend(backend)
        try:
            out = dialect.apply(argv, context_length=4096)
        except Exception as exc:  # AmbiguousInsertion is a legitimate refusal
            damage.append(f"{argv.source}: {type(exc).__name__}: {exc}")
            continue
        got = dialect.read(out, "context_length")
        if not got.is_known or got.require() != "4096":
            damage.append(f"{argv.source}: read back {got.status.value}")
    assert damage == []
