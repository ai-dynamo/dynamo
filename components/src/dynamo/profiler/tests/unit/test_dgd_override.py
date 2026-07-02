# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the profiler's DGD override CLI adapter."""

from __future__ import annotations

import copy
import logging
import textwrap
from pathlib import Path

import pytest
import yaml

from dynamo.profiler.utils.dgd_override import (
    DGD_OVERRIDE_BINARY_ENV,
    DGDOverrideError,
    _verify_protocol,
    apply_dgd_overrides,
    resolve_dgd_override_binary,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
    pytest.mark.timeout(45),
]


@pytest.fixture(autouse=True)
def clear_protocol_cache():
    _verify_protocol.cache_clear()
    yield
    _verify_protocol.cache_clear()


@pytest.fixture
def blueprint() -> dict:
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "generated"},
        "spec": {"components": []},
    }


def _write_fake_cli(tmp_path: Path) -> Path:
    path = tmp_path / "dgd-apply-overrides"
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys

            if "--protocol-version" in sys.argv:
                print(os.environ.get("FAKE_PROTOCOL_VERSION", "1"))
                raise SystemExit(0)
            if os.environ.get("FAKE_MERGE_FAILURE"):
                print("synthetic merge failure", file=sys.stderr)
                raise SystemExit(2)

            def value(flag):
                return sys.argv[sys.argv.index(flag) + 1]

            with open(value("--blueprint"), encoding="utf-8") as source:
                result = json.load(source)
            with open(value("--override"), encoding="utf-8") as source:
                override = json.load(source)
            labels = result.setdefault("metadata", {}).setdefault("labels", {})
            labels["merged"] = "true"
            labels["override-api-version"] = override["apiVersion"]
            with open(value("--output"), "w", encoding="utf-8") as destination:
                json.dump(result, destination)
            print("warning: synthetic warning", file=sys.stderr)
            print("synthetic diagnostic", file=sys.stderr)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_default_profiler_dgd_templates_have_type_meta() -> None:
    repository_root = Path(__file__).resolve().parents[6]
    template_paths = [
        repository_root / "examples" / "backends" / backend / "deploy" / filename
        for backend in ("vllm", "sglang", "trtllm")
        for filename in ("agg.yaml", "disagg.yaml")
    ]
    template_paths.append(
        repository_root / "examples" / "backends" / "mocker" / "deploy" / "disagg.yaml"
    )

    for template_path in template_paths:
        dgd = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        assert dgd["apiVersion"] in {
            "nvidia.com/v1alpha1",
            "nvidia.com/v1beta1",
        }, template_path
        assert dgd["kind"] == "DynamoGraphDeployment", template_path


def test_apply_dgd_overrides_invokes_cli_without_mutating_inputs(
    tmp_path: Path,
    blueprint: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    binary = _write_fake_cli(tmp_path)
    overrides = {"spec": {"services": {}}}
    original_blueprint = copy.deepcopy(blueprint)
    original_overrides = copy.deepcopy(overrides)

    with caplog.at_level(logging.INFO):
        effective = apply_dgd_overrides(
            blueprint,
            overrides,
            binary_path=str(binary),
        )

    assert effective["metadata"]["labels"]["merged"] == "true"
    assert (
        effective["metadata"]["labels"]["override-api-version"]
        == "nvidia.com/v1alpha1"
    )
    assert blueprint == original_blueprint
    assert overrides == original_overrides
    assert "treating it as nvidia.com/v1alpha1" in caplog.text
    assert "DGD override: synthetic warning" in caplog.text
    diagnostic_record = next(
        record for record in caplog.records if "synthetic diagnostic" in record.message
    )
    assert diagnostic_record.levelno == logging.INFO


def test_explicit_versioned_override_does_not_use_legacy_fallback(
    tmp_path: Path,
    blueprint: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    binary = _write_fake_cli(tmp_path)
    overrides = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "spec": {"components": []},
    }

    with caplog.at_level(logging.WARNING):
        effective = apply_dgd_overrides(
            blueprint,
            overrides,
            binary_path=str(binary),
        )

    assert "missing apiVersion and kind" not in caplog.text
    assert (
        effective["metadata"]["labels"]["override-api-version"]
        == "nvidia.com/v1beta1"
    )


def test_resolve_dgd_override_binary_prefers_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = _write_fake_cli(tmp_path)
    monkeypatch.setenv(DGD_OVERRIDE_BINARY_ENV, str(binary))

    assert resolve_dgd_override_binary() == str(binary.resolve())


def test_resolve_dgd_override_binary_rejects_invalid_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv(DGD_OVERRIDE_BINARY_ENV, str(missing))

    with pytest.raises(DGDOverrideError, match="does not point to a file"):
        resolve_dgd_override_binary()


def test_apply_dgd_overrides_rejects_partial_type_metadata(
    tmp_path: Path,
    blueprint: dict,
) -> None:
    binary = _write_fake_cli(tmp_path)

    with pytest.raises(DGDOverrideError, match="both apiVersion and kind"):
        apply_dgd_overrides(
            blueprint,
            {"apiVersion": "nvidia.com/v1beta1", "spec": {}},
            binary_path=str(binary),
        )


def test_apply_dgd_overrides_rejects_incompatible_protocol(
    tmp_path: Path,
    blueprint: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = _write_fake_cli(tmp_path)
    monkeypatch.setenv("FAKE_PROTOCOL_VERSION", "2")

    with pytest.raises(DGDOverrideError, match="expected '1', got '2'"):
        apply_dgd_overrides(
            blueprint,
            {"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoGraphDeployment"},
            binary_path=str(binary),
        )


def test_apply_dgd_overrides_surfaces_cli_failure(
    tmp_path: Path,
    blueprint: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = _write_fake_cli(tmp_path)
    monkeypatch.setenv("FAKE_MERGE_FAILURE", "1")

    with pytest.raises(DGDOverrideError, match="synthetic merge failure"):
        apply_dgd_overrides(
            blueprint,
            {"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoGraphDeployment"},
            binary_path=str(binary),
        )


def test_apply_dgd_overrides_requires_versioned_blueprint(tmp_path: Path) -> None:
    binary = _write_fake_cli(tmp_path)

    with pytest.raises(DGDOverrideError, match="must specify apiVersion and kind"):
        apply_dgd_overrides(
            {"spec": {}},
            {"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoGraphDeployment"},
            binary_path=str(binary),
        )
