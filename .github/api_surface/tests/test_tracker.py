# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contract tests for the shared API-surface tracker."""

from __future__ import annotations

import json
from pathlib import Path

from api_surface import cli as api_surface_cli
from api_surface.annotations import apply_annotations
from api_surface.extractors.http import extract as extract_http_config
from api_surface.extractors.metrics import _scan as scan_metrics
from api_surface.extractors.python_pyi import extract as extract_python_pyi
from api_surface.ledger import merge_changes, save_ledger
from api_surface.models import (
    LedgerEntry,
    SurfaceLedger,
    SurfaceSnapshot,
    SurfaceSymbol,
)
from api_surface.pr_analysis import analyze_impact
from api_surface.results import OperationResult, OpError
from api_surface.snapshot import (
    build_snapshot,
    default_ledger_path,
    default_provenance_path,
    default_snapshot_path,
    save_snapshot,
)
from api_surface.validate import validate_ledger


def _symbol(
    symbol_id: str,
    *,
    signature: str = "",
    stability: str = "stable",
    deprecated: bool = False,
) -> SurfaceSymbol:
    return SurfaceSymbol(
        surface=symbol_id.split(":", 1)[0],
        kind="function",
        id=symbol_id,
        signature=signature,
        stability=stability,
        deprecated=deprecated,
    )


def test_default_artifacts_live_under_the_project_api_surface_directory() -> None:
    root = Path("project")
    assert default_snapshot_path("1.4.0", root) == (
        root / ".github/api-surface/snapshots/1.4.0.json"
    )
    assert default_ledger_path(root) == root / ".github/api-surface/ledger.json"
    assert default_provenance_path(root) == (
        root / ".github/api-surface/crate-provenance.yaml"
    )


def test_snapshot_extracts_openapi_and_helm_surfaces(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    chart = repo / "deploy/helm/charts/platform"
    chart.mkdir(parents=True)
    chart.joinpath("values.yaml").write_text(
        "frontend:\n  replicas: 2\n", encoding="utf-8"
    )
    repo.joinpath("openapi.json").write_text(
        json.dumps(
            {
                "openapi": "3.1.0",
                "paths": {
                    "/v1/models": {
                        "get": {
                            "responses": {"200": {"description": "ok"}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = build_snapshot(repo, "1.4.0", ref="head")
    snapshot = SurfaceSnapshot.from_dict(result.data["snapshot"])
    ids = {symbol.id for symbol in snapshot.symbols}

    assert "http:GET /v1/models" in ids
    assert "helm:platform:frontend.replicas" in ids
    assert snapshot.coverage["http"] is True
    assert snapshot.coverage["helm"] is True


def test_openapi_signature_tracks_refs_and_only_success_responses(
    tmp_path: Path,
) -> None:
    tmp_path.joinpath("openapi.json").write_text(
        json.dumps(
            {
                "paths": {
                    "/v1/models": {
                        "post": {
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/Request"}
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "content": {
                                        "application/json": {
                                            "schema": {"$ref": "#/components/Response"}
                                        }
                                    }
                                },
                                "400": {
                                    "content": {
                                        "application/json": {
                                            "schema": {"$ref": "#/components/Error"}
                                        }
                                    }
                                },
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = extract_http_config(tmp_path, "1.4.0")
    endpoint = next(
        item for item in result.data["symbols"] if item["id"] == "http:POST /v1/models"
    )

    assert "#/components/Request" in endpoint["signature"]
    assert "#/components/Response" in endpoint["signature"]
    assert "#/components/Error" not in endpoint["signature"]
    assert result.metadata["coverage_detail"]["http:schema"] is True


def test_project_annotations_override_tier_and_record_removal_target(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("openapi.json").write_text(
        json.dumps(
            {
                "openapi": "3.1.0",
                "paths": {
                    "/v1/models": {
                        "get": {
                            "responses": {"200": {"description": "ok"}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    annotations = repo / ".github/api-surface"
    annotations.mkdir(parents=True)
    annotations.joinpath("annotations.yaml").write_text(
        "symbols:\n"
        '  "http:GET /v1/models":\n'
        "    stability: preview\n"
        "    deprecated: true\n"
        '    note: "Use POST /v1/responses"\n'
        '    removal_target: "2.0.0"\n',
        encoding="utf-8",
    )

    result = build_snapshot(repo, "1.4.0")
    snapshot = SurfaceSnapshot.from_dict(result.data["snapshot"])
    endpoint = next(
        symbol for symbol in snapshot.symbols if symbol.id == "http:GET /v1/models"
    )
    ledger = merge_changes(SurfaceLedger(), [], "1.4.0", snapshot=snapshot)
    entry = next(item for item in ledger.entries if item.id == endpoint.id)

    assert endpoint.stability == "preview"
    assert endpoint.deprecated is True
    assert endpoint.deprecated_note == "Use POST /v1/responses"
    assert entry.removal_target == "2.0.0"


def test_native_http_and_crd_deprecation_markers_are_captured(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    crd_dir = repo / "deploy/operator/config/crd/bases"
    crd_dir.mkdir(parents=True)
    crd_dir.joinpath("widgets.yaml").write_text(
        "apiVersion: apiextensions.k8s.io/v1\n"
        "kind: CustomResourceDefinition\n"
        "spec:\n"
        "  names:\n"
        "    kind: Widget\n"
        "  versions:\n"
        "    - name: v1\n"
        "      served: true\n"
        "      storage: true\n"
        "      deprecated: true\n"
        '      deprecationWarning: "Use v2"\n'
        "      schema:\n"
        "        openAPIV3Schema:\n"
        "          type: object\n"
        "          properties:\n"
        "            mode:\n"
        "              type: string\n"
        "              enum: [fast, accurate]\n"
        "    - name: v1internal\n"
        "      served: false\n"
        "      storage: false\n"
        "      schema:\n"
        "        openAPIV3Schema:\n"
        "          type: object\n"
        "          properties:\n"
        "            secret:\n"
        "              type: string\n",
        encoding="utf-8",
    )
    repo.joinpath("openapi.json").write_text(
        json.dumps(
            {
                "openapi": "3.1.0",
                "paths": {
                    "/v1/legacy": {
                        "get": {
                            "deprecated": True,
                            "description": "Use /v1/current",
                            "responses": {"200": {"description": "ok"}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = SurfaceSnapshot.from_dict(build_snapshot(repo, "1.4.0").data["snapshot"])
    symbols = {symbol.id: symbol for symbol in snapshot.symbols}

    assert symbols["http:GET /v1/legacy"].deprecated is True
    assert symbols["crd:Widget@v1"].deprecated is True
    assert symbols["crd:Widget@v1"].deprecated_note == "Use v2"
    assert '"enum":["fast","accurate"]' in symbols["crd:Widget@v1.mode"].signature
    assert "crd:Widget@v1internal" not in symbols
    assert "crd:Widget@v1internal.secret" not in symbols


def test_hidden_cli_flags_are_not_part_of_the_config_surface(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    source.joinpath("args.rs").write_text(
        "#[derive(Parser)]\n"
        "struct Args {\n"
        "    #[arg(long)]\n"
        "    public_flag: bool,\n"
        "    #[arg(long, hide = true)]\n"
        "    internal_flag: bool,\n"
        "    #[command(flatten)]\n"
        "    common: CommonArgs,\n"
        "}\n",
        encoding="utf-8",
    )
    shared = tmp_path / "lib"
    shared.mkdir()
    shared.joinpath("common.rs").write_text(
        "#[derive(clap::Args)]\n"
        "struct Args {\n"
        "    #[arg(long)]\n"
        "    public_flag: String,\n"
        "}\n",
        encoding="utf-8",
    )

    result = extract_http_config(tmp_path, "1.4.0")
    symbols = {item["id"]: item for item in result.data["symbols"]}

    assert "config:src/args::Args.public_flag" in symbols
    assert "config:lib/common::Args.public_flag" in symbols
    assert "config:src/args::Args.internal_flag" not in symbols
    assert "config:src/args::Args.common" not in symbols
    assert (
        symbols["config:src/args::Args.public_flag"]["metadata"]["source_file"]
        == "src/args.rs"
    )


def test_python_accessors_and_overloads_have_unique_ids(tmp_path: Path) -> None:
    stub = tmp_path / "api.pyi"
    stub.write_text(
        "class Client:\n"
        "    @property\n"
        "    def value(self) -> int: ...\n"
        "    @value.setter\n"
        "    def value(self, value: int) -> None: ...\n"
        "    @overload\n"
        "    def run(self, value: int) -> int: ...\n"
        "    @overload\n"
        "    def run(self, value: str) -> str: ...\n",
        encoding="utf-8",
    )

    result = extract_python_pyi(
        tmp_path,
        "1.4.0",
        stubs=[("api.pyi", "dynamo.api")],
    )
    symbols = result.data["symbols"]
    by_id = {item["id"]: item for item in symbols}

    assert len(symbols) == len(by_id)
    assert "python:dynamo.api.Client.value" in by_id
    assert "python:dynamo.api.Client.value@setter" in by_id
    assert (
        "(self, value: int) -> int"
        in by_id["python:dynamo.api.Client.run"]["signature"]
    )
    assert (
        "(self, value: str) -> str"
        in by_id["python:dynamo.api.Client.run"]["signature"]
    )


def test_snapshot_metadata_uses_repo_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    source.joinpath("routes.rs").write_text(
        'const ENV: &str = "DYN_PUBLIC_FLAG";\n'
        'let route = RouteDoc::new(Method::GET, "/v1/models");\n',
        encoding="utf-8",
    )

    result = extract_http_config(tmp_path, "1.4.0")
    serialized = json.dumps(result.data["symbols"])

    assert str(tmp_path) not in serialized
    assert "src/routes.rs" in serialized


def test_metric_registry_includes_label_contracts() -> None:
    symbols = {
        symbol.id: symbol
        for symbol in scan_metrics(
            "pub mod labels {\n"
            '    pub const MODEL: &str = "model";\n'
            "}\n"
            "pub mod frontend {\n"
            '    pub const REQUESTS: &str = "requests_total";\n'
            '    pub const STATUS_LABEL: &str = "status";\n'
            "}\n"
        )
    }

    assert symbols["metric:labels::MODEL"].kind == "label"
    assert symbols["metric:frontend::STATUS_LABEL"].kind == "label"
    assert symbols["metric:frontend::REQUESTS"].kind == "metric"


def test_annotation_boolean_must_be_a_real_boolean(tmp_path: Path) -> None:
    symbol = _symbol("python:dynamo.Client.run")
    path = tmp_path / "annotations.yaml"
    path.write_text(
        'symbols:\n  "python:dynamo.Client.run":\n    deprecated: "false"\n',
        encoding="utf-8",
    )

    result = apply_annotations([symbol], path)

    assert result.errors
    assert symbol.deprecated is False


def test_pr_analysis_fails_closed_for_stable_signature_changes() -> None:
    base = SurfaceSnapshot(
        release="1.4.0",
        ref="base",
        coverage={"python": True},
        symbols=[_symbol("python:dynamo.Client.run", signature="(self, x: int)")],
    )
    head = SurfaceSnapshot(
        release="1.5.0",
        ref="head",
        coverage={"python": True},
        symbols=[_symbol("python:dynamo.Client.run", signature="(self, x: str)")],
    )

    result = analyze_impact(base, head)

    assert result.metadata["has_breaking"] is True
    assert result.data["counts"]["breaking"] == 1
    assert result.data["breaking"][0]["change_type"] == "signature_changed"


def test_pr_analysis_treats_experimental_removal_as_informational() -> None:
    base = SurfaceSnapshot(
        release="1.4.0",
        ref="base",
        coverage={"python": True},
        symbols=[
            _symbol(
                "python:dynamo.experimental.helper",
                stability="experimental",
            )
        ],
    )
    head = SurfaceSnapshot(
        release="1.5.0",
        ref="head",
        coverage={"python": True},
    )

    result = analyze_impact(base, head)

    assert result.metadata["has_breaking"] is False
    assert result.data["counts"]["informational"] == 1


def test_uncovered_surface_does_not_fabricate_a_removal() -> None:
    base = SurfaceSnapshot(
        release="1.4.0",
        ref="base",
        coverage={"python": True},
        symbols=[_symbol("python:dynamo.Client.run")],
    )
    head = SurfaceSnapshot(
        release="1.5.0",
        ref="head",
        coverage={"python": False},
    )

    result = analyze_impact(base, head)

    assert result.data["counts"]["total"] == 0
    assert result.data["coverage_gaps"] == ["python"]


def test_validator_rejects_removal_without_prior_deprecation() -> None:
    ledger = SurfaceLedger(
        entries=[
            LedgerEntry(
                id="python:dynamo.Client.run",
                surface="python",
                status="removed",
                stability="stable",
                removed_in="2.0.0",
            )
        ]
    )

    result = validate_ledger(ledger)

    assert [item["kind"] for item in result.data["violations"]] == [
        "removed_without_deprecation"
    ]


def test_validator_rejects_same_release_deprecation_and_removal() -> None:
    ledger = SurfaceLedger(
        entries=[
            LedgerEntry(
                id="python:dynamo.Client.run",
                surface="python",
                status="removed",
                stability="stable",
                deprecated_in="2.0.0",
                removed_in="2.0.0",
            )
        ]
    )

    result = validate_ledger(ledger)

    assert [item["kind"] for item in result.data["violations"]] == [
        "minimum_window_not_met"
    ]


def test_validator_rejects_early_contract_surface_target() -> None:
    ledger = SurfaceLedger(
        entries=[
            LedgerEntry(
                id="http:POST /v1/chat/completions",
                surface="http",
                status="deprecated",
                stability="stable",
                deprecated_in="1.4.0",
                removal_target="1.5.0",
                note="Use POST /v1/responses",
            )
        ]
    )

    result = validate_ledger(ledger)

    assert [item["kind"] for item in result.data["violations"]] == [
        "minimum_window_not_met"
    ]


def test_validator_requires_a_removal_target_for_stable_deprecation() -> None:
    ledger = SurfaceLedger(
        entries=[
            LedgerEntry(
                id="python:dynamo.Client.run",
                surface="python",
                status="deprecated",
                stability="stable",
                deprecated_in="1.4.0",
            )
        ]
    )

    result = validate_ledger(ledger)

    assert [item["kind"] for item in result.data["violations"]] == [
        "missing_removal_target"
    ]


def test_validator_requires_migration_guidance_for_stable_deprecation() -> None:
    ledger = SurfaceLedger(
        entries=[
            LedgerEntry(
                id="python:dynamo.Client.run",
                surface="python",
                status="deprecated",
                stability="stable",
                deprecated_in="1.4.0",
                removal_target="2.0.0",
            )
        ]
    )

    result = validate_ledger(ledger)

    assert [item["kind"] for item in result.data["violations"]] == [
        "missing_migration_guidance"
    ]


def test_validator_allows_stable_removal_in_a_later_minor_after_deprecation() -> None:
    ledger = SurfaceLedger(
        entries=[
            LedgerEntry(
                id="python:dynamo.Client.run",
                surface="python",
                status="removed",
                stability="stable",
                deprecated_in="1.4.0",
                removal_target="1.5.0",
                removed_in="1.5.0",
            )
        ]
    )

    result = validate_ledger(ledger)

    assert result.data["violations"] == []


def test_cli_analyze_pr_returns_nonzero_for_a_stable_break(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    head_path = tmp_path / "head.json"
    save_snapshot(
        SurfaceSnapshot(
            release="1.4.0",
            ref="base",
            coverage={"python": True},
            symbols=[_symbol("python:dynamo.Client.run")],
        ),
        base_path,
    )
    save_snapshot(
        SurfaceSnapshot(
            release="1.5.0",
            ref="head",
            coverage={"python": True},
        ),
        head_path,
    )

    assert (
        api_surface_cli.main(
            [
                "analyze-pr",
                "--base-snapshot",
                str(base_path),
                "--head-snapshot",
                str(head_path),
                "--fail-on-breaking",
            ]
        )
        == 1
    )


def test_cli_analyze_pr_fails_closed_on_extraction_errors(tmp_path: Path) -> None:
    base_repo = tmp_path / "base"
    head_repo = tmp_path / "head"
    base_repo.mkdir()
    annotations = head_repo / ".github/api-surface"
    annotations.mkdir(parents=True)
    annotations.joinpath("annotations.yaml").write_text(
        'symbols:\n  "http:GET /unknown":\n    stability: preview\n',
        encoding="utf-8",
    )

    exit_code = api_surface_cli.main(
        [
            "analyze-pr",
            "--base-repo",
            str(base_repo),
            "--head-repo",
            str(head_repo),
            "--fail-on-breaking",
        ]
    )

    assert exit_code == 1


def test_cli_update_ledger_does_not_write_after_extraction_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    ledger_path = tmp_path / "ledger.json"
    save_snapshot(
        SurfaceSnapshot(release="1.4.0", ref="head", coverage={}),
        snapshot_path,
    )
    save_ledger(
        SurfaceLedger(
            entries=[LedgerEntry(id="python:dynamo.Client.run", surface="python")]
        ),
        ledger_path,
    )
    before = ledger_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        api_surface_cli,
        "gather_changes",
        lambda *args, **kwargs: OperationResult(
            data={"changes": [], "coverage_gaps": []},
            errors=[OpError(operation="extract", message="failed")],
        ),
    )

    exit_code = api_surface_cli.main(
        [
            "update-ledger",
            "--new",
            str(snapshot_path),
            "--release",
            "1.4.0",
            "--ledger",
            str(ledger_path),
        ]
    )

    assert exit_code == 1
    assert ledger_path.read_text(encoding="utf-8") == before


def test_cli_render_from_json_matches_analyze_pr_md(tmp_path: Path, capsys) -> None:
    base_path = tmp_path / "base.json"
    head_path = tmp_path / "head.json"
    data_path = tmp_path / "impact.json"
    save_snapshot(
        SurfaceSnapshot(
            release="1.4.0",
            ref="base",
            coverage={"python": True},
            symbols=[
                SurfaceSymbol(
                    surface="python",
                    kind="function",
                    id="python:dynamo.Client.run",
                    stability="stable",
                )
            ],
        ),
        base_path,
    )
    save_snapshot(
        SurfaceSnapshot(release="1.4.0", ref="head", coverage={"python": True}),
        head_path,
    )

    assert (
        api_surface_cli.main(
            [
                "analyze-pr",
                "--base-snapshot",
                str(base_path),
                "--head-snapshot",
                str(head_path),
                "--format",
                "md",
            ]
        )
        == 0
    )
    md_out = capsys.readouterr().out

    assert (
        api_surface_cli.main(
            [
                "analyze-pr",
                "--base-snapshot",
                str(base_path),
                "--head-snapshot",
                str(head_path),
                "--format",
                "json",
                "--output",
                str(data_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        api_surface_cli.main(
            [
                "render",
                "--from-json",
                str(data_path),
                "--format",
                "md",
            ]
        )
        == 0
    )
    rendered = capsys.readouterr().out

    assert rendered == md_out
