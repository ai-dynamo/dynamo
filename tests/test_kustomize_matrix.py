# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/kustomize-matrix.py"
MODULE_PATH = REPO_ROOT / "scripts/kustomize-matrix.py"

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def load_matrix_module():
    spec = importlib.util.spec_from_file_location("kustomize_matrix", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_kustomization(path: Path, content: str) -> None:
    path.mkdir(parents=True)
    (path / "kustomization.yaml").write_text(content, encoding="utf-8")


def run_matrix(
    *arguments: str, cwd: Path = REPO_ROOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *arguments],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_compose_applies_positional_components_and_forwards_options(
    tmp_path, monkeypatch
):
    target = tmp_path / "target"
    write_kustomization(target, "resources: []\n")

    component = tmp_path / "component"
    write_kustomization(
        component,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\nkind: Component\n",
    )

    output = tmp_path / "manifest.yaml"
    calls = []

    def fake_run(command, **_):
        calls.append(command)
        generated = Path(command[2]) / "kustomization.yaml"
        assert generated.read_text(encoding="utf-8") == (
            "apiVersion: kustomize.config.k8s.io/v1beta1\n"
            "kind: Kustomization\n"
            "sortOptions:\n"
            "  order: fifo\n"
            "resources:\n"
            '  - "../target"\n'
            "components:\n"
            '  - "../component"\n'
        )
        Path(command[command.index("--output") + 1]).write_text(
            "rendered\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    kustomize_matrix = load_matrix_module()
    monkeypatch.setattr(
        kustomize_matrix, "kustomize_command", lambda: ["kustomize", "build"]
    )
    monkeypatch.setattr(kustomize_matrix.subprocess, "run", fake_run)

    assert (
        kustomize_matrix.compose(
            str(target), [str(component)], ["--output", str(output)]
        )
        == 0
    )

    assert calls[0][:2] == ["kustomize", "build"]
    assert calls[0][3:] == ["--output", str(output)]
    assert output.read_text(encoding="utf-8") == "rendered\n"


def test_compose_requires_target_first():
    result = run_matrix("compose", "--enable-helm")

    assert result.returncode == 2
    assert "the following arguments are required: target" in result.stderr


def test_scan_yaml_uses_name_selectors_for_list_comments():
    kustomize_matrix = load_matrix_module()
    document = kustomize_matrix.scan_yaml(
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: app\n"
        "items:\n"
        "  # Applies to UCX only\n"
        "  - name: UCX_NET_DEVICES\n"
        "    value: mlx5_0:1\n"
    )[0]

    path = ("items", "name=UCX_NET_DEVICES")
    assert document.comments[0].path == path
    assert path in document.targets


def test_unfold_expands_matrix_and_check_detects_stale_overlay(tmp_path):
    recipe = tmp_path / "recipe"
    base = recipe / "kustomize/base"
    write_kustomization(base, "resources:\n  - config-map.yaml\n")
    (base / "config-map.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: app\n",
        encoding="utf-8",
    )
    for component_name in ("provider", "telemetry"):
        component = recipe / "components" / component_name
        write_kustomization(
            component,
            "apiVersion: kustomize.config.k8s.io/v1alpha1\nkind: Component\n",
        )

    matrix = recipe / ".kustomize-matrix.yaml"
    matrix.write_text(
        "source: kustomize/base\n"
        'nameTemplate: "${variant}-${observability}"\n'
        "matrix:\n"
        "  variant:\n"
        "    - name: aws\n"
        "      components:\n"
        "        - components/provider\n"
        "  observability:\n"
        "    - name: otel\n"
        "      components:\n"
        "        - components/telemetry\n",
        encoding="utf-8",
    )

    result = run_matrix("unfold", str(matrix))

    assert result.returncode == 0, result.stderr
    overlay = recipe / "kustomize/overlays/aws-otel/kustomization.yaml"
    assert overlay.read_text(encoding="utf-8") == (
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        "# Generated file. Do not edit this checked-in copy.\n"
        f"# Regenerate: scripts/kustomize-matrix.py unfold {matrix}\n\n"
        "apiVersion: kustomize.config.k8s.io/v1beta1\n"
        "kind: Kustomization\n"
        "sortOptions:\n"
        "  order: fifo\n"
        "resources:\n"
        '  - "../../base"\n'
        "components:\n"
        '  - "../../../components/provider"\n'
        '  - "../../../components/telemetry"\n'
    )
    assert run_matrix("unfold", "--check", str(matrix)).returncode == 0

    overlay.write_text("stale\n", encoding="utf-8")
    result = run_matrix("unfold", "--check", str(matrix))

    assert result.returncode == 1
    assert "Generated Kustomize overlays are stale" in result.stderr


def test_render_uses_leaf_component_and_preserves_source_comments(
    tmp_path, monkeypatch
):
    recipe = tmp_path / "recipe"
    base = recipe / "kustomize/base"
    write_kustomization(base, "resources:\n  - config-map.yaml\n")
    (base / "config-map.yaml").write_text(
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: app\n"
        "data:\n"
        "  # Base comment\n"
        "  source: base\n",
        encoding="utf-8",
    )
    parent = recipe / "components/parent"
    write_kustomization(
        parent,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\n"
        "kind: Component\n"
        "patches:\n"
        "  - target:\n"
        "      version: v1\n"
        "      kind: ConfigMap\n"
        "    path: patch.yaml\n",
    )
    (parent / "patch.yaml").write_text(
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: component\n"
        "data:\n"
        "  # Parent comment\n"
        "  parent: value\n",
        encoding="utf-8",
    )
    leaf = recipe / "components/leaf"
    write_kustomization(
        leaf,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\n"
        "kind: Component\n"
        "components:\n"
        "  - ../parent\n"
        "patches:\n"
        "  - path: patch.yaml\n",
    )
    (leaf / "patch.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: app\ndata:\n  leaf: value\n",
        encoding="utf-8",
    )
    matrix = recipe / ".kustomize-matrix.yaml"
    matrix.write_text(
        "source: kustomize/base\n"
        'nameTemplate: "${variant}"\n'
        "matrix:\n"
        "  variant:\n"
        "    - name: aws-efa-p8d16\n"
        "      components:\n"
        "        - components/leaf\n",
        encoding="utf-8",
    )

    def fake_kustomize_build(command, **_):
        assert command[:2] == ["kustomize", "build"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "data:\n"
                "  source: base\n"
                "  parent: value\n"
                "  leaf: value\n"
                "metadata:\n"
                "  name: app\n"
                "kind: ConfigMap\n"
                "apiVersion: v1\n"
            ),
            stderr="",
        )

    kustomize_matrix = load_matrix_module()
    monkeypatch.setattr(
        kustomize_matrix, "generate_kustomize_openapi", lambda *, check: None
    )
    monkeypatch.setattr(
        kustomize_matrix, "kustomize_command", lambda: ["kustomize", "build"]
    )
    monkeypatch.setattr(kustomize_matrix.subprocess, "run", fake_kustomize_build)
    matrix_path = matrix

    def unfold(*, check=False, clean=False):
        return kustomize_matrix.unfold_matrix(
            kustomize_matrix.load_matrix(str(matrix_path)), check=check, clean=clean
        )

    def render(*, check=False, clean=False):
        return kustomize_matrix.render_matrix(
            kustomize_matrix.load_matrix(str(matrix_path)), check=check, clean=clean
        )

    unfold()
    render()

    rendered = (recipe / "deploy-aws-efa-p8d16.yaml").read_text(encoding="utf-8")
    assert (
        "# Generated file. For repository contributors, do not edit this checked-in copy.\n"
        "# Regenerate every public overlay and rendered manifest of this matrix (from the repository root):\n"
        f"#   scripts/kustomize-matrix.py unfold {matrix}\n"
        f"#   scripts/kustomize-matrix.py render {matrix}\n"
        "# Inspect only this Kustomize overlay (from the repository root):\n"
        f"#   kustomize build {recipe / 'kustomize/overlays/aws-efa-p8d16'}\n"
        "# You may edit a copy before applying it.\n" in rendered
    )
    assert "# Base comment\n  source: base" in rendered
    assert "# Parent comment\n  parent: value" in rendered
    assert "  leaf: value" in rendered
    assert "  parent: value" in rendered

    matrix.write_text(
        matrix.read_text(encoding="utf-8").replace("aws-efa-p8d16", "renamed"),
        encoding="utf-8",
    )
    unfold()
    render()
    assert not (recipe / "deploy-aws-efa-p8d16.yaml").exists()
    assert (recipe / "deploy-renamed.yaml").exists()
    assert render(check=True) == []

    relocated_matrix = recipe / "relocated-matrix.yaml"
    matrix.rename(relocated_matrix)
    relocated_matrix.write_text(
        relocated_matrix.read_text(encoding="utf-8").replace("renamed", "current"),
        encoding="utf-8",
    )
    matrix_path = relocated_matrix

    unfold()
    stale_overlays = unfold(check=True)
    assert recipe / "kustomize/overlays/renamed/kustomization.yaml" in stale_overlays
    assert (recipe / "kustomize/overlays/renamed").exists()

    unfold(clean=True)
    assert not (recipe / "kustomize/overlays/renamed").exists()

    stale_manifests = render(check=True)
    assert recipe / "deploy-renamed.yaml" in stale_manifests
    render(clean=True)
    assert not (recipe / "deploy-renamed.yaml").exists()
    assert (recipe / "deploy-current.yaml").exists()

    manual_manifest = recipe / "deploy-manual.yaml"
    manual_manifest.write_text("apiVersion: v1\nkind: ConfigMap\n", encoding="utf-8")
    render(clean=True)
    assert manual_manifest.exists()


def test_help():
    result = run_matrix("--help")

    assert result.returncode == 0
    assert "{unfold,render,check,compose}" in result.stdout
