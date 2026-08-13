# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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


def write_template(path: Path, content: str, values: str = "") -> None:
    path.mkdir(parents=True)
    (path / "kustomization.yaml.j2").write_text(content, encoding="utf-8")
    if values:
        (path / "values.yaml").write_text(values, encoding="utf-8")


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


def mock_kustomize_base_build(kustomize_matrix, monkeypatch, source: Path, output: str):
    def fake_kustomize_build(command, **_):
        assert command == ["kustomize", "build", str(source)]
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(
        kustomize_matrix, "kustomize_command", lambda: ["kustomize", "build"]
    )
    monkeypatch.setattr(kustomize_matrix.subprocess, "run", fake_kustomize_build)


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
        "# Generated file. For repository contributors, do not edit this checked-in copy.\n"
        "# Regenerate this matrix's public overlays and template Components from the repository root:\n"
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


def test_unfold_materializes_template_component_with_base_and_variant_values(
    tmp_path, monkeypatch
):
    recipe = tmp_path / "recipe"
    base = recipe / "kustomize/base"
    write_kustomization(base, "resources:\n  - resources.yaml\n")
    (base / "resources.yaml").write_text(
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: app-config\n"
        "data:\n"
        "  setting: from-base\n"
        "---\n"
        "apiVersion: nvidia.com/v1alpha1\n"
        "kind: DynamoGraphDeployment\n"
        "metadata:\n"
        "  name: app\n"
        "spec:\n"
        "  replicas: 2\n",
        encoding="utf-8",
    )
    template = recipe / "templates/provider/instance"
    write_template(
        template,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\n"
        "kind: Component\n"
        "patches:\n"
        "  # This comment is copied into the generated Component.\n"
        "  - target:\n"
        "      group: nvidia.com\n"
        "      version: v1alpha1\n"
        "      kind: DynamoGraphDeployment\n"
        "    patch: |\n"
        "      {% set dgd = base.dynamographdeployment | only %}\n"
        "      apiVersion: nvidia.com/v1alpha1\n"
        "      kind: DynamoGraphDeployment\n"
        "      metadata:\n"
        "        name: {{ dgd.metadata.name }}\n"
        "        labels:\n"
        "          setting: {{ base.configmap[values.CONFIG_MAP].data.setting }}\n"
        '          replicas: "{{ dgd.spec.replicas * values.MULTIPLIER }}"\n'
        "  - target:\n"
        "      group: nvidia.com\n"
        "      version: v1alpha1\n"
        "      kind: DynamoGraphDeployment\n"
        "    path: patch.yaml\n",
        "CONFIG_MAP: app-config\nMULTIPLIER: 2\n",
    )
    (template / "patch.yaml.j2").write_text(
        "apiVersion: nvidia.com/v1alpha1\n"
        "kind: DynamoGraphDeployment\n"
        "metadata:\n"
        "  name: app\n"
        "  labels:\n"
        "    static-patch: {{ values.STATIC_PATCH }}\n",
        encoding="utf-8",
    )
    (template / "assets/runtime.conf").parent.mkdir()
    (template / "assets/runtime.conf").write_text("plain asset\n", encoding="utf-8")
    (template / "assets/metadata.txt.j2").write_text(
        "model={{ values.MODEL_NAME }}\n", encoding="utf-8"
    )
    matrix = recipe / ".kustomize-matrix.yaml"
    matrix.write_text(
        "source: kustomize/base\n"
        'nameTemplate: "${variant}"\n'
        "matrix:\n"
        "  variant:\n"
        "    - name: instance\n"
        "      templates:\n"
        "        - source: templates/provider/instance\n"
        "          path: components/provider\n"
        "      values:\n"
        "        MULTIPLIER: 3\n"
        "        STATIC_PATCH: from-variant\n"
        "        MODEL_NAME: qwen\n",
        encoding="utf-8",
    )

    kustomize_matrix = load_matrix_module()
    mock_kustomize_base_build(
        kustomize_matrix,
        monkeypatch,
        base,
        (base / "resources.yaml").read_text(encoding="utf-8"),
    )
    config = kustomize_matrix.load_matrix(str(matrix))

    kustomize_matrix.unfold_matrix(config, check=False)
    overlay = recipe / "kustomize/overlays/instance/kustomization.yaml"
    component = (
        recipe / "kustomize/overlays/instance/components/provider/kustomization.yaml"
    )
    assert '  - "components/provider"\n' in overlay.read_text(encoding="utf-8")
    rendered_component = component.read_text(encoding="utf-8")
    assert "# Template source: " in rendered_component
    assert rendered_component.count("# SPDX-License-Identifier") == 1
    assert (
        "# This comment is copied into the generated Component." in rendered_component
    )
    parsed_component = yaml.safe_load(rendered_component)
    assert "setting: from-base" in parsed_component["patches"][0]["patch"]
    assert 'replicas: "6"' in parsed_component["patches"][0]["patch"]
    assert parsed_component["patches"][1]["path"] == "patch.yaml"
    assert (component.parent / "patch.yaml").read_text(encoding="utf-8") == (
        "apiVersion: nvidia.com/v1alpha1\n"
        "kind: DynamoGraphDeployment\n"
        "metadata:\n"
        "  name: app\n"
        "  labels:\n"
        "    static-patch: from-variant\n"
    )
    assert (component.parent / "assets/runtime.conf").read_text(
        encoding="utf-8"
    ) == "plain asset\n"
    assert (component.parent / "assets/metadata.txt").read_text(
        encoding="utf-8"
    ) == "model=qwen\n"
    assert kustomize_matrix.unfold_matrix(config, check=True) == []


def test_template_only_and_undefined_values_fail_clearly(tmp_path):
    kustomize_matrix = load_matrix_module()
    resources = kustomize_matrix.ResourceCollection("ConfigMap")
    resources["first"] = {}
    resources["second"] = {}

    with pytest.raises(ValueError, match="exactly one ConfigMap resource"):
        kustomize_matrix.only(resources)

    template = tmp_path / "template"
    write_template(
        template,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\n"
        "kind: Component\n"
        "patches:\n"
        "  - patch: |\n"
        "      value: {{ values.NOT_DEFINED }}\n",
    )

    selection = kustomize_matrix.TemplateSelection(
        source=template, output_path=Path("components/provider")
    )
    with pytest.raises(ValueError, match="NOT_DEFINED"):
        kustomize_matrix.render_template_component(selection, {}, {})


def test_template_rejects_plain_kustomization_asset(tmp_path):
    kustomize_matrix = load_matrix_module()
    template = tmp_path / "template"
    write_template(
        template,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\nkind: Component\n",
    )
    (template / "kustomization.yaml").write_text("resources: []\n", encoding="utf-8")
    selection = kustomize_matrix.TemplateSelection(
        source=template, output_path=Path("components/provider")
    )

    with pytest.raises(ValueError, match="must not contain a plain kustomization.yaml"):
        kustomize_matrix.render_template_assets(
            selection, tmp_path / "component", {}, {}
        )


def test_template_path_is_a_nested_overlay_component_path(tmp_path):
    kustomize_matrix = load_matrix_module()
    template = tmp_path / "template"
    write_template(
        template,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\nkind: Component\n",
    )

    selections = kustomize_matrix.resolve_template_selections(
        [{"source": "template", "path": "components/fabric/efa"}],
        tmp_path / ".kustomize-matrix.yaml",
        "templates",
    )

    assert selections[0].source == template
    assert selections[0].output_path == Path("components/fabric/efa")
    with pytest.raises(ValueError, match="under components"):
        kustomize_matrix.resolve_template_selections(
            [{"source": "template", "path": "templates/efa"}],
            tmp_path / ".kustomize-matrix.yaml",
            "templates",
        )


def test_unfold_rebases_external_component_paths(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe"
    base = recipe / "kustomize/base"
    write_kustomization(base, "resources:\n  - deployment.yaml\n")
    (base / "deployment.yaml").write_text(
        "apiVersion: nvidia.com/v1alpha1\n"
        "kind: DynamoGraphDeployment\n"
        "metadata:\n"
        "  name: app\n",
        encoding="utf-8",
    )
    external = recipe / "shared-component"
    write_kustomization(
        external,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\n"
        "kind: Component\n"
        "patches:\n"
        "  - target:\n"
        "      group: nvidia.com\n"
        "      version: v1alpha1\n"
        "      kind: DynamoGraphDeployment\n"
        "    patch: |\n"
        "      apiVersion: nvidia.com/v1alpha1\n"
        "      kind: DynamoGraphDeployment\n"
        "      metadata:\n"
        "        name: app\n"
        "        labels:\n"
        "          from-external-component: applies\n",
    )
    template = recipe / "templates/provider/instance"
    write_template(
        template,
        "apiVersion: kustomize.config.k8s.io/v1alpha1\n"
        "kind: Component\n"
        "# This comment remains in the generated Component.\n"
        "components:\n"
        "  - ../../../shared-component\n",
    )
    matrix = recipe / ".kustomize-matrix.yaml"
    matrix.write_text(
        "source: kustomize/base\n"
        'nameTemplate: "${variant}"\n'
        "matrix:\n"
        "  variant:\n"
        "    - name: instance\n"
        "      templates:\n"
        "        - source: templates/provider/instance\n"
        "          path: components/external\n",
        encoding="utf-8",
    )

    kustomize_matrix = load_matrix_module()
    mock_kustomize_base_build(
        kustomize_matrix,
        monkeypatch,
        base,
        (base / "deployment.yaml").read_text(encoding="utf-8"),
    )
    config = kustomize_matrix.load_matrix(str(matrix))

    kustomize_matrix.unfold_matrix(config, check=False)
    component = (
        recipe / "kustomize/overlays/instance/components/external/kustomization.yaml"
    )
    rendered_component = component.read_text(encoding="utf-8")
    assert "# This comment remains in the generated Component." in rendered_component
    parsed_component = yaml.safe_load(rendered_component)
    assert parsed_component["components"] == ["../../../../../shared-component"]


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
