# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render gates for the release/1.3 CUDA EFA image composition."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
    pytest.mark.core,
]

_REPO = Path(__file__).resolve().parents[2]
_CONTAINER = _REPO / "container"
_EFA_INSTALLER = _CONTAINER / "deps/efa/install_efa.sh"
_CUDA_FRAMEWORKS = (
    pytest.param("vllm", "13.0", id="vllm"),
    pytest.param("sglang", "13.0", id="sglang"),
    pytest.param("trtllm", "13.1", id="trtllm"),
)


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "dynamo_container_render", _CONTAINER / "render.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(
    framework: str,
    *,
    cuda_version: str = "13.0",
    make_efa: bool,
    platform: str = "multi",
) -> str:
    renderer = _load_renderer()
    context = yaml.safe_load((_CONTAINER / "context.yaml").read_text())
    args = SimpleNamespace(
        framework=framework,
        device="cuda",
        target="runtime",
        platform=platform,
        cuda_version=cuda_version,
        make_efa=make_efa,
    )
    template = renderer._make_jinja_env(_CONTAINER).get_template("Dockerfile.template")
    rendered = template.render(
        context=context,
        **renderer._render_context(args, context),
    )
    return re.sub(r"\n{3,}", "\n\n", rendered)


@pytest.mark.parametrize(("framework", "cuda_version"), _CUDA_FRAMEWORKS)
@pytest.mark.parametrize("platform", ["amd64", "arm64", "multi"])
def test_release13_efa_stack_is_common_and_pinned(
    framework: str, cuda_version: str, platform: str
):
    rendered = _render(
        framework,
        cuda_version=cuda_version,
        make_efa=True,
        platform=platform,
    )

    for required in (
        "ARG EFA_VERSION=1.49.0",
        "cf2e9281a2328a243c76f911a490faed43ca0fecfe4733c25e34b2e92a32c309",
        "ARG EFA_INSTALLER_SIZE=702811844",
        "source=./container/deps/efa/install_efa.sh",
        '"${EFA_VERSION}" "${EFA_INSTALLER_SHA256}" "${EFA_INSTALLER_SIZE}"',
        '-Dlibfabric_path="/opt/amazon/efa"',
        'libfabric1-aws)" = "2.4.0amzn5.0"',
        'rdma-core)" = "63.0-1"',
        'libnccl-ofi-ngc-v3)" = "1.20.0-1"',
        "dpkg-query -S /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so",
        "5d5f48961cf9b30def1c8cdb0961b0683f0102ca9898641ef90f2d205379a5fd",
        "24291af4829bcdddc2d47d7eb35366c19246170091202b93814fda9b67f32248",
        "FROM ubuntu:24.04 AS efa_sdk",
        "COPY --from=efa_sdk /opt/amazon/efa /opt/amazon/efa",
        "FROM ${EFA_BASE_IMAGE} AS aws_base",
        "FROM aws_base AS aws_framework",
        "FROM aws_framework AS aws",
        "ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH}",
    ):
        assert required in rendered

    # wheel_builder and the final AWS stage install from the same exact archive.
    assert rendered.count("source=./container/deps/efa/install_efa.sh") == 2
    assert rendered.count("--skip-nccl") == 1

    # The retired generic path built OFIWG 2.5.1 into /usr/local/libfabric and
    # copied it over the installer stack. It must not return for any framework.
    assert "libfabric overlay:" not in rendered
    assert "libfabric.so.1.31.1 =>" not in rendered
    assert "https://github.com/ofiwg/libfabric" not in rendered
    assert "/usr/local/libfabric" not in rendered
    assert "NIXL_LIBFABRIC" not in rendered


def test_release13_efa_installer_uses_the_stock_149_stack():
    helper = _EFA_INSTALLER.read_text()

    assert "aws-efa-installer-${efa_version}.tar.gz" in helper
    assert (
        'archive_url="https://efa-installer.amazonaws.com/'
        'aws-efa-installer-${efa_version}.tar.gz"'
    ) in helper
    for required_option in (
        "--build-ngc",
        "--skip-kmod",
        "--skip-limit-conf",
        "--skip-mpi",
        "--no-verify",
    ):
        assert required_option in helper
    assert 'if [[ "${skip_nccl}" == "--skip-nccl" ]]' in helper
    assert "installer_args+=(--skip-plugin)" in helper
    assert "apt-get purge -y" in helper
    assert "libnccl-ofi-ngc-v3" in helper


@pytest.mark.parametrize(("framework", "cuda_version"), _CUDA_FRAMEWORKS)
def test_non_efa_cuda_render_disables_libfabric(framework: str, cuda_version: str):
    rendered = _render(
        framework,
        cuda_version=cuda_version,
        make_efa=False,
        platform="amd64",
    )

    assert "container/deps/efa/install_efa.sh" not in rendered
    assert "ARG EFA_VERSION=" not in rendered
    assert '-Dlibfabric_path="/opt/amazon/efa"' not in rendered
    assert "-Ddisable_plugins=LIBFABRIC" in rendered
    assert "nixl-pr1966" not in rendered
    assert "sglang_nixl_efa_builder" not in rendered


@pytest.mark.parametrize("platform", ["amd64", "arm64", "multi"])
def test_release13_sglang_alone_backports_nixl_pr1966(platform: str):
    rendered = _render("sglang", make_efa=True, platform=platform)

    for required in (
        "0f3c0aec918502aaad4343b99f1cc94cea1e85e0",
        "d1e60ec95140ff018a1db6057e3f51dfa4c0f562",
        "659b5b21f4d0d7ad0d79c54a8bd47be642ca21a2",
        "d605856c2409b5a2204a780ca4e1738445a0496b68a52d83f7f5350403687524",
        "validate_pr1966_semantics.py",
        ".nv_fatbin",
        "FROM aws_base AS sglang_nixl_efa_builder",
        "ENV LD_PRELOAD=/opt/amazon/efa/lib/libfabric.so.1",
    ):
        assert required in rendered


@pytest.mark.parametrize(
    ("framework", "cuda_version"),
    [
        pytest.param("vllm", "13.0", id="vllm"),
        pytest.param("trtllm", "13.1", id="trtllm"),
    ],
)
def test_other_frameworks_do_not_receive_the_sglang_nixl_patch(
    framework: str, cuda_version: str
):
    rendered = _render(framework, cuda_version=cuda_version, make_efa=True)

    assert "nixl-pr1966" not in rendered
    assert "sglang_nixl_efa_builder" not in rendered


@pytest.mark.parametrize(
    ("framework", "cuda_version", "plugin"),
    [
        pytest.param(
            "vllm",
            "13.0",
            "/opt/dynamo/nixl/plugins/libplugin_LIBFABRIC.so",
            id="vllm",
        ),
        pytest.param(
            "trtllm",
            "13.1",
            "/opt/nvidia/nvda_nixl/lib64/plugins/libplugin_LIBFABRIC.so",
            id="trtllm",
        ),
    ],
)
def test_vllm_and_trtllm_gate_the_final_nixl_plugin(
    framework: str, cuda_version: str, plugin: str
):
    rendered = _render(framework, cuda_version=cuda_version, make_efa=True)

    assert f"PLUGIN={plugin}" in rendered
    assert 'ldd -v "$PLUGIN"' in rendered
    assert (
        'if grep -Fq "not found" /tmp/nixl-libfabric.ldd; then exit 1; fi' in rendered
    )
    assert (
        'if grep -Fq "FABRIC_1.9" /tmp/nixl-libfabric.ldd; then exit 1; fi' in rendered
    )
    assert "libfabric.so.1 => /opt/amazon/efa/lib/libfabric.so.1" in rendered
    assert (
        'ctypes.CDLL(os.environ["PLUGIN"], mode=os.RTLD_NOW | os.RTLD_LOCAL)'
        in rendered
    )


def test_vllm_cuda_runtime_installs_nixl_plugin_runtime_dependency():
    rendered = _render("vllm", make_efa=True)

    assert "libhwloc15" in rendered
    assert rendered.index("libhwloc15") < rendered.index("FROM pre_runtime AS licenses")

    non_efa = _render("vllm", make_efa=False)
    assert "libhwloc15" not in non_efa
