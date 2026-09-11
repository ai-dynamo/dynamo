# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the SGLang NIXL/UCX runtime layout (NVBug 6541324)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_SCRIPT = (
    REPO_ROOT / "container/deps/sglang/discover_nixl_ucx_layout.py"
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.framework_agnostic,
]


class _FakeDistribution:
    def __init__(self, root: Path, name: str, files: list[str]) -> None:
        self.root = root
        self.metadata = {"Name": name}
        self.files = [Path(item) for item in files]

    def locate_file(self, item: str | Path) -> Path:
        return self.root / item


def _load_discovery_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_sglang_nixl_ucx_layout_discovery", DISCOVERY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_files(root: Path, files: list[str]) -> _FakeDistribution:
    for item in files:
        path = root / item
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return _FakeDistribution(root, "", files)


def test_cuda_distributions_normalizes_names(monkeypatch, tmp_path: Path) -> None:
    module = _load_discovery_module()
    distributions = [
        _FakeDistribution(tmp_path / "nixl13", "nixl_cu13", []),
        _FakeDistribution(tmp_path / "nixl12", "NIXL.CU12", []),
        _FakeDistribution(tmp_path / "nvshmem", "nvidia-nvshmem-cu13", []),
        _FakeDistribution(tmp_path / "plain", "nixl", []),
    ]
    monkeypatch.setattr(module, "distributions", lambda: distributions)

    matches = module.cuda_distributions("nixl")

    assert [(name, cuda_major) for name, cuda_major, _ in matches] == [
        ("nixl_cu13", "13"),
        ("NIXL.CU12", "12"),
    ]


def test_main_resolves_one_coherent_private_ucx_layout(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_discovery_module()
    nixl_files = [
        "nixl/_bindings/libnixl_capi.so",
        "nixl_cu13.libs/libucp-a1B2c3.so.0.0.0",
        "nixl_cu13.libs/libucs-d4E5f6.so.0.0.0",
    ]
    nvshmem_files = [
        "nvidia_nvshmem_cu13/lib/nvshmem_transport_ucx.so.3",
    ]
    nixl = _stub_files(tmp_path / "aarch64-site-packages", nixl_files)
    nixl.metadata["Name"] = "nixl-cu13"
    nvshmem = _stub_files(tmp_path / "nvshmem-site-packages", nvshmem_files)
    nvshmem.metadata["Name"] = "nvidia-nvshmem-cu13"

    def fake_cuda_distributions(package_prefix: str):
        if package_prefix == "nixl":
            return [("nixl-cu13", "13", nixl)]
        if package_prefix == "nvidia-nvshmem":
            return [("nvidia-nvshmem-cu13", "13", nvshmem)]
        raise AssertionError(f"unexpected package prefix: {package_prefix}")

    version_probes: list[Path] = []

    def fake_read_ucx_version(libucp_path: Path) -> str:
        version_probes.append(libucp_path)
        return "1.19.0"

    monkeypatch.setattr(module, "cuda_distributions", fake_cuda_distributions)
    monkeypatch.setattr(module, "read_ucx_version", fake_read_ucx_version)

    module.main()

    nixl_root = nixl.root.resolve()
    nvshmem_root = nvshmem.root.resolve()
    lib_dir = nixl_root / "nixl_cu13.libs"
    capi_dir = nixl_root / "nixl/_bindings"
    libucp = lib_dir / "libucp-a1B2c3.so.0.0.0"
    libucs = lib_dir / "libucs-d4E5f6.so.0.0.0"
    plugin = nvshmem_root / nvshmem_files[0]

    assert version_probes == [libucp]
    assert capsys.readouterr().out.splitlines() == [
        "nixl\tnixl-cu13\t13",
        "ucx\t1.19.0",
        f"libdir\t{lib_dir}",
        f"capidir\t{capi_dir}",
        f"plugin\t{plugin}",
        f"alias\tlibucp.so.0\t{libucp}",
        f"alias\tlibucs.so.0\t{libucs}",
    ]


@pytest.mark.parametrize(
    ("files", "error_match"),
    [
        ([], "expected one libnixl_capi[.]so"),
        (
            ["nixl/_bindings/libnixl_capi.so"],
            "expected one private libucp library",
        ),
        (
            [
                "nixl/_bindings/libnixl_capi.so",
                "nixl_cu13.libs/libucp-a1B2c3.so.0.0.0",
            ],
            "expected one private libucs library",
        ),
    ],
)
def test_main_rejects_incomplete_layout_without_partial_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
    files: list[str],
    error_match: str,
) -> None:
    module = _load_discovery_module()
    nixl = _stub_files(tmp_path / "incomplete-site-packages", files)
    nixl.metadata["Name"] = "nixl-cu13"
    monkeypatch.setattr(
        module,
        "cuda_distributions",
        lambda package_prefix: [("nixl-cu13", "13", nixl)]
        if package_prefix == "nixl"
        else [],
    )

    with pytest.raises(SystemExit, match=error_match):
        module.main()

    assert capsys.readouterr().out == ""


def test_main_rejects_ucx_libraries_from_different_directories(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_discovery_module()
    files = [
        "nixl/_bindings/libnixl_capi.so",
        "nixl_cu13.libs/libucp-a1B2c3.so.0.0.0",
        "other.libs/libucs-d4E5f6.so.0.0.0",
    ]
    nixl = _stub_files(tmp_path / "mixed-site-packages", files)
    nixl.metadata["Name"] = "nixl-cu13"
    monkeypatch.setattr(
        module,
        "cuda_distributions",
        lambda package_prefix: [("nixl-cu13", "13", nixl)]
        if package_prefix == "nixl"
        else [],
    )

    with pytest.raises(SystemExit, match="UCX libraries span multiple directories"):
        module.main()

    assert capsys.readouterr().out == ""


def test_main_rejects_empty_environment_without_partial_output(
    monkeypatch, capsys
) -> None:
    module = _load_discovery_module()
    monkeypatch.setattr(module, "cuda_distributions", lambda _prefix: [])

    with pytest.raises(SystemExit, match="no nixl-cu[*] distribution found"):
        module.main()

    assert capsys.readouterr().out == ""
