# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the SGLang NIXL/UCX runtime layout (NVBug 6541324)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_SCRIPT = REPO_ROOT / "container/deps/sglang/discover_nixl_ucx_layout.py"
INSTALL_SCRIPT = REPO_ROOT / "container/deps/sglang/install_nixl_ucx_compat.sh"
SGLANG_DOCKERFILE = REPO_ROOT / "container/templates/sglang_runtime.Dockerfile"

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


def _run_installer_helper(
    root: Path,
    command: str,
    *,
    ldd_output: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run only the installer's helper definitions against a temporary root."""
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")
    helpers, separator, _main = installer.partition(
        '[[ "${OUTPUT_DIR}" == /* && "${OUTPUT_DIR}" != "/" ]]'
    )
    assert separator, "installer helper/main boundary changed"

    environment = os.environ.copy()
    environment["CANNED_LDD_OUTPUT"] = ldd_output
    script = helpers + '\nNIXL_LIB_DIR="$1"\n' + command + "\n"
    return subprocess.run(
        ["bash", "-s", "--", str(root)],
        input=script,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


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


def test_read_ucx_version_probes_the_selected_library(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_discovery_module()
    libucp = tmp_path / "libucp-private.so.0"
    calls: list[tuple[str, int]] = []

    class FakeGetVersion:
        argtypes = None
        restype = object()

        def __call__(self, major, minor, release) -> None:
            major._obj.value = 1
            minor._obj.value = 21
            release._obj.value = 0

    class FakeLibrary:
        ucp_get_version = FakeGetVersion()

    def fake_cdll(path: str, mode: int):
        calls.append((path, mode))
        return FakeLibrary()

    monkeypatch.setattr(module.ctypes, "CDLL", fake_cdll)

    assert module.read_ucx_version(libucp) == "1.21.0"
    assert calls == [(str(libucp), module.ctypes.RTLD_LOCAL)]
    assert FakeLibrary.ucp_get_version.argtypes == [
        module.ctypes.POINTER(module.ctypes.c_uint),
        module.ctypes.POINTER(module.ctypes.c_uint),
        module.ctypes.POINTER(module.ctypes.c_uint),
    ]
    assert FakeLibrary.ucp_get_version.restype is None


def test_read_ucx_version_rejects_an_unloadable_library(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_discovery_module()
    libucp = tmp_path / "libucp-broken.so.0"

    def fail_cdll(_path: str, mode: int):
        assert mode == module.ctypes.RTLD_LOCAL
        raise OSError("wrong ELF class")

    monkeypatch.setattr(module.ctypes, "CDLL", fail_cdll)

    with pytest.raises(SystemExit, match="failed to load.*wrong ELF class"):
        module.read_ucx_version(libucp)


def test_read_ucx_version_requires_the_version_symbol(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_discovery_module()
    libucp = tmp_path / "libucp-without-version-symbol.so.0"
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    with pytest.raises(SystemExit, match="does not export ucp_get_version"):
        module.read_ucx_version(libucp)


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
    # Wheel RECORD files can retain irrelevant or stale paths. Discovery must
    # ignore them instead of treating them as usable native libraries.
    nixl.files.extend(
        [
            Path("nixl/_bindings/missing/libnixl_capi.so"),
            Path("nixl_cu13.libs/missing/libucp-deadbeef.so.0.0.0"),
        ]
    )
    nvshmem = _stub_files(tmp_path / "nvshmem-site-packages", nvshmem_files)
    nvshmem.metadata["Name"] = "nvidia-nvshmem-cu13"
    nvshmem.files.extend(
        [
            Path("nvidia_nvshmem_cu13/lib/missing/nvshmem_transport_ucx.so.3"),
            Path("nvidia_nvshmem_cu13/README.txt"),
        ]
    )

    def fake_cuda_distributions(package_prefix: str):
        if package_prefix == "nixl":
            return [("nixl-cu13", "13", nixl)]
        if package_prefix == "nvidia-nvshmem":
            return [("nvidia-nvshmem-cu13", "13", nvshmem)]
        raise AssertionError(f"unexpected package prefix: {package_prefix}")

    version_probes: list[Path] = []

    def fake_read_ucx_version(libucp_path: Path) -> str:
        version_probes.append(libucp_path)
        return "1.21.0"

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
        "ucx\t1.21.0",
        f"libdir\t{lib_dir}",
        f"capidir\t{capi_dir}",
        f"plugin\t{plugin}",
        f"alias\tlibucp.so.0\t{libucp}",
        f"alias\tlibucs.so.0\t{libucs}",
    ]


def test_main_rejects_ambiguous_nixl_distributions_without_partial_output(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    module = _load_discovery_module()
    first = _FakeDistribution(tmp_path / "first", "nixl-cu12", [])
    second = _FakeDistribution(tmp_path / "second", "nixl-cu13", [])
    monkeypatch.setattr(
        module,
        "cuda_distributions",
        lambda _prefix: [
            ("nixl-cu12", "12", first),
            ("nixl-cu13", "13", second),
        ],
    )

    with pytest.raises(SystemExit, match="expected one nixl-cu[*] distribution"):
        module.main()

    assert capsys.readouterr().out == ""


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
        lambda package_prefix: (
            [("nixl-cu13", "13", nixl)] if package_prefix == "nixl" else []
        ),
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
        lambda package_prefix: (
            [("nixl-cu13", "13", nixl)] if package_prefix == "nixl" else []
        ),
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


def test_installer_accepts_only_private_ucx_dependency_resolution(
    tmp_path: Path,
) -> None:
    private_ldd = "\n".join(
        [
            f"libucp.so.0 => {tmp_path}/libucp-a1B2c3.so.0 (0x01)",
            f"libucs.so.0 => {tmp_path}/libucs-d4E5f6.so.0 (0x02)",
        ]
    )
    accepted = _run_installer_helper(
        tmp_path,
        'validate_ucx_dependencies "nvshmem_transport_ucx.so.3" "${CANNED_LDD_OUTPUT}"',
        ldd_output=private_ldd,
    )

    assert accepted.returncode == 0, accepted.stderr

    system_ldd = private_ldd.replace(
        f"{tmp_path}/libucp-a1B2c3.so.0",
        "/usr/lib/aarch64-linux-gnu/libucp.so.0",
    )
    rejected = _run_installer_helper(
        tmp_path,
        'validate_ucx_dependencies "nvshmem_transport_ucx.so.3" "${CANNED_LDD_OUTPUT}"',
        ldd_output=system_ldd,
    )

    assert rejected.returncode != 0
    assert "resolved outside NIXL's UCX" in rejected.stderr
    assert (
        rejected.stdout != accepted.stdout or rejected.returncode != accepted.returncode
    )


@pytest.mark.parametrize(
    ("ldd_output", "error_match"),
    [
        ("linux-vdso.so.1 (0x01)", "no UCX dependencies found"),
        ("libucp.so.0 => not found", "unresolved UCX dependency libucp.so.0"),
    ],
)
def test_installer_rejects_incomplete_dependency_reports(
    tmp_path: Path, ldd_output: str, error_match: str
) -> None:
    result = _run_installer_helper(
        tmp_path,
        'validate_ucx_dependencies "nvshmem_transport_ucx.so.3" "${CANNED_LDD_OUTPUT}"',
        ldd_output=ldd_output,
    )

    assert result.returncode != 0
    assert error_match in result.stderr


@pytest.mark.parametrize("module_name", ["libuct_cuda", "libucm_cuda"])
def test_installer_rejects_missing_ucx_cuda_modules(
    tmp_path: Path, module_name: str
) -> None:
    supported_root = tmp_path / "supported"
    cuda_module = supported_root / f"ucx/{module_name}.so.1.21.0"
    cuda_module.parent.mkdir(parents=True)
    cuda_module.touch()

    supported = _run_installer_helper(
        supported_root,
        f"resolve_ucx_module {module_name}",
    )
    unsupported = _run_installer_helper(
        tmp_path / "unsupported",
        f"resolve_ucx_module {module_name}",
    )

    assert supported.returncode == 0, supported.stderr
    assert supported.stdout.strip() == str(cuda_module)
    assert unsupported.returncode != 0
    assert "missing NIXL UCX CUDA module" in unsupported.stderr
    assert unsupported.stdout != supported.stdout


def test_sglang_runtime_wires_the_validated_compatibility_layout() -> None:
    dockerfile = SGLANG_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "install_nixl_ucx_compat.sh,target=/tmp/install_nixl_ucx_compat.sh"
        in dockerfile
    )
    assert (
        "discover_nixl_ucx_layout.py,target=/tmp/discover_nixl_ucx_layout.py"
        in dockerfile
    )
    assert (
        "bash /tmp/install_nixl_ucx_compat.sh "
        "/opt/dynamo/nixl-ucx-compat /opt/dynamo/nixl-capi"
    ) in dockerfile
    assert (
        "ENV LD_LIBRARY_PATH=/opt/dynamo/nixl-ucx-compat:/opt/dynamo/nixl-capi"
        "${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    ) in dockerfile
