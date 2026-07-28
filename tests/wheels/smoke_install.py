# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.parse import unquote, urlparse

GLIBC_FLOOR = (2, 28)
MANYLINUX_POLICY = "manylinux_2_28"
RUNTIME_PY_TAGS = {"cp310", "cp311", "cp312"}
# First-party optional wheel that must ship with the core wheels.
REQUIRED_OPTIONAL_DIST = "kvbm"
# Wheels we ship but do not own / do not validate as manylinux: nixl lives under the
# nixl/ subdir (excluded by considering only top-level wheels); gpu-memory-service is
# not a shipped artifact and is a non-manylinux wheel.
IGNORED_BINARY_DISTS = ("gpu-memory-service",)


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def wheel_dist_name(wheel: Path) -> str:
    return canonical_name(wheel.name.split("-", 1)[0])


def all_wheels(wheelhouse: Path) -> list[Path]:
    return sorted(wheelhouse.rglob("*.whl"))


def find_wheels(wheelhouse: Path, dist_name: str) -> list[Path]:
    wanted = canonical_name(dist_name)
    return [
        wheel for wheel in all_wheels(wheelhouse) if wheel_dist_name(wheel) == wanted
    ]


def require_one_wheel(wheelhouse: Path, dist_name: str) -> Path:
    matches = find_wheels(wheelhouse, dist_name)
    if not matches:
        raise AssertionError(f"missing required {dist_name!r} wheel in {wheelhouse}")
    if len(matches) > 1:
        names = "\n".join(f"  {wheel}" for wheel in matches)
        raise AssertionError(f"expected one {dist_name!r} wheel, found:\n{names}")
    return matches[0]


def wheel_tags(wheel: Path) -> tuple[str, str, str]:
    parts = wheel.name.removesuffix(".whl").split("-")
    if len(parts) < 5:
        raise AssertionError(
            f"wheel filename does not include PEP 427 tags: {wheel.name}"
        )
    return parts[-3], parts[-2], parts[-1]


def wheel_metadata(wheel: Path) -> dict[str, list[str] | str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_members = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_members) != 1:
            raise AssertionError(
                f"expected one METADATA file in {wheel.name}, found {metadata_members}"
            )
        message = Parser().parsestr(archive.read(metadata_members[0]).decode())

    result: dict[str, list[str] | str] = {}
    for key in ("Name", "Version", "Requires-Python"):
        value = message.get(key)
        if value:
            result[key] = value
    result["Requires-Dist"] = message.get_all("Requires-Dist") or []
    result["Provides-Extra"] = message.get_all("Provides-Extra") or []
    return result


def assert_arch_tag(wheel: Path, target_arch: str | None) -> None:
    if not target_arch:
        return
    _, _, platform_tag = wheel_tags(wheel)
    if platform_tag == "any":
        return

    expected_arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(target_arch)
    if not expected_arch:
        raise AssertionError(f"unsupported target arch: {target_arch}")
    if expected_arch not in platform_tag:
        raise AssertionError(
            f"{wheel.name} platform tag {platform_tag!r} does not match {target_arch}"
        )


def assert_core_wheel_metadata(wheelhouse: Path, target_arch: str | None) -> None:
    ai_dynamo = require_one_wheel(wheelhouse, "ai-dynamo")
    runtime = require_one_wheel(wheelhouse, "ai-dynamo-runtime")

    py_tag, abi_tag, platform_tag = wheel_tags(ai_dynamo)
    if (py_tag, abi_tag, platform_tag) != ("py3", "none", "any"):
        raise AssertionError(
            f"{ai_dynamo.name} should be a pure py3-none-any wheel, "
            f"got {py_tag}-{abi_tag}-{platform_tag}"
        )

    runtime_py_tag, runtime_abi_tag, runtime_platform_tag = wheel_tags(runtime)
    if runtime_abi_tag != "abi3":
        raise AssertionError(f"{runtime.name} should use abi3, got {runtime_abi_tag}")
    if MANYLINUX_POLICY not in runtime_platform_tag:
        raise AssertionError(
            f"{runtime.name} should target {MANYLINUX_POLICY}, got {runtime_platform_tag}"
        )
    if runtime_py_tag not in RUNTIME_PY_TAGS:
        raise AssertionError(
            f"{runtime.name} has unexpected Python tag {runtime_py_tag}"
        )

    ai_meta = wheel_metadata(ai_dynamo)
    runtime_meta = wheel_metadata(runtime)
    requires = "\n".join(ai_meta.get("Requires-Dist", []))
    runtime_version = runtime_meta["Version"]
    if f"ai-dynamo-runtime=={runtime_version}" not in requires.replace(" ", ""):
        raise AssertionError(
            f"{ai_dynamo.name} does not pin local runtime version {runtime_version}"
        )

    for wheel in (ai_dynamo, runtime):
        assert_arch_tag(wheel, target_arch)


def assert_no_bundled_libraries(wheelhouse: Path) -> None:
    """The runtime wheel must not vendor shared libraries.

    All .so dependencies should come from the runtime image / pip packages, not be
    bundled inside the wheel by auditwheel. If this fails, add --exclude flags to the
    auditwheel repair command in container/templates/wheel_builder.Dockerfile.
    """
    runtime = require_one_wheel(wheelhouse, "ai-dynamo-runtime")
    with zipfile.ZipFile(runtime) as archive:
        bundled = [
            name for name in archive.namelist() if ".libs/" in name and ".so" in name
        ]
    if bundled:
        details = "\n".join(f"  {name}" for name in bundled)
        raise AssertionError(
            f"{runtime.name} bundles shared libraries (add --exclude to the auditwheel "
            f"repair in wheel_builder.Dockerfile):\n{details}"
        )


def check_optional_wheels(wheelhouse: Path, target_arch: str | None) -> None:
    kvbm = require_one_wheel(wheelhouse, REQUIRED_OPTIONAL_DIST)
    metadata = wheel_metadata(kvbm)
    print(
        f"required wheel present: {kvbm.name} "
        f"({metadata.get('Name')} {metadata.get('Version')})"
    )
    assert_arch_tag(kvbm, target_arch)

    for wheel in find_wheels(wheelhouse, "nixl"):
        print(f"third-party wheel present (not asserted): {wheel.name}")


def binary_wheels(wheelhouse: Path) -> list[Path]:
    # Top-level wheels only: the nixl/ subdir is third-party and excluded here.
    ignored = {canonical_name(dist) for dist in IGNORED_BINARY_DISTS}
    result = []
    for wheel in sorted(wheelhouse.glob("*.whl")):
        if wheel_dist_name(wheel) in ignored:
            continue
        _, _, platform_tag = wheel_tags(wheel)
        if platform_tag != "any":
            result.append(wheel)
    return result


def assert_auditwheel_show(wheelhouse: Path) -> None:
    for wheel in binary_wheels(wheelhouse):
        proc = run(
            [sys.executable, "-m", "auditwheel", "show", str(wheel)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(proc.stdout)
        if MANYLINUX_POLICY not in proc.stdout:
            raise AssertionError(
                f"{wheel.name}: auditwheel did not report a {MANYLINUX_POLICY} "
                f"policy:\n{proc.stdout}"
            )


def extracted_shared_libraries(wheel: Path, destination: Path) -> list[Path]:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination)
    return sorted(destination.rglob("*.so")) + sorted(destination.rglob("*.so.*"))


def parse_glibc_versions(version_info: str) -> set[tuple[int, int]]:
    versions = set()
    for major, minor in re.findall(r"GLIBC_(\d+)\.(\d+)", version_info):
        versions.add((int(major), int(minor)))
    return versions


def glibc_version_needs(shared_library: Path) -> set[tuple[int, int]]:
    from elftools.common.exceptions import ELFError
    from elftools.elf.elffile import ELFFile

    names: list[str] = []
    try:
        with open(shared_library, "rb") as handle:
            section = ELFFile(handle).get_section_by_name(".gnu.version_r")
            if section is not None and hasattr(section, "iter_versions"):
                for _verneed, aux_iter in section.iter_versions():
                    names.extend(aux.name for aux in aux_iter)
    except ELFError:
        return set()
    return parse_glibc_versions("\n".join(names))


def assert_glibc_floor(wheelhouse: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="dynamo-wheel-symbols-") as tmp:
        tmp_path = Path(tmp)
        offenders: list[str] = []
        for wheel in binary_wheels(wheelhouse):
            wheel_tmp = tmp_path / wheel.name.removesuffix(".whl")
            shared_libraries = extracted_shared_libraries(wheel, wheel_tmp)
            if not shared_libraries:
                raise AssertionError(
                    f"{wheel.name} is tagged binary but has no .so files"
                )

            for shared_library in shared_libraries:
                versions = glibc_version_needs(shared_library)
                too_new = sorted(
                    version for version in versions if version > GLIBC_FLOOR
                )
                if too_new:
                    offenders.append(
                        f"{wheel.name}:{shared_library.relative_to(wheel_tmp)} "
                        f"requires GLIBC_{too_new[-1][0]}.{too_new[-1][1]}"
                    )

        if offenders:
            details = "\n".join(f"  {offender}" for offender in offenders)
            raise AssertionError(
                f"binary wheels exceed GLIBC_{GLIBC_FLOOR[0]}.{GLIBC_FLOOR[1]}:\n"
                f"{details}"
            )


def create_venv(python_spec: str) -> Path:
    # python_spec is a version ("3.10") or an interpreter path; uv fetches the
    # interpreter if needed and --seed installs pip into the fresh venv.
    venv_dir = Path(tempfile.mkdtemp(prefix="dynamo-wheel-venv-"))
    run(["uv", "venv", "--seed", "--python", python_spec, str(venv_dir)])
    return venv_dir / "bin" / "python"


def pip_install(venv_python: Path, wheelhouse: Path, requirements: list[str]) -> None:
    # --find-links only registers search paths; nixl is pulled from nixl/ only when a
    # requirement (e.g. kvbm's nixl[cu12]) resolves to it, not installed on its own.
    find_links = []
    for path in (wheelhouse, wheelhouse / "nixl"):
        if path.exists():
            find_links.extend(["--find-links", str(path)])
    run([str(venv_python), "-m", "pip", "install", *find_links, *requirements])


def pip_check(venv_python: Path) -> None:
    command = [str(venv_python), "-m", "pip", "check"]
    print("+", " ".join(command), flush=True)
    proc = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout)
    # `pip check` flags a package when its internal WHEEL tag is disjoint from the
    # platform's supported tags ("<pkg> is not supported on this platform"). NVIDIA's
    # CUDA wheels (pulled transitively via torch) tag arm64 as `sbsa`, which trips this
    # heuristic even though the wheel installed and works. Ignore those and fail only on
    # genuine dependency problems (missing or conflicting requirements).
    conflicts = [
        line
        for line in proc.stdout.splitlines()
        if "which is not installed." in line or "but you have" in line
    ]
    if conflicts:
        details = "\n".join(f"  {line}" for line in conflicts)
        raise AssertionError(f"pip check found dependency conflicts:\n{details}")


def direct_url_for(venv_python: Path, dist_name: str) -> dict[str, object]:
    code = textwrap.dedent(
        f"""
        import importlib.metadata as metadata
        import json

        direct_url = metadata.distribution({dist_name!r}).read_text("direct_url.json")
        if not direct_url:
            raise SystemExit("missing direct_url.json for {dist_name}")
        print(direct_url)
        """
    )
    proc = subprocess.run(
        [str(venv_python), "-c", code],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def assert_local_direct_url(
    venv_python: Path,
    dist_name: str,
    expected_wheel: Path,
    wheelhouse: Path,
) -> None:
    data = direct_url_for(venv_python, dist_name)
    url = data.get("url")
    if not isinstance(url, str):
        raise AssertionError(f"{dist_name} direct_url.json does not contain a URL")

    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise AssertionError(
            f"{dist_name} was not installed from a local file URL: {url}"
        )

    installed_path = Path(unquote(parsed.path)).resolve()
    expected_path = expected_wheel.resolve()
    wheelhouse_path = wheelhouse.resolve()
    if installed_path != expected_path:
        raise AssertionError(
            f"{dist_name} installed from {installed_path}, expected {expected_path}"
        )
    if not installed_path.is_relative_to(wheelhouse_path):
        raise AssertionError(
            f"{dist_name} installed from {installed_path}, outside {wheelhouse_path}"
        )


def assert_dynamo_local_install(
    venv_python: Path,
    wheelhouse: Path,
    ai_dynamo: Path,
    runtime: Path,
) -> None:
    assert_local_direct_url(venv_python, "ai-dynamo", ai_dynamo, wheelhouse)
    assert_local_direct_url(venv_python, "ai-dynamo-runtime", runtime, wheelhouse)


def run_core_import_smoke(venv_python: Path) -> None:
    code = r"""
import importlib.metadata as metadata

import dynamo.runtime as runtime
from dynamo._core import __version__

assert runtime
assert __version__[0].isdigit()
assert metadata.version("ai-dynamo") == metadata.version("ai-dynamo-runtime")
"""
    run([str(venv_python), "-c", code])


OPTIONAL_IMPORT_NAMES = {"kvbm": "kvbm"}


def install_core(
    wheelhouse: Path, python_spec: str, also: tuple[str, ...] = ()
) -> None:
    ai_dynamo = require_one_wheel(wheelhouse, "ai-dynamo")
    runtime = require_one_wheel(wheelhouse, "ai-dynamo-runtime")
    also_wheels = {dist: require_one_wheel(wheelhouse, dist) for dist in also}

    venv_python = create_venv(python_spec)
    try:
        requirements = [
            str(runtime),
            str(ai_dynamo),
            *(str(w) for w in also_wheels.values()),
        ]
        pip_install(venv_python, wheelhouse, requirements)
        pip_check(venv_python)
        assert_dynamo_local_install(venv_python, wheelhouse, ai_dynamo, runtime)
        run_core_import_smoke(venv_python)

        for dist, wheel in also_wheels.items():
            assert_local_direct_url(venv_python, dist, wheel, wheelhouse)
            import_name = OPTIONAL_IMPORT_NAMES.get(dist, dist)
            run([str(venv_python), "-c", f"import {import_name}; assert {import_name}"])
    finally:
        shutil.rmtree(venv_python.parent.parent, ignore_errors=True)
