# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.parse import unquote, urlparse


GLIBC_FLOOR = (2, 28)
OPTIONAL_WHEEL_DISTS = ("kvbm", "gpu-memory-service", "nixl")
EXTRA_EXPECTED_DISTS = {
    "mocker": ("aiconfigurator",),
    "vllm": ("vllm", "nixl"),
    "sglang": ("sglang", "nixl"),
    "trtllm": ("tensorrt-llm",),
}
EXTRA_DYNAMO_IMPORTS = {
    "mocker": "dynamo.mocker",
    "vllm": "dynamo.vllm",
    "sglang": "dynamo.sglang",
    "trtllm": "dynamo.trtllm",
}


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
    return [wheel for wheel in all_wheels(wheelhouse) if wheel_dist_name(wheel) == wanted]


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
        raise AssertionError(f"wheel filename does not include PEP 427 tags: {wheel.name}")
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
    if "manylinux_2_28" not in runtime_platform_tag:
        raise AssertionError(
            f"{runtime.name} should target manylinux_2_28, got {runtime_platform_tag}"
        )
    if runtime_py_tag not in {"cp310", "cp311", "cp312"}:
        raise AssertionError(f"{runtime.name} has unexpected Python tag {runtime_py_tag}")

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


def report_optional_wheels(wheelhouse: Path, target_arch: str | None) -> None:
    for dist_name in OPTIONAL_WHEEL_DISTS:
        matches = find_wheels(wheelhouse, dist_name)
        if not matches:
            print(f"optional wheel absent: {dist_name}")
            continue
        for wheel in matches:
            metadata = wheel_metadata(wheel)
            print(
                f"optional wheel present: {wheel.name} "
                f"({metadata.get('Name')} {metadata.get('Version')})"
            )
            assert_arch_tag(wheel, target_arch)


def binary_wheels(wheelhouse: Path) -> list[Path]:
    result = []
    for wheel in all_wheels(wheelhouse):
        _, _, platform_tag = wheel_tags(wheel)
        if platform_tag != "any":
            result.append(wheel)
    return result


def assert_auditwheel_show(wheelhouse: Path) -> None:
    for wheel in binary_wheels(wheelhouse):
        run(["auditwheel", "show", str(wheel)])


def extracted_shared_libraries(wheel: Path, destination: Path) -> list[Path]:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination)
    return sorted(destination.rglob("*.so")) + sorted(destination.rglob("*.so.*"))


def parse_glibc_versions(version_info: str) -> set[tuple[int, int]]:
    versions = set()
    for major, minor in re.findall(r"GLIBC_(\d+)\.(\d+)", version_info):
        versions.add((int(major), int(minor)))
    return versions


def assert_glibc_floor(wheelhouse: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="dynamo-wheel-symbols-") as tmp:
        tmp_path = Path(tmp)
        offenders: list[str] = []
        for wheel in binary_wheels(wheelhouse):
            wheel_tmp = tmp_path / wheel.name.removesuffix(".whl")
            shared_libraries = extracted_shared_libraries(wheel, wheel_tmp)
            if not shared_libraries:
                raise AssertionError(f"{wheel.name} is tagged binary but has no .so files")

            for shared_library in shared_libraries:
                proc = subprocess.run(
                    ["readelf", "--version-info", str(shared_library)],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if proc.returncode != 0:
                    print(proc.stdout)
                    raise AssertionError(f"readelf failed for {shared_library}")
                versions = parse_glibc_versions(proc.stdout)
                too_new = sorted(version for version in versions if version > GLIBC_FLOOR)
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


def create_venv(base_python: str) -> Path:
    venv_dir = Path(tempfile.mkdtemp(prefix="dynamo-wheel-venv-"))
    run([base_python, "-m", "venv", str(venv_dir)])
    return venv_dir / "bin" / "python"


def pip_install(venv_python: Path, wheelhouse: Path, requirements: list[str]) -> None:
    find_links = []
    for path in (wheelhouse, wheelhouse / "nixl"):
        if path.exists():
            find_links.extend(["--find-links", str(path)])
    run([str(venv_python), "-m", "pip", "install", *find_links, *requirements])


def pip_check(venv_python: Path) -> None:
    run([str(venv_python), "-m", "pip", "check"])


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
        raise AssertionError(f"{dist_name} was not installed from a local file URL: {url}")

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
import json

import dynamo.runtime as runtime
from dynamo._core import (
    Context,
    __version__,
    get_reasoning_parser_names,
    get_tool_parser_names,
    parse_reasoning_batch,
)

assert runtime
assert metadata.version("ai-dynamo") == metadata.version("ai-dynamo-runtime")
assert __version__[0].isdigit()

ctx = Context(id="req-1", metadata={"tenant": "alpha"})
ctx.metadata["region"] = "us-west"
assert dict(ctx.metadata.items()) == {"region": "us-west", "tenant": "alpha"}

assert get_tool_parser_names()
assert get_reasoning_parser_names()
assert json.loads(parse_reasoning_batch("qwen3", "<think>thinking</think>answer")) == {
    "reasoning_text": "thinking",
    "normal_text": "answer",
}

runtime_files = metadata.files("ai-dynamo-runtime")
assert runtime_files is not None
bundled_libs = [
    str(path) for path in runtime_files if ".libs/" in str(path) and ".so" in str(path)
]
assert not bundled_libs, bundled_libs
"""
    run([str(venv_python), "-c", code])


def install_core(wheelhouse: Path, base_python: str) -> Path:
    ai_dynamo = require_one_wheel(wheelhouse, "ai-dynamo")
    runtime = require_one_wheel(wheelhouse, "ai-dynamo-runtime")
    venv_python = create_venv(base_python)
    pip_install(venv_python, wheelhouse, [str(runtime), str(ai_dynamo)])
    pip_check(venv_python)
    assert_dynamo_local_install(venv_python, wheelhouse, ai_dynamo, runtime)
    run_core_import_smoke(venv_python)
    return venv_python


def install_extra(wheelhouse: Path, base_python: str, extra: str) -> Path:
    ai_dynamo = require_one_wheel(wheelhouse, "ai-dynamo")
    runtime = require_one_wheel(wheelhouse, "ai-dynamo-runtime")
    venv_python = create_venv(base_python)
    pip_install(venv_python, wheelhouse, [str(runtime), f"{ai_dynamo}[{extra}]"])
    pip_check(venv_python)
    assert_dynamo_local_install(venv_python, wheelhouse, ai_dynamo, runtime)
    run_extra_import_smoke(venv_python, extra)
    return venv_python


def run_extra_import_smoke(venv_python: Path, extra: str) -> None:
    expected_dists = EXTRA_EXPECTED_DISTS.get(extra)
    module_name = EXTRA_DYNAMO_IMPORTS.get(extra)
    if expected_dists is None or module_name is None:
        raise AssertionError(f"no smoke assertion is defined for extra {extra!r}")

    code = textwrap.dedent(
        f"""
        import importlib
        import importlib.metadata as metadata

        for dist_name in {expected_dists!r}:
            assert metadata.version(dist_name)
        assert importlib.import_module({module_name!r})
        """
    )
    run([str(venv_python), "-c", code])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("core", "metadata", "extra"))
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--target-arch", default="")
    parser.add_argument("--extra", default="")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    wheelhouse = args.wheelhouse.resolve()
    if not wheelhouse.exists():
        raise AssertionError(f"wheelhouse does not exist: {wheelhouse}")

    print("wheelhouse:", wheelhouse)
    print("wheels:")
    for wheel in all_wheels(wheelhouse):
        print(" ", wheel.relative_to(wheelhouse))

    if args.scenario == "metadata":
        assert_core_wheel_metadata(wheelhouse, args.target_arch)
        report_optional_wheels(wheelhouse, args.target_arch)
        assert_auditwheel_show(wheelhouse)
        assert_glibc_floor(wheelhouse)
    elif args.scenario == "core":
        install_core(wheelhouse, args.python)
    elif args.scenario == "extra":
        if not args.extra:
            raise AssertionError("--extra is required for the extra scenario")
        install_extra(wheelhouse, args.python, args.extra)


if __name__ == "__main__":
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    os.environ.setdefault("PIP_ROOT_USER_ACTION", "ignore")
    main()
