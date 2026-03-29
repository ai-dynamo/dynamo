#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import glob
import json
import os
from importlib import metadata
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Optional


DEFAULT_PINNED_CHECKOUT = Path("/tmp/trtllm-latest/tensorrt_llm")
DEFAULT_REPO_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"
DEFAULT_LIBRARY_DIR_PATTERNS = (
    "/usr/local/cuda*/targets/*/lib",
    "/usr/local/cuda*/targets/*/lib/stubs",
    "/usr/local/cuda*/lib64",
    "/usr/local/cuda*/lib",
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/opt/*/lib",
    "/opt/*/lib64",
)


def _parse_expected_cuda_major(requirements: Iterable[str]) -> Optional[int]:
    for requirement in requirements:
        match = re.search(r"nvidia-nccl-cu(\d+)", requirement)
        if match is not None:
            return int(match.group(1))
    for requirement in requirements:
        match = re.search(r"cuda-python\s*>=\s*(\d+)", requirement)
        if match is not None:
            return int(match.group(1))
    return None


def _read_repo_declared_trtllm_version(pyproject_path: Path) -> Optional[str]:
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None

    in_trtllm_group = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not in_trtllm_group:
            if line.startswith("trtllm") and line.endswith("["):
                in_trtllm_group = True
            continue
        if line == "]":
            return None
        match = re.search(r'"tensorrt-llm==([^"]+)"', line)
        if match is not None:
            return match.group(1)
    return None


def _package_surface(package_root: Path) -> dict[str, Any]:
    return {
        "package_root": str(package_root),
        "has_pyexecutor": (package_root / "_torch" / "pyexecutor").is_dir(),
        "has_disaggregation": (package_root / "_torch" / "disaggregation").is_dir(),
    }


def _resolve_distribution(name: str) -> Any:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _distribution_summary(distribution: Any) -> dict[str, Any]:
    if distribution is None:
        return {
            "installed": False,
            "version": None,
            "expected_cuda_major": None,
            "package_root": None,
            "has_pyexecutor": False,
            "has_disaggregation": False,
            "requirements": [],
        }

    requirements = list(distribution.requires or [])
    package_root = Path(distribution.locate_file("tensorrt_llm"))
    summary = _package_surface(package_root)
    summary.update(
        {
            "installed": True,
            "version": distribution.version,
            "expected_cuda_major": _parse_expected_cuda_major(requirements),
            "requirements": requirements,
        }
    )
    return summary


def _expand_library_dirs(patterns: Iterable[str]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            if not path.is_dir() or path in seen:
                continue
            seen.add(path)
            dirs.append(path)
    return dirs


def _extract_soname_major(path: Path) -> Optional[int]:
    match = re.search(r"\.so\.(\d+)(?:$|\.)", path.name)
    if match is not None:
        return int(match.group(1))
    real_path = Path(os.path.realpath(path))
    match = re.search(r"\.so\.(\d+)(?:$|\.)", real_path.name)
    if match is not None:
        return int(match.group(1))
    return None


def _library_summary(library_dirs: Iterable[Path]) -> dict[str, Any]:
    entries = []
    majors = set()
    for directory in library_dirs:
        for path in sorted(directory.glob("libcublasLt.so*")):
            major = _extract_soname_major(path)
            if major is not None:
                majors.add(major)
            entries.append(
                {
                    "path": str(path),
                    "resolved_path": os.path.realpath(path),
                    "major": major,
                }
            )
    return {
        "search_dirs": [str(path) for path in library_dirs],
        "libcublaslt": entries,
        "available_majors": sorted(majors),
    }


def _normalize_subprocess_stream(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _probe_python_import(
    *,
    python_executable: str,
    module: str,
    python_path: Optional[Path] = None,
    timeout_s: float = 20.0,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    env = os.environ.copy()
    if python_path is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{python_path}{os.pathsep}{existing}" if existing else str(python_path)
        )

    command = [
        python_executable,
        "-c",
        (
            "import importlib, json; "
            f"mod = importlib.import_module({module!r}); "
            "print(json.dumps({'module': mod.__name__, 'file': getattr(mod, '__file__', None)}))"
        ),
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "module": module,
            "pythonpath": str(python_path) if python_path is not None else None,
            "status": "timeout",
            "returncode": None,
            "stdout": _normalize_subprocess_stream(exc.stdout),
            "stderr": _normalize_subprocess_stream(exc.stderr),
        }

    stdout = _normalize_subprocess_stream(result.stdout)
    stderr = _normalize_subprocess_stream(result.stderr)
    summary = None
    if stdout:
        try:
            summary = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            summary = None

    return {
        "module": module,
        "pythonpath": str(python_path) if python_path is not None else None,
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "summary": summary,
    }


def _probe_failure_summary(probe: dict[str, Any]) -> str:
    stderr = probe.get("stderr") or ""
    if "PMIx server's listener thread failed to start" in stderr:
        return "Open MPI / PMIx listener startup failed during import"
    if "MPI_Init_thread" in stderr:
        return "MPI_Init_thread failed during import"
    if stderr:
        return stderr.splitlines()[0]
    if probe.get("status") == "timeout":
        return "import probe timed out"
    return "import probe failed"


def build_runtime_report(
    *,
    distribution: Any = None,
    pinned_checkout: Path = DEFAULT_PINNED_CHECKOUT,
    repo_pyproject: Path = DEFAULT_REPO_PYPROJECT,
    library_dirs: Optional[Iterable[Path]] = None,
    probe_imports: bool = False,
    python_executable: str = sys.executable,
    probe_timeout_s: float = 20.0,
    probe_runner: Any = subprocess.run,
) -> dict[str, Any]:
    distribution = _resolve_distribution("tensorrt_llm") if distribution is None else distribution
    installed = _distribution_summary(distribution)
    checkout = {
        "path": str(pinned_checkout),
        "exists": pinned_checkout.is_dir(),
        "has_pyexecutor": False,
        "has_disaggregation": False,
    }
    if pinned_checkout.is_dir():
        checkout.update(_package_surface(pinned_checkout))
    repo_declared_version = _read_repo_declared_trtllm_version(repo_pyproject)

    resolved_library_dirs = (
        list(library_dirs)
        if library_dirs is not None
        else _expand_library_dirs(DEFAULT_LIBRARY_DIR_PATTERNS)
    )
    libraries = _library_summary(resolved_library_dirs)
    import_probes = []
    if probe_imports:
        import_probes.append(
            _probe_python_import(
                python_executable=python_executable,
                module="tensorrt_llm",
                timeout_s=probe_timeout_s,
                runner=probe_runner,
            )
        )
        if pinned_checkout.is_dir():
            import_probes.append(
                _probe_python_import(
                    python_executable=python_executable,
                    module="tensorrt_llm._torch.disaggregation.transceiver",
                    python_path=pinned_checkout.parent,
                    timeout_s=probe_timeout_s,
                    runner=probe_runner,
                )
            )

    findings = []
    if (
        installed["installed"]
        and repo_declared_version is not None
        and installed["version"] != repo_declared_version
    ):
        findings.append(
            "installed tensorrt_llm package version "
            f"{installed['version']} does not match repo-declared trtllm extra "
            f"version {repo_declared_version}"
        )
    if installed["installed"] and checkout["has_disaggregation"] and not installed["has_disaggregation"]:
        findings.append(
            "installed tensorrt_llm package does not expose _torch.disaggregation, "
            "but the pinned checkout does"
        )

    expected_cuda_major = installed["expected_cuda_major"]
    if expected_cuda_major is not None and expected_cuda_major not in libraries["available_majors"]:
        findings.append(
            "installed tensorrt_llm package expects CUDA major "
            f"{expected_cuda_major}, but libcublasLt majors "
            f"{libraries['available_majors'] or '[]'} were found"
        )
    for probe in import_probes:
        if probe["status"] == "ok":
            continue
        findings.append(
            f"subprocess import of {probe['module']} failed before runtime validation: "
            f"{_probe_failure_summary(probe)}"
        )

    return {
        "python_executable": python_executable,
        "probe_timeout_s": probe_timeout_s,
        "repo_declared_tensorrt_llm_version": repo_declared_version,
        "installed_tensorrt_llm": installed,
        "pinned_checkout": checkout,
        "libraries": libraries,
        "import_probes": import_probes,
        "findings": findings,
        "status": "blocked" if findings else "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the installed TRT-LLM runtime without importing tensorrt_llm "
            "into the current process. This helps detect package-surface, CUDA-"
            "library, and optional subprocess import mismatches before running "
            "the KVBM TRT-LLM smoke path."
        )
    )
    parser.add_argument(
        "--pinned-checkout",
        type=Path,
        default=DEFAULT_PINNED_CHECKOUT,
        help="Pinned TRT-LLM checkout package root",
    )
    parser.add_argument(
        "--library-dir",
        action="append",
        type=Path,
        help="Additional library directory to scan for libcublasLt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter used for subprocess import probes",
    )
    parser.add_argument(
        "--probe-timeout-s",
        type=float,
        default=20.0,
        help="Timeout in seconds for each subprocess import probe",
    )
    parser.add_argument(
        "--probe-imports",
        action="store_true",
        help="Run subprocess import probes for installed and pinned TRT-LLM modules",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return a non-zero exit status when the audit reports blocked",
    )
    args = parser.parse_args()

    report = build_runtime_report(
        pinned_checkout=args.pinned_checkout,
        repo_pyproject=DEFAULT_REPO_PYPROJECT,
        library_dirs=args.library_dir,
        probe_imports=args.probe_imports,
        python_executable=args.python_executable,
        probe_timeout_s=args.probe_timeout_s,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {report['status']}")
        print(f"python: {report['python_executable']}")
        print(f"probe timeout (s): {report['probe_timeout_s']}")
        print(
            "repo-declared tensorrt_llm version: "
            f"{report['repo_declared_tensorrt_llm_version'] or 'unknown'}"
        )
        installed = report["installed_tensorrt_llm"]
        print(
            "installed tensorrt_llm: "
            f"{installed['version'] or 'missing'} "
            f"(disaggregation={installed['has_disaggregation']}, "
            f"pyexecutor={installed['has_pyexecutor']}, "
            f"expected_cuda_major={installed['expected_cuda_major']})"
        )
        checkout = report["pinned_checkout"]
        print(
            "pinned checkout: "
            f"{checkout['path']} "
            f"(exists={checkout['exists']}, "
            f"disaggregation={checkout['has_disaggregation']}, "
            f"pyexecutor={checkout['has_pyexecutor']})"
        )
        available = report["libraries"]["available_majors"]
        print(f"libcublasLt majors: {available or '[]'}")
        if report["import_probes"]:
            print("import probes:")
            for probe in report["import_probes"]:
                print(
                    f"- {probe['module']}: status={probe['status']} "
                    f"returncode={probe['returncode']}"
                )
        if report["findings"]:
            print("findings:")
            for finding in report["findings"]:
                print(f"- {finding}")
    if args.fail_on_blocked and report["status"] == "blocked":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
