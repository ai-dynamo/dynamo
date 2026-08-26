# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate comparable DGDs from conventional v1beta1 and Sweeper case inputs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dynamo.profiler.sweeper.output.atomic import replace_text
from dynamo.profiler.sweeper.renderers import DGDGenerationOptions, render_dgd
from dynamo.profiler.sweeper.runner import load_sweep_config, run_sweep

_CASES_ROOT = Path(__file__).parent / "cases"
_DGDR_INPUT = "dgdr-v1beta1.yaml"
_SWEEPER_INPUT = "sweeper.yaml"
_RENDERERS = ("aic", "direct")


@dataclass(frozen=True)
class ComparisonCase:
    """One convention-based comparison case."""

    name: str
    path: Path
    dgdr_path: Path
    sweeper_path: Path
    dgdr_input: dict[str, Any]
    sweeper_input: dict[str, Any]
    generation_options: DGDGenerationOptions

    @property
    def generated_dir(self) -> Path:
        return self.path / "generated"

    @property
    def cache_dir(self) -> Path:
        return self.generated_dir / ".cache"


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one YAML mapping")
    return value


def _single_sweeper_backend(sweeper_input: dict[str, Any]) -> str:
    search_space = sweeper_input.get("search_space")
    if not isinstance(search_space, dict):
        raise ValueError("sweeper.yaml must define search_space")
    backends = search_space.get("backend")
    if not isinstance(backends, list) or len(backends) != 1:
        raise ValueError(
            "comparison cases require exactly one search_space.backend so the "
            "DGDR runtime image is unambiguous"
        )
    backend = backends[0]
    if not isinstance(backend, str):
        raise ValueError("search_space.backend entries must be strings")
    return backend


def _derive_runtime_image(profiler_image: str, backend: str) -> str:
    """Use the v1 profiler's published-image convention without loading it eagerly."""
    from dynamo.profiler.utils.profile_common import derive_backend_image

    return derive_backend_image(profiler_image, backend)


def _validate_same_intent(
    dgdr_input: dict[str, Any], sweeper_input: dict[str, Any], backend: str
) -> None:
    search_space = sweeper_input["search_space"]
    hardware = dgdr_input.get("hardware")
    if not isinstance(hardware, dict):
        hardware = {}
    expected = {
        "model_name": dgdr_input.get("model"),
        "hardware_sku": hardware.get("gpuSku"),
        "gpu_budget": hardware.get("totalGpus"),
    }
    mismatches = [
        f"{field}: DGDR={value!r}, Sweeper={search_space.get(field)!r}"
        for field, value in expected.items()
        if value is None or search_space.get(field) != value
    ]
    dgdr_backend = dgdr_input.get("backend", "auto")
    if backend != dgdr_backend:
        mismatches.append(f"backend: DGDR={dgdr_backend!r}, Sweeper={backend!r}")

    dgdr_workload = dgdr_input.get("workload")
    sweeper_workload = sweeper_input.get("workload")
    if not isinstance(sweeper_workload, dict):
        mismatches.append("workload: Sweeper input does not define a mapping")
    elif not isinstance(dgdr_workload, dict):
        mismatches.append("workload: DGDR input does not define a workload")
    else:
        for field in ("isl", "osl"):
            dgdr_value = dgdr_workload.get(field)
            if sweeper_workload.get(field) != dgdr_value:
                mismatches.append(
                    f"workload.{field}: DGDR={dgdr_value!r}, "
                    f"Sweeper={sweeper_workload.get(field)!r}"
                )

    if mismatches:
        raise ValueError(
            "case inputs describe different core intent:\n  " + "\n  ".join(mismatches)
        )


def load_case(case_path: Path) -> ComparisonCase:
    """Load one case using only its directory name and conventional files."""
    dgdr_path = case_path / _DGDR_INPUT
    sweeper_path = case_path / _SWEEPER_INPUT
    missing = [path.name for path in (dgdr_path, sweeper_path) if not path.is_file()]
    if missing:
        raise ValueError(
            f"{case_path}: missing conventional input(s): {', '.join(missing)}"
        )

    dgdr_input = _read_mapping(dgdr_path)
    if "spec" in dgdr_input:
        raise ValueError(
            f"{dgdr_path} must contain the v1beta1 spec passed to the profiler, "
            "not a Kubernetes resource wrapper"
        )
    sweeper_input = _read_mapping(sweeper_path)
    backend = _single_sweeper_backend(sweeper_input)
    _validate_same_intent(dgdr_input, sweeper_input, backend)

    image = dgdr_input.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(
            f"{dgdr_path}: image is required to derive the DGD runtime image"
        )
    hardware = dgdr_input.get("hardware")
    num_gpus_per_node = (
        hardware.get("numGpusPerNode") if isinstance(hardware, dict) else None
    )
    if not isinstance(num_gpus_per_node, int):
        raise ValueError(
            f"{dgdr_path}: hardware.numGpusPerNode is required for DGD generation"
        )
    runtime_version_override = dgdr_input.get("runtimeVersionOverride")
    if runtime_version_override is not None and not isinstance(
        runtime_version_override, str
    ):
        raise ValueError(f"{dgdr_path}: runtimeVersionOverride must be a string")

    return ComparisonCase(
        name=case_path.name,
        path=case_path,
        dgdr_path=dgdr_path,
        sweeper_path=sweeper_path,
        dgdr_input=dgdr_input,
        sweeper_input=sweeper_input,
        generation_options=DGDGenerationOptions(
            runtime_image=_derive_runtime_image(image, backend),
            runtime_version_override=runtime_version_override,
            num_gpus_per_node=num_gpus_per_node,
        ),
    )


def _run_v1_profiler(case: ComparisonCase) -> None:
    case.generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = case.generated_dir / "dgdr-v1beta1-dgd.yaml"
    output_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(
        dir=case.generated_dir, prefix=".dgdr-v1beta1-"
    ) as temporary_dir:
        command = [
            sys.executable,
            "-m",
            "dynamo.profiler",
            "--config",
            str(case.dgdr_path),
            "--output-dir",
            temporary_dir,
        ]
        environment = os.environ.copy()
        environment.update(_cache_environment(case))
        environment["DGDR_NAME"] = case.name
        print(f"[{case.name}] v1beta1: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True, env=environment)

        final_config = Path(temporary_dir) / "final_config.yaml"
        if not final_config.is_file():
            raise RuntimeError(
                f"v1beta1 profiler succeeded without writing {final_config}"
            )
        replace_text(
            output_path,
            final_config.read_text(),
        )


def _cache_environment(case: ComparisonCase) -> dict[str, str]:
    return {
        "HF_HOME": str(case.cache_dir / "huggingface"),
        "MPLCONFIGDIR": str(case.cache_dir / "matplotlib"),
        "XDG_CACHE_HOME": str(case.cache_dir),
    }


@contextmanager
def _case_environment(case: ComparisonCase):
    overrides = _cache_environment(case)
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _candidate_document(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "model_dump"):
        value = candidate.model_dump(mode="json")
        if isinstance(value, dict):
            return value
    return {
        "config": candidate.config,
        "used_gpus": candidate.used_gpus,
        "score": candidate.score,
        "metrics": candidate.metrics,
        "objectives": candidate.objectives,
    }


def _best_scalar_candidate(candidates: list[Any]) -> Any:
    return max(
        candidates, key=lambda candidate: (candidate.score, -candidate.used_gpus)
    )


def _run_sweeper_renderers(case: ComparisonCase) -> None:
    config = load_sweep_config(case.sweeper_path)
    if config.goal.is_pareto:
        raise ValueError(
            f"{case.sweeper_path}: comparison cases currently require a scalar goal"
        )

    print(f"[{case.name}] sweeper: {case.sweeper_path}", flush=True)
    result = run_sweep(config)
    if not result.candidates:
        raise RuntimeError("Sweeper returned no feasible candidate")
    candidate = _best_scalar_candidate(result.candidates)
    candidate_path = case.generated_dir / "sweeper-candidate.yaml"
    candidate_path.unlink(missing_ok=True)
    replace_text(
        candidate_path,
        yaml.safe_dump(_candidate_document(candidate), sort_keys=False),
    )

    failures = []
    for renderer in _RENDERERS:
        output_path = case.generated_dir / f"sweeper-{renderer}-dgd.yaml"
        error_path = case.generated_dir / f"sweeper-{renderer}-error.txt"
        output_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
        print(f"[{case.name}] renderer: {renderer}", flush=True)
        try:
            rendered = render_dgd(
                candidate,
                config.workload,
                case.generation_options,
                dgd_name=case.name,
                renderer=renderer,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(f"{renderer}: {exc}")
            replace_text(error_path, f"{exc}\n")
            print(f"[{case.name}] renderer failed: {renderer}: {exc}", file=sys.stderr)
            continue
        replace_text(
            output_path,
            rendered,
        )
    if failures:
        raise RuntimeError("renderer(s) failed:\n  " + "\n  ".join(failures))


def run_case(case: ComparisonCase) -> None:
    """Generate v1beta1, Sweeper-AIC, and Sweeper-direct DGDs for one case."""
    print(f"[{case.name}] output: {case.generated_dir}", flush=True)
    with _case_environment(case):
        _run_v1_profiler(case)
        _run_sweeper_renderers(case)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DGDs for manual v1beta1/Sweeper renderer comparison"
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="case directory names (default: every case)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    names = args.cases or sorted(
        path.name for path in _CASES_ROOT.iterdir() if path.is_dir()
    )
    if not names:
        print(f"no cases found under {_CASES_ROOT}", file=sys.stderr)
        return 2

    try:
        for name in names:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"invalid case name: {name!r}")
            run_case(load_case(_CASES_ROOT / name))
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"sweeper comparison: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
