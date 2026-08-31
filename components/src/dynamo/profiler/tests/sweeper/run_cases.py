# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate comparable DGDs from case inputs composed with hardware profiles.

The portable case-selection and cluster-test shape builds on Ashna Mehrotra's
comparison suite in https://github.com/ai-dynamo/dynamo/pull/14031.
"""

from __future__ import annotations

import argparse
import copy
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

_ROOT = Path(__file__).parent
_REPOSITORY_ROOT = _ROOT.parents[5]
_CASES_ROOT = _ROOT / "cases"
_HARDWARE_ROOT = _ROOT / "hardware"
_DEFAULT_OUTPUT_ROOT = _ROOT / "generated"
_DEFAULT_MANUAL_OUTPUT_ROOT = _DEFAULT_OUTPUT_ROOT / "manual"
_DGDR_INPUT = "dgdr-v1beta1.yaml"
_SWEEPER_INPUT = "sweeper.yaml"
_DGDR_HARDWARE_PATCH = "dgdr-v1beta1.patch.yaml"
_SWEEPER_HARDWARE_PATCH = "sweeper.patch.yaml"
_RENDERERS = ("aic", "direct")


@dataclass(frozen=True)
class HardwareConfig:
    """Provider-independent profiler patches for one accelerator target."""

    name: str
    path: Path
    dgdr_patch: dict[str, Any]
    sweeper_patch: dict[str, Any]


@dataclass(frozen=True)
class Recipe:
    """Optional maintained recipe and its known hardware requirements."""

    source: str
    path: Path
    metadata_path: Path
    requirements: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SuiteEntry:
    """One explicit case and hardware selection."""

    case: str
    hardware: str


@dataclass(frozen=True)
class ComparisonCase:
    """One case composed for one hardware configuration."""

    name: str
    path: Path
    hardware: HardwareConfig
    dgdr_input: dict[str, Any]
    sweeper_input: dict[str, Any]
    recipe: Recipe | None
    generation_options: DGDGenerationOptions
    output_root: Path

    @property
    def generated_dir(self) -> Path:
        return self.output_root / self.hardware.name / self.name

    @property
    def cache_dir(self) -> Path:
        return self.generated_dir / ".cache"

    @property
    def composed_dgdr_path(self) -> Path:
        return self.generated_dir / "dgdr-v1beta1-composed.yaml"

    @property
    def composed_sweeper_path(self) -> Path:
        return self.generated_dir / "sweeper-composed.yaml"


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one YAML mapping")
    return value


def _required_name(value: Any, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{path}: {field} must be one non-empty directory name")
    return value


def _case_path(name: str) -> Path:
    name = _required_name(name, field="case", path=_CASES_ROOT)
    path = _CASES_ROOT / name
    if not path.is_dir():
        raise ValueError(f"{path}: case directory does not exist")
    return path


def missing_case_inputs(name: str) -> tuple[str, ...]:
    """Return native profiler inputs that keep one matrix row from running."""
    path = _case_path(name)
    return tuple(
        filename
        for filename in (_DGDR_INPUT, _SWEEPER_INPUT)
        if not (path / filename).is_file()
    )


def _merge_patch(base: Any, patch: Any) -> Any:
    """Apply RFC 7386-style merge semantics to YAML-compatible values."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = _merge_patch(result.get(key), value)
    return result


def load_hardware(name: str) -> HardwareConfig:
    """Load both native-schema patches for one named hardware target."""
    name = _required_name(name, field="hardware", path=_HARDWARE_ROOT)
    path = _HARDWARE_ROOT / name
    dgdr_path = path / _DGDR_HARDWARE_PATCH
    sweeper_path = path / _SWEEPER_HARDWARE_PATCH
    missing = [
        candidate.name
        for candidate in (dgdr_path, sweeper_path)
        if not candidate.is_file()
    ]
    if missing:
        raise ValueError(f"{path}: missing hardware patch(es): {', '.join(missing)}")
    return HardwareConfig(
        name=name,
        path=path,
        dgdr_patch=_read_mapping(dgdr_path),
        sweeper_patch=_read_mapping(sweeper_path),
    )


def _load_recipe(case_path: Path) -> Recipe | None:
    recipe_path = case_path / "recipe.yaml"
    if not recipe_path.is_file():
        return None
    value = _read_mapping(recipe_path)
    source = value.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{recipe_path}: source must be a non-empty repository path")
    source_path = (_REPOSITORY_ROOT / source).resolve()
    try:
        source_path.relative_to(_REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{recipe_path}: source escapes the repository") from exc
    requirements = value.get("requirements", {})
    if not isinstance(requirements, dict) or any(
        not isinstance(name, str) or not isinstance(requirement, dict)
        for name, requirement in requirements.items()
    ):
        raise TypeError(
            f"{recipe_path}: requirements must map hardware names to mappings"
        )
    return Recipe(
        source=source,
        path=source_path,
        metadata_path=recipe_path,
        requirements=requirements,
    )


def load_recipe(case_name: str) -> Recipe | None:
    """Load the optional recipe provenance for one matrix row."""
    return _load_recipe(_case_path(case_name))


def write_discovered_recipe_requirement(
    recipe: Recipe, hardware: str, gpus: int
) -> Path | None:
    """Write a proposed requirement without changing the checked-in recipe file."""
    proposed = {"gpus": gpus}
    original = _read_mapping(recipe.metadata_path)
    if original.get("requirements", {}).get(hardware) == proposed:
        return None
    output_path = recipe.metadata_path.with_name("recipe.new.yaml")
    value = _read_mapping(output_path) if output_path.is_file() else original
    value.setdefault("requirements", {})[hardware] = proposed
    replace_text(output_path, yaml.safe_dump(value, sort_keys=False))
    return output_path


def recipe_gpu_count(recipe: Recipe) -> int:
    """Return the GPU footprint declared by one maintained DGD recipe."""
    documents = [
        document for document in yaml.safe_load_all(recipe.path.read_text()) if document
    ]
    dgds = [
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    ]
    if len(dgds) != 1:
        raise ValueError(
            f"{recipe.path} must contain exactly one DynamoGraphDeployment"
        )
    spec = dgds[0].get("spec", {})
    total = 0
    services = spec.get("services")
    if isinstance(services, dict):
        for service in services.values():
            replicas = service.get("replicas", 1)
            gpus = service.get("resources", {}).get("limits", {}).get("gpu", 0)
            total += int(replicas) * int(gpus)
    components = spec.get("components")
    if isinstance(components, list):
        for component in components:
            replicas = component.get("replicas", 1)
            containers = (
                component.get("podTemplate", {}).get("spec", {}).get("containers", [])
            )
            main = next(
                (
                    container
                    for container in containers
                    if container.get("name") == "main"
                ),
                containers[0] if containers else {},
            )
            gpus = main.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0)
            total += int(replicas) * int(gpus)
    if total <= 0:
        raise ValueError(f"cannot determine GPU requirement from {recipe.path}")
    return total


def _single_sweeper_backend(sweeper_input: dict[str, Any]) -> str:
    search_space = sweeper_input.get("search_space")
    if not isinstance(search_space, dict):
        raise TypeError("sweeper input must define search_space")
    backends = search_space.get("backend")
    if (
        not isinstance(backends, list)
        or len(backends) != 1
        or not isinstance(backends[0], str)
    ):
        raise ValueError("comparison cases require exactly one search_space.backend")
    return backends[0]


def _derive_runtime_image(profiler_image: str, backend: str) -> str:
    """Use the v1 profiler's published-image convention without loading it eagerly."""
    from dynamo.profiler.utils.profile_common import derive_backend_image

    return derive_backend_image(profiler_image, backend)


def _validate_same_intent(
    dgdr_input: dict[str, Any], sweeper_input: dict[str, Any], backend: str
) -> None:
    search_space = sweeper_input.get("search_space")
    hardware = dgdr_input.get("hardware")
    if not isinstance(search_space, dict) or not isinstance(hardware, dict):
        raise TypeError("composed inputs must define search_space and hardware")
    expected = {
        "model_name": dgdr_input.get("model"),
        "gpu_budget": hardware.get("totalGpus"),
    }
    mismatches = [
        f"{field}: DGDR={value!r}, Sweeper={search_space.get(field)!r}"
        for field, value in expected.items()
        if value is None or search_space.get(field) != value
    ]
    dgdr_gpu_sku = hardware.get("gpuSku")
    sweeper_gpu_sku = search_space.get("hardware_sku")
    normalized_dgdr_gpu_sku = {"gb200_sxm": "gb200"}.get(dgdr_gpu_sku, dgdr_gpu_sku)
    if normalized_dgdr_gpu_sku != sweeper_gpu_sku:
        mismatches.append(
            f"hardware_sku: DGDR={dgdr_gpu_sku!r}, Sweeper={sweeper_gpu_sku!r}"
        )
    if backend != dgdr_input.get("backend", "auto"):
        mismatches.append(
            f"backend: DGDR={dgdr_input.get('backend')!r}, Sweeper={backend!r}"
        )

    dgdr_workload = dgdr_input.get("workload")
    sweeper_workload = sweeper_input.get("workload")
    if not isinstance(dgdr_workload, dict) or not isinstance(sweeper_workload, dict):
        mismatches.append("workload: both inputs must define a mapping")
    else:
        for field in ("isl", "osl"):
            if dgdr_workload.get(field) != sweeper_workload.get(field):
                mismatches.append(
                    f"workload.{field}: DGDR={dgdr_workload.get(field)!r}, "
                    f"Sweeper={sweeper_workload.get(field)!r}"
                )
    if mismatches:
        raise ValueError(
            "composed inputs describe different core intent:\n  "
            + "\n  ".join(mismatches)
        )


def load_case(
    case_name: str,
    hardware: HardwareConfig,
    *,
    output_root: Path = _DEFAULT_MANUAL_OUTPUT_ROOT,
) -> ComparisonCase:
    """Compose and validate one conventional case for one hardware target."""
    case_path = _case_path(case_name)
    dgdr_path = case_path / _DGDR_INPUT
    sweeper_path = case_path / _SWEEPER_INPUT
    missing = [
        candidate.name
        for candidate in (dgdr_path, sweeper_path)
        if not candidate.is_file()
    ]
    if missing:
        raise ValueError(f"{case_path}: missing case input(s): {', '.join(missing)}")

    dgdr_base = _read_mapping(dgdr_path)
    if "spec" in dgdr_base:
        raise ValueError(
            f"{dgdr_path} must contain the v1beta1 spec, not a resource wrapper"
        )
    dgdr_input = _merge_patch(dgdr_base, hardware.dgdr_patch)
    sweeper_input = _merge_patch(_read_mapping(sweeper_path), hardware.sweeper_patch)
    backend = _single_sweeper_backend(sweeper_input)
    _validate_same_intent(dgdr_input, sweeper_input, backend)

    image = dgdr_input.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(
            f"{dgdr_path}: image is required to derive the DGD runtime image"
        )
    hardware_input = dgdr_input.get("hardware")
    num_gpus_per_node = (
        hardware_input.get("numGpusPerNode")
        if isinstance(hardware_input, dict)
        else None
    )
    if not isinstance(num_gpus_per_node, int):
        raise TypeError(f"{hardware.path}: hardware.numGpusPerNode is required")
    runtime_version_override = dgdr_input.get("runtimeVersionOverride")
    if runtime_version_override is not None and not isinstance(
        runtime_version_override, str
    ):
        raise ValueError(f"{dgdr_path}: runtimeVersionOverride must be a string")

    return ComparisonCase(
        name=case_name,
        path=case_path,
        hardware=hardware,
        dgdr_input=dgdr_input,
        sweeper_input=sweeper_input,
        recipe=_load_recipe(case_path),
        generation_options=DGDGenerationOptions(
            runtime_image=_derive_runtime_image(image, backend),
            runtime_version_override=runtime_version_override,
            num_gpus_per_node=num_gpus_per_node,
        ),
        output_root=output_root,
    )


def load_suite(path: Path) -> list[SuiteEntry]:
    """Load an explicit list of case and hardware combinations."""
    value = _read_mapping(path)
    tests = value.get("tests")
    if not isinstance(tests, list) or not tests:
        raise TypeError(f"{path}: tests must be a non-empty list")
    entries = []
    for index, item in enumerate(tests):
        if not isinstance(item, dict):
            raise TypeError(f"{path}: tests[{index}] must be a mapping")
        entries.append(
            SuiteEntry(
                case=_required_name(
                    item.get("case"), field=f"tests[{index}].case", path=path
                ),
                hardware=_required_name(
                    item.get("hardware"), field=f"tests[{index}].hardware", path=path
                ),
            )
        )
    return entries


def default_suite_output_root(path: Path) -> Path:
    """Return the checked-in golden root owned by one suite file."""
    return _DEFAULT_OUTPUT_ROOT / path.stem


def _write_composed_inputs(case: ComparisonCase) -> None:
    case.generated_dir.mkdir(parents=True, exist_ok=True)
    replace_text(
        case.composed_dgdr_path, yaml.safe_dump(case.dgdr_input, sort_keys=False)
    )
    replace_text(
        case.composed_sweeper_path,
        yaml.safe_dump(case.sweeper_input, sort_keys=False),
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


def _run_v1_profiler(case: ComparisonCase) -> None:
    output_path = case.generated_dir / "dgd-profiler-v1beta1.yaml"
    output_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{case.name}-dgdr-v1beta1-"
    ) as temporary_dir:
        command = [
            sys.executable,
            "-m",
            "dynamo.profiler",
            "--config",
            str(case.composed_dgdr_path),
            "--output-dir",
            temporary_dir,
        ]
        environment = os.environ.copy()
        environment.update(_cache_environment(case))
        environment["DGDR_NAME"] = case.name
        print(
            f"[{case.hardware.name}/{case.name}] v1beta1: {shlex.join(command)}",
            flush=True,
        )
        subprocess.run(command, check=True, env=environment)
        final_config = Path(temporary_dir) / "final_config.yaml"
        if not final_config.is_file():
            raise RuntimeError(
                f"v1beta1 profiler succeeded without writing {final_config}"
            )
        replace_text(output_path, final_config.read_text())


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


def _run_sweeper_renderers(case: ComparisonCase) -> list[str]:
    candidate_path = case.generated_dir / "candidate-sweeper.yaml"
    candidate_path.unlink(missing_ok=True)
    for renderer in _RENDERERS:
        (case.generated_dir / f"dgd-sweeper-{renderer}.yaml").unlink(missing_ok=True)

    config = load_sweep_config(case.composed_sweeper_path)
    if config.goal.is_pareto:
        raise ValueError(
            f"{case.composed_sweeper_path}: comparison cases require a scalar goal"
        )

    print(f"[{case.hardware.name}/{case.name}] sweeper", flush=True)
    result = run_sweep(config)
    if not result.candidates:
        raise RuntimeError("Sweeper returned no feasible candidate")
    candidate = _best_scalar_candidate(result.candidates)
    replace_text(
        candidate_path,
        yaml.safe_dump(_candidate_document(candidate), sort_keys=False),
    )

    failures = []
    for renderer in _RENDERERS:
        output_path = case.generated_dir / f"dgd-sweeper-{renderer}.yaml"
        error_path = case.generated_dir / f"error-sweeper-{renderer}.txt"
        print(f"[{case.hardware.name}/{case.name}] renderer: {renderer}", flush=True)
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
            print(
                f"[{case.hardware.name}/{case.name}] renderer failed: {renderer}: {exc}",
                file=sys.stderr,
            )
            continue
        error_path.unlink(missing_ok=True)
        replace_text(output_path, rendered)
    return failures


def run_case(case: ComparisonCase) -> list[str]:
    """Generate every available DGD variant for one case and hardware target."""
    print(
        f"[{case.hardware.name}/{case.name}] output: {case.generated_dir}", flush=True
    )
    _write_composed_inputs(case)
    failures = []
    with _case_environment(case):
        try:
            _run_v1_profiler(case)
        except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
            TypeError,
            ValueError,
        ) as exc:
            failures.append(f"v1beta1: {exc}")
            replace_text(case.generated_dir / "error-profiler-v1beta1.txt", f"{exc}\n")
        else:
            (case.generated_dir / "error-profiler-v1beta1.txt").unlink(missing_ok=True)
        try:
            failures.extend(_run_sweeper_renderers(case))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"sweeper: {exc}")
            replace_text(case.generated_dir / "error-sweeper.txt", f"{exc}\n")
        else:
            (case.generated_dir / "error-sweeper.txt").unlink(missing_ok=True)
    return failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DGD goldens from v1beta1 and Sweeper for selected case/hardware pairs"
    )
    parser.add_argument(
        "cases", nargs="*", help="case directory names used with --hardware"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--suite", type=Path, help="suite YAML containing case/hardware pairs"
    )
    selection.add_argument(
        "--hardware", help="hardware directory name for positional cases"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="generated output root (default: suite-named goldens or generated/manual)",
    )
    return parser


def _selections(args: Any) -> list[SuiteEntry]:
    if args.suite is not None:
        if args.cases:
            raise ValueError("positional cases cannot be combined with --suite")
        return load_suite(args.suite)
    names = args.cases or sorted(
        path.name for path in _CASES_ROOT.iterdir() if path.is_dir()
    )
    if not names:
        raise ValueError(f"no cases found under {_CASES_ROOT}")
    return [SuiteEntry(case=name, hardware=args.hardware) for name in names]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_dir
    if output_root is None:
        output_root = (
            default_suite_output_root(args.suite)
            if args.suite is not None
            else _DEFAULT_MANUAL_OUTPUT_ROOT
        )
    failures = []
    skipped = 0
    try:
        hardware_cache: dict[str, HardwareConfig] = {}
        for entry in _selections(args):
            missing = missing_case_inputs(entry.case)
            if missing:
                if load_recipe(entry.case) is None:
                    raise ValueError(
                        f"{entry.case}: missing {', '.join(missing)} and recipe.yaml"
                    )
                print(
                    f"[{entry.hardware}/{entry.case}] coverage gap: "
                    f"missing {', '.join(missing)}",
                    flush=True,
                )
                skipped += 1
                continue
            if entry.hardware not in hardware_cache:
                hardware_cache[entry.hardware] = load_hardware(entry.hardware)
            hardware = hardware_cache[entry.hardware]
            case = load_case(entry.case, hardware, output_root=output_root)
            failures.extend(
                f"{entry.hardware}/{entry.case}: {failure}"
                for failure in run_case(case)
            )
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"sweeper comparison: error: {exc}", file=sys.stderr)
        return 2
    if failures:
        print(
            "sweeper comparison failures:\n  " + "\n  ".join(failures), file=sys.stderr
        )
        return 2
    if skipped:
        print(f"sweeper comparison: {skipped} coverage gap(s) not run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
