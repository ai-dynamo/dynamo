# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from aisimulate.output_adapter import validate_output_adapter
from aisimulate.sweeper.result import SweepResult
from pydantic import ValidationError

from dynamo.aisimulate.output.dgd import adapter as adapter_module
from dynamo.aisimulate.output.dgd.adapter import DGDOutputConfig, create_adapter
from dynamo.aisimulate.output.dgd.renderers import CandidateMaterializationError

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


class _Candidate:
    def __init__(self, score: float) -> None:
        self.config = {"backend": "vllm", "backend_version": "0.20.1"}
        self.used_gpus = 2
        self.score = score


def _config(**overrides):
    config = {
        "name": "qwen",
        "output_file": "deployment.yaml",
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "num_gpus_per_node": 8,
    }
    config.update(overrides)
    return config


def _result(candidates, *, pareto: bool = False):
    return SimpleNamespace(
        selected_candidates=candidates,
        views=SimpleNamespace(pareto_front=["candidate-000001"] if pareto else []),
        provenance=SimpleNamespace(config={}),
    )


def _real_scalar_result() -> SweepResult:
    smart_config = {
        "search_space": {
            "model_name": "Qwen/Qwen3-32B",
            "deployment_mode": ["agg"],
            "backend": ["vllm"],
            "hardware_sku": "h200_sxm",
            "gpu_budget": 8,
            "context_length": 131072,
        },
        "workload": {
            "isl": 1024,
            "osl": 1024,
            "request_rate": 10,
            "num_request_ratio": 10,
        },
        "goal": {
            "target": "goodput_per_gpu",
            "sla": {"ttft_ms": 2000, "itl_ms": 25},
        },
        "sweep": {
            "max_rounds": 1,
            "candidates_per_round": 2,
            "parallel_evals": 1,
            "max_eval_seconds": 120,
        },
    }

    def candidate_record(candidate_id: str, score: float, used_gpus: int):
        return {
            "candidate_id": candidate_id,
            "status": "feasible",
            "config": {"backend": "vllm", "backend_version": "0.20.1"},
            "used_gpus": used_gpus,
            "score": score,
            "metrics": {},
            "provenance": {
                "model": "Qwen/Qwen3-32B",
                "hardware": "h200_sxm",
                "backend": "vllm",
            },
        }

    candidate_records = [
        candidate_record("candidate-000001", score=2.0, used_gpus=2),
        candidate_record("candidate-000002", score=1.0, used_gpus=4),
    ]
    return SweepResult.model_validate(
        {
            "counts": {
                "evaluated": 2,
                "feasible": 2,
                "infeasible": 0,
                "unsupported": 0,
                "timed_out": 0,
                "failed": 0,
            },
            "candidates": candidate_records,
            "views": {
                "top_n": ["candidate-000002", "candidate-000001"],
            },
            "provenance": {
                "search_strategy": "optimizer_guided",
                "run_id": "adapter-test",
                "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                "input_fingerprint": "sha256:adapter-test",
                "config": smart_config,
            },
        }
    )


def _rendered_dgd(name: str, score: float) -> str:
    return f"""apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: {name}
  annotations:
    test-score: "{score}"
spec:
  components: []
"""


def test_adapter_matches_aisimulate_contract_and_defaults() -> None:
    adapter = create_adapter()
    config = DGDOutputConfig.model_validate(_config())

    assert validate_output_adapter(adapter, requested_name="dgd") is adapter
    assert config.generator == "aic"
    assert config.format == "manifest"
    assert config.output_file == "deployment.yaml"
    assert config.output_dir is None
    assert config.output_format == "dgd"


def test_adapter_subscription_validates_config_before_search() -> None:
    adapter = create_adapter()

    assert adapter.subscribe(_config()) is None
    with pytest.raises(ValidationError, match="num_gpus_per_node"):
        adapter.subscribe(_config(num_gpus_per_node=0))
    with pytest.raises(ValueError, match="canonical MAJOR.MINOR.PATCH"):
        adapter.subscribe(
            _config(runtime_image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest")
        )


def test_adapter_rejects_unknown_dgd_fields() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        DGDOutputConfig.model_validate(_config(unknown=True))


def test_adapter_requires_output_file() -> None:
    config = _config()
    del config["output_file"]
    with pytest.raises(ValidationError, match="output_file"):
        DGDOutputConfig.model_validate(config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            _config(output_dir="deployment"),
            "output_dir is only valid for format kustomize",
        ),
        (_config(format="kustomize"), "output_dir is required for format kustomize"),
        (
            _config(format="kustomize", output_dir="deployment"),
            "output_file is only valid for format manifest",
        ),
    ],
)
def test_output_target_must_match_format(config, message) -> None:
    with pytest.raises(ValidationError, match=message):
        DGDOutputConfig.model_validate(config)


def test_kustomize_requires_only_output_dir() -> None:
    config = DGDOutputConfig.model_validate(
        _config(format="kustomize", output_file=None, output_dir="deployment")
    )

    assert config.output_file is None
    assert config.output_dir == "deployment"
    assert config.output_format == "kustomize"


def test_adapter_requires_name() -> None:
    config = _config()
    del config["name"]
    with pytest.raises(ValidationError, match="name"):
        DGDOutputConfig.model_validate(config)


@pytest.mark.parametrize(
    "output_file", ["deployment", "../deployment.yaml", "{name}.yaml"]
)
def test_adapter_rejects_invalid_output_file(output_file) -> None:
    with pytest.raises(ValidationError, match="YAML filename"):
        DGDOutputConfig.model_validate(_config(output_file=output_file))


@pytest.mark.parametrize("output_dir", [".", "..", "../deployment", "{name}"])
def test_adapter_rejects_invalid_output_dir(output_dir) -> None:
    with pytest.raises(ValidationError, match="path separators|templated"):
        DGDOutputConfig.model_validate(
            _config(format="kustomize", output_file=None, output_dir=output_dir)
        )


@pytest.mark.parametrize(
    ("pareto", "config", "message"),
    [
        (False, _config(name="qwen-{index}"), "only valid for Pareto"),
        (
            True,
            _config(output_file="candidate-{index}.yaml"),
            r"require \{index\} in dgd.name",
        ),
    ],
)
def test_name_form_must_match_result_view(
    monkeypatch, tmp_path, pareto, config, message
) -> None:
    monkeypatch.setattr(adapter_module, "_workload", lambda _result: object())
    with pytest.raises(ValueError, match=message):
        create_adapter().write(
            config,
            result=_result([_Candidate(1.0)], pareto=pareto),
            output_dir=tmp_path,
        )


def test_scalar_respects_aisimulate_selection_order(monkeypatch, tmp_path) -> None:
    # The selected view is authoritative even if local score ranking would differ.
    result = _real_scalar_result()
    rendered_scores = []

    def fake_render(candidate, workload, options, *, dgd_name, renderer):
        assert workload.isl == 1024
        assert workload.osl == 1024
        assert options.dynamo_runtime_version == "1.5.0"
        assert dgd_name == "qwen"
        assert renderer == "aic"
        rendered_scores.append(candidate.score)
        return _rendered_dgd(dgd_name, candidate.score)

    monkeypatch.setattr(adapter_module, "render_dgd", fake_render)
    artifacts = create_adapter().write(
        _config(),
        result=result,
        output_dir=tmp_path,
    )

    assert artifacts == [Path("deployment.yaml"), Path("index.json")]
    assert rendered_scores == [1.0]
    assert yaml.safe_load((tmp_path / "deployment.yaml").read_text())["kind"] == (
        "DynamoGraphDeployment"
    )


@pytest.mark.parametrize(
    ("pareto", "config", "message"),
    [
        (
            False,
            _config(output_file="candidate-{index}.yaml"),
            "only valid for Pareto",
        ),
        (
            True,
            _config(name="qwen-{index}", output_file="candidate.yaml"),
            r"require \{index\} in dgd.output_file",
        ),
        (
            False,
            _config(
                format="kustomize",
                output_file=None,
                output_dir="candidate-{index}",
            ),
            "only valid for Pareto",
        ),
        (
            True,
            _config(
                name="qwen-{index}",
                format="kustomize",
                output_file=None,
                output_dir="candidate",
            ),
            r"require \{index\} in dgd.output_dir",
        ),
    ],
)
def test_output_target_form_must_match_result_view(
    monkeypatch, tmp_path, pareto, config, message
) -> None:
    monkeypatch.setattr(adapter_module, "_workload", lambda _result: object())
    with pytest.raises(ValueError, match=message):
        create_adapter().write(
            config,
            result=_result([_Candidate(1.0)], pareto=pareto),
            output_dir=tmp_path,
        )


def test_pareto_writes_every_selected_candidate(monkeypatch, tmp_path) -> None:
    candidates = [_Candidate(2.0), _Candidate(1.0)]
    monkeypatch.setattr(adapter_module, "_workload", lambda _result: "workload")
    monkeypatch.setattr(
        adapter_module,
        "render_dgd",
        lambda candidate, _workload, _options, *, dgd_name, renderer: _rendered_dgd(
            dgd_name, candidate.score
        ),
    )

    artifacts = create_adapter().write(
        _config(
            name="qwen-pareto-{index}",
            format="kustomize",
            output_file=None,
            output_dir="candidate-{index}",
        ),
        result=_result(candidates, pareto=True),
        output_dir=tmp_path,
    )

    assert artifacts == [
        Path("candidate-000"),
        Path("candidate-001"),
        Path("index.json"),
    ]
    for index in range(2):
        source = tmp_path / f"candidate-{index:03d}"
        assert yaml.safe_load((source / "deploy.yaml").read_text())["kind"] == (
            "DynamoGraphDeployment"
        )
        assert yaml.safe_load((source / "kustomization.yaml").read_text())[
            "resources"
        ] == ["deploy.yaml"]


def test_pareto_materialization_is_atomic(monkeypatch, tmp_path) -> None:
    candidates = [_Candidate(2.0), _Candidate(1.0)]
    monkeypatch.setattr(adapter_module, "_workload", lambda _result: "workload")

    def fake_render(candidate, _workload, _options, *, dgd_name, renderer):
        if candidate.score == 1.0:
            raise CandidateMaterializationError("candidate cannot become a DGD")
        return _rendered_dgd(dgd_name, candidate.score)

    monkeypatch.setattr(adapter_module, "render_dgd", fake_render)
    with pytest.raises(
        CandidateMaterializationError, match="candidate cannot become a DGD"
    ):
        create_adapter().write(
            _config(
                name="qwen-pareto-{index}",
                output_file="candidate-{index}.yaml",
            ),
            result=_result(candidates, pareto=True),
            output_dir=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_adapter_rejects_empty_selection(tmp_path) -> None:
    with pytest.raises(CandidateMaterializationError, match="no feasible candidate"):
        create_adapter().write(
            _config(),
            result=_result([]),
            output_dir=tmp_path,
        )
