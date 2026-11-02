# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One immutable registry and error taxonomy for all optional native profiles."""

import json
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType
from dynamo.llm.exceptions import HttpError, InvalidArgument
from dynamo.vllm import decision_cache_provenance as cache
from dynamo.vllm import decision_execution as causal
from dynamo.vllm import decision_heads
from dynamo.vllm import decision_native_errors as native_errors
from dynamo.vllm import decision_native_loader as native_loader
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_native_wire as native_wire
from dynamo.vllm import decision_reads, decision_runtime_policy
from dynamo.vllm.tests.decision_head_test_support import Engine, NativeApi

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def test_native_execution_error_is_a_typed_backend_protocol_failure():
    error = native_errors.NativeExecutionError("invalid native result")
    assert isinstance(error, HttpError)
    assert error.code == 502
    assert error.message == "invalid native result"


def test_native_request_validation_is_an_explicit_invalid_argument():
    error = native_errors.NativeValidationError("native input token budget exceeded")
    assert isinstance(error, InvalidArgument)
    assert str(error) == "native input token budget exceeded"


def test_native_loader_uses_the_causal_owned_source_verifier(monkeypatch):
    assert native_loader.verify_installed_sources is cache.verify_installed_sources
    monkeypatch.setattr(native_loader, "version", lambda _: "0.33.0")
    imports = Mock(side_effect=AssertionError("source verifier must run first"))
    monkeypatch.setattr(native_loader, "import_module", imports)
    verify = Mock(return_value=False)
    monkeypatch.setattr(native_loader, "verify_installed_sources", verify)
    with pytest.raises(native_errors.NativeRuntimeError, match="differs"):
        native_loader.load_native()
    verify.assert_called_once_with(native_loader.PINNED_SOURCE_BLOBS)
    imports.assert_not_called()


@pytest.mark.parametrize("failure", ["malformed-path", "import-value-error"])
def test_native_loader_rejects_real_source_lookup_failures(monkeypatch, failure):
    monkeypatch.setattr(native_loader, "version", lambda _: "0.33.0")

    def malformed_source(name):
        assert name == "vllm"
        if failure == "import-value-error":
            raise ValueError("optional source import failed")
        return SimpleNamespace(__file__="\0")

    monkeypatch.setattr(cache, "import_module", malformed_source)
    monkeypatch.setattr(native_loader, "import_module", malformed_source)
    with pytest.raises(native_errors.NativeRuntimeError, match="source"):
        native_loader.load_native()


@pytest.mark.parametrize(
    "owner,names",
    [
        ("loader", ("NativeApi", "load_native")),
        ("wire", ("NativeRequest", "BudgetTokenizer", "parse_native_request")),
        ("lifecycle", ("NativeAdapter", "run_children", "_create_strategy")),
        ("registry", ("NativeProfile", "publish_native_decision_capability")),
    ],
)
def test_native_contract_implementations_have_explicit_responsibility_owners(
    owner, names
):
    module = import_module(f"dynamo.vllm.decision_native_{owner}")
    for name in names:
        implementation = getattr(module, name)
        assert implementation.__module__ == module.__name__


def test_registry_is_the_single_architecture_task_and_strategy_owner():
    assert len(native_registry.PROFILES) == 5
    for name, profile in native_registry.PROFILES.items():
        assert profile.kind in ("head", "read")
        assert profile.architecture and profile.task and profile.strategy
        assert (
            native_registry.profile_for_architecture(
                profile.architecture, kind=profile.kind
            )
            == profile
        )
        other_kind = "read" if profile.kind == "head" else "head"
        assert (
            native_registry.profile_for_architecture(
                profile.architecture, kind=other_kind
            )
            is None
        )
    with pytest.raises(TypeError):
        native_registry.PROFILES["unqualified"] = None


def test_adapters_share_canonical_errors_without_identity_aliases():
    for module, prefix in ((decision_heads, "Head"), (decision_reads, "Read")):
        for kind in ("Runtime", "Validation", "Execution"):
            assert not hasattr(module, f"{prefix}{kind}Error")
            assert getattr(module, f"Native{kind}Error") is getattr(
                native_errors, f"Native{kind}Error"
            )


def test_rust_owned_v1_golden_preserves_question_contract(pinned_questions):
    fixture = (
        Path(__file__).resolve().parents[5]
        / "tests/fixtures/decisions/native-question-v1.json"
    )
    raw = json.loads(fixture.read_bytes())
    assert [q["id"] for q in raw["questions"]] == [None, "duplicate", "duplicate", ""]
    values = [c["value"] for c in raw["questions"][1]["candidates"]]
    assert [(type(value), value) for value in values] == [(bool, True), (str, "true")]
    api = NativeApi()
    api.questions = pinned_questions
    parsed = native_wire.parse_native_request(
        raw,
        native_registry.D3_PROFILE,
        api,
        SimpleNamespace(max_questions=64, max_options=255),
    )
    assert parsed.state == "Unicode λ state"
    assert [q.id for q in parsed.questions] == [f"question-{i}" for i in range(4)]
    assert [q.type.name for q in parsed.questions] == [
        "noul",
        "choice",
        "score",
        "noul",
    ]
    assert [o.name for o in parsed.questions[1].options] == ["true", '"true"']
    assert [o.name for o in parsed.questions[2].options] == ["low - Bad", "high"]


def _bounds_fixture():
    fixtures = Path(__file__).resolve().parents[5] / "tests/fixtures/decisions"
    return json.loads((fixtures / "native-bounds-v1.json").read_bytes())


def test_rust_owned_native_bounds_match_python_wire_constants():
    assert _bounds_fixture() == {
        "protocol_version": 1,
        "max_questions": native_wire.MAX_NATIVE_QUESTIONS,
        "max_candidates": native_wire.MAX_NATIVE_CANDIDATES,
        "max_input_tokens": native_wire.MAX_NATIVE_INPUT_TOKENS,
        "max_tokenization_work": native_wire.MAX_NATIVE_TOKENIZATION_WORK,
        "max_encoding_bytes": native_wire.MAX_NATIVE_ENCODING_BYTES,
    }


@pytest.mark.parametrize(
    "field", ["max_input_tokens", "max_tokenization_work", "max_encoding_bytes"]
)
def test_native_parser_accepts_exact_global_budget_and_rejects_one_more(
    pinned_questions, field
):
    fixture = Path(__file__).resolve().parents[5] / "tests/fixtures/decisions"
    raw = json.loads((fixture / "native-question-v1.json").read_bytes())
    api = NativeApi()
    api.questions = pinned_questions
    limits = SimpleNamespace(max_questions=64, max_options=255)
    maximum = _bounds_fixture()[field]
    raw[field] = maximum
    parsed = native_wire.parse_native_request(
        raw, native_registry.D3_PROFILE, api, limits
    )
    assert getattr(parsed, field) == maximum
    with pytest.raises(native_errors.NativeValidationError, match=field):
        native_wire.parse_native_request(
            {**raw, field: maximum + 1}, native_registry.D3_PROFILE, api, limits
        )


def test_wire_byte_budget_uses_compact_wrapper_without_changing_prompt_json(
    pinned_questions,
):
    prefix = '{"model":"m","state":['
    suffix = '],"questions":{"q":{"type":"noul","instructions":"x"}}}'
    count = (4 * 1024 * 1024 - len(prefix) - len(suffix) + 1) // 5
    body = prefix + ",".join(["1e15"] * count) + suffix
    assert len(body.encode()) <= 4 * 1024 * 1024
    fixture = Path(__file__).resolve().parents[5] / "tests/fixtures/decisions"
    raw = json.loads((fixture / "native-question-v1.json").read_bytes())
    raw = {
        **raw,
        "model": "m",
        "input": json.loads(body)["state"],
        "questions": [raw["questions"][0]],
        "cache_salt": "0" * 64,
    }
    limit = raw["max_encoding_bytes"]
    compact = json.dumps(
        {"decision_head": raw}, ensure_ascii=False, separators=(",", ":")
    )
    assert len(compact.encode()) < limit < len(native_wire._json(raw).encode())
    runtime = ModelRuntimeConfig()
    runtime.set_engine_specific("native_wire_size_probe", compact)
    rust_encoded = runtime.runtime_data["native_wire_size_probe"]
    assert len(rust_encoded.encode()) == len(compact.encode())
    api = NativeApi()
    api.questions = pinned_questions
    native_wire.parse_native_request(
        raw,
        native_registry.D3_PROFILE,
        api,
        SimpleNamespace(max_questions=64, max_options=255),
    )
    assert native_wire._text({"a": [1, 2]}) == '{"a": [1, 2]}'


@pytest.mark.parametrize(
    "profile,key",
    [
        (native_registry.D3_PROFILE, "decision_head"),
        (native_registry.LIQUID_PROFILE, "decision_read"),
    ],
)
def test_native_wire_budget_includes_its_selected_wrapper(
    pinned_questions, profile, key
):
    fixture = Path(__file__).resolve().parents[5] / "tests/fixtures/decisions"
    raw = json.loads((fixture / "native-question-v1.json").read_bytes())
    raw["profile"] = profile
    for _ in range(3):
        raw["max_encoding_bytes"] = len(native_wire._wire_json(raw, profile).encode())
    expected = json.dumps(
        {key: raw}, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    assert len(expected.encode()) == raw["max_encoding_bytes"]
    api = NativeApi()
    api.questions = pinned_questions
    limits = SimpleNamespace(max_questions=64, max_options=255)
    native_wire.parse_native_request(raw, profile, api, limits)
    with pytest.raises(
        native_errors.NativeValidationError, match="encoding byte budget"
    ):
        native_wire.parse_native_request(
            {**raw, "max_encoding_bytes": raw["max_encoding_bytes"] - 1},
            profile,
            api,
            limits,
        )


def test_native_wire_budget_rejects_an_unknown_internal_profile():
    with pytest.raises(native_errors.NativeExecutionError, match="wire profile"):
        native_wire._wire_json({}, "unknown")


@pytest.mark.parametrize(
    "value", [float("nan"), object()], ids=["nonfinite", "not-json"]
)
def test_native_wire_budget_still_rejects_invalid_json(value):
    with pytest.raises(native_errors.NativeValidationError, match="valid JSON"):
        native_wire._wire_json({"input": value}, native_registry.D3_PROFILE)


@pytest.mark.parametrize("field", ["max_questions", "max_candidates"])
def test_native_parser_enforces_cross_language_collection_bounds(
    pinned_questions, field
):
    fixture = Path(__file__).resolve().parents[5] / "tests/fixtures/decisions"
    raw = json.loads((fixture / "native-question-v1.json").read_bytes())
    api = NativeApi()
    api.questions = pinned_questions
    limits = SimpleNamespace(max_questions=1000, max_options=1000)
    maximum = _bounds_fixture()[field]
    for count in (maximum, maximum + 1):
        if field == "max_questions":
            questions = [{**raw["questions"][0], "ordinal": i} for i in range(count)]
        else:
            questions = [
                {
                    "ordinal": 0,
                    "kind": "Choice",
                    "candidates": [{"value": str(i)} for i in range(count)],
                }
            ]
        request = {**raw, "questions": questions}
        if count == maximum:
            parsed = native_wire.parse_native_request(
                request, native_registry.D3_PROFILE, api, limits
            )
            assert parsed.questions
        else:
            with pytest.raises(native_errors.NativeValidationError, match="count"):
                native_wire.parse_native_request(
                    request, native_registry.D3_PROFILE, api, limits
                )


@pytest.mark.parametrize("cached", [None, True, -1, 5, 1.0])
def test_native_cache_measurement_is_required_and_bounded(cached):
    with pytest.raises(native_errors.NativeExecutionError) as failure:
        native_wire.measured_cache_tokens(cached, 4)
    assert failure.value.code == 502


@pytest.mark.parametrize("cached,expected", [(0, 2), (3, 5)])
def test_named_usage_sums_measured_work_without_inventing_cache(cached, expected):
    rows = [
        native_wire.NativeUsage(input_tokens=4, output_tokens=0, cached_tokens=2),
        native_wire.NativeUsage(input_tokens=8, output_tokens=1, cached_tokens=cached),
    ]
    assert native_wire.aggregate_usage(rows) == {
        "input_tokens": 12,
        "output_tokens": 1,
        "cached_tokens": expected,
    }


def test_unknown_native_architecture_has_no_profile():
    assert (
        native_registry.profile_for_architecture("UnqualifiedModel", kind="head")
        is None
    )
    assert (
        native_registry.profile_for_architecture("UnqualifiedModel", kind="read")
        is None
    )


@pytest.mark.parametrize(
    "installed",
    [
        "0.33.0rc1",
        "0.33.0rc2",
        "0.33.0rc1.dev5+g123",
        "0.33.0.dev1+g4ba787ec9",
        "0.33.0.post1",
        "1!0.33.0",
        "0.32.0",
    ],
)
def test_unqualified_release_rejected_before_native_imports_or_blob_reads(
    monkeypatch, installed
):
    monkeypatch.setattr(native_loader, "version", lambda name: installed)
    imports = Mock(side_effect=AssertionError("must not import native runtime"))
    monkeypatch.setattr(native_loader, "import_module", imports)
    reads = Mock(side_effect=AssertionError("must not read native source"))
    monkeypatch.setattr(cache.Path, "read_bytes", reads)
    with pytest.raises(native_errors.NativeRuntimeError, match="0.33.0"):
        native_loader.load_native()
    imports.assert_not_called()
    reads.assert_not_called()


@pytest.mark.parametrize("installed", ["0.33.0", "0.33.0+cu130"])
def test_exact_tag_and_local_wheel_suffix_reach_semantic_source_gate(
    monkeypatch, installed
):
    monkeypatch.setattr(native_loader, "version", lambda name: installed)
    monkeypatch.setattr(
        cache, "import_module", lambda name: SimpleNamespace(__file__=None)
    )
    with pytest.raises(native_errors.NativeRuntimeError, match="source is unavailable"):
        native_loader.load_native()


def test_installed_stable_runtime_matches_pinned_sources():
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        pytest.skip("Optional vLLM runtime is not installed")
    if not causal.is_exact_vllm_release(installed, "0.33.0"):
        pytest.skip("Optional stable vLLM 0.33.0 runtime is not installed")
    api = native_loader.load_native()
    assert isinstance(api, native_loader.NativeApi)
    assert api.template_error is import_module("jinja2").TemplateError
    assert (
        api.retryable_request_error
        is import_module("vllm.exceptions").RetryableRequestError
    )


def test_native_runtime_without_source_path_fails_closed(monkeypatch):
    monkeypatch.setattr(native_loader, "version", lambda name: "0.33.0")
    monkeypatch.setattr(
        cache, "import_module", lambda name: SimpleNamespace(__file__=None)
    )
    with pytest.raises(native_errors.NativeRuntimeError, match="source is unavailable"):
        native_loader.load_native()


@pytest.mark.parametrize(
    "name",
    [
        native_registry.D3_PROFILE,
        native_registry.CLEF_FLASH_PROFILE,
        native_registry.CLEF_OMNI_PROFILE,
    ],
)
def test_typed_native_head_metadata_includes_actual_candidate_limit(name):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    profile = native_registry.PROFILES[name]
    assert native_registry.publish_native_decision_capability(
        runtime,
        native_registry.NativeReady(profile),
        ModelInput.Text,
        ModelType.Classify | ModelType.Pooling,
        WorkerType.Aggregated,
    )
    runtime.set_engine_specific.assert_any_call(name, "true")
    assert runtime.set_engine_specific.call_count == 2
    runtime.set_engine_specific.assert_any_call(
        "vllm_native_decision_max_candidates", str(profile.max_candidates)
    )


@pytest.mark.parametrize(
    "profile", native_registry.PROFILES.values(), ids=lambda profile: profile.name
)
def test_failed_native_qualification_publishes_only_a_typed_failure(profile):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    publication = native_registry.NativeFailed(profile)
    assert (
        native_registry.publish_native_decision_capability(
            runtime,
            publication,
            ModelInput.Text if profile.kind == "head" else ModelInput.Tokens,
            (
                ModelType.Classify | ModelType.Pooling
                if profile.kind == "head"
                else ModelType.Chat
            ),
            WorkerType.Aggregated,
        )
        is publication
    )
    runtime.set_engine_specific.assert_called_once_with(
        "vllm_native_decision_qualification_failure_v1",
        json.dumps({"profile": profile.name}),
    )


def test_native_failure_marker_crosses_the_real_runtime_metadata_binding():
    runtime = ModelRuntimeConfig()
    profile = native_registry.PROFILES[native_registry.D3_PROFILE]
    publication = native_registry.NativeFailed(profile)
    assert (
        native_registry.publish_native_decision_capability(
            runtime,
            publication,
            ModelInput.Text,
            ModelType.Classify | ModelType.Pooling,
            WorkerType.Aggregated,
        )
        is publication
    )
    assert {key: json.loads(value) for key, value in runtime.runtime_data.items()} == {
        "vllm_native_decision_qualification_failure_v1": {"profile": profile.name}
    }


@pytest.mark.parametrize(
    "failure",
    [
        None,
        native_registry.NativeProfile(
            "unknown", "head", "Unknown", "classify", "Unknown"
        ),
    ],
)
def test_no_native_failure_marker_without_a_recognized_profile(failure):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert not native_registry.publish_native_decision_capability(
        runtime,
        native_registry.NativeFailed(failure) if failure is not None else None,
        ModelInput.Text,
        ModelType.Classify | ModelType.Pooling,
        WorkerType.Aggregated,
    )
    runtime.set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "profile,model_input,model_type,worker,needs",
    [
        (
            None,
            ModelInput.Text,
            ModelType.Classify | ModelType.Pooling,
            WorkerType.Aggregated,
            None,
        ),
        (
            native_registry.PROFILES[native_registry.D3_PROFILE],
            ModelInput.Text,
            ModelType.Classify | ModelType.Pooling,
            WorkerType.Prefill,
            None,
        ),
        (
            native_registry.PROFILES[native_registry.D3_PROFILE],
            ModelInput.Text,
            ModelType.Classify | ModelType.Pooling,
            WorkerType.Aggregated,
            [[WorkerType.Encode]],
        ),
        (
            native_registry.PROFILES[native_registry.D3_PROFILE],
            ModelInput.Tokens,
            ModelType.Classify | ModelType.Pooling,
            WorkerType.Aggregated,
            None,
        ),
        (
            native_registry.PROFILES[native_registry.D3_PROFILE],
            ModelInput.Text,
            ModelType.Chat,
            WorkerType.Aggregated,
            None,
        ),
        (
            native_registry.PROFILES[native_registry.LIQUID_PROFILE],
            ModelInput.Text,
            ModelType.Classify | ModelType.Pooling,
            WorkerType.Aggregated,
            None,
        ),
    ],
)
def test_native_metadata_is_absent_on_ineligible_surface(
    profile, model_input, model_type, worker, needs
):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert not native_registry.publish_native_decision_capability(
        runtime,
        native_registry.NativeReady(profile) if profile is not None else None,
        model_input,
        model_type,
        worker,
        needs,
    )
    runtime.set_engine_specific.assert_not_called()
    assert not native_registry.publish_native_decision_capability(
        runtime,
        native_registry.NativeFailed(profile) if profile is not None else None,
        model_input,
        model_type,
        worker,
        needs,
    )
    runtime.set_engine_specific.assert_not_called()


def test_native_success_and_failure_cannot_be_published_together():
    runtime = SimpleNamespace(set_engine_specific=Mock())
    profile = native_registry.PROFILES[native_registry.D3_PROFILE]
    for variant in (native_registry.NativeReady, native_registry.NativeFailed):
        with pytest.raises(TypeError):
            variant(profile, failed_profile=profile)
    with pytest.raises(TypeError, match="native publication"):
        native_registry.publish_native_decision_capability(
            runtime,
            profile,
            ModelInput.Text,
            ModelType.Classify | ModelType.Pooling,
            WorkerType.Aggregated,
        )
    runtime.set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "model_type", [ModelType.Completions, ModelType.Completions | ModelType.Classify]
)
def test_completions_combined_worker_publishes_native_read_profile(model_type):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    assert native_registry.publish_native_decision_capability(
        runtime,
        native_registry.NativeReady(
            native_registry.PROFILES[native_registry.LIQUID_PROFILE]
        ),
        ModelInput.Tokens,
        model_type,
        WorkerType.Aggregated,
    )
    runtime.set_engine_specific.assert_any_call(native_registry.LIQUID_PROFILE, "true")
    runtime.set_engine_specific.assert_any_call(
        "vllm_native_decision_max_candidates", "26"
    )


@pytest.mark.parametrize(
    "scheduler",
    [
        "custom.Plugin",
        type("Plugin", (), {}),
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
    ],
)
@pytest.mark.asyncio
async def test_adapter_rejects_custom_or_mismatched_scheduler(native, scheduler):
    engine = Engine()
    engine.vllm_config.scheduler_config.scheduler_cls = scheduler
    engine.vllm_config.scheduler_config.async_scheduling = False
    with pytest.raises(native_errors.NativeRuntimeError, match="effective runtime"):
        await decision_heads.HeadExecutionAdapter.create(
            engine,
            native_registry.PROFILES[native_registry.D3_PROFILE],
            abort_monitor=Mock(),
            max_concurrency=1,
        )


@pytest.mark.parametrize("async_mode", [None, "auto", 1])
@pytest.mark.asyncio
async def test_adapter_rejects_unresolved_async_scheduler_mode(native, async_mode):
    engine = Engine()
    engine.vllm_config.scheduler_config.scheduler_cls = None
    engine.vllm_config.scheduler_config.async_scheduling = async_mode
    with pytest.raises(native_errors.NativeRuntimeError, match="effective runtime"):
        await decision_heads.HeadExecutionAdapter.create(
            engine,
            native_registry.PROFILES[native_registry.D3_PROFILE],
            abort_monitor=Mock(),
            max_concurrency=1,
        )


@pytest.mark.parametrize(
    "diffusion,async_mode,scheduler",
    [
        (False, False, None),
        (False, True, None),
        (False, False, "vllm.v1.core.sched.scheduler.Scheduler"),
        (False, True, "vllm.v1.core.sched.async_scheduler.AsyncScheduler"),
        (True, False, "vllm.v1.core.sched.diffusion_scheduler.DiffusionScheduler"),
        (True, True, "vllm.v1.core.sched.diffusion_scheduler.DiffusionAsyncScheduler"),
    ],
)
def test_canonical_runtime_accepts_pinned_default_scheduler_variants(
    diffusion, async_mode, scheduler
):
    engine = Engine()
    engine.model_config.is_diffusion = diffusion
    engine.vllm_config.scheduler_config.scheduler_cls = scheduler
    engine.vllm_config.scheduler_config.async_scheduling = async_mode
    assert decision_runtime_policy.qualified_runtime(
        engine.vllm_config,
        runner_type="pooling",
        diffusion=diffusion,
        min_model_len=1,
    )


@pytest.mark.asyncio
async def test_adapter_checks_scheduler_once_before_strategy_initialization(
    native, monkeypatch
):
    engine = Engine()
    check = Mock(wraps=decision_runtime_policy.pinned_default_scheduler)
    monkeypatch.setattr(decision_runtime_policy, "pinned_default_scheduler", check)
    adapter = await decision_heads.HeadExecutionAdapter.create(
        engine,
        native_registry.PROFILES[native_registry.D3_PROFILE],
        abort_monitor=Mock(),
        max_concurrency=1,
    )
    check.assert_called_once_with(engine.vllm_config)
    assert adapter.concurrency == 1


def test_native_diffusion_schedulers_share_standard_cache_projection():
    pytest.importorskip("vllm")
    if not causal.is_exact_vllm_release(version("vllm"), "0.33.0"):
        pytest.skip("Optional stable vLLM 0.33.0 runtime is not installed")
    native_loader.load_native()
    from vllm.v1.core.sched.diffusion_scheduler import (
        DiffusionAsyncScheduler,
        DiffusionScheduler,
    )
    from vllm.v1.core.sched.scheduler import Scheduler

    assert DiffusionScheduler.update_from_output is Scheduler.update_from_output
    assert DiffusionAsyncScheduler.update_from_output is Scheduler.update_from_output


@pytest.mark.parametrize("supplied", [True, False])
def test_actual_output_processor_preserves_first_prefill_cache_evidence(supplied):
    pytest.importorskip("vllm")
    if not causal.is_exact_vllm_release(version("vllm"), "0.33.0"):
        pytest.skip("Optional stable vLLM 0.33.0 runtime is not installed")
    native_loader.load_native()
    import torch
    from vllm import PoolingParams
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.request import Request

    params = PoolingParams(task="classify")
    request = Request("cache-evidence", [1, 2, 3], None, params)
    request.prefill_stats.set(
        num_prompt_tokens=3, num_local_cached_tokens=2, num_external_cached_tokens=0
    )
    stats = request.take_prefill_stats()
    assert stats.num_cached_tokens == 2
    assert request.take_prefill_stats() is None
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    core_request = EngineCoreRequest(
        request_id="cache-evidence",
        external_req_id="cache-evidence",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=None,
        pooling_params=params,
        arrival_time=0,
        lora_request=None,
        cache_salt="isolated",
        data_parallel_rank=None,
    )
    processor.add_request(core_request, prompt=None)
    output = EngineCoreOutput(
        request_id="cache-evidence",
        new_token_ids=[],
        pooling_output=torch.tensor([0.2, 0.8]),
        finish_reason=FinishReason.STOP,
        prefill_stats=stats if supplied else None,
    )
    result = processor.process_outputs([output]).request_outputs[0]
    # Missing first-frame evidence really becomes upstream's ambiguous default.
    # Only the source-pinned standard scheduler establishes its provenance.
    assert result.num_cached_tokens == (2 if supplied else 0)
