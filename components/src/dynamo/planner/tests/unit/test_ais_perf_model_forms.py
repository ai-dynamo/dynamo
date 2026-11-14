# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two forms of the Planner's ``ais_perf_model`` value.

The old form is a mapping with ``roles`` and ``dataset_profiles``. The new
form is one perf model config or a list of configs with one ``worker_type``
each.
"""

import json
import re
import sys
import warnings
from dataclasses import asdict, fields

import pytest
import yaml
from aisimulate_core.sdk import ForwardPassPerfModelConfig
from pydantic import ValidationError

from dynamo.planner import __main__ as planner_main
from dynamo.planner.config import planner_config
from dynamo.planner.config.planner_config import (
    AISPerfModelSpec,
    PlannerConfig,
    ais_perf_model_old_form_names,
)
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.tests.unit.test_ais_perf_adapter import (
    _caps,
    _FakeEngineModel,
    _install_fake_engine,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_CONTENT = {"model": "test-model", "system": "test-system", "backend": "vllm"}
_DATASET_PROFILE = {"root": "/not-mounted-yet/dataset", "manifest": "profile.json"}
# The Planner mode that scales only workers of one type.
_MODE_OF = {"prefill": "prefill", "decode": "decode", "aggregated": "agg"}
_NEW_FORM = (
    "ais_perf_model with one perf model config or a list of configs "
    "(one per worker_type)"
)


def _planner(ais_perf_model, mode="decode", optimization_target="sla"):
    return PlannerConfig(
        mode=mode,
        optimization_target=optimization_target,
        ais_perf_model=ais_perf_model,
    )


def _load_failure(ais_perf_model, optimization_target="sla"):
    """Return the errors loading a config with this value raises."""
    with pytest.raises(ValidationError) as caught:
        _planner(ais_perf_model, optimization_target=optimization_target)
    assert type(caught.value) is ValidationError
    return caught.value.errors()


def _load_error(ais_perf_model, optimization_target="sla"):
    """Return the one exception reading this value raises when a config is loaded."""
    (error,) = _load_failure(ais_perf_model, optimization_target)
    assert (error["loc"], error["type"]) == (("ais_perf_model",), "value_error")
    return error["ctx"]["error"]


def _mode_check_error(ais_perf_model, mode="decode"):
    """Return the one exception the check that knows the mode raises at load."""
    with pytest.raises(ValidationError) as caught:
        _planner(ais_perf_model, mode=mode)
    (error,) = caught.value.errors()
    assert (error["loc"], error["type"]) == ((), "value_error")
    return error["ctx"]["error"]


def _roles_form_error(content, worker_type="decode"):
    """Return the exception the old ``roles`` form raises at load for this config."""
    with pytest.raises(ValidationError) as caught:
        _planner({"roles": {worker_type: content}}, mode=_MODE_OF[worker_type])
    (error,) = caught.value.errors()
    assert error["loc"] == ("ais_perf_model", "roles")
    return error["ctx"]["error"]


@pytest.fixture
def adapter(fake_engine_factory):
    """Build the Planner's perf model adapter for a config.

    The tests read only the config the adapter builds, so the engine model is
    a stand-in. With ``limits=False`` the adapter is built without reading
    the perf model, as when the workers have not reported their limits yet.
    """
    _install_fake_engine(
        fake_engine_factory, _FakeEngineModel(diagnostics={"readiness": "ready"})
    )

    def build(config, worker_type="decode", limits=True):
        return PlannerEnginePerfModel(
            worker_type=worker_type,
            config=config,
            capabilities=_caps() if limits else None,
        )

    return build


def test_old_form_roles_stays_typed():
    config = _planner({"roles": {"decode": {**_CONTENT, "tp": 2}}})

    assert isinstance(config.ais_perf_model, AISPerfModelSpec)
    expected = asdict(
        ForwardPassPerfModelConfig(**_CONTENT, tp=2, worker_type="decode")
    )
    assert config.ais_perf_model.roles == {"decode": expected}
    assert config.model_dump()["ais_perf_model"] == {
        "roles": {"decode": expected},
        "dataset_profiles": {},
    }


def test_old_form_dataset_profiles_stays_typed():
    config = _planner({"dataset_profiles": {"decode": _DATASET_PROFILE}})

    assert isinstance(config.ais_perf_model, AISPerfModelSpec)
    assert config.model_dump()["ais_perf_model"] == {
        "roles": {},
        "dataset_profiles": {"decode": _DATASET_PROFILE},
    }


def test_old_form_mode_check_text():
    text = (
        "ais_perf_model.roles.prefill or dataset_profiles.prefill "
        "is required for mode='disagg'"
    )
    with pytest.raises(ValidationError, match=re.escape(text)):
        _planner({"roles": {"decode": _CONTENT}}, mode="disagg")


def test_old_form_empty_mapping_fails_mode_check():
    text = (
        "ais_perf_model.roles.prefill or dataset_profiles.prefill "
        "is required for mode='disagg'"
    )
    with pytest.raises(ValidationError, match=re.escape(text)):
        _planner({}, mode="disagg")


def test_old_form_same_role_twice_is_rejected():
    with pytest.raises(
        ValidationError,
        match="each AIS role must have exactly one configuration source",
    ):
        _planner(
            {
                "roles": {"decode": _CONTENT},
                "dataset_profiles": {"decode": _DATASET_PROFILE},
            }
        )


def test_one_config_is_stored_as_written():
    written = {
        **_CONTENT,
        "tp": 2,
        "estimator_config": {"correction": {"enabled": False}},
    }

    config = _planner(written, mode="agg")

    assert config.ais_perf_model == written
    assert config.ais_perf_model is not written
    assert config.ais_perf_model["estimator_config"] is not written["estimator_config"]


def test_list_is_stored_as_written():
    written = [
        {**_CONTENT, "worker_type": "prefill"},
        {"worker_type": "decode", "from": "decode.yaml"},
    ]

    config = _planner(written, mode="disagg")

    assert config.ais_perf_model == written
    assert config.ais_perf_model is not written
    assert config.ais_perf_model[0] is not written[0]


def test_loader_syntax_errors_reach_the_user():
    # A list entry without a worker type: one of the errors of the loader
    # that the tests of dynamo.common.ais_perf_model list.
    error = _load_error([_CONTENT])

    # The loader's own error class.
    assert isinstance(error, ValueError)
    assert type(error) is not ValueError


def test_new_form_mode_check():
    text = (
        "ais_perf_model: mode='disagg' needs an entry for worker type 'prefill'; "
        "it has: decode"
    )
    with pytest.raises(ValidationError, match=re.escape(text)):
        _planner([{**_CONTENT, "worker_type": "decode"}], mode="disagg")

    for mode in ("agg", "prefill", "decode", "disagg"):
        assert _planner(dict(_CONTENT), mode=mode).ais_perf_model == _CONTENT

    text = (
        "ais_perf_model: mode='agg' needs an entry for worker type 'aggregated'; "
        "it has: prefill"
    )
    with pytest.raises(ValidationError, match=re.escape(text)):
        _planner({**_CONTENT, "worker_type": "prefill"}, mode="agg")


def test_validation_opens_no_file():
    written = {"from": "/missing/template.yaml"}

    assert _planner(written).ais_perf_model == written


@pytest.mark.parametrize("worker_type", ["prefill", "decode", "aggregated"])
def test_adapter_config_equals_old_form(adapter, worker_type):
    content = {
        "model": "Qwen/Qwen3-0.6B",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "tp": 2,
        "attention_dp": 1,
        "estimator_config": {"correction": {"enabled": False}},
    }
    mode = _MODE_OF[worker_type]

    def built(ais_perf_model):
        config = _planner(ais_perf_model, mode=mode)
        return adapter(config, worker_type)._build_ais_config()

    old_form = built({"roles": {worker_type: content}})

    assert old_form["estimator_config"]["correction"]["enabled"] is False
    assert built([{"worker_type": worker_type, **content}]) == old_form
    assert built(dict(content)) == old_form


def test_adapter_resolves_once(adapter, tmp_path):
    template = tmp_path / "template.yaml"
    controls = {"correction": {"sampling": {"max_observations": 256}}}
    template.write_text(
        yaml.safe_dump({**_CONTENT, "attention_dp": 2, "estimator_config": controls})
    )
    model = adapter(_planner({"from": str(template)}))

    def read():
        return (
            model._build_ais_config(),
            model._attention_dp_size(),
            model._observation_history_limit(),
        )

    first = read()
    assert first[1:] == (2, 256)

    # A file that changes after the first use is not read again.
    template.write_text(yaml.safe_dump({**_CONTENT, "attention_dp": 4}))

    assert read() == first


def test_adapter_reads_template_entry(adapter, tmp_path):
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({**_CONTENT, "tp": 2, "pp": 2}))
    config = _planner({"from": str(template), "pp": 1})

    built = adapter(config)._build_ais_config()

    assert "from" not in built
    assert {key: built[key] for key in _CONTENT} == _CONTENT
    assert built["tp"] == 2
    # A key of the entry wins over the same key of the template.
    assert built["pp"] == 1
    assert built["worker_type"] == "decode"


def test_adapter_unknown_key_text(adapter, tmp_path):
    # A key written in the config is checked when the config is loaded; a key
    # of a template file is seen only when the file is read.
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({**_CONTENT, "tp_size": 2}))
    config = _planner({"from": str(template)})
    model = adapter(config, limits=False)

    with pytest.raises(ValueError) as caught:
        model._configured_role()

    assert type(caught.value) is ValueError
    assert str(caught.value).startswith("invalid AIS config for decode: ")


# Values AISimulate rejects. The last one names a directory that does not exist.
_WRONG_VALUES = [
    pytest.param({"tp": 0}, id="tp-zero"),
    pytest.param({"tp": "x"}, id="tp-text"),
    pytest.param({"backend": "nope"}, id="backend"),
    pytest.param({"estimation_mode": "nope"}, id="estimation-mode"),
    pytest.param({"fallback_policy": "nope"}, id="fallback-policy"),
    pytest.param({"nextn": -1}, id="nextn"),
    pytest.param({"estimator_config": {"nope": 1}}, id="estimator-config"),
    pytest.param({"systems_paths": ["/missing/systems"]}, id="systems-path"),
]


@pytest.mark.parametrize("wrong", _WRONG_VALUES)
@pytest.mark.parametrize("as_list", [False, True], ids=["one-config", "list-entry"])
def test_config_without_from_is_checked_when_loaded_like_roles(wrong, as_list):
    content = {**_CONTENT, **wrong}
    expected = _roles_form_error(content)
    value = [{**content, "worker_type": "decode"}] if as_list else content

    error = _mode_check_error(value)

    assert type(error) is ValueError
    assert str(error) == f"ais_perf_model entry for decode: {expected}"


def test_missing_state_snapshot_is_a_load_error():
    content = {
        **_CONTENT,
        "estimation_mode": "fpm_regression",
        "fallback_policy": "deny",
        "estimator_config": {"state_snapshot": {"path": "/missing/snapshot"}},
    }

    error = _mode_check_error(content)

    assert type(error) is ValueError
    assert isinstance(error.__cause__, FileNotFoundError)
    assert str(error) == f"ais_perf_model entry for decode: {error.__cause__}"


@pytest.mark.parametrize(
    "mode,value,checked",
    [
        pytest.param("disagg", dict(_CONTENT), ["prefill", "decode"], id="one-config"),
        pytest.param(
            "decode",
            [
                {**_CONTENT, "worker_type": "prefill", "tp": 0},
                {**_CONTENT, "worker_type": "decode"},
                {**_CONTENT, "worker_type": "aggregated", "backend": "nope"},
            ],
            ["decode"],
            id="only-the-entry-of-the-mode",
        ),
        pytest.param(
            "disagg",
            [
                {"worker_type": "prefill", "from": "/missing/template.yaml", "tp": 0},
                {**_CONTENT, "worker_type": "decode"},
            ],
            ["decode"],
            id="not-an-entry-with-from",
        ),
    ],
)
def test_load_checks_the_configs_of_the_mode_without_from(
    monkeypatch, mode, value, checked
):
    calls = []

    def record(worker_type, config):
        calls.append((worker_type, dict(config)))
        return real(worker_type, config)

    real = planner_config.ais_perf_model_role_config
    monkeypatch.setattr(planner_config, "ais_perf_model_role_config", record)

    assert _planner(value, mode=mode).ais_perf_model == value

    assert [worker_type for worker_type, _ in calls] == checked
    for worker_type, config in calls:
        assert config == {**_CONTENT, "worker_type": worker_type}


def test_first_wrong_config_of_the_mode_is_reported():
    value = [
        {**_CONTENT, "worker_type": "prefill", "tp": 0},
        {**_CONTENT, "worker_type": "decode", "backend": "nope"},
    ]

    error = _mode_check_error(value, mode="disagg")

    expected = _roles_form_error({**_CONTENT, "tp": 0}, "prefill")
    assert str(error) == f"ais_perf_model entry for prefill: {expected}"


def test_missing_entry_is_reported_before_a_wrong_config():
    error = _mode_check_error(
        [{**_CONTENT, "worker_type": "prefill", "tp": 0}], mode="disagg"
    )

    assert str(error) == (
        "ais_perf_model: mode='disagg' needs an entry for worker type 'decode'; "
        "it has: prefill"
    )


@pytest.mark.parametrize("wrong", _WRONG_VALUES)
def test_template_is_checked_at_first_use_like_roles(adapter, tmp_path, wrong):
    content = {**_CONTENT, **wrong}
    expected = _roles_form_error(content)
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(content))
    # Loading the config does not open the template.
    model = adapter(_planner({"from": str(template)}), limits=False)

    # A failure is not remembered: the second call checks the template again.
    for _ in range(2):
        with pytest.raises(ValueError) as when_first_used:
            model._configured_role()
        assert type(when_first_used.value) is ValueError
        assert str(when_first_used.value) == str(expected)


def test_mixed_forms_are_rejected():
    text = (
        "ais_perf_model mixes the old form (roles, dataset_profiles) with other "
        "keys (model); write one form"
    )
    with pytest.raises(ValidationError, match=re.escape(text)):
        _planner({"roles": {"decode": _CONTENT}, "model": "m"})

    text = (
        "ais_perf_model mixes the old form (roles, dataset_profiles) with other "
        "keys (from, zzz); write one form"
    )
    with pytest.raises(ValidationError, match=re.escape(text)):
        _planner({"roles": {"decode": _CONTENT}, "zzz": 1, "from": "x.yaml"})


def test_values_outside_both_forms_keep_old_errors():
    with pytest.raises(ValidationError) as misspelled:
        _planner({"roles": {"decode": _CONTENT}, "dataset_profile": {}})
    assert [
        (error["loc"], error["type"], error["msg"])
        for error in misspelled.value.errors()
    ] == [
        (
            ("ais_perf_model", "dataset_profile"),
            "extra_forbidden",
            "Extra inputs are not permitted",
        )
    ]

    with pytest.raises(ValidationError) as number:
        _planner(3)
    assert [
        (error["loc"], error["type"], error["msg"]) for error in number.value.errors()
    ] == [
        (
            ("ais_perf_model",),
            "model_type",
            "Input should be a valid dictionary or instance of AISPerfModelSpec",
        )
    ]


@pytest.mark.parametrize("optimization_target", ["sla", "throughput", "latency"])
@pytest.mark.parametrize(
    "key,content",
    [
        pytest.param("rolez", {"decode": _CONTENT}, id="rolez"),
        pytest.param("role", {"decode": _CONTENT}, id="role"),
        pytest.param(
            "dataset_profile", {"decode": _DATASET_PROFILE}, id="dataset-profile"
        ),
    ],
)
def test_misspelled_old_key_keeps_old_error(key, content, optimization_target):
    # The error does not depend on whether the target uses the perf model.
    errors = _load_failure({key: content}, optimization_target)

    assert [(error["loc"], error["type"], error["msg"]) for error in errors] == [
        (("ais_perf_model", key), "extra_forbidden", "Extra inputs are not permitted")
    ]


@pytest.mark.parametrize("optimization_target", ["sla", "throughput"])
@pytest.mark.parametrize(
    "value,text",
    [
        pytest.param(
            {**_CONTENT, "rolez": {"decode": _CONTENT}},
            "ais_perf_model: unknown keys: rolez",
            id="misspelled-old-key-next-to-config-key",
        ),
        pytest.param(
            {**_CONTENT, "tp_size": 2},
            "ais_perf_model: unknown keys: tp_size",
            id="one-key",
        ),
        pytest.param(
            {**_CONTENT, "zzz": 1, "aaa": 2},
            "ais_perf_model: unknown keys: aaa, zzz",
            id="two-keys",
        ),
        pytest.param(
            {"from": "/missing/template.yaml", "tpp": 8},
            "ais_perf_model: unknown keys: tpp",
            id="next-to-from",
        ),
        pytest.param(
            {**_CONTENT, "tp_size": 2, "zzz": None},
            "ais_perf_model: unknown keys: tp_size",
            id="next-to-key-without-value",
        ),
        pytest.param(
            [
                {**_CONTENT, "worker_type": "prefill"},
                {**_CONTENT, "worker_type": "decode", "roles": {"decode": _CONTENT}},
            ],
            "ais_perf_model: entry 2: unknown keys: roles",
            id="list-entry-with-roles",
        ),
        pytest.param(
            [
                {**_CONTENT, "worker_type": "prefill"},
                {**_CONTENT, "worker_type": "decode", "zzz": 1, "aaa": 2},
                {**_CONTENT, "worker_type": "aggregated", "tp_size": 2},
            ],
            "ais_perf_model: entry 2: unknown keys: aaa, zzz",
            id="first-of-two-list-entries",
        ),
    ],
)
def test_unknown_keys_are_rejected_when_loaded(value, text, optimization_target):
    error = _load_error(value, optimization_target)

    assert type(error) is ValueError
    assert str(error) == text


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({**_CONTENT, "tp_size": None}, id="one-config"),
        pytest.param(
            [{**_CONTENT, "worker_type": "decode", "tp_size": None}], id="list-entry"
        ),
    ],
)
def test_key_without_value_is_not_checked(adapter, value):
    # The loader drops a key written without a value, so its name is not checked.
    config = _planner(value)

    assert config.ais_perf_model == value
    # Nor is it an error when the config is first used.
    built = adapter(config)._build_ais_config()
    assert built == adapter(_planner(dict(_CONTENT)))._build_ais_config()
    assert "tp_size" not in built


def test_loader_error_comes_before_unknown_keys():
    error = _load_error([{**_CONTENT, "worker_type": "prefill", "zzz": 1}, 3])

    # The loader's own error class: the key names were not looked at.
    assert type(error) is not ValueError


def test_every_config_key_is_accepted():
    names = [field.name for field in fields(ForwardPassPerfModelConfig)]
    written = {
        **dict.fromkeys(names, 1),
        **_CONTENT,
        "worker_type": "decode",
        "systems_paths": ["default"],
        "from": "/missing/template.yaml",
    }

    assert _planner(written).ais_perf_model == written


def test_mixing_and_string_errors_keep_their_texts():
    error = _load_error({"roles": {"decode": _CONTENT}, "model": "m", "rolez": {}})
    assert type(error) is ValueError
    assert str(error) == (
        "ais_perf_model mixes the old form (roles, dataset_profiles) with other "
        "keys (model, rolez); write one form"
    )

    error = _load_error("perf_model.yaml")
    assert type(error) is ValueError
    assert str(error) == (
        "ais_perf_model must be a perf model config (a mapping) or a list of "
        "configs, not a string; a file path is not read here"
    )


def test_keys_are_not_checked_without_the_key_set(monkeypatch):
    # Without the estimator runtime the key set is empty.
    monkeypatch.setattr(
        "dynamo.planner.config.planner_config._ais_perf_model_config_keys", set
    )
    # With `from`, so that the content is not checked either.
    unknown_key = {"from": "/missing/template.yaml", "tp_size": 2}
    misspelled_old_key = {"from": "/missing/template.yaml", "rolez": {}}

    assert _planner(unknown_key).ais_perf_model == unknown_key
    assert _planner(misspelled_old_key).ais_perf_model == misspelled_old_key


def test_instance_value_is_old_form():
    config = _planner(AISPerfModelSpec(roles={"decode": _CONTENT}))

    assert isinstance(config.ais_perf_model, AISPerfModelSpec)
    expected = asdict(ForwardPassPerfModelConfig(**_CONTENT, worker_type="decode"))
    assert config.model_dump()["ais_perf_model"] == {
        "roles": {"decode": expected},
        "dataset_profiles": {},
    }


def test_none_stays_none():
    assert PlannerConfig().ais_perf_model is None
    assert PlannerConfig(ais_perf_model=None).ais_perf_model is None


@pytest.mark.parametrize(
    "value",
    [
        # systems_paths is written: its default is a tuple, which JSON gives
        # back as a list.
        pytest.param(
            {"roles": {"decode": {**_CONTENT, "systems_paths": []}}}, id="old-roles"
        ),
        pytest.param(
            {"dataset_profiles": {"decode": _DATASET_PROFILE}},
            id="old-dataset-profiles",
        ),
        pytest.param({**_CONTENT, "tp": 2}, id="one-config"),
        pytest.param(
            [
                {**_CONTENT, "worker_type": "decode"},
                {"worker_type": "prefill", "from": "prefill.yaml"},
            ],
            id="list",
        ),
    ],
)
def test_round_trips_for_every_form(value):
    config = _planner(value)

    assert PlannerConfig.model_validate(config.model_dump()) == config
    assert PlannerConfig.model_validate(config.model_dump(), strict=True) == config
    assert PlannerConfig.model_validate_json(config.model_dump_json()) == config


def test_json_schema_lists_the_shapes():
    field = PlannerConfig.model_json_schema()["properties"]["ais_perf_model"]
    shapes = field["anyOf"]

    assert shapes[0] == {"$ref": "#/$defs/AISPerfModelSpec"}
    assert [shape.get("type") for shape in shapes] == [None, "object", "array", "null"]
    assert shapes[2]["items"]["type"] == "object"


def test_retired_key_hint_names_new_form():
    text = f"aic_perf_model is no longer supported; use {_NEW_FORM}"
    with pytest.raises(ValidationError, match=re.escape(text)):
        PlannerConfig(aic_perf_model={"roles": {}})


def test_old_form_names():
    roles = {"prefill": _CONTENT}
    dataset_profiles = {"decode": _DATASET_PROFILE}

    assert ais_perf_model_old_form_names(AISPerfModelSpec(roles=roles)) == [
        "ais_perf_model.roles"
    ]
    assert ais_perf_model_old_form_names(
        AISPerfModelSpec(dataset_profiles=dataset_profiles)
    ) == ["ais_perf_model.dataset_profiles"]
    assert ais_perf_model_old_form_names(
        AISPerfModelSpec(roles=roles, dataset_profiles=dataset_profiles)
    ) == ["ais_perf_model.roles", "ais_perf_model.dataset_profiles"]
    assert ais_perf_model_old_form_names(AISPerfModelSpec()) == ["ais_perf_model.roles"]
    assert ais_perf_model_old_form_names(dict(_CONTENT)) == []
    assert ais_perf_model_old_form_names([{**_CONTENT, "worker_type": "decode"}]) == []
    assert ais_perf_model_old_form_names(None) == []


def test_assignment_of_list_dumps_as_list():
    config = _planner(None)
    written = [{**_CONTENT, "worker_type": "decode"}]

    config.ais_perf_model = written

    assert json.loads(config.model_dump_json())["ais_perf_model"] == written


def _set_config_argument(monkeypatch, ais_perf_model):
    config = {
        "mode": "disagg",
        "optimization_target": "sla",
        "ais_perf_model": ais_perf_model,
    }
    monkeypatch.setattr(sys, "argv", ["planner", "--config", json.dumps(config)])


@pytest.mark.parametrize(
    "value,text",
    [
        pytest.param(
            {"roles": {"prefill": _CONTENT, "decode": _CONTENT}},
            f"ais_perf_model.roles is deprecated and will be removed in v1.9; "
            f"use {_NEW_FORM}.",
            id="roles",
        ),
        pytest.param(
            {
                "roles": {"prefill": _CONTENT},
                "dataset_profiles": {"decode": _DATASET_PROFILE},
            },
            f"ais_perf_model.roles and ais_perf_model.dataset_profiles are "
            f"deprecated and will be removed in v1.9; use {_NEW_FORM}.",
            id="roles-and-dataset-profiles",
        ),
    ],
)
def test_planner_entry_point_warns_for_old_form(monkeypatch, value, text):
    _set_config_argument(monkeypatch, value)

    with pytest.warns(FutureWarning, match=re.escape(text)) as caught:
        config = planner_main._parse_config()

    assert [str(warning.message) for warning in caught] == [text]
    assert isinstance(config.ais_perf_model, AISPerfModelSpec)


def test_planner_main_warns_once_for_old_form(monkeypatch):
    started = []

    async def worker(config):
        started.append(config)

    monkeypatch.setattr(planner_main, "worker", worker)
    _set_config_argument(
        monkeypatch, {"roles": {"prefill": _CONTENT, "decode": _CONTENT}}
    )

    with pytest.warns(
        FutureWarning,
        match=r"^ais_perf_model\.roles is deprecated and will be removed in v1\.9; ",
    ) as caught:
        planner_main.main()

    assert len(caught) == 1
    # The warning points at the line of main() that reads the config.
    assert caught[0].filename == planner_main.__file__
    assert len(started) == 1


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param(dict(_CONTENT), id="one-config"),
        pytest.param(
            [
                {**_CONTENT, "worker_type": "prefill"},
                {**_CONTENT, "worker_type": "decode"},
            ],
            id="list",
        ),
    ],
)
def test_planner_entry_point_is_silent_for_new_form(monkeypatch, value):
    _set_config_argument(monkeypatch, value)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = planner_main._parse_config()

    assert config.ais_perf_model == value


def test_relative_template_starts_at_working_directory(adapter, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    other_dir = tmp_path / "other"
    run_dir.mkdir()
    other_dir.mkdir()
    (run_dir / "template.yaml").write_text(yaml.safe_dump({**_CONTENT, "tp": 2}))
    config = _planner({"from": "template.yaml"})
    assert config.ais_perf_model == {"from": "template.yaml"}
    model = adapter(config, limits=False)

    # A folder without the file.
    monkeypatch.chdir(other_dir)
    with pytest.raises(ValueError):
        model._configured_role()

    monkeypatch.chdir(run_dir)
    assert model._configured_role()["tp"] == 2
