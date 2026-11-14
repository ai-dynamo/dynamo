# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate configuration input; the upstream SDK owns the estimator schema."""

import argparse
import json
import os
from pathlib import Path

import yaml

from dynamo._internal.ais import _pad_nextn_accept_rates
from dynamo.common.ais_perf_model import (
    ENV_VAR,
    REMOVE_IN,
    ROUTER_WORKER_TYPES,
    WORKER_TYPE_ENV_VAR,
    env_value,
    router_perf_config,
)
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import Deprecated, join_names

_FIELDS = {
    "backend": (str, None, "backend"),
    "system": (str, None, "system"),
    "backend_version": (str, None, "backend_version"),
    "tp_size": (int, None, "tp"),
    "model_path": (str, None, "model"),
    "moe_tp_size": (int, None, "moe_tp_size"),
    "moe_ep_size": (int, None, "moe_ep_size"),
    "attention_dp_size": (int, None, "attention_dp"),
    "nextn": (int, None, "nextn"),
    "nextn_accept_rates": (str, None, None),
    "mtp_seed": (int, 42, None),
}


def parse_ais_perf_config(value):
    """Read a complete canonical mapping from JSON or a JSON/YAML file."""
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if not isinstance(value, str):
        raise ValueError(
            "AIS perf config must be a mapping, SDK config, JSON, or file path"
        )
    if value.lstrip().startswith("{"):
        result = json.loads(value)
    else:
        try:
            result = yaml.safe_load(Path(value).read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read AIS perf config {value!r}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("AIS perf config must contain an object")
    return result


class _UniqueAlias(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        marker = "_seen_" + self.dest
        seen = getattr(namespace, marker, None)
        if seen is not None and seen != option_string:
            raise argparse.ArgumentError(
                self, "new and legacy spellings cannot be combined"
            )
        setattr(namespace, marker, option_string)
        setattr(namespace, self.dest, values)


def _env(name, convert, default):
    value = os.environ.get("DYN_AIS_" + name.upper())
    return default if value is None else convert(value)


class AisPerfConfigBase(ConfigBase):
    ais_perf_config = None
    ais_perf_model = None
    router_prefill_load_worker_type = None

    def new_ais_perf_input(self) -> tuple[str | None, str | None]:
        """Return the AIS perf model value and the name of the input that gave it.

        A typed ``--ais-perf-model`` wins over ``DYN_AIS_PERF_MODEL``. Both
        items are ``None`` when neither is set.
        """
        if self.ais_perf_model is not None:
            return self.ais_perf_model, "--ais-perf-model"
        value = env_value(ENV_VAR)
        if value is not None:
            return value, ENV_VAR
        return None, None

    def old_ais_perf_inputs(self) -> list[str]:
        """Return the names of the old AIS inputs that are set.

        An input is named as the user gave it: the flag when it was typed, the
        environment variable when the value came from it. The acceptance rates
        and the seed are not old inputs.
        """
        names = []
        if self.ais_perf_config is not None:
            # The parser keeps no mark of a typed --ais-perf-config.
            names.append(
                "DYN_AIS_PERF_CONFIG"
                if "DYN_AIS_PERF_CONFIG" in os.environ
                else "--ais-perf-config"
            )
        for name, (_, _, key) in _FIELDS.items():
            if key is None or getattr(self, "ais_" + name, None) is None:
                continue
            flag = "--ais-" + name.replace("_", "-")
            variable = "DYN_AIS_" + name.upper()
            # The mark holds the spelling that was typed, --ais- or --aic-.
            typed = getattr(self, "_seen_ais_" + name, None)
            names.append(typed or (variable if variable in os.environ else flag))
        return names

    def router_worker_type(self) -> str | None:
        """Return the worker type chosen for the AIS perf model, or ``None``.

        A typed ``--router-prefill-load-worker-type`` wins over
        ``DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE``.
        """
        if self.router_prefill_load_worker_type is not None:
            return self.router_prefill_load_worker_type
        value = env_value(WORKER_TYPE_ENV_VAR)
        if value is not None and value not in ROUTER_WORKER_TYPES:
            raise ValueError(
                f"{WORKER_TYPE_ENV_VAR} is {value!r}; use prefill or aggregated"
            )
        return value

    def validate_ais_perf_inputs(
        self, *, replacement: str = "--ais-perf-model", stacklevel: int = 3
    ) -> None:
        """Check which AIS perf model inputs are given together.

        ``replacement`` is what the component reads instead of the old inputs;
        the one deprecation warning for old inputs names it. ``stacklevel=3``
        is the caller of the ``validate()`` that calls this method. No value
        is read here: ``ais_perf_kwargs()`` does that.
        """
        ais = getattr(self, "router_prefill_load_model", None) == "ais"
        value, name = self.new_ais_perf_input()
        old = self.old_ais_perf_inputs()
        if not ais:
            if self.ais_perf_config is not None:
                raise ValueError(
                    "--ais-perf-config requires --router-prefill-load-model=ais"
                )
            if self.ais_perf_model is not None:
                raise ValueError(
                    "--ais-perf-model requires --router-prefill-load-model=ais"
                )
            if self.router_prefill_load_worker_type is not None:
                raise ValueError(
                    "--router-prefill-load-worker-type requires "
                    "--router-prefill-load-model=ais"
                )
            # The two environment variables are ignored here, so that processes
            # with and without the AIS load model can share one environment.
        elif old:
            if value is not None:
                raise ValueError(f"{name} cannot be combined with {join_names(old)}")
            if self.router_prefill_load_worker_type is not None:
                raise ValueError(
                    f"--router-prefill-load-worker-type requires {replacement}"
                )
        elif value is not None:
            self.router_worker_type()
        # Last statement: every error above is raised before the warning.
        if old:
            Deprecated(replacement, remove_in=REMOVE_IN).warn(
                old, stacklevel=stacklevel
            )

    def ais_perf_model_kwargs(self, value) -> dict:
        """Return the ``AisPerfConfig`` arguments for one AIS perf model value.

        A relative path in ``value`` starts at the working directory.
        """
        config = router_perf_config(value, self.router_worker_type())
        self._check_nextn_accept_rates(config)
        return {"config": config}

    def _check_nextn_accept_rates(self, config) -> None:
        rates = getattr(self, "ais_nextn_accept_rates", None)
        if config.get("nextn", 0) and rates is not None:
            _pad_nextn_accept_rates(rates)

    def check_renamed_env_vars(self) -> None:
        """Reject a ``DYN_AIC_*`` variable: it was renamed to ``DYN_AIS_*``."""
        legacy = [
            "DYN_AIC_" + name.upper()
            for name in _FIELDS
            if "DYN_AIC_" + name.upper() in os.environ
        ]
        if legacy:
            raise ValueError(
                "renamed environment variables are no longer read: "
                f"{', '.join(legacy)}; use the DYN_AIS_* names"
            )

    def ais_perf_kwargs(self) -> dict:
        self.check_renamed_env_vars()
        value, _ = self.new_ais_perf_input()
        if value is not None:
            return self.ais_perf_model_kwargs(value)
        authored = getattr(self, "ais_perf_config", None)
        shorthand = {
            key: getattr(self, "ais_" + name, None)
            for name, (_, _, key) in _FIELDS.items()
            if key is not None
        }
        shorthand = {
            key: value for key, value in shorthand.items() if value is not None
        }
        if authored is not None:
            if shorthand:
                raise ValueError(
                    "--ais-perf-config cannot be combined with flat --ais-* / --aic-* identity flags"
                )
            config = parse_ais_perf_config(authored)
        else:
            config = {"worker_type": "aggregated", **shorthand}
        missing = [
            key
            for key in ("backend", "system", "model", "worker_type")
            if not config.get(key)
        ]
        if missing:
            if authored is None and not shorthand:
                raise ValueError(
                    "--router-prefill-load-model=ais requires an AIS perf model: "
                    f"set --ais-perf-model or {ENV_VAR}"
                )
            raise ValueError("AIS perf config requires " + ", ".join(missing))
        # The upstream class preserves estimator_config, ordered roots and policy.
        from aisimulate_core.sdk import ForwardPassPerfModelConfig

        self._check_nextn_accept_rates(config)
        return {"config": ForwardPassPerfModelConfig(**config).to_dict()}


class AisPerfArgGroup(ArgGroup):
    def __init__(self, *, replacement: str | None = None) -> None:
        """``replacement`` is for a component that does not read
        ``--ais-perf-model``: what it reads instead, as it gives it to
        ``validate_ais_perf_inputs``. The help texts then point to it.
        """
        self.replacement = replacement

    def add_arguments(self, parser) -> None:
        group = parser.add_argument_group("AISimulate Perf Model Options")
        if self.replacement is None:
            new_input = "--ais-perf-model"
            new_input_help = (
                "AIS perf model for --router-prefill-load-model=ais: a file path "
                "or inline JSON, holding one perf model config or a list of "
                f"configs with one worker_type each; {ENV_VAR}."
            )
        else:
            new_input = self.replacement
            new_input_help = f"Not read here; write {new_input}."
        deprecated = Deprecated(new_input, remove_in=REMOVE_IN)
        # The two environment variables are read when the config is validated.
        group.add_argument(
            "--ais-perf-model",
            dest="ais_perf_model",
            default=None,
            metavar="VALUE",
            help=new_input_help,
        )
        group.add_argument(
            "--router-prefill-load-worker-type",
            dest="router_prefill_load_worker_type",
            choices=ROUTER_WORKER_TYPES,
            default=None,
            help=(
                f"Worker type whose config the router takes from {new_input}; "
                "needed when the value has both or names none; "
                f"{WORKER_TYPE_ENV_VAR}."
            ),
        )
        group.add_argument(
            "--ais-perf-config",
            type=parse_ais_perf_config,
            default=os.environ.get("DYN_AIS_PERF_CONFIG"),
            help=(
                deprecated.message("--ais-perf-config")
                + " Complete ForwardPassPerfModelConfig as JSON or a JSON/YAML path."
            ),
        )
        for name, (convert, default, key) in _FIELDS.items():
            flag = name.replace("_", "-")
            text = f"AISimulate {flag}; DYN_AIS_{name.upper()}. Legacy aic spelling accepted."
            if key is not None:
                text = deprecated.message("--ais-" + flag) + " " + text
            group.add_argument(
                "--ais-" + flag,
                "--aic-" + flag,
                dest="ais_" + name,
                type=convert,
                default=_env(name, convert, default),
                action=_UniqueAlias,
                help=(
                    "Comma-separated conditional acceptance rates: entry i is P(draft i accepted | all earlier drafts were accepted)."
                    if name == "nextn_accept_rates"
                    else text
                ),
            )
