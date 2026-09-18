# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate configuration input; the upstream SDK owns the estimator schema."""

import argparse
import json
import os
from pathlib import Path

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase

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
        import yaml

        result = yaml.safe_load(Path(value).read_text())
    if not isinstance(result, dict):
        raise ValueError("AIS perf config must contain an object")
    return result


class _UniqueAlias(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        marker = "_seen_" + self.dest
        if getattr(namespace, marker, False):
            raise argparse.ArgumentError(
                self, "new and legacy spellings cannot be combined"
            )
        setattr(namespace, marker, True)
        setattr(namespace, self.dest, values)


def _env(name, convert, default):
    new, old = "DYN_AIS_" + name.upper(), "DYN_AIC_" + name.upper()
    if new in os.environ and old in os.environ:
        raise ValueError(f"{new} and {old} cannot be combined")
    value = os.environ.get(new, os.environ.get(old))
    return default if value is None else convert(value)


class AisPerfConfigBase(ConfigBase):
    ais_perf_config = None

    def __getattr__(self, name):
        if name.startswith("aic_") and name[4:] in _FIELDS:
            return getattr(self, "ais_" + name[4:], None)
        raise AttributeError(name)

    def ais_perf_kwargs(self) -> dict:
        authored = getattr(self, "ais_perf_config", None)
        shorthand = {
            key: getattr(self, "ais_" + name, getattr(self, "aic_" + name, None))
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
            raise ValueError("AIS perf config requires " + ", ".join(missing))
        # The upstream class preserves estimator_config, ordered roots and policy.
        from aisimulate_core.sdk import ForwardPassPerfModelConfig

        rates = getattr(self, "ais_nextn_accept_rates", None)
        if config.get("nextn", 0) and rates is not None:
            from dynamo._internal.ais import _pad_nextn_accept_rates

            _pad_nextn_accept_rates(rates)
        return {"config": ForwardPassPerfModelConfig(**config).to_dict()}

    def aic_perf_kwargs(self) -> dict:
        """Deprecated constructor-keyword adapter for existing callers."""
        if getattr(self, "ais_perf_config", None) is not None:
            return self.ais_perf_kwargs()
        result = {
            "aic_" + name: getattr(self, "ais_" + name, None)
            for name in _FIELDS
            if name != "mtp_seed"
        }
        if result["aic_tp_size"] is None:
            result["aic_tp_size"] = 1
        return result


class AisPerfArgGroup(ArgGroup):
    def add_arguments(self, parser) -> None:
        group = parser.add_argument_group("AISimulate Perf Model Options")
        group.add_argument(
            "--ais-perf-config",
            type=parse_ais_perf_config,
            default=_env("perf_config", parse_ais_perf_config, None),
            help="Complete ForwardPassPerfModelConfig as JSON or a JSON/YAML path.",
        )
        for name, (convert, default, _) in _FIELDS.items():
            flag = name.replace("_", "-")
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
                    else f"AISimulate {flag}; DYN_AIS_{name.upper()}. Legacy aic spelling accepted."
                ),
            )


# Existing imports and authored objects are accepted at the input boundary.
AicPerfConfigBase = AisPerfConfigBase
AicPerfArgGroup = AisPerfArgGroup
