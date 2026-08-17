# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for native diffusion worker argument parsing.

The real sglang.multimodal_gen ServerArgs resolves model info at parse time,
so these tests inject a lightweight fake through sys.modules. They cover the
Dynamo layer only: dynamo-side flag registration, adapter delegation, and the
handoff contract with the engine parser.
"""

import argparse
import dataclasses
import sys
from types import ModuleType
from typing import Optional

import pytest

from dynamo.sglang.diffusion_args import DiffusionWorkerArgs

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


@dataclasses.dataclass
class FakeEngineArgs:
    """Stand-in for sglang.multimodal_gen ServerArgs (0.5.17 field subset)."""

    model_path: str
    tp_size: Optional[int] = None
    dp_size: int = 1
    log_level: str = "info"
    enable_trace: bool = False
    attention_backend: Optional[str] = None
    performance_mode: str = "auto"

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--model-path", type=str)
        parser.add_argument("--tp-size", type=int, default=None)
        parser.add_argument("--dp-size", type=int, default=1)
        parser.add_argument("--log-level", type=str, default="info")
        parser.add_argument("--enable-trace", action="store_true")
        parser.add_argument("--attention-backend", type=str, default=None)
        parser.add_argument("--performance-mode", type=str, default="auto")
        return parser

    @classmethod
    def from_cli_args(cls, args, unknown_args=None):
        if unknown_args:
            raise SystemExit(f"error: unrecognized arguments: {' '.join(unknown_args)}")
        field_names = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in vars(args).items() if k in field_names}
        return cls(**values)


def _install_fake_sglang(monkeypatch, engine_cls=FakeEngineArgs):
    """Register fake sglang.multimodal_gen modules for the lazy imports."""
    server_args_mod = ModuleType(
        "sglang.multimodal_gen.runtime.server_args.server_args"
    )
    server_args_mod.ServerArgs = engine_cls

    utils_mod = ModuleType("sglang.multimodal_gen.utils")
    utils_mod.FlexibleArgumentParser = argparse.ArgumentParser

    for name, mod in {
        "sglang": ModuleType("sglang"),
        "sglang.multimodal_gen": ModuleType("sglang.multimodal_gen"),
        "sglang.multimodal_gen.runtime": ModuleType("sglang.multimodal_gen.runtime"),
        "sglang.multimodal_gen.runtime.server_args": ModuleType(
            "sglang.multimodal_gen.runtime.server_args"
        ),
        "sglang.multimodal_gen.runtime.server_args.server_args": server_args_mod,
        "sglang.multimodal_gen.utils": utils_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


class TestDiffusionWorkerArgs:
    """Adapter behavior: delegation, fallbacks, inert defaults."""

    def _engine(self, **overrides):
        return FakeEngineArgs(model_path="org/model", **overrides)

    def test_engine_fields_delegate(self):
        adapter = DiffusionWorkerArgs(
            self._engine(tp_size=4, log_level="debug"),
            served_model_name="my-model",
            enable_metrics=True,
        )
        assert adapter.model_path == "org/model"
        assert adapter.tp_size == 4
        assert adapter.log_level == "debug"
        assert adapter.served_model_name == "my-model"
        assert adapter.enable_metrics is True

    def test_served_model_name_falls_back_to_model_path(self):
        adapter = DiffusionWorkerArgs(
            self._engine(), served_model_name=None, enable_metrics=False
        )
        assert adapter.served_model_name == "org/model"

    def test_inert_defaults_present(self):
        """Fields Dynamo's shared worker code probes must exist and be inert."""
        adapter = DiffusionWorkerArgs(
            self._engine(), served_model_name=None, enable_metrics=False
        )
        assert adapter.speculative_algorithm is None
        assert adapter.disaggregation_mode is None
        assert adapter.dllm_algorithm is False
        assert adapter.load_format is None
        assert adapter.kv_events_config is None
        assert adapter.enable_forward_pass_metrics is False

    def test_missing_engine_field_raises(self):
        adapter = DiffusionWorkerArgs(
            self._engine(), served_model_name=None, enable_metrics=False
        )
        with pytest.raises(AttributeError):
            _ = adapter.not_a_real_field

    def test_native_field_wins_over_inert_default(self):
        """If a future engine ServerArgs grows one of the pinned fields,
        its native value must win over the inert default."""

        @dataclasses.dataclass
        class EngineWithLoadFormat(FakeEngineArgs):
            load_format: Optional[str] = "auto"

        adapter = DiffusionWorkerArgs(
            EngineWithLoadFormat(model_path="org/model"),
            served_model_name=None,
            enable_metrics=False,
        )
        assert adapter.load_format == "auto"


class TestParseDiffusionArgs:
    """Parsing behavior against a fake engine parser."""

    def test_engine_args_parse_natively(self, monkeypatch):
        _install_fake_sglang(monkeypatch)
        from dynamo.sglang.diffusion_args import parse_diffusion_args

        _, adapter = parse_diffusion_args(
            [
                "--model-path",
                "org/model",
                "--attention-backend",
                "flash",
                "--performance-mode",
                "fast",
                "--tp-size",
                "2",
            ]
        )
        assert adapter.engine_args.attention_backend == "flash"
        assert adapter.engine_args.performance_mode == "fast"
        assert adapter.engine_args.tp_size == 2

    def test_dynamo_side_flags_split_out(self, monkeypatch):
        """--served-model-name / --enable-metrics must not leak into the
        engine args when the engine parser does not define them."""
        _install_fake_sglang(monkeypatch)
        from dynamo.sglang.diffusion_args import parse_diffusion_args

        _, adapter = parse_diffusion_args(
            [
                "--model-path",
                "org/model",
                "--served-model-name",
                "alias",
                "--enable-metrics",
            ]
        )
        assert adapter.served_model_name == "alias"
        assert adapter.enable_metrics is True
        assert not hasattr(adapter.engine_args, "served_model_name")
        assert not hasattr(adapter.engine_args, "enable_metrics")

    def test_native_served_model_name_preferred(self, monkeypatch):
        """When the engine parser defines --served-model-name natively (sglang
        main does), the Dynamo-side flag must not be registered twice."""

        @dataclasses.dataclass
        class EngineWithServedName(FakeEngineArgs):
            served_model_name: Optional[str] = None

            @staticmethod
            def add_cli_args(parser):
                FakeEngineArgs.add_cli_args(parser)
                parser.add_argument("--served-model-name", type=str, default=None)
                return parser

        _install_fake_sglang(monkeypatch, engine_cls=EngineWithServedName)
        from dynamo.sglang.diffusion_args import parse_diffusion_args

        _, adapter = parse_diffusion_args(
            ["--model-path", "org/model", "--served-model-name", "native-name"]
        )
        assert adapter.engine_args.served_model_name == "native-name"
        assert adapter.served_model_name == "native-name"

    def test_unknown_flag_rejected(self, monkeypatch):
        """Typos and unsupported flags fail loudly instead of being silently
        absorbed (the failure mode the old stub had)."""
        _install_fake_sglang(monkeypatch)
        from dynamo.sglang.diffusion_args import parse_diffusion_args

        with pytest.raises(SystemExit):
            parse_diffusion_args(
                ["--model-path", "org/model", "--not-a-real-flag", "value"]
            )
