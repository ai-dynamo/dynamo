# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.vllm.args import parse_args

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
    pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator"),
]


class TestParseArgsLoraExclusivity:
    """The --enable-lora exclusivity rules must fire on the real CLI path.

    Direct validator tests supply engine_args themselves, so they pass even
    when parse_args never supplies engine_args before validate() runs. These
    tests drive the whole command line to exercise that ordering.
    """

    @staticmethod
    def _parse(extra_argv):
        return parse_args(["--model", "Qwen/Qwen3-0.6B", *extra_argv])

    def test_realtime_with_enable_lora_is_rejected(self):
        with pytest.raises(ValueError, match="enable-lora"):
            self._parse(["--realtime", "--enable-lora"])

    def test_classify_worker_with_enable_lora_is_rejected(self):
        """Kept separate from the --realtime case: the classify rule may be
        removed once LoRA is supported on pooling-family workers, and the
        --realtime rule is independent of that."""
        with pytest.raises(ValueError, match="enable-lora"):
            self._parse(["--classify-worker", "--enable-lora"])

    def test_enable_lora_alone_is_accepted(self):
        config = self._parse(["--enable-lora"])
        assert config.engine_args.enable_lora is True

    def test_realtime_alone_is_accepted(self):
        config = self._parse(["--realtime"])
        assert config.realtime is True
        assert not config.engine_args.enable_lora
