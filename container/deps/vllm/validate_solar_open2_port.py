# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the Solar Open2 250B forward-port against the installed vLLM.

Run during the image build, after the patches in
``container/patches/vllm/solar_open2`` are applied. Every check here fails the
build rather than deferring the failure to a serving request, where a missing
registration surfaces as an unsupported-architecture error or, worse, as silently
wrong output.
"""

import sys


def main() -> int:
    import vllm

    if not vllm.__version__.startswith("0.26."):
        raise SystemExit(f"expected vLLM 0.26.x, found {vllm.__version__}")

    from vllm.model_executor.models.registry import _VLLM_MODELS
    from vllm.model_executor.models.solar_open2 import SolarOpen2ForCausalLM
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.transformers_utils.configs.solar_open2 import SolarOpen2Config
    from vllm.v1.sample.logits_processor.solar_open2 import (
        SolarOpen2TemplateLogitsProcessor,
    )

    failures = []
    if "SolarOpen2ForCausalLM" not in _VLLM_MODELS:
        failures.append("SolarOpen2ForCausalLM not in the model registry")
    if "solar_open2" not in ReasoningParserManager.list_registered():
        failures.append("solar_open2 reasoning parser not registered")
    if "solar_open2" not in ToolParserManager.list_registered():
        failures.append("solar_open2 tool parser not registered")

    # The KDA attention op must be known to the compilation config. Solar Open2 is
    # a hybrid model whose linear-attention layers register as a custom op; if the
    # op is not listed the graph is split incorrectly during compilation.
    from vllm.config.compilation import CompilationConfig

    if "vllm::solar_open2_kda_attention" not in getattr(
        CompilationConfig, "_attention_ops", []
    ):
        failures.append("vllm::solar_open2_kda_attention missing from _attention_ops")

    if failures:
        raise SystemExit("solar_open2 port validation failed: " + "; ".join(failures))

    print(
        "SOLAR-OPEN2-PORT-VALIDATED",
        vllm.__version__,
        SolarOpen2ForCausalLM.__name__,
        SolarOpen2Config.__name__,
        SolarOpen2TemplateLogitsProcessor.__name__,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
