# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Union

from tensorrt_llm import LLM, MultimodalEncoder

from dynamo.trtllm.constants import DisaggregationMode
logging.basicConfig(level=logging.DEBUG)


class TensorRTLLMEngine:
    def __init__(self, engine_args, disaggregation_mode: DisaggregationMode):
        self.engine_args = engine_args
        self._llm: Optional[LLM] = None
        self._disaggregation_mode = disaggregation_mode

    async def initialize(self):
        if not self._llm:
            model = self.engine_args.pop("model")
            if self._disaggregation_mode == DisaggregationMode.ENCODE:
                # Initialize the multimodal encoder for full EPD
                max_batch_size = self.engine_args.pop("max_batch_size", 1)
                logging.info(f"Initializing multimodal encoder with max_batch_size: {max_batch_size}")
                self._llm = MultimodalEncoder(
                    model=model,
                    max_batch_size=max_batch_size,
                )
            else:
                # Initialize the regular LLM for decode-only or prefill-decode
                self._llm = LLM(
                    model=model,
                    **self.engine_args,
                )

    async def cleanup(self):
        if self._llm:
            try:
                self._llm.shutdown()
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
            finally:
                self._llm = None

    @property
    def llm(self) -> Union[LLM, MultimodalEncoder]:
        if not self._llm:
            raise RuntimeError("Engine not initialized")
        return self._llm


@asynccontextmanager
async def get_llm_engine(engine_args, disaggregation_mode) -> AsyncGenerator[TensorRTLLMEngine, None]:
    engine = TensorRTLLMEngine(engine_args, disaggregation_mode)
    try:
        await engine.initialize()
        yield engine
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await engine.cleanup()
