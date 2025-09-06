# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Union

from tensorrt_llm import LLM, MultimodalEncoder

from dynamo.trtllm.constants import DisaggregationMode

logging.basicConfig(level=logging.DEBUG)


class TensorRTLLMEngine:
    def __init__(self, engine_args, disaggregation_mode=None):
        self.engine_args = engine_args
        self._llm: Optional[Union[LLM, MultimodalEncoder]] = None
        self.disaggregation_mode = disaggregation_mode

    async def initialize(self):
        if not self._llm:
            model = self.engine_args.pop("model")
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                self._llm = MultimodalEncoder(
                    model=model,
                    max_batch_size=self.engine_args.pop("max_batch_size"),
                )
            else:
                self._llm = LLM(
                    model=model,
                    **self.engine_args,
                )

    async def cleanup(self):
        if self._llm:
            try:
                if hasattr(self._llm, "shutdown"):
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
async def get_llm_engine(
    engine_args, disaggregation_mode=None
) -> AsyncGenerator[TensorRTLLMEngine, None]:
    engine = TensorRTLLMEngine(engine_args, disaggregation_mode)
    try:
        await engine.initialize()
        yield engine
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await engine.cleanup()
