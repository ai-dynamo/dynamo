# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from tensorrt_llm import LLM

logging.basicConfig(level=logging.DEBUG)


class TensorRTLLMEngine:
    def __init__(self, engine_args):
        self.engine_args = engine_args
        self._llm: Optional[LLM] = None

    async def initialize(self):
        if not self._llm:
            model = self.engine_args.pop("model")
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
    def llm(self):
        if not self._llm:
            raise RuntimeError("Engine not initialized")
        return self._llm


@asynccontextmanager
async def get_llm_engine(engine_args) -> AsyncGenerator[TensorRTLLMEngine, None]:
    engine = TensorRTLLMEngine(engine_args)
    try:
        await engine.initialize()
        yield engine
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await engine.cleanup()
