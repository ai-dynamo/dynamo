# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import sglang as sgl
from sglang.srt.server_args import ServerArgs


class SGLangEngine:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self._engine: Optional[sgl.Engine]

    async def initialize(self):
        if not self._engine:
            self._engine = sgl.Engine(self.server_args)

    async def cleanup(self):
        if self._engine:
            try:
                self._engine.shutdown()
            except Exception as e:
                logging.error(f"Error during shutdown: {e}")
            finally:
                self._engine = None

    @property
    def engine(self):
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        return self._engine


@asynccontextmanager
async def get_llm_engine(engine_args) -> AsyncGenerator[SGLangEngine, None]:
    engine = SGLangEngine(engine_args)
    try:
        await engine.initialize()
        yield engine
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await engine.cleanup()
