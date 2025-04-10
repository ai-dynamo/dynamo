# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import logging
import threading
import traceback
import weakref
from enum import Enum
from queue import Queue
from typing import Callable, Optional, TypedDict, Union

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    PREFIX = "prefix"


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class ServerType(Enum):
    # Generation server used for disaggregated and aggregated requests
    GEN = "gen"
    # Context server used for disaggregated requests
    CTX = "ctx"
    # Dynamo run server used for Dynamo run requests
    DYN_RUN = "dyn_run"


class ConversationMessage(TypedDict):
    role: str
    content: str


class ManagedThread(threading.Thread):
    def __init__(
        self,
        task: Optional[Union[Callable[..., bool], weakref.WeakMethod]],
        error_queue: Optional[Queue] = None,
        name: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.task = task
        self.error_queue = error_queue
        self.kwargs = kwargs
        self.loop = loop
        self.daemon = True

        self.stop_event = threading.Event()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def run(self):
        while not self.stop_event.is_set():
            task: Optional[Union[Callable[..., bool], weakref.WeakMethod]] = self.task
            if isinstance(task, weakref.WeakMethod):
                task = task()
                if task is None:
                    # Normally, this should not happen.
                    logger.warning("WeakMethod is expired.")
                    break

            if task is None:
                break

            try:
                if self.loop is None:
                    logger.error("[ManagedThread] Loop not initialized!")
                    break
                future = asyncio.run_coroutine_threadsafe(
                    task(**self.kwargs), self.loop
                )
                _ = future.result()
            except Exception as e:
                logger.error(
                    f"Error in thread {self.name}: {e}\n{traceback.format_exc()}"
                )
                if self.error_queue is not None:
                    self.error_queue.put(e)

        logger.info(f"Thread {self.name} stopped.")

    def stop(self):
        self.stop_event.set()
