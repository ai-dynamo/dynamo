# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS logs in vLLM worker processes reach vLLM's handlers as configured at emit."""

import logging
import sys
from types import ModuleType

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip("GMS is required", allow_module_level=True)

from gpu_memory_service.integrations.vllm import utils

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


class _Capture(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.seen = []

    def emit(self, record):
        self.seen.append(record.getMessage())


@pytest.fixture
def gms_logging(monkeypatch):
    """Isolated "gpu_memory_service"/"vllm"/root loggers; returns configure()."""
    if "vllm.logger" not in sys.modules:
        # Only the import side effect is needed; no vLLM install required.
        monkeypatch.setitem(
            sys.modules, "vllm", sys.modules.get("vllm", ModuleType("vllm"))
        )
        monkeypatch.setitem(sys.modules, "vllm.logger", ModuleType("vllm.logger"))
    loggers = (
        logging.getLogger("gpu_memory_service"),
        logging.getLogger("vllm"),
        logging.getLogger(),
    )
    for logger in loggers:
        monkeypatch.setattr(logger, "propagate", logger.propagate)
        monkeypatch.setattr(logger, "level", logger.level)

    def configure():
        # Here, not at fixture setup: pytest attaches its capture handlers to
        # non-propagating loggers when the test call starts.
        for logger in loggers:
            monkeypatch.setattr(logger, "handlers", [])
        utils.configure_gms_worker_logging()

    return configure


def test_gms_logs_reach_vllm_handlers_added_after_configuration(gms_logging):
    """A process that imports the GMS worker before vLLM sets up its handlers (a
    pre-importing zygote) must still emit GMS INFO through vLLM's handlers."""
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.propagate = False
    gms_logging()  # nothing to copy yet
    capture = _Capture(level=logging.INFO)
    vllm_logger.handlers.append(capture)
    logging.getLogger("gpu_memory_service.integrations.vllm.model_loader").info(
        "copy-out line"
    )
    assert capture.seen == ["copy-out line"]


def test_gms_logs_follow_vllm_propagation_to_application_handlers(gms_logging):
    """With VLLM_CONFIGURE_LOGGING=0 the application configures root logging and
    "vllm" propagates to it; GMS records must go there too, not to a bare
    stderr handler."""
    logging.getLogger("vllm").propagate = True
    gms_logging()
    capture = _Capture()
    logging.getLogger().handlers.append(capture)
    logging.getLogger("gpu_memory_service.client").info("attached")
    assert capture.seen == ["attached"]
