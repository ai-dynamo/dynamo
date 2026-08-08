# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Subprocess-coverage bootstrap for the Dynamo test suite.
#
# CI installs this module into a site directory and points a ``.pth`` at it
# (``import coverage_subprocess_bootstrap``) so it runs at interpreter startup for every
# python process. It only does anything when ``COVERAGE_PROCESS_START`` is set in the
# environment, which tests/utils/managed_process.py injects into the product processes it
# launches (and which propagates through launch shell scripts to the real
# ``python -m dynamo.*`` workers). No pytest process ever gets ``COVERAGE_PROCESS_START``,
# so this is a no-op there and pytest-cov is unaffected.
#
# Beyond starting coverage, it runs a daemon thread that flushes the data every few
# seconds. Backend workers (vLLM/SGLang/TensorRT-LLM) are launched by scripts whose
# ``trap 'kill 0' EXIT`` double-signals the worker on shutdown; the framework force-exits
# on the second signal before coverage's atexit flush runs, so without a periodic flush
# the worker's coverage would be lost. The last periodic save survives a forced exit.
import os
import threading
import time

# Seconds between background flushes; small enough that a forced exit loses little.
_SAVE_INTERVAL = float(os.environ.get("DYN_SUBPROCESS_COVERAGE_SAVE_INTERVAL", "5"))

try:
    import coverage

    coverage.process_startup()
    _cov = getattr(coverage.process_startup, "coverage", None)
    if _cov is not None and os.environ.get("COVERAGE_PROCESS_START"):

        def _periodic_save():
            while True:
                time.sleep(_SAVE_INTERVAL)
                try:
                    _cov.save()
                except Exception:
                    pass

        threading.Thread(
            target=_periodic_save, name="dynamo-cov-periodic-save", daemon=True
        ).start()
except Exception:
    # Never let coverage bootstrap break a product process.
    pass
