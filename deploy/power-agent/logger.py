# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shim for the `logger` module the NGC DCGM runtime image's vendored
# pydcgm bindings expect to import (e.g. `DcgmGroup.py:20`).
#
# Upstream DCGM ships `logger.py` under
# `datacenter-gpu-manager-4/testing/python3/logger.py` — that path is
# NOT included in the `nvcr.io/nvidia/cloud-native/dcgm:*-ubuntu22.04`
# runtime image, only its top-level bindings under
# `/usr/share/datacenter-gpu-manager-4/bindings/python3/` are. The
# bindings' `import logger` therefore fails as
# `ModuleNotFoundError: No module named 'logger'` the moment any
# DcgmGroup is constructed.
#
# This file is COPYied into `/opt/dcgm/python/logger.py` by the Power
# Agent's Dockerfile (alongside the vendored bindings); `PYTHONPATH`
# resolves `import logger` to it. The contract pydcgm needs is just the
# call-through surface — debug/info/warning/error/critical/log functions
# that accept the same args as Python's `logging` module — so we route
# straight to a stdlib logger.
import logging as _logging

_LOG = _logging.getLogger("dcgm.bindings")

debug = _LOG.debug
info = _LOG.info
warning = _LOG.warning
error = _LOG.error
critical = _LOG.critical
log = _LOG.log
