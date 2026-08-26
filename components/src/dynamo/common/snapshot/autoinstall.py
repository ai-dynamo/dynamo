# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-start installers for snapshot-mode backend hooks.

SGLang has no ``kv_connector_module_path``. Spawned scheduler children re-import
sglang, so the parent wrap of ``get_kv_class`` is invisible to them. The wheel
installs ``dynamo_snapshot.pth`` which calls :func:`install_snapshot_backends`
when ``DYN_SNAPSHOT_CONTROL_DIR`` is set.
"""

from __future__ import annotations

import logging
import os

from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV

logger = logging.getLogger(__name__)


def install_snapshot_backends() -> None:
    if not os.environ.get(SNAPSHOT_CONTROL_DIR_ENV):
        return
    try:
        from dynamo.sglang.snapshot_nixl import install_snapshot_nixl
    except ImportError:
        return
    install_snapshot_nixl()
