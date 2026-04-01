# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Load Thompson router configuration from the bundled config.yaml or an override path.

Priority:
  1. THOMPSON_CONFIG_PATH env var (absolute path to a custom YAML file)
  2. Bundled config.yaml shipped with dynamo.thompson_router
"""

import importlib.resources
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_BUNDLED_PACKAGE = "dynamo.thompson_router"
_BUNDLED_FILENAME = "config.yaml"
_ENV_OVERRIDE = "THOMPSON_CONFIG_PATH"


def load_config() -> dict[str, Any]:
    """Load Thompson router config, preferring env override over bundled default."""
    override_path = os.environ.get(_ENV_OVERRIDE)

    if override_path:
        try:
            with open(override_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info("Loaded Thompson config from %s", override_path)
            return config
        except Exception as e:
            logger.warning(
                "Failed to load config from %s (%s), falling back to bundled default",
                override_path,
                e,
            )

    try:
        ref = importlib.resources.files(_BUNDLED_PACKAGE).joinpath(_BUNDLED_FILENAME)
        text = ref.read_text(encoding="utf-8")
        config = yaml.safe_load(text) or {}
        logger.info("Loaded bundled Thompson config from %s/%s", _BUNDLED_PACKAGE, _BUNDLED_FILENAME)
        return config
    except Exception as e:
        logger.warning("Failed to load bundled config (%s), using empty defaults", e)
        return {}
