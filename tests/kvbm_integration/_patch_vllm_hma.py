#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Patch vLLM config to conditionally disable HMA based on connector support.

vLLM 0.16.0 unconditionally disables HMA (Hybrid Memory Allocator) when
--kv-transfer-config is set. This breaks hybrid Mamba+Attention models
(e.g. Nemotron) whose KV specs can't be unified without HMA.

This patch makes the disable conditional: only turn off HMA if the
connector class does not subclass SupportsHMA.
"""

import site
import sys
from pathlib import Path


def find_vllm_config() -> Path:
    """Locate vllm/config/vllm.py in the installed packages."""
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = Path(sp) / "vllm" / "config" / "vllm.py"
        if candidate.exists():
            return candidate
    for p in sys.path:
        candidate = Path(p) / "vllm" / "config" / "vllm.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find vllm/config/vllm.py")


OLD_BLOCK_V015 = """\
            if self.kv_transfer_config is not None:
                # NOTE(Kuntai): turn HMA off for connector for now.
                # TODO(Kuntai): have a more elegent solution to check and
                # turn off HMA for connector that does not support HMA.
                logger.warning(
                    "Turning off hybrid kv cache manager because "
                    "`--kv-transfer-config` is set. This will reduce the "
                    "performance of vLLM on LLMs with sliding window attention "
                    "or Mamba attention. If you are a developer of kv connector"
                    ", please consider supporting hybrid kv cache manager for "
                    "your connector by making sure your connector is a subclass"
                    " of `SupportsHMA` defined in kv_connector/v1/base.py."
                )
                self.scheduler_config.disable_hybrid_kv_cache_manager = True"""

# vLLM 0.16.0 changed the structure: uses need_disable_hybrid_kv_cache_manager flag
OLD_BLOCK_V016 = """\
            if self.kv_transfer_config is not None:
                # NOTE(Kuntai): turn HMA off for connector unless specifically enabled.
                need_disable_hybrid_kv_cache_manager = True
                logger.warning(
                    "Turning off hybrid kv cache manager because "
                    "`--kv-transfer-config` is set. This will reduce the "
                    "performance of vLLM on LLMs with sliding window attention "
                    "or Mamba attention. If you are a developer of kv connector"
                    ", please consider supporting hybrid kv cache manager for "
                    "your connector by making sure your connector is a subclass"
                    " of `SupportsHMA` defined in kv_connector/v1/base.py and"
                    " use --no-disable-hybrid-kv-cache-manager to start vLLM."
                )"""

NEW_BLOCK = """\
            if self.kv_transfer_config is not None:
                try:
                    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
                    from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma
                    _conn_cls = KVConnectorFactory.get_connector_class(self.kv_transfer_config)
                    if not supports_hma(_conn_cls):
                        need_disable_hybrid_kv_cache_manager = True
                        logger.warning(
                            "Turning off hybrid kv cache manager because "
                            "the connector does not subclass SupportsHMA."
                        )
                    else:
                        logger.info(
                            "Connector %s supports HMA; keeping hybrid kv "
                            "cache manager enabled.", _conn_cls.__name__
                        )
                except Exception as _e:
                    need_disable_hybrid_kv_cache_manager = True
                    logger.warning(
                        "Could not determine HMA support for connector (%s); "
                        "disabling hybrid kv cache manager.", _e
                    )"""

# Backwards-compatible replacement for vLLM <= 0.15 (sets scheduler field directly)
NEW_BLOCK_V015 = """\
            if self.kv_transfer_config is not None:
                try:
                    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
                    from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma
                    _conn_cls = KVConnectorFactory.get_connector_class(self.kv_transfer_config)
                    if not supports_hma(_conn_cls):
                        logger.warning(
                            "Turning off hybrid kv cache manager because "
                            "the connector does not subclass SupportsHMA."
                        )
                        self.scheduler_config.disable_hybrid_kv_cache_manager = True
                    else:
                        logger.info(
                            "Connector %s supports HMA; keeping hybrid kv "
                            "cache manager enabled.", _conn_cls.__name__
                        )
                except Exception as _e:
                    logger.warning(
                        "Could not determine HMA support for connector (%s); "
                        "disabling hybrid kv cache manager.", _e
                    )
                    self.scheduler_config.disable_hybrid_kv_cache_manager = True"""


def main():
    config_path = find_vllm_config()
    print(f"Patching: {config_path}")

    content = config_path.read_text()

    if "supports_hma(_conn_cls)" in content:
        print("Already patched — skipping.")
        return

    if OLD_BLOCK_V016 in content:
        patched = content.replace(OLD_BLOCK_V016, NEW_BLOCK, 1)
        config_path.write_text(patched)
        print("SUCCESS: Patched vLLM 0.16.x config for conditional HMA support.")
    elif OLD_BLOCK_V015 in content:
        patched = content.replace(OLD_BLOCK_V015, NEW_BLOCK_V015, 1)
        config_path.write_text(patched)
        print("SUCCESS: Patched vLLM 0.15.x config for conditional HMA support.")
    else:
        print("ERROR: Could not find the expected code block to patch.")
        print("The vLLM version may have changed. Manual patching required.")
        sys.exit(1)


if __name__ == "__main__":
    main()
