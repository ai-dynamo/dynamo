#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Configuration loading and pool selection logic for the Global Router.

The config file defines:
- Prefill and decode pool namespaces
- Grid-based pool selection strategies mapping (ISL, TTFT) -> prefill pool
  and (context_length, ITL) -> decode pool
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PrefillPoolSelectionStrategy:
    """Strategy for selecting prefill pools based on ISL and TTFT target."""

    ttft_min: float
    ttft_max: float
    ttft_resolution: int
    isl_min: int
    isl_max: int
    isl_resolution: int
    prefill_pool_mapping: List[List[int]]

    @property
    def ttft_step(self) -> float:
        """Step size for TTFT grid."""
        return (self.ttft_max - self.ttft_min) / self.ttft_resolution

    @property
    def isl_step(self) -> float:
        """Step size for ISL grid."""
        return (self.isl_max - self.isl_min) / self.isl_resolution

    def select_pool(self, isl: int, ttft_target: Optional[float] = None) -> int:
        """
        Select prefill pool based on ISL and TTFT target.

        Args:
            isl: Input sequence length (number of tokens)
            ttft_target: Target time to first token in ms. If None, uses middle of range.

        Returns:
            Pool index from prefill_pool_mapping
        """
        if ttft_target is None:
            ttft_target = (self.ttft_min + self.ttft_max) / 2

        # Compute grid indices with clamping
        isl_idx = self._clamp_index(
            (isl - self.isl_min) / self.isl_step, self.isl_resolution
        )
        ttft_idx = self._clamp_index(
            (ttft_target - self.ttft_min) / self.ttft_step, self.ttft_resolution
        )

        pool_idx = self.prefill_pool_mapping[isl_idx][ttft_idx]
        logger.debug(
            f"Prefill pool selection: ISL={isl}, TTFT={ttft_target} -> "
            f"grid[{isl_idx}][{ttft_idx}] -> pool {pool_idx}"
        )
        return pool_idx

    @staticmethod
    def _clamp_index(value: float, resolution: int) -> int:
        """Clamp index to valid grid range."""
        return max(0, min(int(value), resolution - 1))


@dataclass
class DecodePoolSelectionStrategy:
    """Strategy for selecting decode pools based on context length and ITL target."""

    itl_min: float
    itl_max: float
    itl_resolution: int
    context_length_min: int
    context_length_max: int
    context_length_resolution: int
    decode_pool_mapping: List[List[int]]

    @property
    def itl_step(self) -> float:
        """Step size for ITL grid."""
        return (self.itl_max - self.itl_min) / self.itl_resolution

    @property
    def context_length_step(self) -> float:
        """Step size for context length grid."""
        return (
            self.context_length_max - self.context_length_min
        ) / self.context_length_resolution

    def select_pool(
        self, context_length: int, itl_target: Optional[float] = None
    ) -> int:
        """
        Select decode pool based on context length and ITL target.

        Args:
            context_length: Total context length (prompt + generated tokens so far)
            itl_target: Target inter-token latency in ms. If None, uses middle of range.

        Returns:
            Pool index from decode_pool_mapping
        """
        if itl_target is None:
            itl_target = (self.itl_min + self.itl_max) / 2

        # Compute grid indices with clamping
        ctx_idx = self._clamp_index(
            (context_length - self.context_length_min) / self.context_length_step,
            self.context_length_resolution,
        )
        itl_idx = self._clamp_index(
            (itl_target - self.itl_min) / self.itl_step, self.itl_resolution
        )

        pool_idx = self.decode_pool_mapping[ctx_idx][itl_idx]
        logger.debug(
            f"Decode pool selection: context_length={context_length}, ITL={itl_target} -> "
            f"grid[{ctx_idx}][{itl_idx}] -> pool {pool_idx}"
        )
        return pool_idx

    @staticmethod
    def _clamp_index(value: float, resolution: int) -> int:
        """Clamp index to valid grid range."""
        return max(0, min(int(value), resolution - 1))


@dataclass
class GlobalRouterConfig:
    """Configuration for the Global Router."""

    num_prefill_pools: int
    num_decode_pools: int
    prefill_pool_dynamo_namespaces: List[str]
    decode_pool_dynamo_namespaces: List[str]
    prefill_pool_selection_strategy: PrefillPoolSelectionStrategy
    decode_pool_selection_strategy: DecodePoolSelectionStrategy

    def validate(self) -> None:
        """Validate configuration consistency."""
        if len(self.prefill_pool_dynamo_namespaces) != self.num_prefill_pools:
            raise ValueError(
                f"num_prefill_pools ({self.num_prefill_pools}) does not match "
                f"prefill_pool_dynamo_namespaces length ({len(self.prefill_pool_dynamo_namespaces)})"
            )

        if len(self.decode_pool_dynamo_namespaces) != self.num_decode_pools:
            raise ValueError(
                f"num_decode_pools ({self.num_decode_pools}) does not match "
                f"decode_pool_dynamo_namespaces length ({len(self.decode_pool_dynamo_namespaces)})"
            )

        # Validate prefill strategy ranges and resolutions
        prefill_strategy = self.prefill_pool_selection_strategy
        if prefill_strategy.isl_resolution <= 0:
            raise ValueError(
                f"isl_resolution must be positive, got {prefill_strategy.isl_resolution}"
            )
        if prefill_strategy.ttft_resolution <= 0:
            raise ValueError(
                f"ttft_resolution must be positive, got {prefill_strategy.ttft_resolution}"
            )
        if prefill_strategy.isl_min >= prefill_strategy.isl_max:
            raise ValueError(
                f"isl_min ({prefill_strategy.isl_min}) must be less than "
                f"isl_max ({prefill_strategy.isl_max})"
            )
        if prefill_strategy.ttft_min >= prefill_strategy.ttft_max:
            raise ValueError(
                f"ttft_min ({prefill_strategy.ttft_min}) must be less than "
                f"ttft_max ({prefill_strategy.ttft_max})"
            )

        # Validate decode strategy ranges and resolutions
        decode_strategy = self.decode_pool_selection_strategy
        if decode_strategy.context_length_resolution <= 0:
            raise ValueError(
                f"context_length_resolution must be positive, got {decode_strategy.context_length_resolution}"
            )
        if decode_strategy.itl_resolution <= 0:
            raise ValueError(
                f"itl_resolution must be positive, got {decode_strategy.itl_resolution}"
            )
        if decode_strategy.context_length_min >= decode_strategy.context_length_max:
            raise ValueError(
                f"context_length_min ({decode_strategy.context_length_min}) must be less than "
                f"context_length_max ({decode_strategy.context_length_max})"
            )
        if decode_strategy.itl_min >= decode_strategy.itl_max:
            raise ValueError(
                f"itl_min ({decode_strategy.itl_min}) must be less than "
                f"itl_max ({decode_strategy.itl_max})"
            )

        # Validate mapping dimensions match resolution
        if (
            len(prefill_strategy.prefill_pool_mapping)
            != prefill_strategy.isl_resolution
        ):
            raise ValueError(
                f"prefill_pool_mapping rows ({len(prefill_strategy.prefill_pool_mapping)}) "
                f"does not match isl_resolution ({prefill_strategy.isl_resolution})"
            )

        for i, row in enumerate(prefill_strategy.prefill_pool_mapping):
            if len(row) != prefill_strategy.ttft_resolution:
                raise ValueError(
                    f"prefill_pool_mapping row {i} length ({len(row)}) "
                    f"does not match ttft_resolution ({prefill_strategy.ttft_resolution})"
                )
            for pool_idx in row:
                if pool_idx < 0 or pool_idx >= self.num_prefill_pools:
                    raise ValueError(
                        f"Invalid prefill pool index {pool_idx} in mapping "
                        f"(must be 0 to {self.num_prefill_pools - 1})"
                    )

        decode_strategy = self.decode_pool_selection_strategy
        if (
            len(decode_strategy.decode_pool_mapping)
            != decode_strategy.context_length_resolution
        ):
            raise ValueError(
                f"decode_pool_mapping rows ({len(decode_strategy.decode_pool_mapping)}) "
                f"does not match context_length_resolution ({decode_strategy.context_length_resolution})"
            )

        for i, row in enumerate(decode_strategy.decode_pool_mapping):
            if len(row) != decode_strategy.itl_resolution:
                raise ValueError(
                    f"decode_pool_mapping row {i} length ({len(row)}) "
                    f"does not match itl_resolution ({decode_strategy.itl_resolution})"
                )
            for pool_idx in row:
                if pool_idx < 0 or pool_idx >= self.num_decode_pools:
                    raise ValueError(
                        f"Invalid decode pool index {pool_idx} in mapping "
                        f"(must be 0 to {self.num_decode_pools - 1})"
                    )


def load_config(config_path: str | Path) -> GlobalRouterConfig:
    """
    Load Global Router configuration from JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        GlobalRouterConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = json.load(f)

    logger.info(f"Loading global router config from {config_path}")

    # Parse prefill selection strategy
    prefill_strategy_data = data["prefill_pool_selection_strategy"]
    prefill_strategy = PrefillPoolSelectionStrategy(
        ttft_min=prefill_strategy_data["ttft_min"],
        ttft_max=prefill_strategy_data["ttft_max"],
        ttft_resolution=prefill_strategy_data["ttft_resolution"],
        isl_min=prefill_strategy_data["isl_min"],
        isl_max=prefill_strategy_data["isl_max"],
        isl_resolution=prefill_strategy_data["isl_resolution"],
        prefill_pool_mapping=prefill_strategy_data["prefill_pool_mapping"],
    )

    # Parse decode selection strategy
    decode_strategy_data = data["decode_pool_selection_strategy"]
    decode_strategy = DecodePoolSelectionStrategy(
        itl_min=decode_strategy_data["itl_min"],
        itl_max=decode_strategy_data["itl_max"],
        itl_resolution=decode_strategy_data["itl_resolution"],
        context_length_min=decode_strategy_data["context_length_min"],
        context_length_max=decode_strategy_data["context_length_max"],
        context_length_resolution=decode_strategy_data["context_length_resolution"],
        decode_pool_mapping=decode_strategy_data["decode_pool_mapping"],
    )

    config = GlobalRouterConfig(
        num_prefill_pools=data["num_prefill_pools"],
        num_decode_pools=data["num_decode_pools"],
        prefill_pool_dynamo_namespaces=data["prefill_pool_dynamo_namespaces"],
        decode_pool_dynamo_namespaces=data["decode_pool_dynamo_namespaces"],
        prefill_pool_selection_strategy=prefill_strategy,
        decode_pool_selection_strategy=decode_strategy,
    )

    config.validate()
    logger.info(
        f"Loaded config: {config.num_prefill_pools} prefill pools, "
        f"{config.num_decode_pools} decode pools"
    )

    return config
