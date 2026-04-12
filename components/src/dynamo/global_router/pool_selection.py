#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Configuration loading and pool selection logic for the Global Router.

Supports two modes:
- "disagg" (default): Separate prefill and decode pools with independent
  grid-based selection strategies mapping (ISL, TTFT) -> prefill pool
  and (context_length, ITL) -> decode pool.
- "agg": Unified pools handling both prefill and decode (chunked prefill),
  with grid-based selection mapping (ISL, ITL) -> agg pool.

Both modes support optional priority-based pool overrides from agent hints.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PriorityPoolOverride:
    """Override pool selection based on request priority from agent hints."""

    min_priority: int  # inclusive lower bound
    max_priority: int  # inclusive upper bound
    target_pool: int  # pool index to route to when priority matches


def _apply_priority_overrides(
    base_pool: int,
    priority: Optional[int],
    overrides: List[PriorityPoolOverride],
) -> int:
    """Apply priority-based pool overrides. First matching rule wins."""
    if priority is None or not overrides:
        return base_pool
    for rule in overrides:
        if rule.min_priority <= priority <= rule.max_priority:
            return rule.target_pool
    return base_pool


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
    priority_overrides: List[PriorityPoolOverride] = field(default_factory=list)

    @property
    def ttft_step(self) -> float:
        """Step size for TTFT grid."""
        return (self.ttft_max - self.ttft_min) / self.ttft_resolution

    @property
    def isl_step(self) -> float:
        """Step size for ISL grid."""
        return (self.isl_max - self.isl_min) / self.isl_resolution

    def select_pool(
        self,
        isl: int,
        ttft_target: Optional[float] = None,
        priority: Optional[int] = None,
    ) -> int:
        """
        Select prefill pool based on ISL, TTFT target, and optional priority.

        Args:
            isl: Input sequence length (number of tokens)
            ttft_target: Target time to first token in ms. If None, uses middle of range.
            priority: Request priority from agent hints. If set and a priority
                override rule matches, the override takes precedence over the grid.

        Returns:
            Pool index from prefill_pool_mapping or a priority override
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
        pool_idx = _apply_priority_overrides(
            pool_idx, priority, self.priority_overrides
        )
        logger.debug(
            f"Prefill pool selection: ISL={isl}, TTFT={ttft_target}, "
            f"priority={priority} -> pool {pool_idx}"
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
    priority_overrides: List[PriorityPoolOverride] = field(default_factory=list)

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
        self,
        context_length: int,
        itl_target: Optional[float] = None,
        priority: Optional[int] = None,
    ) -> int:
        """
        Select decode pool based on context length, ITL target, and optional priority.

        Args:
            context_length: Total context length (prompt + generated tokens so far)
            itl_target: Target inter-token latency in ms. If None, uses middle of range.
            priority: Request priority from agent hints. If set and a priority
                override rule matches, the override takes precedence over the grid.

        Returns:
            Pool index from decode_pool_mapping or a priority override
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
        pool_idx = _apply_priority_overrides(
            pool_idx, priority, self.priority_overrides
        )
        logger.debug(
            f"Decode pool selection: context_length={context_length}, ITL={itl_target}, "
            f"priority={priority} -> pool {pool_idx}"
        )
        return pool_idx

    @staticmethod
    def _clamp_index(value: float, resolution: int) -> int:
        """Clamp index to valid grid range."""
        return max(0, min(int(value), resolution - 1))


@dataclass
class AggPoolSelectionStrategy:
    """Strategy for selecting agg pools based on TTFT and ITL targets.

    In aggregated mode, each pool handles both prefill and decode. Since both
    phases happen in the same pool, both SLA targets matter for a single routing
    decision. The grid maps (TTFT target, ITL target) -> pool index.

    This works regardless of whether chunked prefill is enabled:
    - With chunked prefill: ITL reflects combined prefill+decode contention.
    - Without chunked prefill: TTFT captures the blocking prefill cost,
      ITL captures pure decode performance.
    """

    ttft_min: float
    ttft_max: float
    ttft_resolution: int
    itl_min: float
    itl_max: float
    itl_resolution: int
    agg_pool_mapping: List[List[int]]
    priority_overrides: List[PriorityPoolOverride] = field(default_factory=list)

    @property
    def ttft_step(self) -> float:
        """Step size for TTFT grid."""
        return (self.ttft_max - self.ttft_min) / self.ttft_resolution

    @property
    def itl_step(self) -> float:
        """Step size for ITL grid."""
        return (self.itl_max - self.itl_min) / self.itl_resolution

    def select_pool(
        self,
        ttft_target: Optional[float] = None,
        itl_target: Optional[float] = None,
        priority: Optional[int] = None,
    ) -> int:
        """
        Select agg pool based on TTFT target, ITL target, and optional priority.

        Args:
            ttft_target: Target time to first token in ms. If None, uses middle of range.
            itl_target: Target inter-token latency in ms. If None, uses middle of range.
            priority: Request priority from agent hints. If set and a priority
                override rule matches, the override takes precedence over the grid.

        Returns:
            Pool index from agg_pool_mapping or a priority override
        """
        if ttft_target is None:
            ttft_target = (self.ttft_min + self.ttft_max) / 2
        if itl_target is None:
            itl_target = (self.itl_min + self.itl_max) / 2

        # Compute grid indices with clamping
        ttft_idx = self._clamp_index(
            (ttft_target - self.ttft_min) / self.ttft_step, self.ttft_resolution
        )
        itl_idx = self._clamp_index(
            (itl_target - self.itl_min) / self.itl_step, self.itl_resolution
        )

        pool_idx = self.agg_pool_mapping[ttft_idx][itl_idx]
        pool_idx = _apply_priority_overrides(
            pool_idx, priority, self.priority_overrides
        )
        logger.debug(
            f"Agg pool selection: TTFT={ttft_target}, ITL={itl_target}, "
            f"priority={priority} -> pool {pool_idx}"
        )
        return pool_idx

    @staticmethod
    def _clamp_index(value: float, resolution: int) -> int:
        """Clamp index to valid grid range."""
        return max(0, min(int(value), resolution - 1))


@dataclass
class GlobalRouterConfig:
    """Configuration for the Global Router.

    Supports two modes:
    - "disagg" (default): separate prefill and decode pools
    - "agg": unified pools handling both prefill and decode
    """

    mode: str = "disagg"  # "disagg" or "agg"

    # --- disagg-only fields (required when mode="disagg") ---
    num_prefill_pools: Optional[int] = None
    num_decode_pools: Optional[int] = None
    prefill_pool_dynamo_namespaces: Optional[List[str]] = None
    decode_pool_dynamo_namespaces: Optional[List[str]] = None
    prefill_pool_selection_strategy: Optional[PrefillPoolSelectionStrategy] = None
    decode_pool_selection_strategy: Optional[DecodePoolSelectionStrategy] = None

    # --- agg-only fields (required when mode="agg") ---
    num_agg_pools: Optional[int] = None
    agg_pool_dynamo_namespaces: Optional[List[str]] = None
    agg_pool_selection_strategy: Optional[AggPoolSelectionStrategy] = None

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.mode == "disagg":
            self._validate_disagg()
        elif self.mode == "agg":
            self._validate_agg()
        else:
            raise ValueError(f"Unknown mode '{self.mode}', must be 'disagg' or 'agg'")

    def _validate_disagg(self) -> None:
        """Validate disagg mode configuration."""
        if self.num_prefill_pools is None:
            raise ValueError("num_prefill_pools required for disagg mode")
        if self.num_decode_pools is None:
            raise ValueError("num_decode_pools required for disagg mode")
        if self.prefill_pool_dynamo_namespaces is None:
            raise ValueError("prefill_pool_dynamo_namespaces required for disagg mode")
        if self.decode_pool_dynamo_namespaces is None:
            raise ValueError("decode_pool_dynamo_namespaces required for disagg mode")
        if self.prefill_pool_selection_strategy is None:
            raise ValueError("prefill_pool_selection_strategy required for disagg mode")
        if self.decode_pool_selection_strategy is None:
            raise ValueError("decode_pool_selection_strategy required for disagg mode")

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

        # Validate prefill strategy
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

        for i, override in enumerate(prefill_strategy.priority_overrides):
            if override.min_priority > override.max_priority:
                raise ValueError(
                    f"Prefill priority_overrides[{i}]: min_priority "
                    f"({override.min_priority}) must be <= max_priority "
                    f"({override.max_priority})"
                )
            if (
                override.target_pool < 0
                or override.target_pool >= self.num_prefill_pools
            ):
                raise ValueError(
                    f"Prefill priority_overrides[{i}]: invalid target_pool "
                    f"{override.target_pool} (must be 0 to {self.num_prefill_pools - 1})"
                )

        # Validate decode strategy
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

        for i, override in enumerate(decode_strategy.priority_overrides):
            if override.min_priority > override.max_priority:
                raise ValueError(
                    f"Decode priority_overrides[{i}]: min_priority "
                    f"({override.min_priority}) must be <= max_priority "
                    f"({override.max_priority})"
                )
            if (
                override.target_pool < 0
                or override.target_pool >= self.num_decode_pools
            ):
                raise ValueError(
                    f"Decode priority_overrides[{i}]: invalid target_pool "
                    f"{override.target_pool} (must be 0 to {self.num_decode_pools - 1})"
                )

    def _validate_agg(self) -> None:
        """Validate agg mode configuration."""
        if self.num_agg_pools is None:
            raise ValueError("num_agg_pools required for agg mode")
        if self.agg_pool_dynamo_namespaces is None:
            raise ValueError("agg_pool_dynamo_namespaces required for agg mode")
        if self.agg_pool_selection_strategy is None:
            raise ValueError("agg_pool_selection_strategy required for agg mode")

        if len(self.agg_pool_dynamo_namespaces) != self.num_agg_pools:
            raise ValueError(
                f"num_agg_pools ({self.num_agg_pools}) does not match "
                f"agg_pool_dynamo_namespaces length ({len(self.agg_pool_dynamo_namespaces)})"
            )

        agg_strategy = self.agg_pool_selection_strategy
        if agg_strategy.ttft_resolution <= 0:
            raise ValueError(
                f"ttft_resolution must be positive, got {agg_strategy.ttft_resolution}"
            )
        if agg_strategy.itl_resolution <= 0:
            raise ValueError(
                f"itl_resolution must be positive, got {agg_strategy.itl_resolution}"
            )
        if agg_strategy.ttft_min >= agg_strategy.ttft_max:
            raise ValueError(
                f"ttft_min ({agg_strategy.ttft_min}) must be less than "
                f"ttft_max ({agg_strategy.ttft_max})"
            )
        if agg_strategy.itl_min >= agg_strategy.itl_max:
            raise ValueError(
                f"itl_min ({agg_strategy.itl_min}) must be less than "
                f"itl_max ({agg_strategy.itl_max})"
            )

        # Validate mapping dimensions
        if len(agg_strategy.agg_pool_mapping) != agg_strategy.ttft_resolution:
            raise ValueError(
                f"agg_pool_mapping rows ({len(agg_strategy.agg_pool_mapping)}) "
                f"does not match ttft_resolution ({agg_strategy.ttft_resolution})"
            )

        for i, row in enumerate(agg_strategy.agg_pool_mapping):
            if len(row) != agg_strategy.itl_resolution:
                raise ValueError(
                    f"agg_pool_mapping row {i} length ({len(row)}) "
                    f"does not match itl_resolution ({agg_strategy.itl_resolution})"
                )
            for pool_idx in row:
                if pool_idx < 0 or pool_idx >= self.num_agg_pools:
                    raise ValueError(
                        f"Invalid agg pool index {pool_idx} in mapping "
                        f"(must be 0 to {self.num_agg_pools - 1})"
                    )

        for i, override in enumerate(agg_strategy.priority_overrides):
            if override.min_priority > override.max_priority:
                raise ValueError(
                    f"Agg priority_overrides[{i}]: min_priority "
                    f"({override.min_priority}) must be <= max_priority "
                    f"({override.max_priority})"
                )
            if override.target_pool < 0 or override.target_pool >= self.num_agg_pools:
                raise ValueError(
                    f"Agg priority_overrides[{i}]: invalid target_pool "
                    f"{override.target_pool} (must be 0 to {self.num_agg_pools - 1})"
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

    mode = data.get("mode", "disagg")

    if mode == "disagg":
        config = _load_disagg_config(data, mode)
    elif mode == "agg":
        config = _load_agg_config(data, mode)
    else:
        raise ValueError(f"Unknown mode '{mode}' in config")

    config.validate()
    return config


def _load_disagg_config(data: dict, mode: str) -> GlobalRouterConfig:
    """Load disagg mode configuration from parsed JSON data."""
    # Parse prefill selection strategy
    prefill_strategy_data = data["prefill_pool_selection_strategy"]
    prefill_priority_overrides = [
        PriorityPoolOverride(**rule)
        for rule in prefill_strategy_data.get("priority_overrides", [])
    ]
    prefill_strategy = PrefillPoolSelectionStrategy(
        ttft_min=prefill_strategy_data["ttft_min"],
        ttft_max=prefill_strategy_data["ttft_max"],
        ttft_resolution=prefill_strategy_data["ttft_resolution"],
        isl_min=prefill_strategy_data["isl_min"],
        isl_max=prefill_strategy_data["isl_max"],
        isl_resolution=prefill_strategy_data["isl_resolution"],
        prefill_pool_mapping=prefill_strategy_data["prefill_pool_mapping"],
        priority_overrides=prefill_priority_overrides,
    )

    # Parse decode selection strategy
    decode_strategy_data = data["decode_pool_selection_strategy"]
    decode_priority_overrides = [
        PriorityPoolOverride(**rule)
        for rule in decode_strategy_data.get("priority_overrides", [])
    ]
    decode_strategy = DecodePoolSelectionStrategy(
        itl_min=decode_strategy_data["itl_min"],
        itl_max=decode_strategy_data["itl_max"],
        itl_resolution=decode_strategy_data["itl_resolution"],
        context_length_min=decode_strategy_data["context_length_min"],
        context_length_max=decode_strategy_data["context_length_max"],
        context_length_resolution=decode_strategy_data["context_length_resolution"],
        decode_pool_mapping=decode_strategy_data["decode_pool_mapping"],
        priority_overrides=decode_priority_overrides,
    )

    config = GlobalRouterConfig(
        mode=mode,
        num_prefill_pools=data["num_prefill_pools"],
        num_decode_pools=data["num_decode_pools"],
        prefill_pool_dynamo_namespaces=data["prefill_pool_dynamo_namespaces"],
        decode_pool_dynamo_namespaces=data["decode_pool_dynamo_namespaces"],
        prefill_pool_selection_strategy=prefill_strategy,
        decode_pool_selection_strategy=decode_strategy,
    )

    logger.info(
        f"Loaded disagg config: {config.num_prefill_pools} prefill pools, "
        f"{config.num_decode_pools} decode pools"
    )
    return config


def _load_agg_config(data: dict, mode: str) -> GlobalRouterConfig:
    """Load agg mode configuration from parsed JSON data."""
    agg_strategy_data = data["agg_pool_selection_strategy"]
    agg_priority_overrides = [
        PriorityPoolOverride(**rule)
        for rule in agg_strategy_data.get("priority_overrides", [])
    ]
    agg_strategy = AggPoolSelectionStrategy(
        ttft_min=agg_strategy_data["ttft_min"],
        ttft_max=agg_strategy_data["ttft_max"],
        ttft_resolution=agg_strategy_data["ttft_resolution"],
        itl_min=agg_strategy_data["itl_min"],
        itl_max=agg_strategy_data["itl_max"],
        itl_resolution=agg_strategy_data["itl_resolution"],
        agg_pool_mapping=agg_strategy_data["agg_pool_mapping"],
        priority_overrides=agg_priority_overrides,
    )

    config = GlobalRouterConfig(
        mode=mode,
        num_agg_pools=data["num_agg_pools"],
        agg_pool_dynamo_namespaces=data["agg_pool_dynamo_namespaces"],
        agg_pool_selection_strategy=agg_strategy,
    )

    logger.info(f"Loaded agg config: {config.num_agg_pools} agg pools")
    return config
