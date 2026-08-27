# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic vLLM hybrid KV-cache hit simulation.

This models prefix hits, physical slot occupancy, eviction, and CPU admission.
It does not model inference latency or GPU/CPU transfer duration.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CacheKey = tuple[int, str, str]


@dataclass(frozen=True)
class HybridCacheGroup:
    """One vLLM hybrid KV-cache group."""

    group_index: int
    block_size: int
    sliding_window: int | None = None
    gpu_spec_group: str | None = None
    use_eagle: bool = False
    offload: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HybridCacheGroup:
        return cls(
            group_index=int(raw["group_index"]),
            block_size=int(raw["block_size"]),
            sliding_window=(
                int(raw["sliding_window"])
                if raw.get("sliding_window") is not None
                else None
            ),
            gpu_spec_group=(
                str(raw["gpu_spec_group"])
                if raw.get("gpu_spec_group") is not None
                else None
            ),
            use_eagle=bool(raw.get("use_eagle", False)),
            offload=bool(raw.get("offload", True)),
        )


@dataclass(frozen=True)
class HybridCacheConfig:
    """Physical cache geometry and offload admission policy."""

    scheduler_block_size: int
    hash_block_size: int
    gpu_capacity_slots: int
    cpu_capacity_slots: int
    cpu_slot_bytes: int
    groups: tuple[HybridCacheGroup, ...]
    blocks_per_chunk: int = 1
    store_threshold: int = 1
    retention_interval: int = 8192
    tracker_capacity: int = 4_000_000

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HybridCacheConfig:
        slot_bytes = int(raw["cpu_slot_bytes"])
        if slot_bytes <= 0:
            raise ValueError("cpu_slot_bytes must be greater than zero")
        if raw.get("cpu_capacity_slots") is not None:
            capacity_slots = int(raw["cpu_capacity_slots"])
        elif raw.get("cpu_capacity_bytes") is not None:
            capacity_slots = int(raw["cpu_capacity_bytes"]) // slot_bytes
        else:
            raise ValueError("cpu_capacity_slots or cpu_capacity_bytes is required")
        config = cls(
            scheduler_block_size=int(raw["scheduler_block_size"]),
            hash_block_size=int(raw["hash_block_size"]),
            gpu_capacity_slots=int(raw["gpu_capacity_slots"]),
            cpu_capacity_slots=capacity_slots,
            cpu_slot_bytes=slot_bytes,
            groups=tuple(HybridCacheGroup.from_dict(item) for item in raw["groups"]),
            blocks_per_chunk=int(raw.get("blocks_per_chunk", 1)),
            store_threshold=int(raw.get("store_threshold", 1)),
            retention_interval=int(raw.get("retention_interval", 8192)),
            tracker_capacity=int(raw.get("tracker_capacity", 4_000_000)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_values = {
            "scheduler_block_size": self.scheduler_block_size,
            "hash_block_size": self.hash_block_size,
            "gpu_capacity_slots": self.gpu_capacity_slots,
            "cpu_capacity_slots": self.cpu_capacity_slots,
            "cpu_slot_bytes": self.cpu_slot_bytes,
            "blocks_per_chunk": self.blocks_per_chunk,
            "store_threshold": self.store_threshold,
            "retention_interval": self.retention_interval,
            "tracker_capacity": self.tracker_capacity,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not self.groups:
            raise ValueError("at least one cache group is required")
        indices = [group.group_index for group in self.groups]
        if len(indices) != len(set(indices)):
            raise ValueError("cache group indices must be unique")
        full_groups = [group for group in self.groups if group.sliding_window is None]
        if len(full_groups) != 1:
            raise ValueError("exactly one full-attention cache group is required")
        if full_groups[0].block_size != self.scheduler_block_size:
            raise ValueError(
                "full-attention block size must match scheduler_block_size"
            )
        for group in self.groups:
            if group.block_size <= 0:
                raise ValueError("group block sizes must be greater than zero")
            if group.block_size % self.hash_block_size:
                raise ValueError(
                    "group block sizes must be divisible by hash_block_size"
                )
            if self.scheduler_block_size % group.block_size:
                raise ValueError(
                    "scheduler_block_size must be divisible by every group block size"
                )
            if self.retention_interval % group.block_size:
                raise ValueError(
                    "retention_interval must be divisible by every group block size"
                )
            if group.sliding_window is not None and group.sliding_window <= 0:
                raise ValueError("sliding windows must be greater than zero")


@dataclass(frozen=True)
class HybridCacheRequest:
    """One request expressed as cumulative hashes at hash-block boundaries."""

    request_id: str
    input_length: int
    output_length: int
    lineage: tuple[str, ...]
    trace_block_size: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HybridCacheRequest:
        event = raw.get("event", raw)
        if not isinstance(event, dict):
            raise TypeError("event must be an object")
        if isinstance(event.get("request"), dict):
            request = event["request"]
            replay = request["replay"]
            request_id = str(request["request_id"])
            input_length = int(replay["input_length"])
            output_tokens = request.get("output_tokens")
            output_length = int(output_tokens) if output_tokens is not None else 0
            lineage = replay["input_sequence_hashes"]
            trace_block_size = replay.get("trace_block_size")
        else:
            request_id = str(event["request_id"])
            input_length = int(event["input_length"])
            output_tokens = (
                event["output_length"]
                if "output_length" in event
                else event.get("output_tokens")
            )
            output_length = int(output_tokens) if output_tokens is not None else 0
            lineage = event.get("lineage", event.get("lineage_b4"))
            trace_block_size = event.get("trace_block_size")
        if not isinstance(lineage, list):
            raise TypeError("request must contain cumulative input lineage hashes")
        canonical_lineage = tuple(
            json.dumps(value, separators=(",", ":"), sort_keys=True)
            for value in lineage
        )
        return cls(
            request_id,
            input_length,
            output_length,
            canonical_lineage,
            int(trace_block_size) if trace_block_size is not None else None,
        )


@dataclass(frozen=True)
class HybridCacheRequestResult:
    request_id: str
    input_tokens: int
    output_tokens: int
    gpu_hit_tokens: int
    cpu_hit_tokens: int
    combined_hit_tokens: int
    gpu_occupancy_slots: int
    cpu_occupancy_slots: int
    cpu_reserved_bytes: int
    cpu_store_offers: int
    cpu_store_offers_by_group: dict[int, int]
    cpu_admissions: int
    cpu_evictions: int
    gpu_evictions: int


class VllmHybridCacheSimulator:
    """Replay vLLM hybrid GPU and CPU cache policy in request order."""

    def __init__(self, config: HybridCacheConfig):
        config.validate()
        self.config = config
        self._groups = {group.group_index: group for group in config.groups}
        self._full_group = next(
            group for group in config.groups if group.sliding_window is None
        )
        eagle_specs = {
            group.gpu_spec_group or f"group-{group.group_index}"
            for group in config.groups
            if group.use_eagle
        }
        self.effective_eagle_groups = frozenset(
            group.group_index
            for group in config.groups
            if (group.gpu_spec_group or f"group-{group.group_index}") in eagle_specs
        )
        self._gpu_by_group: dict[int, set[CacheKey]] = {
            group.group_index: set() for group in config.groups
        }
        self._gpu_lru: OrderedDict[CacheKey, None] = OrderedDict()
        self._cpu_lru: OrderedDict[CacheKey, None] = OrderedDict()
        self._tracker: OrderedDict[CacheKey, int] = OrderedDict()

    def _prompt_key(
        self, request: HybridCacheRequest, group: HybridCacheGroup, block_index: int
    ) -> CacheKey:
        token_end = (block_index + 1) * group.block_size
        lineage_index = token_end // self.config.hash_block_size - 1
        if lineage_index >= len(request.lineage):
            raise ValueError(
                f"request {request.request_id} lacks lineage through token {token_end}"
            )
        return group.group_index, "prompt", request.lineage[lineage_index]

    def _effective_eagle(self, group: HybridCacheGroup) -> bool:
        return group.group_index in self.effective_eagle_groups

    def _cpu_chunk_size(self, group: HybridCacheGroup) -> int:
        return group.block_size * self.config.blocks_per_chunk

    def _cpu_chunk_key(
        self, request: HybridCacheRequest, group: HybridCacheGroup, chunk_index: int
    ) -> CacheKey:
        block_index = (chunk_index + 1) * self.config.blocks_per_chunk - 1
        return self._prompt_key(request, group, block_index)

    def _cpu_sliding_need(self, group: HybridCacheGroup) -> int:
        if group.sliding_window is None:
            raise ValueError("sliding-window group required")
        chunk_size = self._cpu_chunk_size(group)
        return (group.sliding_window + chunk_size - 1) // chunk_size

    def _sliding_need(self, group: HybridCacheGroup, *, gpu: bool) -> int:
        if group.sliding_window is None:
            raise ValueError("sliding-window group required")
        blocks = (group.sliding_window + group.block_size - 1) // group.block_size
        return blocks + int(gpu and self._effective_eagle(group))

    def _sliding_gpu_hit(
        self, request: HybridCacheRequest, group: HybridCacheGroup, max_length: int
    ) -> int:
        block_count = max_length // group.block_size
        needed = self._sliding_need(group, gpu=True)
        consecutive = 0
        for block_index in range(block_count - 1, -1, -1):
            key = self._prompt_key(request, group, block_index)
            if key in self._gpu_by_group[group.group_index]:
                if consecutive == 0:
                    post_pop = (
                        block_index if self._effective_eagle(group) else block_index + 1
                    )
                    if post_pop * group.block_size % self.config.scheduler_block_size:
                        continue
                consecutive += 1
                if consecutive >= needed:
                    blocks = (
                        block_index + consecutive - int(self._effective_eagle(group))
                    )
                    while blocks * group.block_size % self.config.scheduler_block_size:
                        blocks -= 1
                    return blocks * group.block_size
            else:
                consecutive = 0
        blocks = consecutive - int(self._effective_eagle(group) and consecutive > 0)
        while (
            blocks > 0 and blocks * group.block_size % self.config.scheduler_block_size
        ):
            blocks -= 1
        return blocks * group.block_size

    def _gpu_hit(self, request: HybridCacheRequest) -> int:
        max_hit = request.input_length - 1
        full_blocks = 0
        for block_index in range(max_hit // self._full_group.block_size):
            key = self._prompt_key(request, self._full_group, block_index)
            if key not in self._gpu_by_group[self._full_group.group_index]:
                break
            full_blocks += 1
        hit_tokens = full_blocks * self._full_group.block_size
        if hit_tokens == 0:
            return 0
        while True:
            previous = hit_tokens
            for group in self.config.groups:
                if group.sliding_window is None:
                    continue
                query_max = min(
                    max_hit,
                    hit_tokens
                    + (group.block_size if self._effective_eagle(group) else 0),
                )
                hit_tokens = min(
                    hit_tokens, self._sliding_gpu_hit(request, group, query_max)
                )
                if hit_tokens == 0:
                    return 0
            if hit_tokens >= previous:
                return hit_tokens

    def _track_store_offer(self, key: CacheKey) -> None:
        if key in self._tracker:
            self._tracker.move_to_end(key)
            self._tracker[key] += 1
            return
        if len(self._tracker) >= self.config.tracker_capacity:
            self._tracker.popitem(last=False)
        self._tracker[key] = 1

    def _cpu_lookup(self, request: HybridCacheRequest, gpu_tokens: int) -> int:
        max_hit = request.input_length - 1
        full_chunk_size = self._cpu_chunk_size(self._full_group)
        start_chunk = gpu_tokens // full_chunk_size
        end_chunk = max_hit // full_chunk_size
        full_hits = 0
        for chunk_index in range(start_chunk, end_chunk):
            key = self._cpu_chunk_key(request, self._full_group, chunk_index)
            if key not in self._cpu_lru:
                break
            full_hits += 1
        if full_hits == 0:
            return 0
        candidate = min(max_hit, (start_chunk + full_hits) * full_chunk_size)
        if candidate - gpu_tokens < full_chunk_size:
            return 0
        sliding_groups = sorted(
            (
                group
                for group in self.config.groups
                if group.sliding_window is not None and group.offload
            ),
            key=self._cpu_sliding_need,
            reverse=True,
        )
        for group in sliding_groups:
            chunk_size = self._cpu_chunk_size(group)
            start_chunk = gpu_tokens // chunk_size
            end_chunk = min(
                (candidate + chunk_size - 1) // chunk_size,
                request.input_length // chunk_size,
            )
            needed = self._cpu_sliding_need(group)
            consecutive = 0
            end_of_run = 0
            for chunk_index in range(end_chunk - 1, start_chunk - 1, -1):
                key = self._cpu_chunk_key(request, group, chunk_index)
                consecutive = consecutive + 1 if key in self._cpu_lru else 0
                if consecutive == needed:
                    end_of_run = chunk_index + needed
                    break
            if end_of_run == 0:
                return 0
            candidate = min(candidate, end_of_run * chunk_size)
        external_tokens = max(0, candidate - gpu_tokens)
        if external_tokens:
            for group in self.config.groups:
                if not group.offload:
                    continue
                chunk_size = self._cpu_chunk_size(group)
                first = gpu_tokens // chunk_size
                last = (gpu_tokens + external_tokens + chunk_size - 1) // chunk_size
                chunk_indices = list(range(first, last))
                if group.sliding_window is not None:
                    chunk_indices = chunk_indices[-self._cpu_sliding_need(group) :]
                for chunk_index in chunk_indices:
                    key = self._cpu_chunk_key(request, group, chunk_index)
                    if key in self._cpu_lru:
                        self._cpu_lru.move_to_end(key)
        return external_tokens

    def _store_offer_keys(
        self,
        request: HybridCacheRequest,
        combined_hit_tokens: int,
        external_hit_tokens: int,
        gpu_hit_tokens: int,
    ) -> list[CacheKey]:
        offered: list[CacheKey] = []
        for group in self.config.groups:
            if not group.offload:
                continue
            chunk_size = self._cpu_chunk_size(group)
            chunk_count = request.input_length // chunk_size
            first = combined_hit_tokens // chunk_size if external_hit_tokens else 0
            if group.sliding_window is None:
                chunk_indices = range(first, chunk_count)
            else:
                needed = self._cpu_sliding_need(group)
                full_chunk_size = self._cpu_chunk_size(self._full_group)
                chunks_per_alignment = full_chunk_size // chunk_size
                chunk_indices = (
                    chunk_index
                    for chunk_index in range(first, chunk_count)
                    if chunk_index % chunks_per_alignment
                    >= chunks_per_alignment - needed
                    and (
                        chunk_index * chunk_size >= gpu_hit_tokens
                        or (
                            chunk_index >= gpu_hit_tokens // chunk_size - needed
                            and self._cpu_chunk_key(request, group, chunk_index)
                            in self._gpu_by_group[group.group_index]
                        )
                    )
                )
            for chunk_index in chunk_indices:
                offered.append(self._cpu_chunk_key(request, group, chunk_index))
        return offered

    def _cpu_store(
        self,
        request: HybridCacheRequest,
        combined_hit_tokens: int,
        external_hit_tokens: int,
        gpu_hit_tokens: int,
    ) -> tuple[int, int, list[CacheKey]]:
        offered = self._store_offer_keys(
            request, combined_hit_tokens, external_hit_tokens, gpu_hit_tokens
        )
        for key in offered:
            self._track_store_offer(key)
        eligible = [
            key for key in offered if self._tracker[key] >= self.config.store_threshold
        ]
        to_store = [key for key in eligible if key not in self._cpu_lru]
        protected = set(eligible)
        evictions = max(
            0,
            len(self._cpu_lru) + len(to_store) - self.config.cpu_capacity_slots,
        )
        victims: list[CacheKey] = []
        if evictions:
            for key in self._cpu_lru:
                if key not in protected:
                    victims.append(key)
                    if len(victims) == evictions:
                        break
        # vLLM's CPUOffloadingManager.prepare_store is all-or-nothing: it
        # returns None when the full batch cannot be allocated without
        # evicting a protected or non-evictable block.
        if len(victims) != evictions:
            return 0, 0, offered
        for key in victims:
            del self._cpu_lru[key]
        for key in to_store:
            self._cpu_lru[key] = None
        if to_store:
            for group in self.config.groups:
                if not group.offload:
                    continue
                chunk_size = self._cpu_chunk_size(group)
                chunk_count = request.input_length // chunk_size
                chunk_indices = list(range(chunk_count))
                if group.sliding_window is not None:
                    chunk_indices = chunk_indices[-self._cpu_sliding_need(group) :]
                for chunk_index in reversed(chunk_indices):
                    key = self._cpu_chunk_key(request, group, chunk_index)
                    if key in self._cpu_lru:
                        self._cpu_lru.move_to_end(key)
        return len(to_store), evictions, offered

    def _insert_gpu(self, group_index: int, key: CacheKey) -> int:
        if key in self._gpu_lru:
            self._gpu_lru.move_to_end(key)
            return 0
        evicted = 0
        if len(self._gpu_lru) >= self.config.gpu_capacity_slots:
            victim, _ = self._gpu_lru.popitem(last=False)
            self._gpu_by_group[victim[0]].remove(victim)
            evicted = 1
        self._gpu_lru[key] = None
        self._gpu_by_group[group_index].add(key)
        return evicted

    def _cache_prompt(self, request: HybridCacheRequest) -> int:
        evictions = 0
        aligned = (
            request.input_length
            // self.config.scheduler_block_size
            * self.config.scheduler_block_size
        )
        for group in self.config.groups:
            tokens_to_cache = aligned
            if self._effective_eagle(group) and aligned > 0:
                tokens_to_cache = min(request.input_length, aligned + group.block_size)
            max_blocks = tokens_to_cache // group.block_size
            if group.sliding_window is None:
                block_indices = list(range(max_blocks))
            else:
                needed = self._sliding_need(group, gpu=True)
                shift = int(self._effective_eagle(group))
                blocks_per_retention = (
                    self.config.retention_interval // group.block_size
                )
                selected = {
                    block_index
                    for block_index in range(max_blocks)
                    if block_index >= shift
                    and (block_index - shift) % blocks_per_retention
                    >= blocks_per_retention - needed
                }
                latest = (
                    (request.input_length - 1)
                    // self.config.scheduler_block_size
                    * self.config.scheduler_block_size
                )
                prompt_end = latest // group.block_size + shift
                selected.update(
                    range(max(0, prompt_end - needed), min(max_blocks, prompt_end))
                )
                block_indices = sorted(selected)
            for block_index in block_indices:
                key = self._prompt_key(request, group, block_index)
                evictions += self._insert_gpu(group.group_index, key)
        return evictions

    def _cache_unique_outputs(self, request: HybridCacheRequest) -> int:
        evictions = 0
        total_tokens = request.input_length + request.output_length
        aligned = (
            total_tokens
            // self.config.scheduler_block_size
            * self.config.scheduler_block_size
        )
        for group in self.config.groups:
            tokens_to_cache = aligned
            if self._effective_eagle(group) and aligned > 0:
                tokens_to_cache = min(total_tokens, aligned + group.block_size)
            first = request.input_length // group.block_size
            last = tokens_to_cache // group.block_size
            if group.sliding_window is None:
                block_indices = list(range(first, last))
            else:
                needed = self._sliding_need(group, gpu=True)
                shift = int(self._effective_eagle(group))
                blocks_per_retention = (
                    self.config.retention_interval // group.block_size
                )
                selected = {
                    block_index
                    for block_index in range(first, last)
                    if block_index >= shift
                    and (block_index - shift) % blocks_per_retention
                    >= blocks_per_retention - needed
                }
                latest = (
                    (total_tokens - 1)
                    // self.config.scheduler_block_size
                    * self.config.scheduler_block_size
                )
                prompt_end = latest // group.block_size + shift
                selected.update(
                    range(max(first, prompt_end - needed), min(last, prompt_end))
                )
                block_indices = sorted(selected)
            for block_index in block_indices:
                key = (
                    group.group_index,
                    "output",
                    f"{request.request_id}:{block_index}",
                )
                evictions += self._insert_gpu(group.group_index, key)
                if group.offload:
                    self._track_store_offer(key)
        return evictions

    def process(
        self, request: HybridCacheRequest, *, unique_output_occupancy: bool = False
    ) -> HybridCacheRequestResult:
        """Process one request after all earlier requests have completed."""

        if (
            request.trace_block_size is not None
            and request.trace_block_size != self.config.hash_block_size
        ):
            raise ValueError(
                "request trace_block_size must match the configured hash_block_size"
            )
        gpu_hit_tokens = self._gpu_hit(request)
        cpu_hit_tokens = self._cpu_lookup(request, gpu_hit_tokens)
        combined_hit_tokens = gpu_hit_tokens + cpu_hit_tokens
        admissions, cpu_evictions, offered = self._cpu_store(
            request,
            combined_hit_tokens,
            cpu_hit_tokens,
            gpu_hit_tokens,
        )
        gpu_evictions = self._cache_prompt(request)
        if unique_output_occupancy:
            gpu_evictions += self._cache_unique_outputs(request)
        offers_by_group: dict[int, int] = {}
        for key in offered:
            offers_by_group[key[0]] = offers_by_group.get(key[0], 0) + 1
        return HybridCacheRequestResult(
            request_id=request.request_id,
            input_tokens=request.input_length,
            output_tokens=request.output_length,
            gpu_hit_tokens=gpu_hit_tokens,
            cpu_hit_tokens=cpu_hit_tokens,
            combined_hit_tokens=combined_hit_tokens,
            gpu_occupancy_slots=len(self._gpu_lru),
            cpu_occupancy_slots=len(self._cpu_lru),
            cpu_reserved_bytes=len(self._cpu_lru) * self.config.cpu_slot_bytes,
            cpu_store_offers=len(offered),
            cpu_store_offers_by_group=offers_by_group,
            cpu_admissions=admissions,
            cpu_evictions=cpu_evictions,
            gpu_evictions=gpu_evictions,
        )


def simulate_trace(
    config: HybridCacheConfig,
    requests: list[HybridCacheRequest],
    *,
    unique_output_occupancy: bool = False,
) -> tuple[dict[str, Any], list[HybridCacheRequestResult]]:
    """Simulate a chronological request trace and return aggregate and request rows."""

    simulator = VllmHybridCacheSimulator(config)
    results = [
        simulator.process(request, unique_output_occupancy=unique_output_occupancy)
        for request in requests
    ]
    input_tokens = sum(item.input_tokens for item in results)
    gpu_hits = sum(item.gpu_hit_tokens for item in results)
    cpu_hits = sum(item.cpu_hit_tokens for item in results)
    aggregate = {
        "schema": "dynamo.vllm-hybrid-cache-simulation.v1",
        "requests": len(results),
        "input_tokens": input_tokens,
        "output_tokens": sum(item.output_tokens for item in results),
        "gpu_hit_tokens": gpu_hits,
        "cpu_hit_tokens": cpu_hits,
        "combined_hit_tokens": gpu_hits + cpu_hits,
        "gpu_hit_rate": gpu_hits / input_tokens if input_tokens else 0.0,
        "cpu_hit_rate": cpu_hits / input_tokens if input_tokens else 0.0,
        "combined_hit_rate": (
            (gpu_hits + cpu_hits) / input_tokens if input_tokens else 0.0
        ),
        "cpu_admissions": sum(item.cpu_admissions for item in results),
        "cpu_evictions": sum(item.cpu_evictions for item in results),
        "gpu_evictions": sum(item.gpu_evictions for item in results),
        "final_gpu_occupancy_slots": (
            results[-1].gpu_occupancy_slots if results else 0
        ),
        "final_cpu_occupancy_slots": (
            results[-1].cpu_occupancy_slots if results else 0
        ),
        "final_cpu_reserved_bytes": (results[-1].cpu_reserved_bytes if results else 0),
        "unique_output_occupancy": unique_output_occupancy,
        "config": {
            **asdict(config),
            "groups": [asdict(group) for group in config.groups],
        },
    }
    return aggregate, results


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate vLLM hybrid GPU/CPU KV-cache hits"
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--per-request-jsonl", type=Path)
    parser.add_argument("--unique-output-occupancy", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        config = HybridCacheConfig.from_dict(json.load(stream))
    with args.trace.open(encoding="utf-8") as stream:
        requests = [
            HybridCacheRequest.from_dict(json.loads(line))
            for line in stream
            if line.strip()
        ]
    aggregate, results = simulate_trace(
        config,
        requests,
        unique_output_occupancy=args.unique_output_occupancy,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report_json, aggregate)
    if args.per_request_jsonl is not None:
        args.per_request_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.per_request_jsonl.open("w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(asdict(result), separators=(",", ":")))
                stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
