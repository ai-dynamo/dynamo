# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Redis connector: publishes scaling decisions to Redis and reads back
observed worker state, instead of talking to Kubernetes directly.

Sibling to VirtualConnector for "hand the decision to a non-native
environment," but with no etcd/nats/DistributedRuntime dependency -- just a
Redis connection.

Wire format: one Redis hash per deployment, key
``{prefix}:{dynamo_namespace}:{model_name}``. Two deployments serving the
same model_name under different namespaces get distinct keys -- necessary
since Redis may be shared across more than one deployment. This connector
owns the "desired" fields (``prefill``, ``decode``, ``updated_at``).
An external actuator (not part of this repo) is expected to read those and,
separately, write back "observed" fields (``prefill_active``,
``decode_active``, ``prefill_stable``, ``decode_stable``, ``observed_at``)
reflecting what it actually did. Distinct field names on both sides means
neither side's write can clobber the other's fields, even sharing one key.
The two ``*_stable`` fields are plain strings, ``"true"``/``"false"``
(case-insensitive), not JSON booleans -- kept human-readable for anyone
inspecting the key by hand.

If the external actuator never writes the observed fields -- before its
first write-back, or because it doesn't track this at all --
``get_actual_worker_counts`` reads inactive/unstable defaults rather than
raising or assuming a settled empty deployment.
"""

import logging
import os
import time

import redis.asyncio as redis_asyncio

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.errors import EmptyTargetReplicasError
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = os.environ.get("DYN_REDIS_KEY_PREFIX", "dynamo:planner:target")


def _parse_non_negative_int(raw: dict[str, str], field: str) -> int:
    """Parse one hash field as a non-negative int, defaulting to 0 if absent.

    Raises ValueError naming the offending field on a negative or otherwise
    invalid value -- corrupted or hand-edited Redis data should fail loudly
    rather than silently propagate a nonsensical replica count.
    """
    value = raw.get(field, "0")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"redis field {field!r} is not a valid integer: {value!r}"
        ) from None
    if parsed < 0:
        raise ValueError(f"redis field {field!r} must not be negative, got {parsed}")
    return parsed


class RedisConnector(PlannerConnector):
    """Publishes scaling decisions to Redis; reads observed state back from it."""

    def __init__(
        self,
        dynamo_namespace: str,
        model_name: str | None = None,
        redis_url: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("Model name is required for redis connector")
        # Case-preserving: unlike VirtualConnector/KubernetesConnector, this
        # connector never matches model_name against MDC entries, so there's
        # no reason to fold case -- and doing so would risk merging two
        # distinct, differently-cased model names onto the same Redis key.
        self.model_name = model_name
        self.dynamo_namespace = dynamo_namespace

        redis_url = redis_url or os.environ.get("DYN_REDIS_URL")
        if not redis_url:
            raise ValueError(
                "redis_url is required for redis connector "
                "(pass explicitly or set DYN_REDIS_URL)"
            )
        self._redis = redis_asyncio.from_url(redis_url, decode_responses=True)
        self._closed = False
        # Curly braces are a Redis Cluster hash tag: only the portion inside
        # them is hashed to pick a shard, so every key for one deployment
        # lands on the same node regardless of what prefix precedes it.
        # Namespace is part of the tag, not just the prefix, so two
        # deployments sharing a model_name under different namespaces never
        # collide on the same key.
        self._key = f"{REDIS_KEY_PREFIX}:{{{self.dynamo_namespace}:{self.model_name}}}"

    async def async_init(self) -> None:
        """Nothing to do -- the Redis client connects lazily on first command."""

    async def _read_desired_counts(self) -> tuple[int, int]:
        raw = await self._redis.hgetall(self._key)
        return (
            _parse_non_negative_int(raw, "prefill"),
            _parse_non_negative_int(raw, "decode"),
        )

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        """Add a component by increasing its published replica count by 1.

        Not exercised by the automatic tick loop (that goes through
        ``set_component_replicas`` below); implemented only for parity with
        the other connectors' manual/CLI use. Read-then-write, same as
        ``VirtualConnector``'s equivalent -- fine off the hot path.
        """
        current_p, current_d = await self._read_desired_counts()
        if sub_component_type == SubComponentType.PREFILL:
            await self._redis.hset(
                self._key,
                mapping={"prefill": current_p + 1, "updated_at": time.time()},
            )
        elif sub_component_type == SubComponentType.DECODE:
            await self._redis.hset(
                self._key,
                mapping={"decode": current_d + 1, "updated_at": time.time()},
            )

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        """Remove a component by decreasing its published replica count by 1.

        See ``add_component`` -- not on the automatic scaling path.
        """
        current_p, current_d = await self._read_desired_counts()
        if sub_component_type == SubComponentType.PREFILL:
            await self._redis.hset(
                self._key,
                mapping={"prefill": max(0, current_p - 1), "updated_at": time.time()},
            )
        elif sub_component_type == SubComponentType.DECODE:
            await self._redis.hset(
                self._key,
                mapping={"decode": max(0, current_d - 1), "updated_at": time.time()},
            )

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ) -> None:
        """Publish the tick's final decision to Redis.

        This is the method every planner mode actually calls. A single
        ``HSET`` covers only the roles present in ``target_replicas`` -- no
        read-modify-write, so two planners sharing this model_name's key
        (e.g. a dedicated PrefillPlanner and DecodePlanner) can't race each
        other clobbering the other's field.

        ``blocking`` is accepted for interface compatibility but ignored:
        there's no "wait for the external actuator to finish scaling"
        concept here. Our job ends when the write returns; the external
        control plane picks it up and actuates on its own schedule.
        """
        if not target_replicas:
            raise EmptyTargetReplicasError()

        mapping: dict[str, int | float] = {}
        for target_replica in target_replicas:
            if target_replica.desired_replicas < 0:
                raise ValueError(
                    f"desired_replicas must not be negative, got "
                    f"{target_replica.desired_replicas} for "
                    f"{target_replica.sub_component_type.value}"
                )
            if target_replica.sub_component_type == SubComponentType.PREFILL:
                mapping["prefill"] = target_replica.desired_replicas
            elif target_replica.sub_component_type == SubComponentType.DECODE:
                mapping["decode"] = target_replica.desired_replicas

        if not mapping:
            return
        mapping["updated_at"] = time.time()
        await self._redis.hset(self._key, mapping=mapping)

    async def validate_deployment(
        self,
        prefill_component_name: str | None = None,
        decode_component_name: str | None = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> None:
        """No external deployment for this connector to validate against."""

    async def wait_for_deployment_ready(self, include_planner: bool = True) -> None:
        """No external deployment to wait on."""

    def get_model_name(
        self, require_prefill: bool = True, require_decode: bool = True
    ) -> str:
        """Get the model name (as given at construction)."""
        del require_prefill, require_decode
        return self.model_name

    def get_gpu_counts(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[int | None, int | None]:
        """No live view of GPU shape -- same reasoning as ``VirtualConnector``:
        this connector's Redis hash carries replica counts only, never GPU
        shape. The planner falls back to its own ``prefill_engine_num_gpu``/
        ``decode_engine_num_gpu`` config for its sizing math.
        """
        del require_prefill, require_decode
        return None, None

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        """No live discovery source for this connector -- always defaults.

        ``VirtualConnector`` gets an MDC source wired in after construction
        (a ``VirtualConnector``-specific hook in ``construct_environment``);
        this connector never receives one and always falls back to defaults.
        """
        info = build_worker_info_from_defaults(backend, sub_component_type)
        info.model_name = self.model_name
        return info

    async def get_actual_worker_counts(
        self,
        prefill_component_name: str | None = None,
        decode_component_name: str | None = None,
    ) -> tuple[int, int, bool]:
        """Read observed state written back by the external actuator.

        Per the ``PlannerConnector`` contract, a component whose name arg is
        ``None`` (not required by this planner mode) reports 0 and does not
        count against the returned ``stable`` flag -- only roles actually in
        play for this planner instance affect it.

        If the external actuator hasn't published observed counts for this
        model -- before its first write-back, or because it doesn't track
        this at all -- missing fields default to inactive/unstable: fail
        closed as "still converging," never a false "settled empty."
        """
        raw = await self._redis.hgetall(self._key)

        prefill_active = 0
        decode_active = 0
        stable = True

        if prefill_component_name is not None:
            prefill_active = _parse_non_negative_int(raw, "prefill_active")
            stable = stable and raw.get("prefill_stable", "").lower() == "true"
        if decode_component_name is not None:
            decode_active = _parse_non_negative_int(raw, "decode_active")
            stable = stable and raw.get("decode_stable", "").lower() == "true"

        return prefill_active, decode_active, stable

    async def shutdown(self) -> None:
        """Release the Redis client. Idempotent -- safe to call more than
        once (e.g. if the environment's own shutdown and a caller's
        explicit cleanup both fire)."""
        if self._closed:
            return
        self._closed = True
        await self._redis.aclose()
