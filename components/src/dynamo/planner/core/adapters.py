# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native planner adapter subclasses (one per mode).

Each subclass sets ``require_prefill`` / ``require_decode`` and overrides
``_bootstrap_regression()`` and ``_apply_effects()``.  Everything else
(connector, Prometheus, FPM subscribers, tick loop) is in ``NativePlannerBase``.
"""

import logging

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import PlannerEffects
from dynamo.planner.monitoring.perf_metrics import fetch_pre_deployment_metrics

logger = logging.getLogger(__name__)


class PrefillPlanner(NativePlannerBase):
    """Prefill-only mode."""

    require_prefill = True
    require_decode = False

    async def _bootstrap_regression(self) -> None:
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.namespace,
                worker_info=self.prefill_worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=SubComponentType.PREFILL,
                aic_spec=self.config.aic_interpolation,
            )
            self.state_machine.load_benchmark_fpms(prefill_fpms=fpms)
        except Exception as e:
            if self.config.enable_throughput_scaling:
                raise
            logger.warning(f"No pre-deployment data for prefill: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None or effects.scale_to.num_prefill is None:
            return
        desired = effects.scale_to.num_prefill
        if self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_prefill_replicas.set(desired)
        await self._apply_scaling_targets(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    component_name=self.prefill_worker_info.k8s_name,
                    desired_replicas=desired,
                )
            ]
        )


class DecodePlanner(NativePlannerBase):
    """Decode-only mode."""

    require_prefill = False
    require_decode = True

    async def _bootstrap_regression(self) -> None:
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.namespace,
                worker_info=self.decode_worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=SubComponentType.DECODE,
                aic_spec=self.config.aic_interpolation,
            )
            self.state_machine.load_benchmark_fpms(decode_fpms=fpms)
        except Exception as e:
            if self.config.enable_throughput_scaling:
                raise
            logger.warning(f"No pre-deployment data for decode: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None or effects.scale_to.num_decode is None:
            return
        desired = effects.scale_to.num_decode
        if self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_decode_replicas.set(desired)
        await self._apply_scaling_targets(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name=self.decode_worker_info.k8s_name,
                    desired_replicas=desired,
                )
            ]
        )


class AggPlanner(NativePlannerBase):
    """Aggregated mode (single engine type handles both prefill and decode)."""

    require_prefill = False
    require_decode = True

    async def _bootstrap_regression(self) -> None:
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.namespace,
                worker_info=self.decode_worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=SubComponentType.DECODE,
                aic_spec=self.config.aic_interpolation,
            )
            self.state_machine.load_benchmark_fpms(agg_fpms=fpms)
        except Exception as e:
            if self.config.enable_throughput_scaling:
                raise
            logger.warning(f"No pre-deployment data for agg: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None or effects.scale_to.num_decode is None:
            return
        desired = effects.scale_to.num_decode
        if self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_decode_replicas.set(desired)
        await self._apply_scaling_targets(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name=self.decode_worker_info.k8s_name,
                    desired_replicas=desired,
                )
            ]
        )


class DisaggPlanner(NativePlannerBase):
    """Disaggregated mode (separate prefill and decode engines)."""

    require_prefill = True
    require_decode = True

    async def _bootstrap_regression(self) -> None:
        for component, kwarg in [
            (SubComponentType.PREFILL, "prefill_fpms"),
            (SubComponentType.DECODE, "decode_fpms"),
        ]:
            worker_info = (
                self.prefill_worker_info
                if component == SubComponentType.PREFILL
                else self.decode_worker_info
            )
            try:
                fpms = await fetch_pre_deployment_metrics(
                    runtime=self.runtime,
                    namespace=self.namespace,
                    worker_info=worker_info,
                    profile_results_dir=self.config.profile_results_dir,
                    component_type=component,
                    aic_spec=self.config.aic_interpolation,
                )
                self.state_machine.load_benchmark_fpms(**{kwarg: fpms})
            except Exception as e:
                if self.config.enable_throughput_scaling:
                    raise
                logger.warning(f"No pre-deployment data for {component.value}: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None:
            return
        decision = effects.scale_to

        if decision.num_prefill is not None and self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_prefill_replicas.set(
                decision.num_prefill
            )
        if decision.num_decode is not None and self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_decode_replicas.set(
                decision.num_decode
            )

        targets = []
        if decision.num_prefill is not None:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    component_name=self.prefill_worker_info.k8s_name,
                    desired_replicas=decision.num_prefill,
                )
            )
        if decision.num_decode is not None:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name=self.decode_worker_info.k8s_name,
                    desired_replicas=decision.num_decode,
                )
            )
        await self._apply_scaling_targets(targets)
