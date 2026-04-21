# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Optional

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.core.base import BasePlanner
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class EncodePlanner(BasePlanner):
    component_type = SubComponentType.ENCODE

    def load_plan_adjustment(self) -> Optional[int]:
        raise NotImplementedError(
            "Load-based scaling is not supported for encode mode in Phase 1."
        )

    def _has_required_metrics_for_adjustment(self) -> bool:
        return self.last_metrics.is_valid_for_request_based_throughput()

    def _update_correction_factor(self) -> bool:
        return True

    def update_predictors_from_metrics(self, metrics: Metrics) -> None:
        if metrics.num_req is not None:
            self.num_req_predictor.add_data_point(metrics.num_req)

    def predict_load(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            next_num_req = self.num_req_predictor.predict_next()
            logger.info(f"Predicted encode load: num_req={next_num_req:.2f}")
            return next_num_req, 0.0, 0.0
        except Exception as e:
            logger.error(f"Failed to predict encode load: {e}")
            return None, None, None

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> int:
        del next_isl, next_osl

        predicted_request_rate = (
            next_num_req / self.config.throughput_adjustment_interval
        )
        e_thpt_per_gpu = self.encode_interpolator.interpolate_thpt_per_gpu(
            predicted_request_rate
        )
        if e_thpt_per_gpu <= 0:
            logger.warning(
                f"e_thpt_per_gpu is {e_thpt_per_gpu}, falling back to min_endpoint"
            )
            return self.config.min_endpoint

        next_num_e = math.ceil(
            predicted_request_rate / e_thpt_per_gpu / self._engine_num_gpu()
        )
        next_num_e = max(next_num_e, self.config.min_endpoint)
        logger.info(
            f"Encode calculation: {predicted_request_rate:.2f}(req/s) / "
            f"{e_thpt_per_gpu * self._engine_num_gpu():.2f}(e_engine_cap) = "
            f"{next_num_e}(num_e)"
        )
        return next_num_e

    def update_predicted_replicas_metric(self, desired_replicas: int) -> None:
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.predicted_num_e.set(desired_replicas)
