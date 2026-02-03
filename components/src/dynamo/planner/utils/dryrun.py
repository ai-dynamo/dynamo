# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Optional

from dynamo.planner.utils.dryrun_plot_utils import create_dryrun_plot
from dynamo.planner.utils.planner_core import (
    DecodePlanner,
    PlannerSharedState,
    PrefillPlanner,
    _apply_component_gpu_budget,
    _apply_global_gpu_budget,
)
from dynamo.planner.utils.trace_data_extractor import extract_metrics_from_mooncake


def run_sla_planner_dryrun(args: argparse.Namespace) -> None:
    # Dryrun mode: use defaults if GPU counts not provided (no DGD available)
    if args.prefill_engine_num_gpu is None:
        args.prefill_engine_num_gpu = 1
    if args.decode_engine_num_gpu is None:
        args.decode_engine_num_gpu = 1

    warmup_metrics = None
    if getattr(args, "load_predictor_warmup_trace", None):
        warmup_metrics = extract_metrics_from_mooncake(
            args.load_predictor_warmup_trace,
            args.adjustment_interval,
        )

    metrics = extract_metrics_from_mooncake(args.dataset, args.adjustment_interval)
    if not metrics:
        raise ValueError("Empty metrics dataset: cannot run dryrun")

    mode = getattr(args, "mode", "disagg")
    prefill_planner: Optional[PrefillPlanner] = None
    decode_planner: Optional[DecodePlanner] = None
    if mode == "disagg":
        shared_state = PlannerSharedState()
        prefill_planner = PrefillPlanner(
            None, args, dryrun=True, shared_state=shared_state
        )
        decode_planner = DecodePlanner(
            None, args, dryrun=True, shared_state=shared_state
        )
    elif mode == "prefill":
        prefill_planner = PrefillPlanner(None, args, dryrun=True)
    elif mode == "decode":
        decode_planner = DecodePlanner(None, args, dryrun=True)
    else:
        raise ValueError(f"Invalid planner mode: {mode}")

    def compute_safe_p_thpt(num_p: int, isl: float, ttft: float):
        """safe throughput is maximum throughput that the engine can handle given the TTFT SLA"""
        assert prefill_planner is not None
        actual_ttft = prefill_planner.prefill_interpolator.interpolate_ttft(isl)
        if actual_ttft > ttft:
            return 0
        return num_p * prefill_planner.prefill_interpolator.interpolate_thpt_per_gpu(
            isl
        )

    def compute_safe_d_thpt(num_d: int, isl: float, osl: float, itl: float):
        """safe throughput is maximum throughput that the engine can handle given the ITL SLA"""
        assert decode_planner is not None
        (
            pred_decode_thpt_per_gpu,
            actual_itl,
            _,
        ) = decode_planner.decode_interpolator.find_best_throughput_per_gpu(
            itl=itl, context_length=isl + osl / 2
        )
        if actual_itl > itl:
            return 0
        return num_d * pred_decode_thpt_per_gpu

    time_series = [0]
    rr = [metrics[0]["request_count"]]
    est_rr = [metrics[0]["request_count"]]
    isl = [metrics[0]["avg_isl"]]
    est_isl = [metrics[0]["avg_isl"]]
    osl = [metrics[0]["avg_osl"]]
    est_osl = [metrics[0]["avg_osl"]]

    if prefill_planner is not None:
        num_p = [args.start_num_p]
        p_thpt = [rr[0] * isl[0]]
        safe_p_thpt = [
            compute_safe_p_thpt(args.start_num_p, isl[0], args.ttft)
            * args.adjustment_interval
        ]
        prefill_planner.dryrun_observe_metrics(rr[0], isl[0], osl[0])
    else:
        num_p = [0]
        p_thpt = [0]
        safe_p_thpt = [0]

    if decode_planner is not None:
        num_d = [args.start_num_d]
        d_thpt = [rr[0] * osl[0]]
        safe_d_thpt = [
            compute_safe_d_thpt(args.start_num_d, isl[0], osl[0], args.itl)
            * args.adjustment_interval
        ]
        decode_planner.dryrun_observe_metrics(rr[0], isl[0], osl[0])
    else:
        num_d = [0]
        d_thpt = [0]
        safe_d_thpt = [0]

    predictor_planner = prefill_planner or decode_planner
    assert predictor_planner is not None

    for metric in metrics[1:]:
        # update time
        time_series.append(time_series[-1] + args.adjustment_interval)

        # load prediction
        _est_rr, _est_isl, _est_osl = predictor_planner.predict_load()
        est_rr.append(_est_rr)
        est_isl.append(_est_isl)
        est_osl.append(_est_osl)

        # compute num_p and num_d
        _num_p = (
            prefill_planner._compute_replica_requirements(_est_rr, _est_isl, _est_osl)
            if prefill_planner is not None
            else 0
        )
        _num_d = (
            decode_planner._compute_replica_requirements(_est_rr, _est_isl, _est_osl)
            if decode_planner is not None
            else 0
        )

        # apply GPU budget
        if prefill_planner is not None and decode_planner is not None:
            _num_p, _num_d = _apply_global_gpu_budget(_num_p, _num_d, args)
        elif prefill_planner is not None:
            _num_p = _apply_component_gpu_budget(
                _num_p, args.prefill_engine_num_gpu, args
            )
        elif decode_planner is not None:
            _num_d = _apply_component_gpu_budget(
                _num_d, args.decode_engine_num_gpu, args
            )

        num_p.append(_num_p)
        num_d.append(_num_d)

        # update load predictor
        for planner in [prefill_planner, decode_planner]:
            if planner is not None:
                planner.dryrun_observe_metrics(
                    metric["request_count"], metric["avg_isl"], metric["avg_osl"]
                )

        # fill in ground truth
        rr.append(metric["request_count"])
        isl.append(metric["avg_isl"])
        osl.append(metric["avg_osl"])

        p_thpt.append(rr[-1] * isl[-1] if prefill_planner is not None else 0)
        d_thpt.append(rr[-1] * osl[-1] if decode_planner is not None else 0)

        safe_p_thpt.append(
            compute_safe_p_thpt(num_p[-1], isl[-1], args.ttft)
            * args.adjustment_interval
            if prefill_planner is not None
            else 0
        )
        safe_d_thpt.append(
            compute_safe_d_thpt(num_d[-1], isl[-1], osl[-1], args.itl)
            * args.adjustment_interval
            if decode_planner is not None
            else 0
        )

    warmup_time = None
    warmup_rr = None
    warmup_isl = None
    warmup_osl = None
    if warmup_metrics:
        interval = args.adjustment_interval
        n = len(warmup_metrics)
        warmup_time = [-(n - i) * interval for i in range(n)]
        warmup_rr = [m["request_count"] for m in warmup_metrics]
        warmup_isl = [m["avg_isl"] for m in warmup_metrics]
        warmup_osl = [m["avg_osl"] for m in warmup_metrics]

    create_dryrun_plot(
        time=time_series,
        rr=rr,
        est_rr=est_rr,
        isl=isl,
        est_isl=est_isl,
        osl=osl,
        est_osl=est_osl,
        num_p=num_p,
        p_thpt=p_thpt,
        safe_p_thpt=safe_p_thpt,
        num_d=num_d,
        d_thpt=d_thpt,
        safe_d_thpt=safe_d_thpt,
        output_path=args.output_plot,
        warmup_time=warmup_time,
        warmup_rr=warmup_rr,
        warmup_isl=warmup_isl,
        warmup_osl=warmup_osl,
    )
