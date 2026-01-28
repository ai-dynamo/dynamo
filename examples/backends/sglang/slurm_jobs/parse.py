# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
# pylint: skip-file

import json
import os
import re

### Slurm configs
SLURM_JOB_ID = "slurm id"
### Model Deployment configurations
PREFILL_TP = "Prefill TP"
PREFILL_DP = "Prefill DP"
DECODE_TP = "Decode TP"
DECODE_DP = "Decode DP"
FRONTENDS = "Frontends"
### Profiler configs
PROFILER_TYPE = "Profiler type"
ISL = "ISL"
OSL = "OSL"
REQUEST_RATE = "Request rate"
CONCURRENCIES = "Concurrencies"
OUTPUT_TPS = "Output TPS"
OUTPUT_TPS_PER_USER = "Output TPS/User"
ITL = "Mean ITL (ms)"
TTFT = "Mean TTFT (ms)"
TPOT = "Mean TPOT (ms)"
### FORMAT PRINT ORDERS
KEY_PRINT_ORDER = [
    SLURM_JOB_ID,
    PREFILL_TP,
    PREFILL_DP,
    DECODE_TP,
    DECODE_DP,
    FRONTENDS,
    PROFILER_TYPE,
    ISL,
    OSL,
    REQUEST_RATE,
    CONCURRENCIES,
    OUTPUT_TPS,
    OUTPUT_TPS_PER_USER,
    ITL,
    TTFT,
    TPOT,
]


def format_key_order():
    report = "================\nThe following log will be reported according to this order:\n----\n"
    for key in KEY_PRINT_ORDER:
        report += f"{key}\n"
    print(report[:-1])


def format_print(result):
    report = "================\n"
    for key in KEY_PRINT_ORDER:
        report += f"{result.get(key, '')}\n"
    print(report[:-1])


def analyze_sgl_out(folder):
    result = []
    for file in os.listdir(folder):
        with open(f"{folder}/{file}", "r") as f:
            content = json.load(f)
            res = [
                content["max_concurrency"],
                content["output_throughput"],
                content["mean_itl_ms"],
                content["mean_ttft_ms"],
                content["request_rate"],
            ]

            if "mean_tpot_ms" in content:
                res.append(content["mean_tpot_ms"])
            result.append(res)
    out = {
        REQUEST_RATE: [],
        CONCURRENCIES: [],
        OUTPUT_TPS: [],
        ITL: [],
        TTFT: [],
        TPOT: [],
    }

    for data in sorted(result, key=lambda x: x[0]):
        con, tps, itl, ttft, req_rate = data[0:5]
        out[CONCURRENCIES].append(con)
        out[OUTPUT_TPS].append(tps)
        out[ITL].append(itl)
        out[TTFT].append(ttft)
        out[REQUEST_RATE].append(req_rate)

        if len(data) >= 6:
            if TPOT not in out:
                out[TPOT] = []
            out[TPOT].append(data[5])

    return out


def analyze_gap_out(folder):
    result = []
    for file in os.listdir(folder):
        with open(f"{folder}/{file}", "r") as f:
            content = json.load(f)
            result.append(
                (
                    content["input_config"]["perf_analyzer"]["stimulus"]["concurrency"],
                    content["output_token_throughput_per_user"]["avg"],
                    content["output_token_throughput"]["avg"],
                )
            )

    out = {CONCURRENCIES: [], OUTPUT_TPS: [], OUTPUT_TPS_PER_USER: []}

    for con, tpspuser, tps in sorted(result, key=lambda x: x[0]):
        out[CONCURRENCIES].append(con)
        out[OUTPUT_TPS].append(tps)
        out[OUTPUT_TPS_PER_USER].append(tpspuser)

    return out


def analyze(p):
    files = os.listdir(p)

    prefill_nodes = {}
    decode_nodes = {}
    frontends = []

    profile_result = {}

    for file in files:
        p_re = re.search(
            "([-_A-Za-z0-9]+)_(prefill|decode|nginx|frontend)_([a-zA-Z0-9]+).out", file
        )
        if p_re is not None:
            _, node_type, number = p_re.groups()
            if node_type == "prefill":
                if number not in prefill_nodes:
                    prefill_nodes[number] = []
                prefill_nodes[number].append(file)
            elif node_type == "decode":
                if number not in decode_nodes:
                    decode_nodes[number] = []
                decode_nodes[number].append(file)
            elif node_type == "frontend":
                frontends.append(file)

        profiler_match = re.match("(sglang|vllm|gap)_isl_([0-9]+)_osl_([0-9]+)", file)
        if profiler_match:
            profiler, isl, osl = profiler_match.groups()
            if profiler == "gap":
                profile_result = analyze_gap_out(f"{p}/{file}")
            else:
                profile_result = analyze_sgl_out(f"{p}/{file}")

            profile_result[PROFILER_TYPE] = profiler
            profile_result[ISL] = isl
            profile_result[OSL] = osl

    config = {SLURM_JOB_ID: p}
    if len(prefill_nodes.values()) != 0:
        config[PREFILL_TP] = f"{len(list(prefill_nodes.values())[0]) * 4}"
        config[PREFILL_DP] = f"{len(prefill_nodes.keys())}"

    if len(decode_nodes.values()) != 0:
        config[DECODE_TP] = f"{len(list(decode_nodes.values())[0]) * 4}"
        config[DECODE_DP] = f"{len(decode_nodes.keys())}"

    if len(frontends) != 0:
        config[FRONTENDS] = f"{len(frontends)}"

    result = {**config}
    for key, value in profile_result.items():
        result[key] = (
            value
            if type(value) != list
            else ", ".join([str(x) for x in value])  # ignore:
        )
    return result


paths = [x for x in os.listdir(".") if ".py" not in x and os.path.isdir(x)]
format_key_order()


def extract_job_id(dirname):
    """Extract job ID from directory name for sorting.

    Handles formats like:
    - 12345_3P_1D_20250104_123456 (disaggregated)
    - 12345_4A_20250104_123456 (aggregated)
    - 12345 (legacy format)
    """
    try:
        return int(dirname.split("_")[0])
    except (ValueError, IndexError):
        # If directory name doesn't match expected format, return -1
        return -1


for path in sorted(paths, key=extract_job_id, reverse=True):
    result = analyze(path)
    if OUTPUT_TPS not in result:
        pass
    else:
        format_print(result)
