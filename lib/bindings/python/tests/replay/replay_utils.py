# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dynamo.llm import KvRouterConfig, MockEngineArgs

MOONCAKE_TRACE_FIRST20 = """{"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
{"timestamp": 0, "input_length": 7319, "output_length": 490, "hash_ids": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}
{"timestamp": 0, "input_length": 7234, "output_length": 794, "hash_ids": [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]}
{"timestamp": 0, "input_length": 2287, "output_length": 316, "hash_ids": [0, 42, 43, 44, 45]}
{"timestamp": 0, "input_length": 9013, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}
{"timestamp": 0, "input_length": 6506, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64]}
{"timestamp": 0, "input_length": 4824, "output_length": 173, "hash_ids": [0, 65, 66, 67, 68, 69, 70, 71, 72, 73]}
{"timestamp": 0, "input_length": 3119, "output_length": 20, "hash_ids": [74, 75, 76, 77, 78, 79, 80]}
{"timestamp": 0, "input_length": 23090, "output_length": 453, "hash_ids": [0, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]}
{"timestamp": 0, "input_length": 3135, "output_length": 19, "hash_ids": [74, 75, 76, 77, 78, 126, 127]}
{"timestamp": 0, "input_length": 26874, "output_length": 458, "hash_ids": [0, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179]}
{"timestamp": 0, "input_length": 10487, "output_length": 402, "hash_ids": [0, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]}
{"timestamp": 0, "input_length": 17448, "output_length": 610, "hash_ids": [0, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233]}
{"timestamp": 0, "input_length": 6253, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 234]}
{"timestamp": 0, "input_length": 6725, "output_length": 32, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 235, 236]}
{"timestamp": 3052, "input_length": 13538, "output_length": 71, "hash_ids": [0, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262]}
{"timestamp": 3052, "input_length": 87162, "output_length": 402, "hash_ids": [0, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432]}
{"timestamp": 3052, "input_length": 6166, "output_length": 24, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 433]}
{"timestamp": 3052, "input_length": 6320, "output_length": 548, "hash_ids": [0, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445]}
{"timestamp": 3052, "input_length": 2007, "output_length": 354, "hash_ids": [0, 446, 447, 448]}
"""

AIC_PARITY_MODEL = "Qwen/Qwen3-32B"
AIC_PARITY_SYSTEM = "h200_sxm"
AIC_PARITY_VERSIONS = {
    "vllm": "0.12.0",
    "sglang": "0.5.6.post2",
}
AIC_PARITY_BACKENDS = [
    pytest.param("vllm", marks=pytest.mark.vllm, id="vllm"),
    pytest.param("sglang", marks=pytest.mark.sglang, id="sglang"),
]


def _vllm_args_payload():
    return {
        "block_size": 64,
        "speedup_ratio": 1000.0,
    }


def _sglang_args_payload():
    return {
        "engine_type": "sglang",
        "num_gpu_blocks": 512,
        "block_size": 64,
        "speedup_ratio": 1000.0,
        "sglang": {
            "page_size": 64,
        },
    }


def _router_config_payload():
    return {
        "router_queue_threshold": 1.25,
        "router_event_threads": 1,
        "router_queue_policy": "wspt",
        "router_temperature": 0.0,
        "overlap_score_weight": 1.0,
        "use_kv_events": True,
        "durable_kv_events": False,
        "router_replica_sync": False,
        "router_track_active_blocks": True,
        "router_track_output_blocks": False,
        "router_assume_kv_reuse": True,
        "router_track_prefill_tokens": True,
        "router_snapshot_threshold": 1000000,
        "router_reset_states": False,
        "router_ttl_secs": 120.0,
        "router_enable_cache_control": False,
        "skip_initial_worker_wait": False,
        "use_remote_indexer": False,
    }


def _write_trace_and_args(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    records = [
        {
            "timestamp": 1000.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [101],
        },
        {
            "timestamp": 1005.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [101],
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return trace_path


def _write_multiturn_trace(tmp_path):
    trace_path = tmp_path / "multiturn_trace.jsonl"
    records = [
        {
            "session_id": "session-a",
            "timestamp": 1000.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [101],
        },
        {
            "session_id": "session-b",
            "timestamp": 1002.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [202],
        },
        {
            "session_id": "session-a",
            "delay": 5.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [303],
        },
        {
            "session_id": "session-b",
            "delay": 1.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [404],
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return trace_path


def _write_applied_compute_agentic_trace(tmp_path):
    trace_path = tmp_path / "applied_compute_agentic_trace.jsonl"
    records = [
        {
            "num_turns": 2,
            "input_prompt_length": 64,
            "assistant_response_length": [2, 2],
            "tool_call_output_length": [2, 2],
            "tool_call_latency": [0.001, 0.001],
            "final_assistant_response_length": 2,
        },
        {
            "num_turns": 1,
            "input_prompt_length": 64,
            "assistant_response_length": [2],
            "tool_call_output_length": [2],
            "tool_call_latency": [0.001],
            "final_assistant_response_length": 2,
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return trace_path


def _write_cli_smoke_trace(tmp_path):
    trace_path = tmp_path / "cli_smoke_trace.jsonl"
    records = []
    for index in range(10):
        records.append(
            {
                "timestamp": 1000.0 + index,
                "input_length": 250,
                "output_length": 25,
                "hash_ids": [index, index + 1, index + 2, index + 3],
            }
        )
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return trace_path


def _write_vllm_args(tmp_path):
    args_path = tmp_path / "args.json"
    args_path.write_text(
        json.dumps(_vllm_args_payload()),
        encoding="utf-8",
    )
    return args_path


def _vllm_args():
    return MockEngineArgs.from_json(json.dumps(_vllm_args_payload()))


def _write_sglang_args(tmp_path):
    args_path = tmp_path / "sglang_args.json"
    args_path.write_text(
        json.dumps(_sglang_args_payload()),
        encoding="utf-8",
    )
    return args_path


def _sglang_args():
    return MockEngineArgs.from_json(json.dumps(_sglang_args_payload()))


def _prefill_args():
    return MockEngineArgs(block_size=64, speedup_ratio=1000.0, worker_type="prefill")


def _decode_args():
    return MockEngineArgs(block_size=64, speedup_ratio=1000.0, worker_type="decode")


def _write_router_config(tmp_path):
    config_path = tmp_path / "router_config.json"
    config_path.write_text(
        json.dumps(_router_config_payload()),
        encoding="utf-8",
    )
    return config_path


def _router_config():
    return KvRouterConfig.from_json(json.dumps(_router_config_payload()))


def _partial_router_config():
    return KvRouterConfig(
        router_queue_threshold=1.25,
        router_event_threads=1,
        router_queue_policy="wspt",
    )


def _assert_basic_report_counts(report, *, num_requests, input_tokens, output_tokens):
    assert report["num_requests"] == num_requests
    assert report["completed_requests"] == num_requests
    assert report["total_input_tokens"] == num_requests * input_tokens
    assert report["total_output_tokens"] == num_requests * output_tokens


def _assert_basic_report_metrics(report):
    assert report["request_throughput_rps"] > 0
    assert report["output_throughput_tok_s"] > 0
    assert report["duration_ms"] > 0


def _replay_cli_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[5]
    env = os.environ.copy()
    pythonpath_entries = [
        str(repo_root / "lib/bindings/python/src"),
        str(repo_root / "components/src"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    return env


def _planner_profile_data_npz_path() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "benchmarks/results/H200_TP1P_TP1D_perf_data.npz"
    )


def _aic_replay_args(backend_name: str):
    payload = {
        "block_size": 512,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 65536,
        "num_gpu_blocks": 100000,
        "speedup_ratio": 1.0,
        "aic_backend": backend_name,
        "aic_system": AIC_PARITY_SYSTEM,
        "aic_backend_version": AIC_PARITY_VERSIONS[backend_name],
        "aic_tp_size": 1,
        "aic_model_path": AIC_PARITY_MODEL,
    }
    if backend_name == "sglang":
        payload["engine_type"] = "sglang"
        payload["sglang"] = {
            "page_size": 512,
            "max_prefill_tokens": 65536,
            "chunked_prefill_size": 65536,
        }
    return MockEngineArgs.from_json(json.dumps(payload))


def _aic_disagg_replay_args(
    backend_name: str,
    *,
    tp_size: int,
    is_prefill: bool,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    payload = {
        "block_size": 512,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "num_gpu_blocks": 50000,
        "speedup_ratio": 1.0,
        "aic_backend": backend_name,
        "aic_system": AIC_PARITY_SYSTEM,
        "aic_backend_version": AIC_PARITY_VERSIONS[backend_name],
        "aic_tp_size": tp_size,
        "aic_model_path": AIC_PARITY_MODEL,
        "is_prefill": is_prefill,
        "is_decode": not is_prefill,
    }
    if backend_name == "sglang":
        payload["engine_type"] = "sglang"
        payload["sglang"] = {
            "page_size": 512,
            "max_prefill_tokens": 65536,
            "chunked_prefill_size": 65536,
        }
    return MockEngineArgs.from_json(json.dumps(payload))


def _run_aic_static_point(backend_name: str, isl: int, osl: int, batch_size: int):
    aiconfigurator = pytest.importorskip("aiconfigurator")

    database = aiconfigurator.sdk.perf_database.get_database(
        system=AIC_PARITY_SYSTEM,
        backend=backend_name,
        version=AIC_PARITY_VERSIONS[backend_name],
    )
    backend = aiconfigurator.sdk.backends.factory.get_backend(backend_name)
    model = aiconfigurator.sdk.models.get_model(
        model_path=AIC_PARITY_MODEL,
        model_config=aiconfigurator.sdk.config.ModelConfig(tp_size=1),
        backend_name=backend_name,
    )
    session = aiconfigurator.sdk.inference_session.InferenceSession(
        model, database, backend
    )
    summary = session.run_static(
        runtime_config=aiconfigurator.sdk.config.RuntimeConfig(
            batch_size=batch_size,
            beam_width=1,
            isl=isl,
            osl=osl,
            prefix=0,
        ),
        mode="static",
        stride=32,
    )
    return summary.get_summary_df().to_dict(orient="records")[0]


def _planner_profile_data_dir_path() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "components/src/dynamo/planner/tests/data/profiling_results/H200_TP1P_TP1D"
    )


def _write_planner_profile_data_npz(tmp_path: Path) -> Path:
    planner_profile_data = tmp_path / "planner_profile_data.npz"
    np.savez(
        planner_profile_data,
        prefill_isl=np.array([128.0, 256.0]),
        prefill_ttft_ms=np.array([4.0, 8.0]),
        decode_active_kv_tokens=np.array([1024.0, 2048.0]),
        decode_context_length=np.array([128.0, 256.0]),
        decode_itl=np.array([[1.0, 1.5], [2.0, 2.5]]),
    )
    return planner_profile_data


def _run_replay_cli(tmp_path, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamo.replay",
            *args,
        ],
        capture_output=True,
        check=True,
        cwd=str(tmp_path),
        env=_replay_cli_env(),
        text=True,
    )


def _assert_replay_cli_outputs(completed, report_path):
    assert "NVIDIA AIPerf | LLM Metrics" in completed.stdout
    assert "Saved full report to:" in completed.stdout
    assert '"completed_requests"' not in completed.stdout
    return json.loads(report_path.read_text(encoding="utf-8"))
