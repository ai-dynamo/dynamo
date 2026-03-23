# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.constants import ROUTER_MODEL_NAME

MODEL_NAME = ROUTER_MODEL_NAME
MOONCAKE_TRACE_BLOCK_SIZE = 512
MOONCAKE_TRACE_SAMPLE_LINES = [
    '{"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}',
    '{"timestamp": 0, "input_length": 7319, "output_length": 490, "hash_ids": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}',
    '{"timestamp": 0, "input_length": 7234, "output_length": 794, "hash_ids": [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]}',
    '{"timestamp": 0, "input_length": 2287, "output_length": 316, "hash_ids": [0, 42, 43, 44, 45]}',
    '{"timestamp": 0, "input_length": 9013, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}',
    '{"timestamp": 0, "input_length": 6506, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64]}',
    '{"timestamp": 0, "input_length": 4824, "output_length": 173, "hash_ids": [0, 65, 66, 67, 68, 69, 70, 71, 72, 73]}',
    '{"timestamp": 0, "input_length": 3119, "output_length": 20, "hash_ids": [74, 75, 76, 77, 78, 79, 80]}',
    '{"timestamp": 0, "input_length": 23090, "output_length": 453, "hash_ids": [0, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]}',
    '{"timestamp": 0, "input_length": 3135, "output_length": 19, "hash_ids": [74, 75, 76, 77, 78, 126, 127]}',
    '{"timestamp": 0, "input_length": 26874, "output_length": 458, "hash_ids": [0, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179]}',
    '{"timestamp": 0, "input_length": 10487, "output_length": 402, "hash_ids": [0, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]}',
    '{"timestamp": 0, "input_length": 17448, "output_length": 610, "hash_ids": [0, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233]}',
    '{"timestamp": 0, "input_length": 6253, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 234]}',
    '{"timestamp": 0, "input_length": 6725, "output_length": 32, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 235, 236]}',
    '{"timestamp": 3052, "input_length": 13538, "output_length": 71, "hash_ids": [0, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262]}',
    '{"timestamp": 3052, "input_length": 87162, "output_length": 402, "hash_ids": [0, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432]}',
    '{"timestamp": 3052, "input_length": 6166, "output_length": 24, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 433]}',
    '{"timestamp": 3052, "input_length": 6320, "output_length": 548, "hash_ids": [0, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445]}',
    '{"timestamp": 3052, "input_length": 2007, "output_length": 354, "hash_ids": [0, 446, 447, 448]}',
]

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.router,
    pytest.mark.model(MODEL_NAME),
]


@pytest.mark.timeout(120)
def test_mocker_trace_file_replay(tmp_path):
    repo_root = Path.cwd()
    trace_file = tmp_path / "mooncake_trace.jsonl"
    trace_file.write_text(
        "\n".join(MOONCAKE_TRACE_SAMPLE_LINES) + "\n", encoding="utf-8"
    )
    replay_report = trace_file.with_name(f"{trace_file.stem}.replay.json")
    pythonpath_entries = [
        str(repo_root / "components/src"),
        str(repo_root / "lib/bindings/python/src"),
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamo.mocker",
            "--trace-file",
            str(trace_file),
            "--model-path",
            MODEL_NAME,
            "--num-workers",
            "1",
            "--block-size",
            str(MOONCAKE_TRACE_BLOCK_SIZE),
            "--speedup-ratio",
            "0",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"dynamo.mocker trace replay failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert replay_report.exists(), (
        "Expected default replay report next to the temp trace file, "
        f"but {replay_report} was not created.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Replay Summary" in result.stdout
    assert f"JSON report: {replay_report}" in result.stdout

    report = json.loads(replay_report.read_text(encoding="utf-8"))
    assert report["num_requests"] == len(MOONCAKE_TRACE_SAMPLE_LINES)
    assert report["completed_requests"] == len(MOONCAKE_TRACE_SAMPLE_LINES)
    assert report["total_input_tokens"] > 0
    assert report["total_output_tokens"] > 0
    assert report["duration_ms"] > 0
    assert report["wall_time_ms"] >= 0
    assert report["request_throughput_rps"] > 0
