# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from types import SimpleNamespace

import pytest

from dynamo.mocker import config as CONFIG
from dynamo.mocker.args import parse_args

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("serial", [False, True])
async def test_systemone_serial_launch_advertisement_matches_engine(
    monkeypatch, serial
):
    monkeypatch.setenv("DYN_COMPUTE_THREADS", "0")
    mocker_main = importlib.import_module("dynamo.mocker.main")
    flags = ["--engine-type", "sglang", "--sglang-generate"]
    ordinary = CONFIG.load_mocker_engine_args(parse_args(flags))
    args = parse_args([*flags, *(["--systemone-serial"] if serial else [])])
    config = CONFIG.load_mocker_engine_args(args)
    captured = []

    async def make_engine(_runtime, entrypoint):
        captured.append(entrypoint)
        raise RuntimeError("launch config captured")

    monkeypatch.setattr(mocker_main, "create_runtime", lambda *a, **kw: (None, None))
    monkeypatch.setattr(
        mocker_main, "EntrypointArgs", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(mocker_main, "make_engine", make_engine)
    with pytest.raises(RuntimeError, match="launch config captured"):
        await mocker_main.launch_workers(args, config)
    entrypoint = captured[0]
    expected = 1 if serial else ordinary["engine"]["max_num_seqs"]
    assert entrypoint.mocker_engine_args["engine"]["max_num_seqs"] == expected
    assert entrypoint.runtime_config.max_num_seqs == expected
    assert entrypoint.runtime_config.runtime_data["sglang_generate"] == "true"
    assert entrypoint.runtime_config.runtime_data.get("sglang_systemone_serial_v1") == (
        "true" if serial else None
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--engine-type", "sglang"],
        ["--sglang-generate", "--engine-type", "vllm"],
        [
            "--sglang-generate",
            "--engine-type",
            "sglang",
            "--disaggregation-mode",
            "decode",
        ],
        ["--sglang-generate", "--engine-type", "sglang", "--data-parallel-size", "2"],
        ["--sglang-generate", "--engine-type", "sglang", "--max-num-seqs", "2"],
        ["--sglang-generate", "--engine-type", "sglang", "--max-num-seqs=2"],
        ["--sglang-generate", "--engine-type", "sglang", "--max-num-seqs=256"],
        ["--sglang-generate", "--engine-type", "sglang", "--max-num-s=2"],
    ],
)
def test_systemone_serial_rejects_incompatible_launch_options(flags):
    with pytest.raises((ValueError, SystemExit)):
        CONFIG.load_mocker_engine_args(parse_args(["--systemone-serial", *flags]))


@pytest.mark.parametrize(
    "engine_overrides,dp_size,valid",
    [
        ({"max_num_seqs": 1}, 1, True),
        ({"max_num_seqs": 2}, 1, False),
        ({"max_num_seqs": 1, "backend": "vllm", "block_size": 64}, 1, False),
        ({"max_num_seqs": 1, "worker_type": "decode"}, 1, False),
        ({"max_num_seqs": 1}, 2, False),
    ],
)
def test_systemone_serial_validates_canonical_engine_args(
    tmp_path, engine_overrides, dp_size, valid
):
    config = CONFIG.build_mocker_engine_args(parse_args(["--engine-type", "sglang"]))
    config = {
        **config,
        "engine": {**config["engine"], **engine_overrides},
        "dp_size": dp_size,
    }
    path = tmp_path / "engine.json"
    path.write_text(json.dumps(config))
    args = parse_args(
        ["--systemone-serial", "--sglang-generate", "--extra-engine-args", str(path)]
    )
    if valid:
        assert CONFIG.load_mocker_engine_args(args)["engine"]["max_num_seqs"] == 1
    else:
        with pytest.raises(ValueError, match="systemone-serial"):
            CONFIG.load_mocker_engine_args(args)
