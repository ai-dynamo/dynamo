# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dynamo.replay.api as replay_api


def test_replay_api_forwards_policy_model_name(monkeypatch):
    calls = []

    def capture_trace(*args, **kwargs):
        calls.append(("trace", args, kwargs))
        return {}

    def capture_synthetic(*args, **kwargs):
        calls.append(("synthetic", args, kwargs))
        return {}

    monkeypatch.setattr(replay_api, "_run_mocker_trace_replay", capture_trace)
    monkeypatch.setattr(
        replay_api,
        "_run_mocker_synthetic_trace_replay",
        capture_synthetic,
    )

    replay_api.run_trace_replay("trace.jsonl", model_name="model-a")
    replay_api.run_synthetic_trace_replay(64, 8, 2, model_name="model-b")

    assert calls[0][2]["model_name"] == "model-a"
    assert calls[1][2]["model_name"] == "model-b"
