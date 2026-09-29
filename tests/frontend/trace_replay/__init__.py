# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded agent turns through the frontend with the mocker as the engine.

`fixtures` turns agent trajectories into mocker replay rows plus expected parsed
turns, `endpoints` builds and normalizes chat / Responses / Messages traffic, and
`runner` drives a live frontend and reports how its output compares.
"""
