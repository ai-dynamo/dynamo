# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller identities used to validate replay and decision compatibility."""

POLICY_CONTRACTS = {
    "mpc": "paired-slo-feedback-v2",
    "knowline": "joint-fleet-measured-efficiency-v2",
}

#: Replay telemetry cadence (simulated seconds) both feedback planners require.
FEEDBACK_TELEMETRY_INTERVAL_S = 5.0
