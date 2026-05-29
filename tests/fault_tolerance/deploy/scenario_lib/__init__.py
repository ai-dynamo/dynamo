# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# YAML-driven scenario library for router-mode / leak / admission tests.
#
# Each scenario YAML file describes a complete test cell:
#   - deployment (backend, topology, units, image)
#   - router (mode + DYN_ROUTER_* knobs)
#   - admission (optional DYN_TCP_* / DYN_VLLM_REJECT_* knobs)
#   - load (workload shape + rung sequence) OR events (full event sequence)
#   - reports + checks
#   - expectations (documentation, observed history)
#
# Scenarios live in subdirectories of this package, one per kind:
#   scenario_lib/router_memory/    consumed by test_router_modes::test_router_memory
#   scenario_lib/admission_control/  consumed by test_admission_control (future)
#   scenario_lib/endurance/        consumed by test_endurance (future)
#
# The subdirectory name IS the kind. The loader validates each YAML's
# top-level ``kind:`` field matches the parent directory — misplaced
# files fail at pytest collection time.
