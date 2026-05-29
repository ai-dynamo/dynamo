---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "DynoSim: Simulating the Pareto Frontier"
subtitle: "A short pointer to the NVIDIA Technical Blog deep dive - May 2026"
description: "DynoSim helps Dynamo teams simulate serving tradeoffs before spending GPU time on full deployments."
keywords: DynoSim, Dynamo, simulation, Pareto frontier, Planner, Router, KVBM, LLM inference
last-updated: May 29, 2026
---

DynoSim is a workload-driven discrete-event simulation of NVIDIA Dynamo that lets teams explore the serving design space before spending GPU time on full deployments. The NVIDIA Technical Blog post, [DynoSim: Simulating the Pareto Frontier](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/), walks through how DynoSim composes workload replay, scheduler-aware engine timing, Router, Planner, and KVBM behavior on a shared virtual timeline; why a Rust replay can screen thousands of configurations far faster than real time; and how the same loop can map throughput-latency Pareto frontiers, tune Planner behavior, and shortlist candidates for real-cluster validation.
