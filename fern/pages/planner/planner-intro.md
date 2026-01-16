---
title: "Planner"
---

{/*
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/}

The planner monitors the state of the system and adjusts workers to
ensure that the system runs efficiently.

Currently, the planner can scale the number of vllm workers up and down
based on the kv cache load and prefill queue size:

Key features include:

- **SLA-based scaling** that uses predictive modeling and performance
  interpolation to proactively meet TTFT and ITL targets
- **Graceful scaling** that ensures no requests are dropped during
  scale-down operations

<Tip>
**New to SLA Planner?** Start with the [SLA Planner Quick Start Guide](sla-planner-quickstart.md) for a complete, step-by-step workflow.

**Prerequisites**: SLA-based planner requires pre-deployment profiling (2-4 hours on real silicon or a few minutes using simulator) before deployment. The Quick Start guide includes everything you need.
</Tip>
