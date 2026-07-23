<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evidence Before Spend

Treat AIPerf results, Kubernetes evidence, and performance guidance as observations, not proof that another candidate
should consume GPU time.

Every generated optimization hypothesis must cite at least three of these evidence categories, including AIPerf
profiler data:

1. AIPerf profiler data and audited analysis tied to the target metric or SLO.
2. Dynamo or serving-framework source code, official documentation, or performance guidance.
3. Same-series benchmark history from prior candidates.
4. Model architecture details relevant to the observed behavior.
5. Hardware speed-of-light or roofline analysis when the hypothesis concerns compute, memory, or communication limits.
