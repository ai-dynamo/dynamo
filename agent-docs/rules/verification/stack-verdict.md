<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Stack Verdict

Use verdict `stack` when a change clears every verification gate but is insufficient by itself to satisfy the target or
justify promotion. Do not stack noise, inconclusive results, regressions, or changes whose configuration engagement was
not proven.

Compatible, independently verified `stack` changes may later be combined as one integration candidate. Record every
changed field, benchmark the bundle, and retain the individual results as its ablation evidence.
