<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Implausible Speedup

Treat a gain that materially exceeds the changed knob's plausible impact or known headroom as suspect. Recheck config
engagement, workload identity, response correctness, request and output-token counts, errors, active GPU count, and
AIPerf confidence evidence.

Do not accept the gain until it is reproduced and its magnitude has a credible explanation. Otherwise mark the result
`inconclusive` or invalid.
