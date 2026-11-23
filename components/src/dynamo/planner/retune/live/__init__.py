# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live retune: run the retune sweep through DGDR v1beta2 and apply the winner.

The replay retune loop (``RetuneController`` and ``_decide``) is unchanged;
this package supplies its live pieces: the DGDR sweep executor, the
supervisor that owns leadership and persisted state, the router actuator,
the reconciler and the trace window cutter. ``contracts`` holds the types
they share.
"""
