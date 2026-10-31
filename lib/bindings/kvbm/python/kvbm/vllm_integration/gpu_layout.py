# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained-GPU layout identity shared by discovery and storage initialization."""


def parallel_layout_domain(domain, tp_size):
    if type(tp_size) is not int or tp_size not in (1, 2):
        raise ValueError("Retained GPU layout supports TP1 or TP2")
    if (
        not isinstance(domain, str)
        or not domain
        or domain != domain.strip()
        or "\x00" in domain
    ):
        raise ValueError("An explicit certified GPU layout domain is required")
    return domain if tp_size == 1 else f"{domain}|gpu-raw-tp-v1:tp={tp_size}"
