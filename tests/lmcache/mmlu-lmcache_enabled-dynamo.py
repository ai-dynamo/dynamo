# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the MMLU evaluator against an LMCache-enabled Dynamo deployment."""

from _mmlu_dynamo import run

if __name__ == "__main__":
    run("dynamo-lmcache")
