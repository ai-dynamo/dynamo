# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default to the prefill server when invoked as ``python -m kvbm.vllm``."""

from kvbm.vllm.prefill import main

if __name__ == "__main__":
    main()
