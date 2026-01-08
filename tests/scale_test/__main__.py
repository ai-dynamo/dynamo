# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Entry point for running the scale test tool as a module.

Usage:
    python -m tests.scale_test <command> [options]

Commands:
    start   - Start N deployments and wait for manual testing
    run     - Run full test: start + load + cleanup
    cleanup - Cleanup any leftover scale test processes
"""

import sys

from tests.scale_test.cli import main

if __name__ == "__main__":
    sys.exit(main())
