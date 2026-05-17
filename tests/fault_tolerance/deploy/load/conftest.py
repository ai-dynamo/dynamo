# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scoped pytest config for load-focused tests under this directory.
#
# These tests deploy replicated 1F:2P:1D "units" of the prod-mirror
# vLLM disagg DGD. The ``--units`` flag accepts either a single integer
# (``--units=3``) or a comma-separated list (``--units=1,2,3``); each
# value becomes its own parametrized test invocation. That way a single
# pytest call produces distinct named tests (``test[1]``, ``test[2]``,
# ``test[3]``), so ``test.log.txt`` and every other per-test artifact
# lands in its own directory under ``test_outputs/test_…[N]/`` instead
# of colliding across sequential runs of the same test name.
#
# The parent ``deploy/conftest.py`` provides the shared options
# (--namespace, --image, --log-pvc, --model-pvc, --storage-class,
# --prefetch-model, --recreate-log-pvc, etc.) and the ``runtime_env``
# fixture they assemble. ``--units`` is only visible to tests under
# ``load/``.

import pytest


def _parse_units_arg(value):
    """Parse ``--units`` value: ``"3"`` → ``[3]``; ``"1,2,3"`` → ``[1,2,3]``."""
    if value is None:
        return [1]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out = [int(p) for p in parts]
    if not out:
        raise pytest.UsageError("--units cannot be empty")
    if any(n < 1 for n in out):
        raise pytest.UsageError(f"--units values must be >= 1, got {out}")
    return out


def pytest_addoption(parser):
    parser.addoption(
        "--units",
        type=str,
        default="1",
        help="Number of 1F:2P:1D units to replicate inside one DGD. "
        "Accepts a single integer (--units=3) or a comma-separated list "
        "(--units=1,2,3) for parametrized chained sweeps. "
        "N=1 → 1 FE / 2 PF / 1 DE (6 GPUs, fits one 8×H100 node). "
        "N=2 → 2 FE / 4 PF / 2 DE. N=3 → 3 FE / 6 PF / 3 DE. "
        "Only tests under tests/fault_tolerance/deploy/load/ see this flag.",
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests that take a ``units`` fixture, one test per N.

    With ``--units=1,2,3`` pytest produces three test invocations whose
    node names are ``test_load_sweep_…[1]``, ``[2]``, ``[3]``; the root
    conftest's autouse ``logger`` fixture then writes ``test.log.txt``
    to a distinct directory per N.
    """
    if "units" in metafunc.fixturenames:
        units_list = _parse_units_arg(metafunc.config.getoption("--units"))
        # Use ids=str(n) so the dir name reads ``test_…[3]`` not ``test_…[units0]``.
        metafunc.parametrize("units", units_list, ids=[str(n) for n in units_list])
