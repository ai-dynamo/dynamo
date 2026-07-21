# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spica only configures process-wide JAX state when a sampler is requested."""

import subprocess
import sys
import textwrap


def _run_isolated(script: str) -> None:
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        text=True,
        capture_output=True,
    )


def test_import_spica_has_no_jax_process_side_effects():
    _run_isolated(
        """
        import logging
        import os

        os.environ.pop("JAX_PLATFORMS", None)
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
        logging.getLogger("absl").setLevel(logging.WARNING)

        import dynamo.profiler.spica as spica

        assert spica.__name__ == "dynamo.profiler.spica"
        assert "JAX_PLATFORMS" not in os.environ
        assert logging.getLogger("jax._src.xla_bridge").level == logging.WARNING
        assert logging.getLogger("absl").level == logging.WARNING
        """
    )


def test_configure_vizier_runtime_is_explicit_opt_in():
    _run_isolated(
        """
        import logging
        import os

        os.environ.pop("JAX_PLATFORMS", None)
        from dynamo.profiler.spica._quiet import configure_vizier_runtime

        configure_vizier_runtime()

        assert os.environ.get("JAX_PLATFORMS") == "cpu"
        assert logging.getLogger("jax._src.xla_bridge").level == logging.ERROR
        assert logging.getLogger("absl").level == logging.ERROR
        """
    )
