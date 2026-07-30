# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip this directory's tests when the compiled bindings aren't built.

Every test here imports ``dynamo.sample_engine`` (→ ``dynamo.backend`` → the
PyO3 extension ``dynamo._core.backend``) at import time. On a source checkout
without ``maturin develop`` that fails during collection, so ignore the whole
directory. CI (which builds the wheel) still runs everything. This lets each
test import at module top with no per-file ``importorskip`` / ``noqa: E402``.
"""

import importlib

try:
    importlib.import_module("dynamo._core.backend")
    _HAS_BINDINGS = True
except ImportError:
    _HAS_BINDINGS = False


def pytest_ignore_collect(collection_path, config):
    if (
        not _HAS_BINDINGS
        and collection_path.suffix == ".py"
        and collection_path.name.startswith("test_")
    ):
        return True
    return None
