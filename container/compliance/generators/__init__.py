# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-ecosystem NOTICES generators run inside each container's `licenses` Dockerfile stage.

Each module exposes a ``generate(...) -> Result`` function that produces:
- ``NOTICES-<Ecosystem>.txt`` rendered into the output directory
- ``<ecosystem>-deps.csv`` for downstream policy validation
- Per-package ``LICENSE-*`` text files where available

The orchestrator in ``__main__`` dispatches to one or more ecosystems.
"""
