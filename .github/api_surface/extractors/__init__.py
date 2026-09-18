# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-surface extractors for the API surface tracker.

Each module owns one surface and exposes a frozen interface:

    SURFACE: str
    def extract(repo_path: Path, release: str) -> OperationResult

Dispatch/registry lives in the parent-owned CLI layer (Phase C), not here,
so the six Phase B extractor modules can be authored in parallel without
sharing this file. Keep this module import-only (no per-extractor imports).
"""
