#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RETIRED — the install-command selector no longer has a generated data file.

This script used to generate ``docs/data/install-selector.json`` and a TS
module by scraping ``docs/reference/support-matrix.md`` and
``container/context.yaml``. That markdown page was removed when the Reference
pages moved to a single TypeScript source of truth
(``docs/fern/components/releases.data.ts``).

The install selector is now driven at build time by a hand-authored view
module, ``docs/fern/components/install-selector-data.ts``, which imports
``RELEASES``, ``MAIN_TOT``, ``CURRENT_VERSION`` and ``CURRENT_WHEEL`` from
``releases.data.ts`` and formats the install commands directly — there is no
generation step and no JSON artifact to keep in sync. The agent-facing twins
and machine-readable exports come from ``docs/fern/scripts/gen_llms_tables.py``.

No workflow invokes this script. It is kept as a tombstone (rather than a
dangling reference to the deleted markdown) so anyone who runs it gets a clear
pointer to the current flow instead of a ``FileNotFoundError``.

To change install commands, edit ``install-selector-data.ts``. To change the
versions those commands show, edit ``releases.data.ts`` (see its
PER-RELEASE BUMP CHECKLIST).
"""

from __future__ import annotations

import sys

MESSAGE = (
    "build_install_selector.py is retired.\n"
    "The install selector is now sourced from "
    "docs/fern/components/install-selector-data.ts, which reads "
    "docs/fern/components/releases.data.ts directly — no generation step.\n"
    "Edit those files instead; run docs/fern/scripts/gen_llms_tables.py for the "
    "agent-facing twins and machine-readable exports."
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
