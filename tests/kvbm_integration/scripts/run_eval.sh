#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Layer C — run the determinism eval against an already-running vllm+KVBM
# server. Requires KVBM_EXTERNAL_BASE_URL and KVBM_EXTERNAL_METRICS_PORT
# (printed by scripts/run_server.sh). Iterate freely without re-spawning
# the server.

set -euo pipefail

if [[ -z "${KVBM_EXTERNAL_BASE_URL:-}" || -z "${KVBM_EXTERNAL_METRICS_PORT:-}" ]]; then
    cat <<'EOF' >&2
[eval] KVBM_EXTERNAL_BASE_URL and KVBM_EXTERNAL_METRICS_PORT must be set.
Start a server with scripts/run_server.sh <spec-id> first, then export the
values it printed (KVBM_EXTERNAL_BASE_URL, KVBM_EXTERNAL_METRICS_PORT,
KVBM_SPEC_ID) and re-run this script.
EOF
    exit 2
fi

if [[ -z "${KVBM_SPEC_ID:-}" ]]; then
    cat <<'EOF' >&2
[eval] KVBM_SPEC_ID must be set so shell 3 only runs the spec shell 2 launched.
run_server.sh prints the export line for it; copy that into your shell.
EOF
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Default: agg-cache-reset test, filtered to the spec id the server was
# launched with. Positional args override entirely (ad-hoc runs).
TARGET="tests/kvbm_integration/test_determinism_agg.py::TestDeterminismAgg::test_determinism_agg_with_cache_reset"

if [[ $# -eq 0 ]]; then
    set -- "$TARGET" -v --tb=short -k "$KVBM_SPEC_ID"
fi

exec python -m pytest "$@"
