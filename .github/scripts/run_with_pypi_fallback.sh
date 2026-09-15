#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

if (( $# < 2 )); then
  echo "usage: $0 <log-file> <command> [args...]" >&2
  exit 2
fi

log_file=$1
shift

"$@" 2>&1 | tee "$log_file"
primary_rc=${PIPESTATUS[0]}

if (( primary_rc == 0 )); then
  exit 0
fi

# Restrict failover to builds that intentionally use NVIDIA Artifactory.
if [[ ${PYPI_FALLBACK_ENABLED:-false} != true ]] \
  || [[ ${PIP_INDEX_URL:-} != https://artifactory.nvidia.com/* ]]; then
  exit "$primary_rc"
fi

# Large PyPI artifacts are redirected to this CloudFront distribution. Its
# access-policy failures must not make unrelated build failures retryable.
if ! grep -Fqi "d1j32scj9xxftt.cloudfront.net" "$log_file" \
  || ! grep -Eqi "403|Request blocked" "$log_file"; then
  exit "$primary_rc"
fi

# Keep the successful retry log clean for BuildKit metrics parsing.
primary_log="${log_file}.artifactory-failed"
mv "$log_file" "$primary_log"
echo "::warning title=PyPI fallback::Artifactory CDN returned 403; retrying with public PyPI"
export PIP_INDEX_URL="https://pypi.org/simple/"
export UV_DEFAULT_INDEX="$PIP_INDEX_URL"

"$@" 2>&1 | tee "$log_file"
exit "${PIPESTATUS[0]}"
