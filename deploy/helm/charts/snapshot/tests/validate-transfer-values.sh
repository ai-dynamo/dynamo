#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

chart_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary_dir="$(mktemp -d)"
trap 'rm -rf "${temporary_dir}"' EXIT

render_with_values() {
  local name="$1"
  local count="$2"
  local chunk="$3"
  local values_file="${temporary_dir}/${name}.yaml"
  cat >"${values_file}" <<EOF
config:
  cudaCheckpoint:
    transferBufferCount: ${count}
    transferChunkBytes: ${chunk}
EOF
  helm template snapshot-transfer-values "${chart_dir}" --values "${values_file}"
}

expect_failure() {
  local name="$1"
  local count="$2"
  local chunk="$3"
  local expected="$4"
  local output
  if output="$(render_with_values "${name}" "${count}" "${chunk}" 2>&1)"; then
    printf 'expected Helm rendering to reject %s\n' "${name}" >&2
    return 1
  fi
  if [[ "${output}" != *"${expected}"* ]]; then
    printf 'Helm rendering for %s failed without expected message %q:\n%s\n' \
      "${name}" "${expected}" "${output}" >&2
    return 1
  fi
}

expect_rendered_values() {
  local name="$1"
  local count="$2"
  local chunk="$3"
  local expected_count="$4"
  local expected_chunk="$5"
  local output
  output="$(render_with_values "${name}" "${count}" "${chunk}")"
  if ! grep -Fqx "      transferBufferCount: ${expected_count}" <<<"${output}"; then
    printf 'Helm rendering for %s did not contain normalized transferBufferCount %s\n' \
      "${name}" "${expected_count}" >&2
    return 1
  fi
  if ! grep -Fqx "      transferChunkBytes: ${expected_chunk}" <<<"${output}"; then
    printf 'Helm rendering for %s did not contain normalized transferChunkBytes %s\n' \
      "${name}" "${expected_chunk}" >&2
    return 1
  fi
}

helm lint "${chart_dir}"
helm template snapshot-transfer-defaults "${chart_dir}" >/dev/null
expect_rendered_values custom-integers 2 33554432 2 33554432
expect_rendered_values integral-decimal-count 2.0 67108864 2 67108864
expect_rendered_values integral-decimal-chunk 1 67108864.0 1 67108864

for value_type in fractional boolean string null; do
  case "${value_type}" in
    fractional) value=1.5 ;;
    boolean) value=true ;;
    string) value='"2"' ;;
    null) value=null ;;
  esac
  expect_failure \
    "transfer-buffer-count-${value_type}" \
    "${value}" \
    67108864 \
    "transferBufferCount must be an integral numeric value"
done

for value_type in fractional boolean string null; do
  case "${value_type}" in
    fractional) value=67108864.5 ;;
    boolean) value=true ;;
    string) value='"67108864"' ;;
    null) value=null ;;
  esac
  expect_failure \
    "transfer-chunk-bytes-${value_type}" \
    1 \
    "${value}" \
    "transferChunkBytes must be an integral numeric value"
done

expect_failure transfer-buffer-count-negative -1 67108864 \
  "transferBufferCount must be between 1 and 8"
expect_failure transfer-buffer-count-out-of-range 9 67108864 \
  "transferBufferCount must be between 1 and 8"
expect_failure transfer-chunk-bytes-negative 1 -4096 \
  "transferChunkBytes must be a 4096-byte multiple between 1048576 and 268435456"
expect_failure transfer-chunk-bytes-out-of-range 1 268439552 \
  "transferChunkBytes must be a 4096-byte multiple between 1048576 and 268435456"
expect_failure transfer-chunk-bytes-misaligned 1 1048577 \
  "transferChunkBytes must be a 4096-byte multiple between 1048576 and 268435456"

printf 'snapshot transfer-value Helm checks passed\n'
