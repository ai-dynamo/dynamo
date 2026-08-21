#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

artifact_dir="artifacts/buildkite/comparison"
cold_result="artifacts/buildkite/image-build/cold/result.json"
warm_result="artifacts/buildkite/image-build/warm/result.json"
mkdir -p "${artifact_dir}"

buildkite-agent artifact download "${cold_result}" .
buildkite-agent artifact download "${warm_result}" .

cold_seconds="$(jq -er '.duration_seconds' "${cold_result}")"
warm_seconds="$(jq -er '.duration_seconds' "${warm_result}")"

if (( cold_seconds <= 0 || warm_seconds <= 0 )); then
  echo "Build durations must both be positive" >&2
  exit 1
fi

github_seconds="${GITHUB_BASELINE_SECONDS:-}"
if [[ -n "${github_seconds}" ]] && ! [[ "${github_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GITHUB_BASELINE_SECONDS must be a positive integer" >&2
  exit 2
fi

jq -n \
  --argjson cold_seconds "${cold_seconds}" \
  --argjson warm_seconds "${warm_seconds}" \
  --arg github_seconds "${github_seconds}" \
  --arg github_url "${GITHUB_BASELINE_URL:-}" \
  '{
    cold_seconds: $cold_seconds,
    warm_seconds: $warm_seconds,
    cache_speedup: ($cold_seconds / $warm_seconds),
    cache_time_saved_percent: ((($cold_seconds - $warm_seconds) / $cold_seconds) * 100),
    github_baseline_seconds: (if $github_seconds == "" then null else ($github_seconds | tonumber) end),
    github_baseline_url: (if $github_url == "" then null else $github_url end),
    warm_vs_github_speedup: (
      if $github_seconds == "" then null
      else (($github_seconds | tonumber) / $warm_seconds)
      end
    )
  }' > "${artifact_dir}/comparison.json"

cache_speedup="$(jq -r '.cache_speedup * 100 | round / 100' "${artifact_dir}/comparison.json")"
cache_saved="$(jq -r '.cache_time_saved_percent * 10 | round / 10' "${artifact_dir}/comparison.json")"

summary="## Hosted remote-builder benchmark

| Measurement | Seconds |
| --- | ---: |
| First Buildkite build | ${cold_seconds} |
| Repeated Buildkite build | ${warm_seconds} |"

if [[ -n "${github_seconds}" ]]; then
  github_speedup="$(jq -r '.warm_vs_github_speedup * 100 | round / 100' "${artifact_dir}/comparison.json")"
  summary+="
| GitHub Actions baseline | ${github_seconds} |"
fi

summary+="

BuildKit cache speedup: **${cache_speedup}x** (${cache_saved}% time saved)."

if [[ -n "${github_seconds}" ]]; then
  summary+="

Warm Buildkite vs. GitHub Actions speedup: **${github_speedup}x**."
fi

buildkite-agent annotate --style info --context build-comparison "${summary}"
printf '%s\n' "${summary}"
