{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sbom_inject.Dockerfile ===
# Generates the per-runtime CycloneDX 1.6 SBOM and ATTRIBUTIONS-container-*
# files in-place in the final runtime image, where every framework artifact,
# Python site-packages tree, and source-compiled binary is already present.
# Scan targets are discovered dynamically: the script always scans /opt/dynamo
# and then probes a fixed allowlist of framework install roots and the Python
# site-/dist-packages directories on the image, adding each that exists. Pass
# --build-arg SBOM_EXTRA_TARGETS="dir:/foo dir:/bar" to append more targets at
# docker build time.
#
# Outputs:
#   /opt/dynamo/sbom/dynamo.cdx.json       (+ .sha256 sidecar)
#   /opt/dynamo/licenses/ATTRIBUTIONS-container-*.{md,csv}
#   /opt/dynamo/licenses/LICENSE-MANIFEST.csv
#   /opt/dynamo/licenses/LICENSES/<spdx-id>.txt
#
# Include this fragment near the end of a runtime template, after the last
# wheel_builder COPY block and before the final USER/ENTRYPOINT lines.
USER root

ARG DYNAMO_COMMIT_SHA
ARG SBOM_EXTRA_TARGETS=""
RUN --mount=type=bind,from=anchore/syft:v1.16.0,source=/syft,target=/usr/local/bin/syft \
    --mount=type=bind,source=./container/compliance,target=/tmp/compliance,ro \
    set -eux; \
    mkdir -p /opt/dynamo/sbom /opt/dynamo/licenses; \
    # Dynamic scan-target discovery. /opt/dynamo is always present; the rest
    # depend on the framework's install layout, so probe and add what exists.
    SCAN_TARGETS=("dir:/opt/dynamo"); \
    for candidate in /opt/vllm /opt/sglang /opt/tensorrt_llm /opt/trtllm /workspace/sglang /workspace/vllm; do \
        [ -d "$candidate" ] && SCAN_TARGETS+=("dir:$candidate"); \
    done; \
    for d in /usr/local/lib/python*/dist-packages /usr/local/lib/python*/site-packages /usr/lib/python*/dist-packages /opt/conda/lib/python*/site-packages; do \
        [ -d "$d" ] && SCAN_TARGETS+=("dir:$d"); \
    done; \
    for t in ${SBOM_EXTRA_TARGETS}; do SCAN_TARGETS+=("$t"); done; \
    SCAN_ARGS=(); \
    for t in "${SCAN_TARGETS[@]}"; do SCAN_ARGS+=(--scan-target "$t"); done; \
    echo ">> SBOM scan targets: ${SCAN_TARGETS[*]}"; \
    bash /tmp/compliance/sbom/generate_sbom.sh \
        "${SCAN_ARGS[@]}" \
        --framework {{ framework }} \
        --output-dir /opt/dynamo/sbom \
        --name dynamo; \
    python3 /tmp/compliance/sbom/render_attributions.py \
        --bom /opt/dynamo/sbom/dynamo.cdx.json \
        --release "${DYNAMO_COMMIT_SHA:-HEAD}" \
        --output-dir /opt/dynamo/licenses; \
    chmod -R a+rX /opt/dynamo/sbom /opt/dynamo/licenses; \
    chown -R dynamo:0 /opt/dynamo/sbom /opt/dynamo/licenses

USER dynamo
# === END templates/sbom_inject.Dockerfile ===
