{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/compliance.Dockerfile ===
#
# Inline-compliance Dockerfile stages, shared by every shipped runtime
# template (dynamo / vllm / sglang / trtllm / frontend / planner).
#
# This template emits four stages in fixed order:
#
#   1. licenses          -- runs compliance.generators against the
#                           previously-defined build stage, validates output
#                           against policy, and stages the unified /legal tree
#                           (flat NOTICES-<Eco>.txt + osrb-deps.csv + osrb.cdx.json).
#   2. compliance        -- FROM scratch; exposes the unified /legal tree for CI
#                           extraction as a single `-compliance` artifact.
#   3. sources_collect   -- gated on ENABLE_SOURCE_ARCHIVAL; runs
#                           compliance.collect_sources to produce /sources.zip.
#   4. sources_archive   -- FROM scratch; exposes /sources.zip.
#
# The caller (each per-framework runtime template) is expected to:
#   - have defined `pre_runtime` / `pre_frontend` / `pre_planner` already
#   - end with its own final stage (typically `runtime`) that does
#     `COPY --from=licenses /legal /legal` to inherit NOTICES.
#
# Jinja variables consumed:
#
#   compliance_base_stage     -- "pre_runtime" / "pre_frontend" / "pre_planner";
#                                set by container/render.py:_render_context()
#                                from `target`.
#   compliance_baseline_sbom  -- filename under base_sboms/ (or empty string
#                                if no baseline captured); set by
#                                _render_context() from `target`/`framework`/
#                                `device_key`.
#   framework, target, make_efa -- already in render context; control
#                                  ecosystem flags + EFA native attribution.

#######################################
########## Compliance: licenses #######
#######################################
#
# Runs every per-ecosystem generator under container/compliance/generators/
# against the parent build stage's filesystem, applies the license policy
# gate, and exposes /legal/ + /sboms/ for the next two stages to fan out.
#
# Per-framework variations:
#   - sglang uses `--site-packages "$(... sysconfig ...)"` because the
#     upstream image installs into system Python via
#     `pip install --break-system-packages`, not a venv.
#   - frontend additionally feeds the EPP image's CycloneDX Go SBOM via
#     `--go-sbom` (the EPP image is COPY'd from earlier in frontend.Dockerfile).
#   - native always runs (image filter "{framework}-{target}[-efa]"), attributing
#     the from-source binaries the python/rust/dpkg scanners miss (ffmpeg, libvpx,
#     UCX, NIXL, gdrcopy, libfabric, etcd, nats-server) per native_packages.yaml's
#     per-image `images:` lists; make_efa adds the EFA-only entries.

FROM {{ compliance_base_stage }} AS licenses

USER root
RUN mkdir -p /legal /sboms
COPY --chown=root:0 container/compliance /opt/compliance
ENV PYTHONPATH=/opt

# Real crate LICENSE files harvested from the cargo registry in wheel_builder
# (empty when none were harvested -- the rust generator then falls back to
# canonical SPDX text). Keyed "<name>-<version>". wheel_builder_base always
# creates the dir, so this COPY never fails even for wheel-less targets.
COPY --from=wheel_builder /opt/dynamo/rust-licenses /tmp/rust-licenses
{% if target == "frontend" %}
# EPP's Go compliance SBOM + harvested module LICENSE files, read from the build
# CONTEXT (.epp-sbom/) rather than COPY --from the EPP image. The CI EPP-build
# step exports them there via `make sbom-export` while the build cache is warm
# (see deploy/inference-gateway/epp/Dockerfile sbom-export stage). This avoids
# re-pulling the pushed EPP image — whose runtime layer could be served from a
# stale cache and miss these files after the BuildKit builder is refreshed. The
# SBOM is architecture-independent JSON, exported once on amd64.
COPY .epp-sbom/sbom-go.cdx.json /tmp/sbom-go-epp.cdx.json
# Real Go module LICENSE files so the go generator inlines upstream license text
# instead of canonical SPDX fallback.
COPY .epp-sbom/sbom-go-licenses /tmp/go-licenses
{% endif %}

# BASELINE_SBOM_FILE: the per-arch baseline SBOM *stem* (e.g.
# "cuda-dl-base@8315e245") under /opt/compliance/base_sboms/. We append
# "-${TARGETARCH}.cdx.json" so each platform of a multi-arch build subtracts
# its OWN-arch floor — the amd64 baseline would otherwise under-attribute a
# package present in the amd64 base but not the arm64 base that we install on
# arm64. Rendered from context.yaml's baseline_sbom by render.py; empty when no
# baseline is captured (NOTICES then cover the full image — correct but
# unfiltered).
ARG BASELINE_SBOM_FILE="{{ compliance_baseline_sbom }}"
ARG TARGETARCH
# Resolve where this image's Python packages live at runtime rather than per
# framework: venv-based images export VIRTUAL_ENV (trtllm, planner, frontend,
# vllm xpu/cpu, dev), while images that install into system Python leave it
# unset (vllm cuda via `pip --system`, sglang via `pip --break-system-packages`).
# Pick the matching generator flag so it always finds the deps — passing an
# empty `--venv ${VIRTUAL_ENV}` is what broke system-Python images.
# dpkg is scoped out for planner: it ships a distroless RUNTIME image, but its
# licenses stage runs against the python:3.12-slim BUILD stage (pre_planner) so a
# Python interpreter is available. Scanning that build image's full dpkg set
# attributes ~120 OS packages that never ship in the distroless runtime (and trips
# the policy gate on e.g. libcom-err2's GPL, which isn't redistributed). Like the
# operator/snapshot distroless images, the runtime base OS is NGC's responsibility
# (tracked via the baseline corpus); planner attributes only what it adds — the
# venv (python+rust). NOTE: the few raw OS artifacts planner copies into distroless
# (libgomp.so, etcd, nats-server, dash) are a cross-image native-attribution gap
# tracked separately, not unique to planner.
RUN {% if framework == "sglang" %}PKG_ARG="--site-packages $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"{% else %}if [ -n "${VIRTUAL_ENV:-}" ]; then PKG_ARG="--venv ${VIRTUAL_ENV}"; else PKG_ARG="--site-packages $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; fi{% endif %} && \
    python3 -m compliance.generators \
    --ecosystem python,rust{% if target != "planner" %},dpkg{% endif %}{% if target == "frontend" %},go{% endif %},native \
    ${PKG_ARG} \
{% if target == "frontend" %}    --go-sbom /tmp/sbom-go-epp.cdx.json \
    --go-licenses-dir /tmp/go-licenses \
{% endif %}    --rust-licenses-dir /tmp/rust-licenses \
    --output-dir /legal \
    --policy /opt/compliance/policy/licenses.toml \
    --native-yaml /opt/compliance/native_packages.yaml \
    --native-image {{ framework }}-{{ target }}{% if make_efa %}-efa{% endif %} \
    ${BASELINE_SBOM_FILE:+--subtract-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}-${TARGETARCH}.cdx.json} \
    -v
# Policy gate runs on the single unified CSV (its `ecosystem` column scopes each
# row), replacing the per-ecosystem loop. Non-zero exit fails the build.
RUN python3 -m compliance.policy.validate \
        --policy /opt/compliance/policy/licenses.toml \
        --input /legal/osrb-deps.csv


#######################################
####### Compliance: artifact ##########
#######################################
#
# Single FROM-scratch stage exposing the unified compliance tree for CI
# extraction (one `-compliance` artifact): flat NOTICES-<Eco>.txt + the unified
# osrb-deps.csv (with Notes) + osrb.cdx.json (delta CycloneDX). Export is bounded
# by these files' size (a few MB) regardless of runtime image size.

FROM scratch AS compliance
COPY --from=licenses /legal/ /


#######################################
########## Compliance: sources ########
#######################################
#
# Collects third-party source archives on top of the runtime baseline.
# Gated on ENABLE_SOURCE_ARCHIVAL -- default off so PR builds stay fast;
# CI flips it on for nightly + release/*.*.* branch pushes (see
# .github/workflows/post-merge-ci.yml and nightly-ci.yml).

FROM {{ compliance_base_stage }} AS sources_collect

USER root
RUN mkdir -p /sources /opt/compliance /opt/native-sources /opt/dynamo-vendor-full
COPY --chown=root:0 container/compliance /opt/compliance
ENV PYTHONPATH=/opt
COPY --from=wheel_builder /tmp/native-sources/ /opt/native-sources/
COPY --from=wheel_builder /tmp/dynamo-vendor-full/ /opt/dynamo-vendor-full/

ARG ENABLE_SOURCE_ARCHIVAL=false
ARG BASELINE_SBOM_FILE="{{ compliance_baseline_sbom }}"
ARG TARGETARCH
RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        {% if framework == "sglang" %}RUST_PKG_ARG="--rust-site-packages $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"{% else %}if [ -n "${VIRTUAL_ENV:-}" ]; then RUST_PKG_ARG="--rust-venv ${VIRTUAL_ENV}"; else RUST_PKG_ARG="--rust-site-packages $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; fi{% endif %} && \
        python3 -m compliance.collect_sources \
            --ecosystem dpkg --ecosystem rust --ecosystem native \
            --output-zip /sources.zip \
            --sources-root /sources \
            --native-source-dir /opt/native-sources \
            ${RUST_PKG_ARG} \
            --rust-vendor-full /opt/dynamo-vendor-full \
            ${BASELINE_SBOM_FILE:+--baseline-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}-${TARGETARCH}.cdx.json} \
            -v ; \
    else \
        : > /sources.zip ; \
    fi


FROM scratch AS sources_archive
COPY --from=sources_collect /sources.zip /sources.zip

# === END templates/compliance.Dockerfile ===
