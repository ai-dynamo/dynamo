{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/compliance.Dockerfile ===
#
# Inline-compliance Dockerfile stages, shared by every shipped runtime
# template (dynamo / vllm / sglang / trtllm / frontend / planner).
#
# This template emits five stages in fixed order:
#
#   1. licenses          -- runs compliance.generators against the
#                           previously-defined build stage, validates
#                           output against policy, stages /legal/ + /sboms/.
#   2. sboms             -- FROM scratch; exposes /sboms/ for CI extraction.
#   3. legal             -- FROM scratch; exposes /legal/ for CI extraction.
#   4. sources_collect   -- gated on ENABLE_SOURCE_ARCHIVAL; runs
#                           compliance.collect_sources to produce /sources.zip.
#   5. sources_archive   -- FROM scratch; exposes /sources.zip.
#
# The caller (each per-framework runtime template) is expected to:
#   - have defined `runtime_pre` / `frontend_pre` / `planner_builder` already
#   - end with its own final stage (typically `runtime`) that does
#     `COPY --from=licenses /legal /legal` to inherit NOTICES.
#
# Jinja variables consumed:
#
#   compliance_base_stage     -- "runtime_pre" / "frontend_pre" / "planner_builder";
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
#   - vllm + trtllm with make_efa=true also pull in native attribution
#     for libfabric + aws-ofi-nccl via the YAML overlay.

FROM {{ compliance_base_stage }} AS licenses

USER root
RUN mkdir -p /legal /sboms
COPY --chown=root:0 container/compliance /opt/compliance
ENV PYTHONPATH=/opt
{% if target == "frontend" %}
# Approach A: EPP self-describes. frontend.Dockerfile pulls a dedicated
# amd64-pinned EPP stage `epp_sbom` whose only purpose is to expose the
# CycloneDX SBOM (emitted by cyclonedx-gomod in EPP's amd64 go-builder).
# The arm64 EPP view doesn't carry /sbom-go.cdx.json so we MUST pull
# from the amd64-pinned stage regardless of TARGETPLATFORM. The SBOM
# itself is architecture-independent JSON, so this is safe.
COPY --from=epp_sbom /sbom-go.cdx.json /tmp/sbom-go-epp.cdx.json
{% endif %}

# BASELINE_SBOM_FILE: the slim CycloneDX SBOM under
# /opt/compliance/base_sboms/ to subtract before writing NOTICES.
# Rendered from context.yaml by container/render.py:_render_context().
# Empty when no baseline has been captured for this target -- the
# generators then emit NOTICES for the full final image (correct but
# unfiltered).
ARG BASELINE_SBOM_FILE="{{ compliance_baseline_sbom }}"
RUN python3 -m compliance.generators \
    --ecosystem python,rust,dpkg{% if target == "frontend" %},go{% endif %}{% if make_efa %},native{% endif %} \
{% if framework == "sglang" %}    --site-packages "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
{% else %}    --venv ${VIRTUAL_ENV} \
{% endif %}{% if target == "frontend" %}    --go-sbom /tmp/sbom-go-epp.cdx.json \
{% endif %}    --output-dir /legal \
{% if make_efa %}    --native-yaml /opt/compliance/native_packages.yaml \
    --native-image {{ framework }}-runtime-efa \
{% endif %}    ${BASELINE_SBOM_FILE:+--subtract-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}} \
    -v
RUN find /legal -name '*-deps.csv' -print0 | \
    xargs -0 -n1 -I {} python3 -m compliance.policy.validate \
        --policy /opt/compliance/policy/licenses.toml \
        --input {}
RUN find /legal -name '*-deps.csv' -print -exec sh -c \
    'mkdir -p "/sboms/$(basename $(dirname "$1"))" && mv "$1" "/sboms/$(basename $(dirname "$1"))/$(basename "$1")"' _ {} \;


#######################################
########## Compliance: sboms ##########
#######################################

FROM scratch AS sboms
COPY --from=licenses /sboms/ /


#######################################
########## Compliance: legal ##########
#######################################

FROM scratch AS legal
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
RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        python3 -m compliance.collect_sources \
            --ecosystem dpkg --ecosystem rust --ecosystem native \
            --output-zip /sources.zip \
            --sources-root /sources \
            --native-source-dir /opt/native-sources \
{% if framework == "sglang" %}            --rust-site-packages "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
{% else %}            --rust-venv ${VIRTUAL_ENV} \
{% endif %}            --rust-vendor-full /opt/dynamo-vendor-full \
            ${BASELINE_SBOM_FILE:+--baseline-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}} \
            -v ; \
    else \
        : > /sources.zip ; \
    fi


FROM scratch AS sources_archive
COPY --from=sources_collect /sources.zip /sources.zip

# === END templates/compliance.Dockerfile ===
