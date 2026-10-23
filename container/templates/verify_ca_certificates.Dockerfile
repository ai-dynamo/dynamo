{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# Reject base images or mirrors that ship ca-certificates older than the minimum below.
RUN set -eu; \
    . /etc/os-release; \
    case "${ID}:${VERSION_ID:-unknown}" in \
        ubuntu:22.04|ubuntu:24.04) ca_min="20260816ubuntu0.${VERSION_ID}.1" ;; \
        debian:12) ca_min="20250419~deb12u2" ;; \
        debian:13) ca_min="20250419+deb13u1" ;; \
        *) echo "Unsupported CA package baseline: ${ID}:${VERSION_ID:-unknown}" >&2; exit 1 ;; \
    esac; \
    ca_version="$(dpkg-query -W -f='${Version}' ca-certificates)"; \
    dpkg --compare-versions "$ca_version" ge "$ca_min" || { \
        echo "ca-certificates $ca_version is below required $ca_min" >&2; exit 1; \
    }
