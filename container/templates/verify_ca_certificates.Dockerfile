{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shell fragment, not a Dockerfile instruction. It fails unless the installed
# ca-certificates is at or above the floor below, so base images and mirrors
# that ship an older package are rejected.
#
# Include it as the last command of the RUN that installs or upgrades
# ca-certificates, after "&& \" (or after "; \" under set -e). The floor is
# then part of that RUN's text, which is the layer cache key, and a layer below
# the floor fails before BuildKit can cache it. Do not put it in a RUN of its
# own: BuildKit then reuses a cached install layer that holds an older package,
# and the check fails on every build that gets that layer.
-#}
( set -eu; \
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
    } )
