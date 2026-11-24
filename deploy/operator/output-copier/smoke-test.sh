#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

image=${1:?Usage: smoke-test.sh IMAGE PLATFORM}
platform=${2:?Usage: smoke-test.sh IMAGE PLATFORM}

docker run --rm --platform "${platform}" --network none --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=16m --cap-drop ALL \
    --security-opt no-new-privileges "${image}" \
    /bin/busybox timeout 30 /bin/sh -ec '
    set -o pipefail

    echo "Checking the image user and copier utilities"
    [ "$(busybox id -u)" = 1000 ]
    [ "$(busybox id -g)" = 1000 ]
    [ -s /etc/ssl/certs/ca-certificates.crt ]
    for tool in kubectl date grep awk sed tr cat sleep; do
        command -v "$tool"
    done
    if (false | cat); then
        echo "ERROR: sh did not propagate the pipeline failure" >&2
        exit 1
    fi

    echo "Parsing profiler status with the copier shell pipeline"
    printf "status: success\nphase: Done\nmessage: profiling complete\n" >/tmp/profiler_status.yaml
    [ "$(grep "^status:" /tmp/profiler_status.yaml | awk "{print \$2}" | tr -d "\"")" = success ]
    [ "$(grep "^message:" /tmp/profiler_status.yaml | sed "s/^message: *//")" = "profiling complete" ]
    date +%s | grep -Eq "^[0-9]+$"
    sleep 0

    echo "Reading profiler output with the real upstream kubectl client"
    kubectl version --client -o yaml
    printf "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n" >/tmp/final_config.yaml
    kubectl create configmap copier-smoke --dry-run=client \
        --from-file=final_config.yaml=/tmp/final_config.yaml \
        --from-file=profiler_status.yaml=/tmp/profiler_status.yaml -o yaml >/tmp/cm.yaml
    grep -q "kind: ConfigMap" /tmp/cm.yaml
    grep -q "kind: DynamoGraphDeployment" /tmp/cm.yaml
    echo "Output-copier image smoke test passed"
'
