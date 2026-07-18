#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^(a|b|c|m|p|mp)$ ]]; then
    echo "usage: $0 {a|b|c|m|p|mp} EVIDENCE_DIRECTORY" >&2
    exit 2
fi

VARIANT=$1
ART=$2
while [[ "$ART" != / && "$ART" == */ ]]; do
    ART=${ART%/}
done
ROOT=$(git rev-parse --show-toplevel)
EXP="$ROOT/benchmarks/gms_cuda_init_ab"
OVERLAY="$EXP/manifests/variant-$VARIANT"
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
NS=schwinns
DGD=g52-t8-gms-prof-r29604929787-r2
COMPONENT=VllmDecodeWorker
CKPT=checkpoint-57a124961e2a47a2cf9c2712e58a0a2b
CKPT_UID=1fb182f1-1a4c-4c51-aff0-67ab530437ea
CKPT_ID=57a124961e2a47a2cf9c2712e58a0a2b
FRONT=g52-t8-gms-prof-r29604929787-r2-frontend
MODEL=nvidia/GLM-5.2-NVFP4
NODE=cluster-0967a26d-pool-14bee067-prctr-s2877
CACHE_HELPER=gms-cuda-init-cache-helper-root-v2
OP_SELECTOR=app.kubernetes.io/instance=gmsprof-op-760e
POD_SELECTOR="nvidia.com/dynamo-graph-deployment-name=$DGD,nvidia.com/dynamo-component=$COMPONENT"
CLAIM_PREFIX="${DGD}-vllmdecode-intrapod-"
MAIN_IMAGE=dynamoci.azurecr.io/ai-dynamo/dynamo:760e55e21e14f76d7c204920f00ea9144d819b4b-vllm-placeholder-run-29604929787-1@sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2
MAIN_DIGEST=sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2
B_DIGEST=sha256:f0e3d788dca28715674705a7f151636dae3ee868f4df5575c1e284a777a7ab0a
EXPERIMENT_LOADER_DIGEST=REPLACE_LOADER_DIGEST
C_LOADER_DIGEST=sha256:592b70a87779348ce90a53ce7034ec84ea3f8274fc2b4b59177e53e7a4a99fe7
TIMES="$ART/timestamps.tsv"
PIDS=()
TRACKED_PIDS=()
WORKER_POD=
CLAIM=
CLEANUP_OWNED=0
ZERO_CONFIRMED=0

ART_LOCK="${ART}.run-variant.lock"
ART_LOCK_HELD=0

release_evidence_lock() {
    if [[ "$ART_LOCK_HELD" -eq 1 ]]; then
        rmdir -- "$ART_LOCK"
        ART_LOCK_HELD=0
    fi
}

mkdir -p -- "$(dirname -- "$ART")"
if ! mkdir -- "$ART_LOCK" 2>/dev/null; then
    echo "unable to acquire evidence directory lock: $ART_LOCK" >&2
    exit 1
fi
ART_LOCK_HELD=1
trap release_evidence_lock EXIT
trap 'release_evidence_lock; exit 1' INT TERM

if [[ -L "$ART" ]]; then
    echo "refusing evidence path symlink: $ART" >&2
    exit 1
fi
if [[ -e "$ART" && ! -d "$ART" ]]; then
    echo "evidence path exists and is not a directory: $ART" >&2
    exit 1
fi
if [[ -d "$ART" ]]; then
    first_evidence_entry=$(
        find "$ART" -mindepth 1 -maxdepth 1 -print -quit
    )
    if [[ -n "$first_evidence_entry" ]]; then
        echo "refusing to reuse non-empty evidence directory: $ART" >&2
        exit 1
    fi
fi
if [[ ! -e "$ART" ]]; then
    mkdir -- "$ART"
fi
mkdir -p "$ART"/{cache,inference,logs,metrics,objects,preflight,teardown}
release_evidence_lock
trap - EXIT INT TERM

printf 'utc\tevent\tdetails\n' > "$TIMES"
cat > "$ART/expected-uuids.txt" <<'EOF'
GPU-02ff0cc1-647f-dee7-8365-921738e945a6
GPU-0d5ad102-eb8f-922a-173e-e91033320e0f
GPU-9c595f65-4651-0b25-f95c-09a0abd5f5fa
GPU-4fba7684-5a96-6280-91ff-b41f7484564c
GPU-3ef7c092-d55c-ca6c-0018-9fc89ed28683
GPU-32a6c51c-d07d-7513-86b5-813b64e452d2
GPU-56c0be30-f1d3-a00d-8a2a-7fa70da8037f
GPU-ce231b92-c54f-3f0c-af1c-1a696db97f51
EOF
NFS_CACHE_ROOTS=(
    "/checkpoints/$CKPT_ID/versions/1"
    "/checkpoints/gms/$DGD/versions/1"
)
NVME_CACHE_ROOTS=(
    "/cache/nvme2/schwinns/$DGD"
    "/cache/nvme4/schwinns/$DGD"
    "/cache/nvme5/schwinns/$DGD"
    "/cache/nvme6/schwinns/$DGD"
    "/cache/nvme7/schwinns/$DGD"
    "/cache/nvme8/schwinns/$DGD"
    "/cache/nvme9/schwinns/$DGD"
)

now() {
    date -u +%FT%T.%NZ
}

stamp() {
    printf '%s\t%s\t%s\n' "$(now)" "$1" "${2:-}" | tee -a "$TIMES"
}

k() {
    kubectl --context "$CTX" -n "$NS" "$@"
}

choose_ephemeral_port() {
    local port
    if ! port=$(
        python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
    ); then
        echo "failed to ask the local kernel for an ephemeral port" >&2
        return 1
    fi
    if [[ ! "$port" =~ ^[0-9]+$ ]] || ((${#port} > 5)) ||
        ((10#$port < 1 || 10#$port > 65535)); then
        echo "invalid ephemeral port returned by Python: $port" >&2
        return 1
    fi
    printf '%s\n' "$port"
}

descendant_pids() {
    local child parent=$1
    while read -r child; do
        [[ -n "$child" ]] || continue
        descendant_pids "$child"
        printf '%s\n' "$child"
    done < <(pgrep -P "$parent" 2>/dev/null || true)
}

pid_is_running() {
    local pid=$1 state
    [[ -r "/proc/$pid/stat" ]] || return 1
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null) || return 1
    [[ "$state" != Z ]]
}

append_unique_pid() {
    local candidate=$1 existing
    for existing in "${TRACKED_PIDS[@]}"; do
        [[ "$existing" == "$candidate" ]] && return
    done
    TRACKED_PIDS+=("$candidate")
}

refresh_background_descendants() {
    local child pid
    local -a children=()
    for pid in "${PIDS[@]}"; do
        mapfile -t children < <(descendant_pids "$pid")
        for child in "${children[@]}"; do
            append_unique_pid "$child"
        done
    done
}

stop_background() {
    local attempt pid running=0
    local -a descendants=()
    TRACKED_PIDS=()
    for pid in "${PIDS[@]}"; do
        append_unique_pid "$pid"
    done
    refresh_background_descendants
    for pid in "${TRACKED_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for attempt in $(seq 1 50); do
        refresh_background_descendants
        running=0
        for pid in "${TRACKED_PIDS[@]}"; do
            if pid_is_running "$pid"; then
                running=1
                break
            fi
        done
        [[ "$running" -eq 0 ]] && break
        sleep 0.1
    done
    refresh_background_descendants
    descendants=("${TRACKED_PIDS[@]:${#PIDS[@]}}")
    for pid in "${PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    for pid in "${descendants[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    for attempt in $(seq 1 50); do
        running=0
        for pid in "${TRACKED_PIDS[@]}"; do
            if pid_is_running "$pid"; then
                running=1
                break
            fi
        done
        [[ "$running" -eq 0 ]] && break
        sleep 0.1
    done
    if [[ "$running" -ne 0 ]]; then
        echo "background processes remain after termination" >&2
        return 1
    fi
    PIDS=()
    TRACKED_PIDS=()
}

matching_claims() {
    k get resourceclaims -o json |
        jq -r --arg prefix "$CLAIM_PREFIX" \
            '.items[].metadata.name | select(startswith($prefix))'
}

zero_reached() {
    local dcd_nonzero deployment_nonzero pods claims
    if ! k get dynamographdeployment "$DGD" -o json |
        jq -e --arg component "$COMPONENT" '
            [
                .spec.components[] |
                select(.name == $component) |
                .replicas
            ] == [0]
        ' >/dev/null; then
        return 1
    fi
    if ! dcd_nonzero=$(
        k get dynamocomponentdeployments -l "$POD_SELECTOR" -o json |
            jq '[.items[] | select((.spec.replicas // 0) != 0)] | length'
    ); then
        return 1
    fi
    if ! deployment_nonzero=$(
        k get deployments -l "$POD_SELECTOR" -o json |
            jq '[.items[] | select((.spec.replicas // 0) != 0)] | length'
    ); then
        return 1
    fi
    if ! pods=$(k get pods -l "$POD_SELECTOR" -o name); then
        return 1
    fi
    if ! claims=$(matching_claims); then
        return 1
    fi
    [[ "$dcd_nonzero" == 0 && "$deployment_nonzero" == 0 &&
        -z "$pods" && -z "$claims" ]]
}

wait_for_zero() {
    local attempt
    for attempt in $(seq 1 300); do
        if zero_reached; then
            stamp ZERO_AND_DRA_RELEASED
            return
        fi
        sleep 1
    done
    echo "timed out waiting for DGD/DCD/deployment zero, pod deletion, and claim release" >&2
    return 1
}

scale_down() {
    local attempt patch_status
    if [[ "$CLEANUP_OWNED" -ne 1 || "$ZERO_CONFIRMED" -eq 1 ]]; then
        return
    fi
    stamp SCALE_DOWN_SENT
    : > "$ART/teardown/scale-down.txt"
    for attempt in $(seq 1 300); do
        patch_status=0
        k patch dynamographdeployment "$DGD" --type=json \
            -p='[
                {"op":"test","path":"/spec/components/1/name","value":"VllmDecodeWorker"},
                {"op":"replace","path":"/spec/components/1/replicas","value":0}
            ]' \
            >> "$ART/teardown/scale-down.txt" 2>&1 || patch_status=$?
        if zero_reached; then
            ZERO_CONFIRMED=1
            WORKER_POD=
            CLAIM=
            stamp ZERO_AND_DRA_RELEASED
            stamp SCALE_DOWN_CONFIRMED \
                "attempt=$attempt patch_status=$patch_status"
            return
        fi
        sleep 1
    done
    echo "timed out retrying scale-down and waiting for zero resources" >&2
    return 1
}

write_evidence_checksums() {
    local destination=$1
    find "$ART" -type f ! -path "$ART/SHA256SUMS" -print0 |
        sort -z |
        xargs -0 --no-run-if-empty sha256sum > "$destination"
}

finalize_evidence() {
    local attempt
    local first="${ART}.SHA256SUMS.$$.first.tmp"
    local second="${ART}.SHA256SUMS.$$.second.tmp"
    local max_attempts=3
    if [[ -e "$ART/SHA256SUMS" ]]; then
        echo "evidence finalization failed: refusing to replace existing checksum manifest" >&2
        return 1
    fi
    if ((${#PIDS[@]} != 0 || ${#TRACKED_PIDS[@]} != 0)); then
        echo "evidence finalization failed: tracked background writers are not stopped" >&2
        return 1
    fi
    if ! sha256sum \
        "$EXP/gms-fadvise-exact.py" \
        "$EXP/gms-host-sampler.sh" \
        "$EXP/validate-loader-profile.py" \
        "$EXP/validate-server-profile.py" \
        > "$ART/helper-checksums.txt"; then
        echo "evidence finalization failed: unable to checksum runner helpers" >&2
        return 1
    fi
    for attempt in $(seq 1 "$max_attempts"); do
        rm -f -- "$first" "$second"
        if ! write_evidence_checksums "$first"; then
            echo "evidence checksum attempt $attempt/$max_attempts: generation failed" >&2
            continue
        fi
        if [[ ! -s "$first" ]]; then
            echo "evidence checksum attempt $attempt/$max_attempts: no evidence files found" >&2
            continue
        fi
        if ! sha256sum -c "$first" >/dev/null; then
            echo "evidence checksum attempt $attempt/$max_attempts: files changed during first snapshot" >&2
            continue
        fi
        if ! write_evidence_checksums "$second"; then
            echo "evidence checksum attempt $attempt/$max_attempts: second generation failed" >&2
            continue
        fi
        if ! sha256sum -c "$second" >/dev/null; then
            echo "evidence checksum attempt $attempt/$max_attempts: files changed during second snapshot" >&2
            continue
        fi
        if ! cmp -s -- "$first" "$second"; then
            echo "evidence checksum attempt $attempt/$max_attempts: file set or content did not stabilize" >&2
            continue
        fi
        if ! sha256sum -c "$second" >/dev/null; then
            echo "evidence checksum attempt $attempt/$max_attempts: final verification failed" >&2
            continue
        fi
        rm -f -- "$first"
        if ! mv -- "$second" "$ART/SHA256SUMS"; then
            echo "evidence finalization failed: unable to publish stable checksum manifest" >&2
            rm -f -- "$second"
            return 1
        fi
        return 0
    done
    rm -f -- "$first" "$second"
    echo "evidence finalization failed: evidence did not stabilize after $max_attempts attempts" >&2
    return 1
}

cleanup() {
    local status=$?
    local cleanup_complete=1
    trap - EXIT INT TERM
    if ! stop_background; then
        status=1
        cleanup_complete=0
    fi
    if [[ "$CLEANUP_OWNED" -eq 1 && "$ZERO_CONFIRMED" -ne 1 ]]; then
        if ! scale_down; then
            status=1
            cleanup_complete=0
        fi
    fi
    stamp RUNNER_EXIT "status=$status"
    if [[ "$cleanup_complete" -eq 1 ]]; then
        if ! finalize_evidence; then
            echo "SHA256SUMS was not published for evidence directory: $ART" >&2
            status=1
        fi
    else
        echo "evidence finalization skipped because cleanup did not stop all tracked writers and confirm zero state: $ART" >&2
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

checkpoint_json() {
    k get dynamocheckpoint "$CKPT" -o json
}

validate_checkpoint() {
    local output=$1
    checkpoint_json | tee "$output" | jq -e \
        --arg uid "$CKPT_UID" --arg id "$CKPT_ID" '
        .metadata.uid == $uid and
        .metadata.ownerReferences == null and
        .metadata.deletionTimestamp == null and
        .status.phase == "Ready" and
        .status.checkpointID == $id
    ' >/dev/null
}

capture_objects() {
    local prefix=$1
    k get dynamographdeployment "$DGD" -o yaml \
        > "$ART/objects/$prefix-dgd.yaml"
    k get dynamocomponentdeployments -l "$POD_SELECTOR" -o yaml \
        > "$ART/objects/$prefix-dcds.yaml"
    k get deployments -l "$POD_SELECTOR" -o yaml \
        > "$ART/objects/$prefix-deployments.yaml"
    k get resourceclaims -o yaml > "$ART/objects/$prefix-resourceclaims.yaml"
    k get events --sort-by=.metadata.creationTimestamp \
        > "$ART/objects/$prefix-events.txt"
    if [[ -n "$WORKER_POD" ]]; then
        k get pod "$WORKER_POD" -o yaml \
            > "$ART/objects/$prefix-worker-pod.yaml"
        k describe pod "$WORKER_POD" \
            > "$ART/objects/$prefix-worker-pod.describe.txt"
    fi
}

validate_eviction_results() {
    local output=$1
    shift
    local expected
    expected=$(jq -cn --args '$ARGS.positional' "$@")
    jq -e --argjson expected "$expected" '
        .ok == true and
        (.roots | type) == "array" and
        [.roots[].root] == $expected and
        all(.roots[];
            .status == "ok" and
            (.files | type) == "number" and .files > 0 and
            (.bytes | type) == "number" and .bytes >= 0 and
            .errors == 0
        ) and
        .total == {
            roots: ($expected | length),
            files: ([.roots[].files] | add),
            bytes: ([.roots[].bytes] | add),
            errors: 0
        }
    ' "$output" >/dev/null
}

evict_cache_roots() {
    local output=$1
    local error_output=$2
    shift 2
    local status=0
    k exec -i "$CACHE_HELPER" -c helper -- python3 - "$@" \
        < "$EXP/gms-fadvise-exact.py" \
        > "$output" 2> "$error_output" || status=$?
    if ! validate_eviction_results "$output" "$@"; then
        echo "cache eviction returned invalid per-root results: $output" >&2
        return 1
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "cache eviction exited with status $status: $output" >&2
        return 1
    fi
}

validate_cache_helper_access() {
    local output=$1
    local error_output=$2
    shift 2
    local status=0
    k exec "$CACHE_HELPER" -c helper -- sh -eu -c '
        uid=$(id -u)
        gid=$(id -g)
        printf "uid=%s gid=%s\n" "$uid" "$gid"
        [ "$uid" -eq 0 ]
        [ "$gid" -eq 0 ]
        for root do
            find "$root" -print >/dev/null
            printf "traversable\t%s\n" "$root"
        done
    ' sh "$@" > "$output" 2> "$error_output" || status=$?
    if ! diff -u \
        <(
            printf 'uid=0 gid=0\n'
            printf 'traversable\t%s\n' "$@"
        ) \
        "$output"; then
        echo "cache helper identity or exact-root traversal is invalid: $output" >&2
        return 1
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "cache helper access preflight exited with status $status: $output" >&2
        return 1
    fi
}

validate_live_images() {
    local live_images=$1
    local container count expected_digest image_id
    for container in main gms-loader gms-server; do
        count=$(
            awk -F '\t' -v name="$container" \
                '$1 == name {count++} END {print count + 0}' "$live_images"
        )
        [[ "$count" -eq 1 ]]
        image_id=$(
            awk -F '\t' -v name="$container" '$1 == name {print $3}' \
                "$live_images"
        )
        case "$container" in
            main)
                expected_digest=$MAIN_DIGEST
                ;;
            gms-loader)
                expected_digest=$EXPECTED_LOADER_DIGEST
                ;;
            gms-server)
                expected_digest=$EXPECTED_SERVER_DIGEST
                ;;
        esac
        [[ "$image_id" == *"@$expected_digest" ]]
    done
    stamp LIVE_IMAGES_VALIDATED \
        "main=$MAIN_DIGEST loader=$EXPECTED_LOADER_DIGEST server=$EXPECTED_SERVER_DIGEST"
}

stamp PREFLIGHT_BEGIN "variant=${VARIANT^^}"
for command in kubectl jq yq curl pgrep python3 sha256sum cmp; do
    command -v "$command" >/dev/null
done
choose_ephemeral_port >/dev/null
stamp LOCAL_EPHEMERAL_PORT_SELECTION_VALIDATED
validate_checkpoint "$ART/preflight/checkpoint.json"
k get pods --field-selector "spec.nodeName=$NODE" -o wide \
    > "$ART/preflight/node-pods.txt"
k get resourceclaimtemplate "${DGD}-vllmdecodeworker-gpu" -o json \
    > "$ART/preflight/resourceclaimtemplate.json"
jq -e '
    .spec.spec.devices.requests[] |
    select(.name == "gpus") |
    .exactly.allocationMode == "ExactCount" and
    .exactly.count == 8 and
    .exactly.deviceClassName == "gpu.nvidia.com"
' "$ART/preflight/resourceclaimtemplate.json" >/dev/null
kubectl --context "$CTX" -n "$NS" get resourceslices -o json |
    jq --arg node "$NODE" '
        [
            .items[] |
            select(.spec.nodeName == $node and .spec.driver == "gpu.nvidia.com") |
            .spec.devices[].attributes.uuid.string
        ] | sort[]
    ' -r > "$ART/preflight/resourceslice-uuids.txt"
diff -u \
    <(sort "$ART/expected-uuids.txt") \
    "$ART/preflight/resourceslice-uuids.txt"
stamp NODE_DRA_UUIDS_VALIDATED "count=8"
wait_for_zero

k kustomize "$OVERLAY" > "$ART/preflight/rendered.yaml"
RENDERED_MAIN_IMAGE=$(
    yq -r '.spec.components[1].podTemplate.spec.containers[]
        | select(.name == "main") | .image' "$ART/preflight/rendered.yaml"
)
EXPECTED_LOADER_IMAGE=$(
    yq -r '.spec.components[1].podTemplate.spec.containers[]
        | select(.name == "gms-loader") | .image' "$ART/preflight/rendered.yaml"
)
EXPECTED_SERVER_IMAGE=$(
    yq -r '.spec.components[1].podTemplate.spec.initContainers[]
        | select(.name == "gms-server") | .image' "$ART/preflight/rendered.yaml"
)
EXPECTED_LOADER_DIGEST=${EXPECTED_LOADER_IMAGE##*@}
EXPECTED_SERVER_DIGEST=${EXPECTED_SERVER_IMAGE##*@}
case "$VARIANT" in
    a)
        [[ "$EXPECTED_LOADER_DIGEST" == \
            sha256:d0ea4cc1aceeeeef5825c418999ceca00fcde20dbdbc203d4b2bc683a874708a ]]
        [[ "$EXPECTED_SERVER_DIGEST" == "$EXPECTED_LOADER_DIGEST" ]]
        ;;
    b)
        [[ "$EXPECTED_LOADER_DIGEST" == "$EXPERIMENT_LOADER_DIGEST" ]]
        [[ "$EXPECTED_SERVER_DIGEST" == "$B_DIGEST" ]]
        ;;
    c)
        [[ "$EXPECTED_LOADER_DIGEST" == "$C_LOADER_DIGEST" ]]
        [[ "$EXPECTED_SERVER_DIGEST" == "$B_DIGEST" ]]
        ;;
    m | p | mp)
        [[ "$EXPECTED_LOADER_DIGEST" == "$EXPERIMENT_LOADER_DIGEST" ]]
        [[ "$EXPECTED_SERVER_DIGEST" == "$B_DIGEST" ]]
        ;;
esac
[[ "$RENDERED_MAIN_IMAGE" == "$MAIN_IMAGE" ]]
[[ $(
    yq '[.spec.components[1].podTemplate.spec.initContainers[]
        | select(.name == "gms-server")] | length' "$ART/preflight/rendered.yaml"
) == 1 ]]
[[ $(
    yq '[.spec.components[1].podTemplate.spec.containers[]
        | select(.name == "gms-server")] | length' "$ART/preflight/rendered.yaml"
) == 0 ]]
yq -o=json '.spec.components[1].podTemplate.spec' \
    "$ART/preflight/rendered.yaml" |
    jq -e --arg server_image "$EXPECTED_SERVER_IMAGE" '
        [.initContainers[] | select(.name == "gms-server")] == [{
            name: "gms-server",
            image: $server_image,
            command: ["python3", "-m", "gpu_memory_service.cli.server"],
            env: [{
                name: "GMS_SOCKET_DIR",
                value: "/gms-intrapod-control"
            }],
            volumeMounts: [{
                name: "gms-intrapod-control",
                mountPath: "/gms-intrapod-control"
            }],
            resources: {
                claims: [{
                    name: "intrapod-shared-gpu"
                }]
            },
            restartPolicy: "Always"
        }] and
        [.volumes[] | select(.name == "gms-intrapod-control")] == [{
            name: "gms-intrapod-control",
            emptyDir: {}
        }]
    ' >/dev/null
[[ $(
    yq -r '.spec.components[1].experimental.checkpoint.checkpointRef' \
        "$ART/preflight/rendered.yaml"
) == "$CKPT" ]]
[[ $(
    yq -r '.spec.components[1].experimental.checkpoint.startupPolicy' \
        "$ART/preflight/rendered.yaml"
) == WaitForCheckpoint ]]
[[ $(
    yq -r '.spec.components[1].replicas' "$ART/preflight/rendered.yaml"
) == 0 ]]
if yq -e '.spec.components[1].experimental.checkpoint.job' \
    "$ART/preflight/rendered.yaml" >/dev/null 2>&1; then
    echo "rendered overlay unexpectedly contains a checkpoint job" >&2
    exit 1
fi
if [[ "$VARIANT" == c ]]; then
    [[ $(
        yq -o=json '.spec.components[1].podTemplate.spec.containers[]
            | select(.name == "gms-loader")
            | [.env[] | select(.name == "DYN_GMS_SHARDED_SSD_CUDA_MODE")]
            | length' "$ART/preflight/rendered.yaml"
    ) == 1 ]]
    [[ $(
        yq -r '.spec.components[1].podTemplate.spec.containers[]
            | select(.name == "gms-loader")
            | .env[] | select(.name == "DYN_GMS_SHARDED_SSD_CUDA_MODE")
            | .value' "$ART/preflight/rendered.yaml"
    ) == driver ]]
fi
if [[ "$VARIANT" == m || "$VARIANT" == mp ]]; then
    [[ $(
        yq -r '.spec.components[1].podTemplate.spec.containers[]
            | select(.name == "gms-loader")
            | .env[] | select(.name == "DYN_GMS_MAPPING_FIRST")
            | .value' "$ART/preflight/rendered.yaml"
    ) == 1 ]]
fi
if [[ "$VARIANT" == p || "$VARIANT" == mp ]]; then
    [[ $(
        yq -r '.spec.components[1].podTemplate.spec.containers[]
            | select(.name == "gms-loader")
            | .env[] | select(.name == "DYN_GMS_PINNED_REGISTRATION_GROUPS")
            | .value' "$ART/preflight/rendered.yaml"
    ) == 1 ]]
fi

stamp APPLY_ZERO_REPLICA_VARIANT
k apply -k "$OVERLAY" > "$ART/preflight/apply.txt"
CLEANUP_OWNED=1
wait_for_zero
ZERO_CONFIRMED=1
capture_objects pre

k apply -f "$EXP/cache-helper.yaml" > "$ART/preflight/cache-helper-apply.txt"
k wait --for=condition=Ready "pod/$CACHE_HELPER" --timeout=300s
k get pod "$CACHE_HELPER" -o yaml > "$ART/preflight/cache-helper.yaml"
validate_cache_helper_access \
    "$ART/preflight/cache-helper-access.txt" \
    "$ART/preflight/cache-helper-access.err" \
    "${NFS_CACHE_ROOTS[@]}"
stamp CACHE_HELPER_ACCESS_VALIDATED "uid=0 gid=0 roots=2"

stamp FADVISE_NFS_BEGIN
evict_cache_roots \
    "$ART/cache/nfs.txt" "$ART/cache/nfs.err" "${NFS_CACHE_ROOTS[@]}"
stamp FADVISE_NFS_DONE "$(tail -1 "$ART/cache/nfs.txt")"

stamp FADVISE_NVME_BEGIN
evict_cache_roots \
    "$ART/cache/nvme.txt" "$ART/cache/nvme.err" "${NVME_CACHE_ROOTS[@]}"
stamp FADVISE_NVME_DONE "$(tail -1 "$ART/cache/nvme.txt")"

START=$(now)
printf '%s\n' "$START" > "$ART/start.txt"
AGENT=$(
    k get pods -l app.kubernetes.io/component=snapshot-agent \
        --field-selector "spec.nodeName=$NODE" \
        -o jsonpath='{.items[0].metadata.name}'
)
OP_POD=$(
    k get pods -l "$OP_SELECTOR" -o jsonpath='{.items[0].metadata.name}'
)

stamp SAMPLERS_BEGIN
k exec -i "$AGENT" -c agent -- sh -s -- 6000 \
    < "$EXP/gms-host-sampler.sh" \
    > "$ART/metrics/host-proc-200ms.raw" \
    2> "$ART/metrics/host-proc-200ms.err" &
PIDS+=("$!")
k logs "$AGENT" -c agent --timestamps --since-time="$START" -f \
    > "$ART/logs/snapshot-agent.follow.log" 2>&1 &
PIDS+=("$!")
k logs "$OP_POD" -c manager --timestamps --since-time="$START" -f \
    > "$ART/logs/operator.follow.log" 2>&1 &
PIDS+=("$!")

ZERO_CONFIRMED=0
stamp SCALE_UP_SENT
k patch dynamographdeployment "$DGD" --type=json \
    -p='[
        {"op":"test","path":"/spec/components/1/name","value":"VllmDecodeWorker"},
        {"op":"replace","path":"/spec/components/1/replicas","value":1}
    ]' \
    > "$ART/scale-up.txt"

for attempt in $(seq 1 300); do
    WORKER_POD=$(
        k get pods -l "$POD_SELECTOR" \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
    )
    [[ -n "$WORKER_POD" ]] && break
    sleep 1
done
[[ -n "$WORKER_POD" ]]
stamp POD_SEEN "$WORKER_POD"

for attempt in $(seq 1 120); do
    CLAIM=$(
        k get pod "$WORKER_POD" \
            -o jsonpath='{.status.resourceClaimStatuses[0].resourceClaimName}' \
            2>/dev/null || true
    )
    [[ -n "$CLAIM" ]] && break
    sleep 1
done
[[ "$CLAIM" == "$CLAIM_PREFIX"* ]]
k get resourceclaim "$CLAIM" -o yaml > "$ART/objects/allocated-resourceclaim.yaml"
stamp DRA_CLAIM_SEEN "$CLAIM"

for container in main gms-loader gms-server; do
    k logs "$WORKER_POD" -c "$container" --timestamps --since-time="$START" -f \
        > "$ART/logs/$container.follow.log" 2>&1 &
    PIDS+=("$!")
done

(
    while true; do
        printf '%s\t' "$(now)"
        k get pod "$WORKER_POD" \
            -o jsonpath='phase={.status.phase} scheduled={.status.conditions[?(@.type=="PodScheduled")].status} initialized={.status.conditions[?(@.type=="Initialized")].status} ready={.status.conditions[?(@.type=="Ready")].status}{"\n"}' \
            2>/dev/null || true
        sleep 1
    done
) > "$ART/metrics/lifecycle.tsv" 2>&1 &
PIDS+=("$!")

(
    for attempt in $(seq 1 120); do
        if k exec "$WORKER_POD" -c gms-server -- nvidia-smi \
            --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw \
            --format=csv,noheader,nounits -lms 200; then
            exit 0
        fi
        sleep 1
    done
) > "$ART/metrics/nvidia-smi-200ms.csv" \
    2> "$ART/metrics/nvidia-smi-200ms.err" &
PIDS+=("$!")

k wait --for=condition=Ready "pod/$WORKER_POD" --timeout=900s
stamp WORKER_READY
k get pod "$WORKER_POD" -o json > "$ART/objects/ready-worker-pod.json"

jq -e \
    --arg main_image "$MAIN_IMAGE" \
    --arg loader_image "$EXPECTED_LOADER_IMAGE" \
    --arg server_image "$EXPECTED_SERVER_IMAGE" '
    [.spec.containers[] | select(.name == "main") | .image] ==
        [$main_image] and
    [.spec.containers[] | select(.name == "gms-loader") | .image] ==
        [$loader_image] and
    [.spec.containers[] | select(.name == "gms-server")] == [] and
    ([.spec.initContainers[] | select(.name == "gms-server")] | length) == 1 and
    ([.spec.initContainers[] | select(.name == "gms-server")][0] as $server |
        $server.image == $server_image and
        $server.command ==
            ["python3", "-m", "gpu_memory_service.cli.server"] and
        ([$server.env[] | select(
            .name == "GMS_SOCKET_DIR" and
            .value == "/gms-intrapod-control"
        )] | length) == 1 and
        ([$server.resources.claims[] | select(
            .name == "intrapod-shared-gpu"
        )] | length) == 1 and
        ([$server.volumeMounts[] | select(
            .name == "gms-intrapod-control" and
            .mountPath == "/gms-intrapod-control"
        )] | length) == 1 and
        $server.restartPolicy == "Always"
    )
' "$ART/objects/ready-worker-pod.json" >/dev/null

jq -r '
    (
        [.status.containerStatuses[]
            | select(.name == "main" or .name == "gms-loader")]
        +
        [.status.initContainerStatuses[]
            | select(.name == "gms-server")]
    )[] | [.name, .image, .imageID] | @tsv
' "$ART/objects/ready-worker-pod.json" | sort \
    > "$ART/preflight/live-images.tsv"
validate_live_images "$ART/preflight/live-images.tsv"

k exec "$WORKER_POD" -c gms-loader -- \
    nvidia-smi --query-gpu=uuid --format=csv,noheader \
    | tr -d ' ' > "$ART/preflight/loader-uuids.txt"
[[ $(wc -l < "$ART/preflight/loader-uuids.txt") -eq 8 ]]
diff -u \
    <(sort "$ART/expected-uuids.txt") \
    <(sort "$ART/preflight/loader-uuids.txt")
stamp LOADER_VISIBILITY_VALIDATED "count=8"

PORT=$(choose_ephemeral_port)
k port-forward --address 127.0.0.1 "service/$FRONT" "$PORT:8000" \
    > "$ART/inference/port-forward.log" 2>&1 &
PIDS+=("$!")
for attempt in $(seq 1 120); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" \
        > "$ART/inference/health.json" 2>/dev/null; then
        break
    fi
    sleep 1
done
curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" \
    > "$ART/inference/models.json"
jq -e --arg model "$MODEL" '.data[] | select(.id == $model)' \
    "$ART/inference/models.json" >/dev/null

jq -n --arg model "$MODEL" '{
    model: $model,
    messages: [{
        role: "user",
        content: "In one concise sentence, explain why the sky appears blue."
    }],
    temperature: 0,
    max_tokens: 512
}' > "$ART/inference/request.json"
stamp INFERENCE_SENT
curl --silent --show-error --max-time 180 \
    -D "$ART/inference/headers.txt" \
    -o "$ART/inference/response.json" \
    -w 'http_code=%{http_code}\ntime_total_s=%{time_total}\n' \
    -H 'Content-Type: application/json' \
    --data-binary @"$ART/inference/request.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    > "$ART/inference/curl-metrics.txt"
stamp INFERENCE_DONE
[[ $(
    awk -F= '$1 == "http_code" {print $2}' "$ART/inference/curl-metrics.txt"
) == 200 ]]
[[ $(jq -r '.choices[0].finish_reason' "$ART/inference/response.json") == stop ]]
content=$(jq -r '.choices[0].message.content // empty' "$ART/inference/response.json")
[[ -n "$content" ]]
printf '%s\n' "$content" > "$ART/inference/content.txt"
grep -Eqi 'blue' "$ART/inference/content.txt"
grep -Eqi 'atmosphere' "$ART/inference/content.txt"
grep -Eqi 'scatter' "$ART/inference/content.txt"
stamp COHERENT_INFERENCE_VALIDATED "chars=${#content}"

for container in main gms-loader gms-server; do
    k logs "$WORKER_POD" -c "$container" --timestamps --since-time="$START" \
        > "$ART/logs/$container.final.log"
done
k logs "$AGENT" -c agent --timestamps --since-time="$START" \
    > "$ART/logs/snapshot-agent.final.log"
k logs "$OP_POD" -c manager --timestamps --since-time="$START" \
    > "$ART/logs/operator.final.log"

python3 "$EXP/validate-server-profile.py" \
    "$ART/logs/gms-server.final.log" \
    "$ART/expected-uuids.txt" \
    "$VARIANT" | tee "$ART/preflight/server-profile-validation.txt"
python3 "$EXP/validate-loader-profile.py" \
    "$ART/logs/gms-loader.final.log" \
    "$VARIANT" | tee "$ART/preflight/loader-profile-validation.txt"
python3 - "$ART/logs/snapshot-agent.final.log" "$ART/expected-uuids.txt" \
    "$WORKER_POD" <<'PY'
import json
import sys
from pathlib import Path

log_path, expected_path, pod = sys.argv[1:]
expected = Path(expected_path).read_text(encoding="utf-8").splitlines()
observed = None
for line in Path(log_path).read_text(encoding="utf-8").splitlines():
    if "resolved DRA GPU UUIDs in container ordinal order" not in line:
        continue
    record = json.loads(line[line.index("{") :])
    if record.get("pod", "").endswith(f"/{pod}"):
        observed = record["uuids"]
if observed != expected:
    raise SystemExit(f"DRA UUID order mismatch: observed={observed} expected={expected}")
print(f"validated DRA ordinal UUIDs: {len(observed)}")
PY
stamp PHYSICAL_UUID_PROFILES_VALIDATED

capture_objects final
validate_checkpoint "$ART/objects/final-checkpoint.json"
stop_background
scale_down
validate_checkpoint "$ART/objects/post-teardown-checkpoint.json"
capture_objects post-teardown
stamp RUN_COMPLETE
