#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Rewrites Dynamo Operator custom resources through the v1beta1 endpoint so
# Kubernetes persists them using the v1beta1 storage version. CRDs that have
# not promoted v1beta1 to storage are reported and skipped.
#
# Run this script after v1beta1 becomes the storage version and the conversion
# webhook is healthy, but before v1alpha1 is marked served=false, removed from
# the CRD, or removed from the conversion webhook. The script must list objects
# through every obsolete stored version, so those versions must still be served.

set -o errexit
set -o nounset
set -o pipefail

readonly API_GROUP="nvidia.com"
readonly DEFAULT_TARGET_VERSION="v1beta1"
readonly STORAGE_VERSION_ANNOTATION="nvidia.com/storage-version"

# This allowlist prevents the script from touching CRDs owned by other NVIDIA
# operators. Eligibility is still determined from the live CRD before any
# object is written.
readonly DYNAMO_CRDS=(
    "dynamographdeploymentrequests.nvidia.com"
    "dynamographdeployments.nvidia.com"
    "dynamocomponentdeployments.nvidia.com"
    "dynamographdeploymentscalingadapters.nvidia.com"
    "dynamocheckpoints.nvidia.com"
    "dynamomodels.nvidia.com"
    "dynamoworkermetadatas.nvidia.com"
    "podsnapshots.nvidia.com"
    "podsnapshotcontents.nvidia.com"
)

KUBECTL="${KUBECTL:-kubectl}"
TARGET_VERSION="${TARGET_VERSION:-${DEFAULT_TARGET_VERSION}}"
DRY_RUN=""
NAMESPACE_REGEX=""
VERBOSE=false
SELECTED_CRDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Rewrite eligible Dynamo Operator resources using the v1beta1 storage version.
Alpha-only CRDs and CRDs that have not promoted v1beta1 to storage are skipped.

When to run:
  1. Upgrade to an Operator release that serves both v1alpha1 and v1beta1.
  2. Confirm v1beta1 is the CRD storage version and the conversion webhook works.
  3. Run this script to rewrite existing objects.
  4. Only after migration is independently verified may a later lifecycle step
     stop serving or remove v1alpha1 and its conversion support.

This script must run before v1alpha1 is marked served=false or removed. It does
not modify CRDs or status.storedVersions and does not retire the old API version.

Options:
  --dry-run[=client|server]   Preview object writes without persisting changes.
                              The default strategy is client.
  --namespace-regex=REGEX    Migrate only namespaces matching the Bash regex.
  --crd=NAME                 Process one allowlisted CRD. May be repeated.
  --target-version=VERSION   Storage version to migrate to (default: v1beta1).
  -v, --verbose              Print kubectl commands before running them.
  -h, --help                 Show this help text.

The active kubectl context determines the target cluster. Set KUBECTL to use a
specific kubectl-compatible binary.
EOF
}

log() {
    printf '%s\n' "$*"
}

log_verbose() {
    if [[ "${VERBOSE}" == "true" ]]; then
        printf '  +'
        printf ' %q' "$@"
        printf '\n'
    fi
}

run_kubectl() {
    log_verbose "${KUBECTL}" "$@"
    "${KUBECTL}" "$@"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN="client"
                ;;
            --dry-run=client|--dry-run=server)
                DRY_RUN="${1#*=}"
                ;;
            --dry-run=*)
                printf 'error: invalid dry-run strategy: %s\n' "${1#*=}" >&2
                return 2
                ;;
            --namespace-regex=*)
                NAMESPACE_REGEX="${1#*=}"
                ;;
            --crd=*)
                SELECTED_CRDS+=("${1#*=}")
                ;;
            --target-version=*)
                TARGET_VERSION="${1#*=}"
                ;;
            -v|--verbose)
                VERBOSE=true
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                printf 'error: unknown option: %s\n' "$1" >&2
                usage >&2
                return 2
                ;;
        esac
        shift
    done
}

is_allowlisted_crd() {
    local candidate="$1"
    local crd
    for crd in "${DYNAMO_CRDS[@]}"; do
        if [[ "${candidate}" == "${crd}" ]]; then
            return 0
        fi
    done
    return 1
}

validate_selected_crds() {
    local crd
    for crd in "${SELECTED_CRDS[@]}"; do
        if ! is_allowlisted_crd "${crd}"; then
            printf 'error: CRD is not in the Dynamo migration allowlist: %s\n' "${crd}" >&2
            return 2
        fi
    done
}

crd_jsonpath() {
    local crd="$1"
    local expression="$2"
    run_kubectl get crd "${crd}" -o "jsonpath=${expression}"
}

version_is_served() {
    local crd="$1"
    local version="$2"
    local served_version
    local served_versions
    if ! served_versions="$(crd_jsonpath "${crd}" '{range .spec.versions[?(@.served==true)]}{.name}{"\n"}{end}')"; then
        return 2
    fi
    while IFS= read -r served_version; do
        if [[ "${served_version}" == "${version}" ]]; then
            return 0
        fi
    done <<< "${served_versions}"
    return 1
}

namespace_matches() {
    local namespace="$1"
    if [[ -z "${NAMESPACE_REGEX}" ]]; then
        return 0
    fi
    [[ "${namespace}" =~ ${NAMESPACE_REGEX} ]]
}

annotate_object() {
    local resource="$1"
    local name="$2"
    local namespace="$3"
    local command=(annotate "${resource}" "${name}" "${STORAGE_VERSION_ANNOTATION}=${TARGET_VERSION}" --overwrite)

    if [[ -n "${namespace}" ]]; then
        command+=(-n "${namespace}")
    fi
    if [[ -n "${DRY_RUN}" ]]; then
        command+=("--dry-run=${DRY_RUN}")
    fi

    if ! run_kubectl "${command[@]}" >/dev/null; then
        printf 'error: failed to rewrite %s/%s%s\n' \
            "${resource}" "${name}" "${namespace:+ in namespace ${namespace}}" >&2
        return 1
    fi
}

migrate_from_version() {
    local plural="$1"
    local scope="$2"
    local source_version="$3"
    local source_resource="${plural}.${source_version}.${API_GROUP}"
    local target_resource="${plural}.${TARGET_VERSION}.${API_GROUP}"
    local objects
    local namespace
    local name
    local migrated=0

    if [[ "${scope}" == "Namespaced" ]]; then
        if ! objects="$(run_kubectl get "${source_resource}" --all-namespaces --chunk-size=500 \
            -o 'jsonpath={range .items[*]}{.metadata.namespace}{"\t"}{.metadata.name}{"\n"}{end}')"; then
            printf 'error: failed to list %s\n' "${source_resource}" >&2
            return 1
        fi
        while IFS=$'\t' read -r namespace name; do
            [[ -z "${namespace}" || -z "${name}" ]] && continue
            namespace_matches "${namespace}" || continue
            annotate_object "${target_resource}" "${name}" "${namespace}" || return 1
            migrated=$((migrated + 1))
        done <<< "${objects}"
    else
        if ! objects="$(run_kubectl get "${source_resource}" --chunk-size=500 \
            -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')"; then
            printf 'error: failed to list %s\n' "${source_resource}" >&2
            return 1
        fi
        while IFS= read -r name; do
            [[ -z "${name}" ]] && continue
            annotate_object "${target_resource}" "${name}" "" || return 1
            migrated=$((migrated + 1))
        done <<< "${objects}"
    fi

    log "  migrated ${migrated} object(s) through ${source_version}"
}

process_crd() {
    local crd="$1"
    local plural
    local scope
    local storage_version
    local stored_versions
    local source_version
    local served_status
    local old_versions=()

    log "Inspecting ${crd}"
    if ! run_kubectl get crd "${crd}" >/dev/null 2>&1; then
        log "  skipped: CRD is not installed"
        return 0
    fi

    if ! plural="$(crd_jsonpath "${crd}" '{.spec.names.plural}')"; then
        printf 'error: failed to read plural for %s\n' "${crd}" >&2
        return 1
    fi
    if ! scope="$(crd_jsonpath "${crd}" '{.spec.scope}')"; then
        printf 'error: failed to read scope for %s\n' "${crd}" >&2
        return 1
    fi
    if [[ "${scope}" != "Namespaced" && "${scope}" != "Cluster" ]]; then
        printf 'error: %s has unsupported scope: %s\n' "${crd}" "${scope}" >&2
        return 1
    fi
    if ! storage_version="$(crd_jsonpath "${crd}" '{range .spec.versions[?(@.storage==true)]}{.name}{end}')"; then
        printf 'error: failed to read storage version for %s\n' "${crd}" >&2
        return 1
    fi

    served_status=0
    version_is_served "${crd}" "${TARGET_VERSION}" || served_status=$?
    if [[ ${served_status} -eq 2 ]]; then
        printf 'error: failed to read served versions for %s\n' "${crd}" >&2
        return 1
    elif [[ ${served_status} -ne 0 ]]; then
        log "  skipped: ${TARGET_VERSION} is not served (alpha-only or not yet promoted)"
        return 0
    fi
    if [[ "${storage_version}" != "${TARGET_VERSION}" ]]; then
        log "  skipped: configured storage version is ${storage_version}, not ${TARGET_VERSION}"
        return 0
    fi

    if ! stored_versions="$(crd_jsonpath "${crd}" '{range .status.storedVersions[*]}{.}{"\n"}{end}')"; then
        printf 'error: failed to read status.storedVersions for %s\n' "${crd}" >&2
        return 1
    fi
    while IFS= read -r source_version; do
        [[ -z "${source_version}" || "${source_version}" == "${TARGET_VERSION}" ]] && continue
        old_versions+=("${source_version}")
    done <<< "${stored_versions}"

    if [[ ${#old_versions[@]} -eq 0 ]]; then
        log "  complete: no obsolete stored versions"
        return 0
    fi

    for source_version in "${old_versions[@]}"; do
        served_status=0
        version_is_served "${crd}" "${source_version}" || served_status=$?
        if [[ ${served_status} -eq 2 ]]; then
            printf 'error: failed to read served versions for %s\n' "${crd}" >&2
            return 1
        elif [[ ${served_status} -ne 0 ]]; then
            printf 'error: %s still stores %s, but that version is not served\n' \
                "${crd}" "${source_version}" >&2
            return 1
        fi
    done

    for source_version in "${old_versions[@]}"; do
        migrate_from_version "${plural}" "${scope}" "${source_version}" || return 1
    done

    if [[ -n "${DRY_RUN}" ]]; then
        log "  dry-run: status.storedVersions was not changed"
    else
        log "  objects rewritten; status.storedVersions was not changed"
    fi
}

main() {
    parse_args "$@"
    validate_selected_crds

    if ! command -v "${KUBECTL}" >/dev/null 2>&1; then
        printf 'error: kubectl executable not found: %s\n' "${KUBECTL}" >&2
        return 1
    fi

    local crds=("${DYNAMO_CRDS[@]}")
    if [[ ${#SELECTED_CRDS[@]} -gt 0 ]]; then
        crds=("${SELECTED_CRDS[@]}")
    fi

    if [[ -n "${DRY_RUN}" ]]; then
        log "Running with --dry-run=${DRY_RUN}; no cluster state will be changed."
    fi
    log "Target storage version: ${TARGET_VERSION}"

    local crd
    local failures=0
    for crd in "${crds[@]}"; do
        if ! process_crd "${crd}"; then
            failures=$((failures + 1))
        fi
    done

    if [[ ${failures} -ne 0 ]]; then
        printf 'Migration failed for %d CRD(s).\n' "${failures}" >&2
        return 1
    fi
    log "Storage-version migration completed successfully."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
