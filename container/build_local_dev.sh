#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Script to convert dev images to local_dev images
# Usage: ./convert-to-local-dev.sh [OPTIONS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_LOCAL_DEV="${SCRIPT_DIR}/Dockerfile.local_dev"


# Default values
USER_UID=$(id -u)
USER_GID=$(id -g)
RUN_PREFIX=
CUSTOM_TAGS=()

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Convert dev images to local_dev images for Dev Container use.

OPTIONS:
    -i, --dev-image IMAGE   Dev image to convert (required for conversion)
    -t, --tag TAG           Custom tag for the local-dev image (overrides default naming)
    -u, --uid UID           User UID (default: current user UID)
    -g, --gid GID           User GID (default: current user GID)
    -d, --dry-run           Show what would be done without building
    -h, --help              Show this help message

EXAMPLES:
    # Convert a specific dev image
    $0 --dev-image dynamo:latest-vllm

    # Convert with custom UID/GID
    $0 --dev-image dynamo:vllm-dev --uid 1001 --gid 1001

    # Dry run to see what would be done
    $0 --dev-image dynamo:latest-vllm --dry-run

    # Use custom tag
    $0 --dev-image dynamo:latest-vllm --tag my-custom:local-dev

EOF
}



validate_image() {
    local image="$1"

    if [[ "$image" == *"-local-dev" ]]; then
        echo "ERROR: Cannot use local-dev image as dev image input: '$image'"
        echo "Please use a dev image (without -local-dev suffix) instead"
        return 1
    fi

    if ! docker image inspect "$image" &>/dev/null; then
        echo "ERROR: Image '$image' not found locally"
        echo "Available images:"
        docker images --format "table {{.Repository}}:{{.Tag}}"
        return 1
    fi

    return 0
}

validate_uid_gid() {
    # Check for valid UID/GID values (should be positive integers)
    if ! [[ "$USER_UID" =~ ^[0-9]+$ ]] || [ "$USER_UID" -le 0 ]; then
        echo "ERROR: Invalid USER_UID: $USER_UID (should be a positive integer)"
        return 1
    fi

    if ! [[ "$USER_GID" =~ ^[0-9]+$ ]] || [ "$USER_GID" -le 0 ]; then
        echo "ERROR: Invalid USER_GID: $USER_GID (should be a positive integer)"
        return 1
    fi

    return 0
}


# Parse command line arguments
PARSE_ARGS=true
BASE_IMAGE=""

while [[ $# -gt 0 ]] && [[ "$PARSE_ARGS" == "true" ]]; do
    case $1 in
        -i|--dev-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        -t|--tag)
            CUSTOM_TAGS+=("$2")
            shift 2
            ;;
        -u|--uid)
            USER_UID="$2"
            shift 2
            ;;
        -g|--gid)
            USER_GID="$2"
            shift 2
            ;;
        -d|--dry-run|--dryrun)
            RUN_PREFIX="echo"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            PARSE_ARGS=false
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main logic
if [[ "$LIST_IMAGES" == "true" ]]; then
    list_dev_images
    exit 0
fi

if [[ -z "$BASE_IMAGE" ]]; then
    echo "ERROR: No dev image specified"
    echo
    usage
    exit 1
fi

# Validate inputs
validate_image "$BASE_IMAGE" || exit 1
validate_uid_gid || exit 1

# Build tag arguments
if [[ ${#CUSTOM_TAGS[@]} -eq 0 ]]; then
    # No custom tags provided, use default naming
    if [[ "$BASE_IMAGE" == *:* ]]; then
        tag="${BASE_IMAGE#*:}"
        CUSTOM_TAGS=("dynamo:${tag}-local-dev")
    else
        CUSTOM_TAGS=("dynamo:${BASE_IMAGE}-local-dev")
    fi
fi

TAG_ARGS=""
for tag in "${CUSTOM_TAGS[@]}"; do
    TAG_ARGS+=" --tag $tag"
done

echo "Using UID: $USER_UID, GID: $USER_GID"

if [[ ! -f "$DOCKERFILE_LOCAL_DEV" ]]; then
    echo "ERROR: Dockerfile.local_dev not found at: $DOCKERFILE_LOCAL_DEV"
    exit 1
fi

# Show the docker command being executed if not in dry-run mode
if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

# Docker build logic moved to build.sh
echo "Docker build arguments prepared:"
echo "  LOCAL_DEV_BASE: $BASE_IMAGE"
echo "  USER_UID: $USER_UID"
echo "  USER_GID: $USER_GID"
echo "  TAG_ARGS: $TAG_ARGS"
{ set +x; } 2>/dev/null

# Show usage example with run.sh
echo "# To run.sh with --mount-workspace (remember, local-dev images will give you proper local user permissions):"
echo "# ./run.sh --image ${CUSTOM_TAGS[0]} --mount-workspace <... additional options ...>"

