#!/bin/bash

# Script to convert dev images to local_dev images
# Usage: ./convert-to-local-dev.sh [OPTIONS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_LOCAL_DEV="${SCRIPT_DIR}/Dockerfile.local_dev"


# Default values
ARCH="amd64"
USER_UID=$(id -u)
USER_GID=$(id -g)
RUN_PREFIX=
SQUASH=false
CUSTOM_TAGS=()

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Convert dev images to local_dev images for Dev Container use.

OPTIONS:
    -l, --list              List available dev images
    -i, --dev-image IMAGE   Dev image to convert (required for conversion)
    -t, --tag TAG           Custom tag for the local-dev image (overrides default naming)
    -a, --arch ARCH         Architecture (default: amd64)
    -u, --uid UID           User UID (default: current user UID)
    -g, --gid GID           User GID (default: current user GID)
    -d, --dry-run           Show what would be done without building
    --squash                Squash the layers in the resulting image
    -h, --help              Show this help message

EXAMPLES:
    # List available dev images
    $0 --list

    # Convert a specific dev image
    $0 --dev-image dynamo:latest-vllm

    # Convert with custom UID/GID
    $0 --dev-image dynamo:vllm-dev --uid 1001 --gid 1001

    # Convert for arm64 architecture
    $0 --dev-image dynamo:vllm-dev --arch arm64

    # Dry run to see what would be done
    $0 --dev-image dynamo:latest-vllm --dry-run

    # Use custom tag
    $0 --dev-image dynamo:latest-vllm --tag my-custom:local-dev

EOF
}


list_dev_images() {
    echo "Searching for dynamo dev/latest images..."

    # Look for dynamo images with dev patterns or latest patterns
    local dev_images
    dev_images=$(docker images --format "{{.Repository}}:{{.Tag}}|{{.ID}}|{{.CreatedSince}}|{{.Size}}" | \
        grep "dynamo:" | \
        grep -E "(dev|development|latest)" | \
        grep -v "local-dev" | \
        head -20)

    if [[ -z "$dev_images" ]]; then
        echo "No dynamo dev/latest images found. Looking for any dynamo images..."
        dev_images=$(docker images --format "{{.Repository}}:{{.Tag}}|{{.ID}}|{{.CreatedSince}}|{{.Size}}" | \
            grep "dynamo:" | \
            grep -v "local-dev" | \
            head -20)
    fi

    if [[ -n "$dev_images" ]]; then
        echo
        echo "Available dynamo images that could be converted:"
        echo "IMAGE:TAG                                                    IMAGE ID     CREATED        SIZE"
        echo "====================================================================================================="

        # Format each line with proper column alignment
        while IFS='|' read -r image_tag image_id created_since size; do
            printf "%-60s %-12s %-14s %s\n" "$image_tag" "$image_id" "$created_since" "$size"
        done <<< "$dev_images"

        echo
        echo "Use --dev-image <IMAGE:TAG> to convert one of these images"
    else
        echo "No dynamo images found"
        echo "Make sure you have built dynamo images first"
    fi
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
LIST_IMAGES=false
BASE_IMAGE=""

while [[ $# -gt 0 ]] && [[ "$PARSE_ARGS" == "true" ]]; do
    case $1 in
        -l|--list)
            LIST_IMAGES=true
            shift
            ;;
        -i|--dev-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        -t|--tag)
            CUSTOM_TAGS+=("$2")
            shift 2
            ;;
        -a|--arch)
            ARCH="$2"
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
        --squash)
            SQUASH=true
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

# Add squash flag if requested
SQUASH_ARG=""
if [[ "$SQUASH" == "true" ]]; then
    SQUASH_ARG=" --squash"
fi

echo "Using UID: $USER_UID, GID: $USER_GID, ARCH: $ARCH"

if [[ ! -f "$DOCKERFILE_LOCAL_DEV" ]]; then
    echo "ERROR: Dockerfile.local_dev not found at: $DOCKERFILE_LOCAL_DEV"
    exit 1
fi

# Show the docker command being executed if not in dry-run mode
if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

$RUN_PREFIX docker build \
    --build-arg LOCAL_DEV_BASE="$BASE_IMAGE" \
    --build-arg USER_UID="$USER_UID" \
    --build-arg USER_GID="$USER_GID" \
    --build-arg ARCH="$ARCH" \
    --file "$DOCKERFILE_LOCAL_DEV" \
    $TAG_ARGS$SQUASH_ARG \
    "$SCRIPT_DIR" || {
    { set +x; } 2>/dev/null
    echo "ERROR: Failed to build local_dev image"
    exit 1
}
{ set +x; } 2>/dev/null

# Show usage example with run.sh
echo "# To run.sh with --mount-workspace (remember, local-dev images will give you proper local user permissions):"
echo "# ./run.sh --image ${CUSTOM_TAGS[0]} --mount-workspace <... additional options ...>"

