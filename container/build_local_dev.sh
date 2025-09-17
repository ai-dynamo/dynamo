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
DRY_RUN=false
CUSTOM_TAG=""

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

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_warning() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1"
}

list_dev_images() {
    log_info "Searching for dynamo dev/latest images..."

    # Look for dynamo images with dev patterns or latest patterns
    local dev_images
    dev_images=$(docker images --format "{{.Repository}}:{{.Tag}}|{{.ID}}|{{.CreatedSince}}|{{.Size}}" | \
        grep "dynamo:" | \
        grep -E "(dev|development|latest)" | \
        grep -v "local-dev" | \
        head -20)

    if [[ -z "$dev_images" ]]; then
        log_warning "No dynamo dev/latest images found. Looking for any dynamo images..."
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
        log_info "Use --dev-image <IMAGE:TAG> to convert one of these images"
    else
        log_warning "No dynamo images found"
        log_info "Make sure you have built dynamo images first"
    fi
}

validate_image() {
    local image="$1"

    if ! docker image inspect "$image" &>/dev/null; then
        log_error "Image '$image' not found locally"
        log_info "Available images:"
        docker images --format "table {{.Repository}}:{{.Tag}}"
        return 1
    fi

    return 0
}

validate_uid_gid() {
    # Check for valid UID/GID values (should be positive integers)
    if ! [[ "$USER_UID" =~ ^[0-9]+$ ]] || [ "$USER_UID" -le 0 ]; then
        log_error "Invalid USER_UID: $USER_UID (should be a positive integer)"
        return 1
    fi

    if ! [[ "$USER_GID" =~ ^[0-9]+$ ]] || [ "$USER_GID" -le 0 ]; then
        log_error "Invalid USER_GID: $USER_GID (should be a positive integer)"
        return 1
    fi

    return 0
}

convert_image() {
    local base_image="$1"

    # Use custom tag if provided, otherwise use default naming
    if [[ -n "$CUSTOM_TAG" ]]; then
        local output_image="$CUSTOM_TAG"
    else
        # Extract tag from base image and create dynamo:tag-local-dev
        if [[ "$base_image" == *:* ]]; then
            local tag="${base_image#*:}"
            local output_image="dynamo:${tag}-local-dev"
        else
            local output_image="dynamo:${base_image}-local-dev"
        fi
    fi

    log_info "Converting '$base_image' to '$output_image'"
    log_info "Using UID: $USER_UID, GID: $USER_GID, ARCH: $ARCH"

    if [[ ! -f "$DOCKERFILE_LOCAL_DEV" ]]; then
        log_error "Dockerfile.local_dev not found at: $DOCKERFILE_LOCAL_DEV"
        return 1
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo
        log_info "DRY RUN - Would execute the following command:"
        echo "docker build \\"
        echo "    --build-arg LOCAL_DEV_BASE=\"$base_image\" \\"
        echo "    --build-arg USER_UID=\"$USER_UID\" \\"
        echo "    --build-arg USER_GID=\"$USER_GID\" \\"
        echo "    --build-arg ARCH=\"$ARCH\" \\"
        echo "    --file \"$DOCKERFILE_LOCAL_DEV\" \\"
        echo "    --tag \"$output_image\" \\"
        echo "    \"$SCRIPT_DIR\""
        echo
        log_info "No actual build performed (dry run mode)"
        return 0
    fi

    # Build the local_dev image
    log_info "Building local_dev image..."

    # Show the docker command being executed
    set -x
    docker build \
        --build-arg LOCAL_DEV_BASE="$base_image" \
        --build-arg USER_UID="$USER_UID" \
        --build-arg USER_GID="$USER_GID" \
        --build-arg ARCH="$ARCH" \
        --file "$DOCKERFILE_LOCAL_DEV" \
        --tag "$output_image" \
        "$SCRIPT_DIR" || {
        set +x > /dev/null 2>&1
        log_error "Failed to build local_dev image"
        return 1
    }
    set +x > /dev/null 2>&1

    log_info "Successfully added local-dev image: '$output_image'"

    # Show the new image info
    docker images "$output_image" --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}"

    echo
    log_info "Usage options:"
    log_info "  - Dev Container IDE Extension: Use directly with VS Code/Cursor Dev Container extension"
    log_info "  - Command line: run.sh --image $output_image --mount-workspace ..."
    log_info "    where the ubuntu user inside the container is mapped to $(whoami) (UID:$USER_UID, GID:$USER_GID)"
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
            CUSTOM_TAG="$2"
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
            DRY_RUN=true
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
    log_error "No dev image specified"
    echo
    usage
    exit 1
fi

# Validate and convert
validate_image "$BASE_IMAGE" || exit 1
validate_uid_gid || exit 1
convert_image "$BASE_IMAGE" || exit 1

