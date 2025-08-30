#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# copy_devcontainer.sh - Framework-Specific DevContainer Distribution Script
#
# PURPOSE: Distributes devcontainer.json to framework-specific directories
#
# WHAT IT DOES:
#   - Creates .devcontainer/{vllm,sglang,trtllm,none}/ directories
#   - Copies and customizes devcontainer.json for each framework
#   - Substitutes: vllm->$framework, VLLM->$framework_upper
#
# USAGE: ./copy_devcontainer.sh [--dry-run] [--force] [--silent]
#
# DIRECTORY STRUCTURE:
#
# BEFORE running the script:
# .devcontainer/
# ├── devcontainer.json
# └── copy_devcontainer.sh
#
# AFTER running the script:
# .devcontainer/
# ├── devcontainer.json
# ├── copy_devcontainer.sh
# ├── vllm/
# │   └── devcontainer.json
# ├── sglang/
# │   └── devcontainer.json
# ├── trtllm/
# │   └── devcontainer.json
# └── none/
#     └── devcontainer.json
#
# ==============================================================================

set -eu

# Define base directory and source file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="${SCRIPT_DIR}/devcontainer.json"
DEVCONTAINER_DIR="${SCRIPT_DIR}"

# Define frameworks (lowercase for directory names)
FRAMEWORKS=("vllm" "sglang" "trtllm" "none")

# Check for flags
DRYRUN=false
FORCE=false
SILENT=false
while [ $# -gt 0 ]; do
    case $1 in
        --dryrun|--dry-run)
            DRYRUN=true
            ;;
        --force)
            FORCE=true
            ;;
        --silent)
            SILENT=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run, --dryrun  Preview changes without making them"
            echo "  --force              Force sync even if files already exist"
            echo "  --silent             Suppress all output (for cron jobs)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "This script copies devcontainer.json from bin/ to framework-specific"
            echo "directories under .devcontainer/, customizing the Docker image for each."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    shift
done

# Function to handle dry run output
dry_run_echo() {
    if [ "$SILENT" = true ]; then
        return
    fi

    if [ "$DRYRUN" = true ]; then
        echo "[DRYRUN] $*"
    else
        echo "$*"
    fi
}

# Command wrapper that shows commands using set -x format and respects dry-run mode
cmd() {
    if [ "$DRYRUN" = true ]; then
        # Dry run mode: show command but don't execute
        if [ "$SILENT" != true ]; then
            echo "[DRYRUN] $*"
        fi
        # Return success in dryrun mode
        return 0
    else
        # Not dry run: execute the command
        if [ "$SILENT" != true ]; then
            # Show and execute
            ( set -x; "$@" )
        else
            # Execute silently
            "$@"
        fi
    fi
}

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    dry_run_echo "ERROR: Source file not found at $SOURCE_FILE"
    exit 1
fi

dry_run_echo "INFO: Distributing devcontainer.json to framework-specific directories..."
dry_run_echo "INFO: Detected frameworks: ${FRAMEWORKS[*]}"

# Process each framework
SYNC_COUNT=0
TEMP_OUTPUT_FILE=$(mktemp)

for framework in "${FRAMEWORKS[@]}"; do
    FRAMEWORK_DIR="${DEVCONTAINER_DIR}/${framework}"
    DEST_FILE="${FRAMEWORK_DIR}/devcontainer.json"

    # Check if destination already exists (unless force flag is set)
    if [ -f "$DEST_FILE" ] && [ "$FORCE" = false ] && [ "$DRYRUN" = false ]; then
        dry_run_echo "INFO: Skipping ${framework} - file already exists (use --force to overwrite)"
        continue
    fi

    # Create framework directory if it doesn't exist
    if [ ! -d "$FRAMEWORK_DIR" ]; then
        cmd mkdir -p "${FRAMEWORK_DIR}"
    fi

    # Apply customizations to JSON file for this framework
    # Substitute: name, container name, vllm->$framework, VLLM->$framework_upper
    framework_upper="${framework^^}"  # Convert to uppercase for display name
    repo_basename=$(basename "$(dirname "${SCRIPT_DIR}")")  # Get repo basename

    sed "s|\"name\": \"|\"name\": \"[${repo_basename}] |g" "${SOURCE_FILE}" | \
    sed "s|\"--name\", \"dynamo-|\"--name\", \"${repo_basename}-|g" | \
    sed "s|vllm|${framework}|g" | \
    sed "s|VLLM|${framework_upper}|g" > "${TEMP_OUTPUT_FILE}"

    # Copy the modified file to the destination
    if ! cmd cp "${TEMP_OUTPUT_FILE}" "${DEST_FILE}"; then
        dry_run_echo "ERROR: Failed to copy devcontainer.json to ${DEST_FILE}"
    fi

    SYNC_COUNT=$((SYNC_COUNT + 1))
done

# Clean up temporary file
rm -f "${TEMP_OUTPUT_FILE}" 2>/dev/null

dry_run_echo "INFO: Distribution complete. Processed $SYNC_COUNT framework configurations."

# Directory structure AFTER running the script:
# .devcontainer/
# ├── devcontainer.json
# ├── copy_devcontainer.sh
# ├── vllm/
# │   └── devcontainer.json
# ├── sglang/
# │   └── devcontainer.json
# ├── trtllm/
# │   └── devcontainer.json
# └── none/
#     └── devcontainer.json

dry_run_echo "Framework-specific devcontainer.json files created in:"
for framework in "${FRAMEWORKS[@]}"; do
    dry_run_echo "  - ${DEVCONTAINER_DIR}/${framework}/devcontainer.json"
done
