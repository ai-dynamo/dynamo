#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared script to set up MinIO and upload LoRA adapters from Hugging Face Hub.
# Backend-agnostic: symlink from any backend's lora/ directory.
# SCRIPT_DIR resolves to the directory of the symlink, not this file's location,
# so "Next steps" messages correctly reference the backend's launch script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MINIO_DATA_DIR="${HOME}/dynamo_minio_data"
MINIO_ENDPOINT="http://localhost:9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
BUCKET_NAME="my-loras"

# Default LoRA (override via env vars)
HF_LORA_REPO="${HF_LORA_REPO:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"
LORA_NAME="${LORA_NAME:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"
TEMP_DIR=""

# HF CLI: "hf" (v0.34.0+) or "huggingface-cli" (legacy)
HF_CLI_CMD=""

# Parse args
MODE="full"
case "${1:-}" in
    --start)  MODE="start" ;;
    --stop)   MODE="stop" ;;
    -h|--help) MODE="help" ;;
    "")       MODE="full" ;;
    *)        echo -e "${RED}Error: Unknown option '$1'${NC}"; MODE="help" ;;
esac

info()    { echo -e "${YELLOW}-> $1${NC}"; }
success() { echo -e "${GREEN}ok $1${NC}"; }

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Setup MinIO and upload LoRA adapters from Hugging Face Hub.

Options:
  (none)      Full setup: start MinIO, download and upload LoRA
  --start     Start MinIO container only
  --stop      Stop and remove MinIO container
  -h, --help  Show this help

Environment Variables:
  HF_LORA_REPO  HF repository (default: $HF_LORA_REPO)
  LORA_NAME     Name for the LoRA (default: $LORA_NAME)

Examples:
  $0                                # Full setup
  $0 --start                        # Start MinIO only
  $0 --stop                         # Stop MinIO
  HF_LORA_REPO=user/repo $0        # Custom LoRA
EOF
}

check_dependencies() {
    info "Checking dependencies..."
    command -v docker &>/dev/null || { echo "Error: docker not installed"; exit 1; }
    command -v aws &>/dev/null    || { echo "Error: aws-cli not installed (pip install awscli)"; exit 1; }

    if command -v hf &>/dev/null; then
        HF_CLI_CMD="hf"
    elif command -v huggingface-cli &>/dev/null; then
        HF_CLI_CMD="huggingface-cli"
    else
        echo "Error: Neither 'hf' nor 'huggingface-cli' installed (pip install huggingface-hub[cli])"
        exit 1
    fi
    success "Dependencies OK (HF CLI: ${HF_CLI_CMD})"
}

start_minio() {
    info "Setting up MinIO..."
    mkdir -p "${MINIO_DATA_DIR}"
    docker stop dynamo-minio 2>/dev/null || true
    docker rm dynamo-minio 2>/dev/null || true

    docker run -d --name dynamo-minio \
        -p 9000:9000 -p 9001:9001 \
        -v "${MINIO_DATA_DIR}:/data" \
        quay.io/minio/minio server /data --console-address ":9001"

    info "Waiting for MinIO..."
    for i in {1..30}; do
        curl -s ${MINIO_ENDPOINT}/minio/health/live >/dev/null 2>&1 && break
        [ $i -eq 30 ] && { echo "Error: MinIO did not start in time"; exit 1; }
        sleep 1
    done
    success "MinIO ready (API: ${MINIO_ENDPOINT}, Console: http://localhost:9001)"
}

configure_aws_cli() {
    export AWS_ACCESS_KEY_ID="${MINIO_ACCESS_KEY}"
    export AWS_SECRET_ACCESS_KEY="${MINIO_SECRET_KEY}"
    export AWS_ENDPOINT_URL="${MINIO_ENDPOINT}"

    if ! aws --endpoint-url=${MINIO_ENDPOINT} s3 ls s3://${BUCKET_NAME} 2>/dev/null; then
        aws --endpoint-url=${MINIO_ENDPOINT} s3 mb s3://${BUCKET_NAME}
        success "Bucket created: ${BUCKET_NAME}"
    else
        success "Bucket exists: ${BUCKET_NAME}"
    fi
}

download_lora_from_hf() {
    info "Downloading LoRA: ${HF_LORA_REPO}..."
    TEMP_DIR=$(mktemp -d -t lora_download_XXXXXX)

    if [ "${HF_CLI_CMD}" = "huggingface-cli" ]; then
        huggingface-cli download "${HF_LORA_REPO}" \
            --local-dir "${TEMP_DIR}" --local-dir-use-symlinks False
    else
        hf download "${HF_LORA_REPO}" --local-dir "${TEMP_DIR}"
    fi

    rm -rf "${TEMP_DIR}/.cache"
    success "Downloaded to ${TEMP_DIR}"
}

upload_lora_to_minio() {
    info "Uploading to s3://${BUCKET_NAME}/${LORA_NAME}..."
    aws --endpoint-url=${MINIO_ENDPOINT} s3 sync \
        "${TEMP_DIR}" "s3://${BUCKET_NAME}/${LORA_NAME}" --exclude "*.git*"
    success "Upload complete"
}

cleanup() {
    [ -n "${TEMP_DIR}" ] && [ -d "${TEMP_DIR}" ] && rm -rf "${TEMP_DIR}"
}

stop_minio() {
    info "Stopping MinIO..."
    docker stop dynamo-minio 2>/dev/null && success "Stopped" || info "Not running"
    docker rm dynamo-minio 2>/dev/null && success "Removed" || true
    echo "Data preserved in: ${MINIO_DATA_DIR}"
}

# --- Main ---
case "$MODE" in
    help)
        show_help; exit 0 ;;
    stop)
        stop_minio ;;
    start)
        start_minio ;;
    full)
        check_dependencies
        start_minio
        configure_aws_cli
        download_lora_from_hf
        upload_lora_to_minio
        cleanup
        echo ""
        echo "Setup complete. Next steps:"
        echo "  1. Launch:  ${SCRIPT_DIR}/agg_lora.sh"
        echo "  2. Load:    curl -X POST http://localhost:8081/v1/loras \\"
        echo "                -H 'Content-Type: application/json' \\"
        echo "                -d '{\"lora_name\": \"${LORA_NAME}\", \"source\": {\"uri\": \"s3://${BUCKET_NAME}/${LORA_NAME}\"}}'"
        echo "  3. Infer:   curl http://localhost:8000/v1/chat/completions \\"
        echo "                -H 'Content-Type: application/json' \\"
        echo "                -d '{\"model\": \"${LORA_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
        echo "  4. Stop:    $0 --stop"
        ;;
esac
