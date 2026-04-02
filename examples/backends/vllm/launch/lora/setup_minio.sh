#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to setup MinIO and upload LoRA adapters from Hugging Face Hub

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
MINIO_DATA_DIR="${HOME}/dynamo_minio_data"
MINIO_ENDPOINT="http://localhost:9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
BUCKET_NAME="my-loras"

# Default LoRA to download (can be overridden)
HF_LORA_REPO="${HF_LORA_REPO:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"
LORA_NAME="${LORA_NAME:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"
# TEMP_DIR will be created using mktemp when needed
TEMP_DIR=""

# HF_CLI_CMD will be set to either "hf" or "huggingface-cli" based on huggingface-hub python package version
# Starting from HF v0.34.0, the `huggingface-cli` command is deprecated in favor of `hf`.
# Please refer to https://huggingface.co/blog/hf-cli for more details.
HF_CLI_CMD=""

# Parse command line arguments
MODE="full"
if [ "$1" = "--start" ]; then
    MODE="start"
elif [ "$1" = "--stop" ]; then
    MODE="stop"
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    MODE="help"
elif [ -n "$1" ]; then
    echo -e "${RED}Error: Unknown option '$1'${NC}"
    MODE="help"
fi

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Show help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup MinIO and upload LoRA adapters from Hugging Face Hub"
    echo ""
    echo "Options:"
    echo "  (no options)  Run full setup: start MinIO, download and upload LoRA"
    echo "  --start       Only start MinIO container"
    echo "  --stop        Stop and remove MinIO container"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HF_LORA_REPO  Hugging Face repository (default: ${HF_LORA_REPO:-codelion/Qwen3-0.6B-accuracy-recovery-lora})"
    echo "  LORA_NAME     Local name for the LoRA (default: ${LORA_NAME:-codelion/Qwen3-0.6B-accuracy-recovery-lora})"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Full setup"
    echo "  $0 --start                            # Start MinIO only"
    echo "  $0 --stop                             # Stop MinIO"
    echo "  HF_LORA_REPO=user/repo $0             # Use custom LoRA"
    echo ""
}

# Check if required tools are installed
check_dependencies() {
    print_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        echo "Error: docker is not installed"
        exit 1
    fi

    if ! command -v aws &> /dev/null; then
        echo "Error: aws-cli is not installed. Install with: pip install awscli"
        exit 1
    fi

    # Check for either hf or huggingface-cli
    if command -v hf &> /dev/null; then
        HF_CLI_CMD="hf"
        print_success "Found Hugging Face CLI: hf ($(hf version))"
    elif command -v huggingface-cli &> /dev/null; then
        HF_CLI_CMD="huggingface-cli"
        print_success "Found Hugging Face CLI: huggingface-cli ($(huggingface-cli version))"
    else
        echo "Error: Neither 'hf' nor 'huggingface-cli' is installed. Install with: pip install huggingface-hub[cli]"
        exit 1
    fi

    print_success "All dependencies are installed"
}

# Start MinIO using Docker
start_minio() {
    print_info "Setting up MinIO..."

    # Create data directory
    mkdir -p "${MINIO_DATA_DIR}"

    # Stop and remove existing container if it exists
    docker stop dynamo-minio 2>/dev/null || true
    docker rm dynamo-minio 2>/dev/null || true

    # Start MinIO
    print_info "Starting MinIO container..."
    docker run -d \
        --name dynamo-minio \
        -p 9000:9000 \
        -p 9001:9001 \
        -v "${MINIO_DATA_DIR}:/data" \
        quay.io/minio/minio server /data \
        --console-address ":9001"

    # Wait for MinIO to be ready
    print_info "Waiting for MinIO to be ready..."
    for i in {1..30}; do
        if curl -s ${MINIO_ENDPOINT}/minio/health/live > /dev/null 2>&1; then
            print_success "MinIO is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Error: MinIO did not start in time"
            exit 1
        fi
        sleep 1
    done

    print_success "MinIO started successfully"
    echo "  - MinIO API: ${MINIO_ENDPOINT}"
    echo "  - MinIO Console: http://localhost:9001"
    echo "  - Username: ${MINIO_ACCESS_KEY}"
    echo "  - Password: ${MINIO_SECRET_KEY}"
}

# Configure AWS CLI for MinIO
configure_aws_cli() {
    print_info "Configuring AWS CLI for MinIO..."

    export AWS_ACCESS_KEY_ID="${MINIO_ACCESS_KEY}"
    export AWS_SECRET_ACCESS_KEY="${MINIO_SECRET_KEY}"
    export AWS_ENDPOINT_URL="${MINIO_ENDPOINT}"

    # Create bucket if it doesn't exist
    if ! aws --endpoint-url=${MINIO_ENDPOINT} s3 ls s3://${BUCKET_NAME} 2>/dev/null; then
        print_info "Creating bucket: ${BUCKET_NAME}"
        aws --endpoint-url=${MINIO_ENDPOINT} s3 mb s3://${BUCKET_NAME}
        print_success "Bucket created"
    else
        print_success "Bucket already exists: ${BUCKET_NAME}"
    fi
}

# Download LoRA from Hugging Face Hub
download_lora_from_hf() {
    print_info "Downloading LoRA from Hugging Face Hub..."
    echo "  - Repository: ${HF_LORA_REPO}"
    echo "  - Local name: ${LORA_NAME}"

    # Create temporary directory using mktemp (global variable for cleanup)
    TEMP_DIR=$(mktemp -d -t lora_download_XXXXXX)

    # Download LoRA adapter files using the detected CLI
    print_info "Downloading adapter files using ${HF_CLI_CMD}..."
    if [ "${HF_CLI_CMD}" = "huggingface-cli" ]; then
        huggingface-cli download "${HF_LORA_REPO}" \
            --local-dir "${TEMP_DIR}" \
            --local-dir-use-symlinks False
    else
        hf download "${HF_LORA_REPO}" \
            --local-dir "${TEMP_DIR}"
    fi

    print_success "LoRA downloaded to ${TEMP_DIR}"

    rm -rf "${TEMP_DIR}/.cache"
    # List downloaded files
    echo "Downloaded files:"
    ls -lh "${TEMP_DIR}"
}

# Upload LoRA to MinIO
upload_lora_to_minio() {
    print_info "Uploading LoRA to MinIO..."

    # Upload all files to S3
    aws --endpoint-url=${MINIO_ENDPOINT} s3 sync \
        "${TEMP_DIR}" \
        "s3://${BUCKET_NAME}/${LORA_NAME}" \
        --exclude "*.git*"

    print_success "LoRA uploaded to s3://${BUCKET_NAME}/${LORA_NAME}"

    # List uploaded files
    echo "Uploaded files:"
    aws --endpoint-url=${MINIO_ENDPOINT} s3 ls "s3://${BUCKET_NAME}/${LORA_NAME}/" --recursive
}

# Cleanup temp files
cleanup() {
    if [ -n "${TEMP_DIR}" ] && [ -d "${TEMP_DIR}" ]; then
        print_info "Cleaning up temporary files..."
        rm -rf "${TEMP_DIR}"
        print_success "Cleanup complete"
    fi
}

# Stop MinIO
stop_minio() {
    print_info "Stopping MinIO..."

    if docker ps | grep -q dynamo-minio; then
        docker stop dynamo-minio 2>/dev/null
        print_success "MinIO container stopped"
    else
        print_info "MinIO container is not running"
    fi

    if docker ps -a | grep -q dynamo-minio; then
        docker rm dynamo-minio 2>/dev/null
        print_success "MinIO container removed"
    fi

    echo ""
    echo "MinIO has been stopped."
    echo "Data is preserved in: ${MINIO_DATA_DIR}"
    echo ""
    echo "To start MinIO again:"
    echo "  $0 --start"
    echo ""
}

# Start MinIO only (without downloading/uploading LoRA)
start_only() {
    echo "========================================"
    echo "Starting MinIO"
    echo "========================================"
    echo ""

    start_minio
    echo ""

    echo "========================================"
    echo "MinIO Started!"
    echo "========================================"
    echo ""
    echo "MinIO is now running."
    echo ""
    echo "To upload a LoRA, run the full setup:"
    echo "  $0"
    echo ""
    echo "Or manually upload using AWS CLI:"
    echo "  export AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY}"
    echo "  export AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY}"
    echo "  aws --endpoint-url=${MINIO_ENDPOINT} s3 cp your-lora/ s3://${BUCKET_NAME}/your-lora/ --recursive"
    echo ""
    echo "To stop MinIO:"
    echo "  $0 --stop"
    echo ""
}

# Full setup (start MinIO + download/upload LoRA)
full_setup() {
    echo "========================================"
    echo "MinIO Setup & LoRA Upload Script"
    echo "========================================"
    echo ""

    check_dependencies
    echo ""

    start_minio
    echo ""

    configure_aws_cli
    echo ""

    download_lora_from_hf
    echo ""

    upload_lora_to_minio
    echo ""

    cleanup
    echo ""

    echo "========================================"
    echo "Setup Complete!"
    echo "========================================"
    echo ""
    echo "MinIO is running and LoRA has been uploaded."
    echo ""
    echo "Next steps:"
    echo "  1. Run the Dynamo service with LoRA support:"
    echo "     ${SCRIPT_DIR}/agg_lora.sh"
    echo ""
    echo "  2. Load the LoRA adapter:"
    echo "     curl -X POST http://localhost:8081/v1/loras \\"
    echo "       -H \"Content-Type: application/json\" \\"
    echo "       -d '{\"lora_name\": \"${LORA_NAME}\", \"source\": {\"uri\": \"s3://${BUCKET_NAME}/${LORA_NAME}\"}}'"
    echo ""
    echo "  3. Run inference with the LoRA:"
    echo "     curl -X POST http://localhost:8000/v1/chat/completions \\"
    echo "       -H \"Content-Type: application/json\" \\"
    echo "       -d '{\"model\": \"${LORA_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"your prompt here\"}]}'"
    echo ""
    echo "To stop MinIO:"
    echo "  $0 --stop"
    echo ""
}

# Main execution
case "$MODE" in
    start)
        start_only
        ;;
    stop)
        stop_minio
        ;;
    help)
        show_help
        exit 0
        ;;
    full)
        full_setup
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        show_help
        exit 1
        ;;
esac

