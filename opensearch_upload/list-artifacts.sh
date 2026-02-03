#!/bin/bash
#
# GitHub Actions Artifact Lister - Detailed Build Metrics
# Lists artifacts and uploads detailed container, stage, and layer metrics
#

set -u

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration
DEFAULT_REPO="ai-dynamo/dynamo"
DEFAULT_HOURS="1.0"
DEFAULT_CONTAINER_INDEX="http://gpuwa.nvidia.com/dataflow2/swdl-triton-ops-gh-containers/posting"
DEFAULT_STAGE_INDEX="http://gpuwa.nvidia.com/dataflow2/swdl-triton-ops-gh-stages/posting"
DEFAULT_LAYER_INDEX="http://gpuwa.nvidia.com/dataflow2/swdl-triton-ops-gh-layers/posting"

# Colors for output
COLOR_INFO="\033[0;36m"
COLOR_SUCCESS="\033[0;32m"
COLOR_ERROR="\033[0;31m"
COLOR_RESET="\033[0m"

# Logging functions
log_info() {
    echo -e "${COLOR_INFO}[INFO]${COLOR_RESET} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${COLOR_SUCCESS}[SUCCESS]${COLOR_RESET} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${COLOR_ERROR}[ERROR]${COLOR_RESET} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to check dependencies
check_dependencies() {
    # Check for Python 3 in venv first
    if [ -f "$HOME/workflow_env/bin/python3" ]; then
        PYTHON_BIN="$HOME/workflow_env/bin/python3"
        log_info "Using venv Python: $PYTHON_BIN"
        return 0
    fi

    # Fall back to system python3
    if ! command -v python3 &> /dev/null; then
        log_error "python3 is required but not installed"
        log_error "Please install Python 3 or set up the venv at ~/workflow_env"
        exit 1
    fi

    PYTHON_BIN="python3"
    log_info "Using system Python: $PYTHON_BIN"

    # Check if requests module is available
    if ! $PYTHON_BIN -c "import requests" 2>/dev/null; then
        log_error "Python requests module is required but not installed"
        log_error "Install with: pip install requests"
        exit 1
    fi
}

# Function to load GitHub token
load_github_token() {
    if [ -z "${GITHUB_TOKEN:-}" ]; then
        TOKEN_FILE="$HOME/.github-token"
        if [ -f "$TOKEN_FILE" ]; then
            export GITHUB_TOKEN=$(cat "$TOKEN_FILE")
            log_info "Loaded GitHub token from $TOKEN_FILE"
        else
            log_error "GitHub token not found"
            log_error "Set GITHUB_TOKEN environment variable or create $TOKEN_FILE"
            exit 1
        fi
    fi
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

List GitHub Actions artifacts and upload detailed build metrics

OPTIONS:
    --hours HOURS                   Number of hours to look back (default: $DEFAULT_HOURS)
    --repo OWNER/REPO               Repository to query (default: $DEFAULT_REPO)
    --upload-detailed-metrics       Download artifacts and upload detailed metrics to OpenSearch (default: enabled)
    --no-upload                     Disable uploading metrics to OpenSearch
    --container-index URL           Override container index URL
    --stage-index URL               Override stage index URL
    --layer-index URL               Override layer index URL
    -h, --help                      Show this help message

EXAMPLES:
    # List artifacts from the past hour
    $0

    # List artifacts from the past 6 hours
    $0 --hours 6

    # List and upload detailed metrics
    $0 --hours 1 --upload-detailed-metrics

    # List artifacts from a different repository
    $0 --repo myorg/myrepo --hours 24

    # Use custom OpenSearch indices
    $0 --upload-detailed-metrics --container-index http://custom.com/containers

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN        GitHub personal access token (or use ~/.github-token file)
    GITHUB_REPOSITORY   Default repository (can be overridden with --repo)
    CONTAINER_INDEX     OpenSearch index URL for container metrics
                        Default: $DEFAULT_CONTAINER_INDEX
    STAGE_INDEX         OpenSearch index URL for stage metrics
                        Default: $DEFAULT_STAGE_INDEX
    LAYER_INDEX         OpenSearch index URL for layer metrics
                        Default: $DEFAULT_LAYER_INDEX

DETAILED METRICS:
    When --upload-detailed-metrics is enabled, the script will:
    1. Download build-metrics artifacts
    2. Parse the nested JSON structure with container, stages, and layers
    3. Upload container-level metrics (overall build info)
    4. Upload stage-level metrics (per Docker build stage)
    5. Upload layer-level metrics (per Docker layer/step)
    
    All metrics include relational fields (job_id, workflow_id, etc.) for joining data.

EOF
    exit 0
}

# Main function
main() {
    # Parse command line arguments
    REPO="${GITHUB_REPOSITORY:-$DEFAULT_REPO}"
    HOURS="$DEFAULT_HOURS"
    UPLOAD_METRICS="--upload-detailed-metrics"
    CONTAINER_INDEX="${CONTAINER_INDEX:-$DEFAULT_CONTAINER_INDEX}"
    STAGE_INDEX="${STAGE_INDEX:-$DEFAULT_STAGE_INDEX}"
    LAYER_INDEX="${LAYER_INDEX:-$DEFAULT_LAYER_INDEX}"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --hours)
                HOURS="$2"
                shift 2
                ;;
            --repo)
                REPO="$2"
                shift 2
                ;;
            --upload-detailed-metrics)
                UPLOAD_METRICS="--upload-detailed-metrics"
                shift
                ;;
            --no-upload)
                UPLOAD_METRICS=""
                shift
                ;;
            --container-index)
                CONTAINER_INDEX="$2"
                shift 2
                ;;
            --stage-index)
                STAGE_INDEX="$2"
                shift 2
                ;;
            --layer-index)
                LAYER_INDEX="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    log_info "Starting GitHub Actions detailed artifact listing for $REPO"
    log_info "Configuration:"
    log_info "  Repository: $REPO"
    log_info "  Hours Back: $HOURS"
    # Always export index URLs (needed when upload is enabled, which is the default)
    export CONTAINER_INDEX="$CONTAINER_INDEX"
    export STAGE_INDEX="$STAGE_INDEX"
    export LAYER_INDEX="$LAYER_INDEX"
    
    if [ -n "$UPLOAD_METRICS" ]; then
        log_info "  Upload Detailed Metrics: ENABLED"
        log_info "  Container Index: $CONTAINER_INDEX"
        log_info "  Stage Index: $STAGE_INDEX"
        log_info "  Layer Index: $LAYER_INDEX"
    else
        log_info "  Upload Detailed Metrics: DISABLED"
    fi

    # Check dependencies
    check_dependencies

    # Load GitHub token
    load_github_token

    # Set up Python script path
    PYTHON_SCRIPT="$SCRIPT_DIR/list_artifacts.py"

    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi

    log_info "Using Python script: $PYTHON_SCRIPT"

    # Run the Python artifact lister
    log_info "Running Python detailed artifact lister..."
    
    export GITHUB_REPOSITORY="$REPO"
    
    $PYTHON_BIN "$PYTHON_SCRIPT" \
        --hours "$HOURS" \
        --repo "$REPO" \
        $UPLOAD_METRICS
    
    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        log_success "Detailed artifact listing completed successfully"
    else
        log_error "Detailed artifact listing failed with exit code $RESULT"
        exit $RESULT
    fi
}

# Run main function
main "$@"


