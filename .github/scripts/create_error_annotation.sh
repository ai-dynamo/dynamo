#!/bin/bash
# Common script to create GitHub error annotations with LogAI analysis
# Usage: ./create_error_annotation.sh <step_name> <line_number> <log_file> [error_message] [namespace]

set -e

STEP_NAME="${1:-Unknown Step}"
LINE_NUMBER="${2:-1}"
LOG_FILE="${3:-}"
MANUAL_ERROR="${4:-}"
NAMESPACE="${5:-}"

echo "=== Creating error annotation for ${STEP_NAME} ==="

# Install Python and LogAI inline if not already available
if ! command -v python3 &> /dev/null; then
    echo "Python not available, skipping LogAI analysis"
    USE_LOGAI=false
else
    pip install --upgrade pip logai 2>/dev/null || echo "LogAI installation skipped"
    if [ -f "$(dirname "$0")/extract_log_errors.py" ]; then
        chmod +x "$(dirname "$0")/extract_log_errors.py" 2>/dev/null || true
        USE_LOGAI=true
    else
        USE_LOGAI=false
    fi
fi

# Extract errors from log if available
EXTRACTED_ERROR=""
if [ "$USE_LOGAI" = true ] && [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo "Analyzing $LOG_FILE with LogAI..."
    EXTRACTED_ERROR=$(python3 "$(dirname "$0")/extract_log_errors.py" "$LOG_FILE" 2>/dev/null || echo "")
fi

# Use extracted error or fall back to manual error
if [ -n "$EXTRACTED_ERROR" ]; then
    FINAL_ERROR="$EXTRACTED_ERROR"
elif [ -n "$MANUAL_ERROR" ]; then
    FINAL_ERROR="$MANUAL_ERROR"
else
    FINAL_ERROR="Unknown error occurred in ${STEP_NAME}"
fi

# Add Kubernetes pod context if namespace is provided
if [ -n "$NAMESPACE" ]; then
    if [ -f ".kubeconfig" ]; then
        export KUBECONFIG="$(pwd)/.kubeconfig"
    fi
    
    # Get pod status
    POD_STATUS=$(kubectl get pods -n "$NAMESPACE" -o json 2>/dev/null || echo "{}")
    POD_ERRORS=$(echo "$POD_STATUS" | jq -r '.items[] | select(.status.phase != "Running") | "Pod: \(.metadata.name), Status: \(.status.phase), Reason: \(.status.containerStatuses[0].state.waiting.reason // .status.reason // "N/A")"' 2>/dev/null | head -10 || echo "")
    
    if [ -n "$POD_ERRORS" ]; then
        FINAL_ERROR="${FINAL_ERROR}\n\nPod Status:\n${POD_ERRORS}"
    fi
    
    # Get recent error events
    EVENTS=$(kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' -o json 2>/dev/null || echo "{}")
    ERROR_EVENTS=$(echo "$EVENTS" | jq -r '.items[] | select(.type == "Warning" or .type == "Error") | "\(.lastTimestamp) - \(.reason): \(.message)"' 2>/dev/null | tail -5 || echo "")
    
    if [ -n "$ERROR_EVENTS" ]; then
        FINAL_ERROR="${FINAL_ERROR}\n\nRecent Events:\n${ERROR_EVENTS}"
    fi
fi

# Create GitHub annotation using workflow command
echo "::error file=.github/workflows/container-validation-backends.yml,line=${LINE_NUMBER},title=${STEP_NAME} Failed::${FINAL_ERROR}"

echo "=== Annotation created for ${STEP_NAME} ==="

exit 1

