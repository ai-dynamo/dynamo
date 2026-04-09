#!/bin/bash
set -e
TIMESTAMP=$(date +%s)
echo $TIMESTAMP > /tmp/.${POD_NAME}.start_time
mkdir -p {{SERVICE_LOG_DIR}}
LOG_FILE="{{SERVICE_LOG_DIR}}/${POD_NAME}_${TIMESTAMP}.log"
exec {{FULL_COMMAND}} > >(tee -a "$LOG_FILE") 2>&1
