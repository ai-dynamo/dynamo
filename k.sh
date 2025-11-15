#!/bin/bash

# Script to kill dynamo frontend and worker processes
# This script finds and kills processes based on their command patterns

echo "Searching for dynamo processes..."

# Function to kill processes matching a pattern
kill_processes() {
    local pattern="$1"
    local description="$2"
    
    # Find PIDs matching the pattern
    pids=$(ps ax | grep -E "$pattern" | grep -v grep | awk '{print $1}')
    
    if [ -n "$pids" ]; then
        echo "Found $description processes:"
        ps ax | grep -E "$pattern" | grep -v grep
        echo ""
        for pid in $pids; do
            echo "Killing PID $pid..."
            kill -9 "$pid" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ✓ Successfully killed PID $pid"
            else
                echo "  ✗ Failed to kill PID $pid (may require sudo)"
            fi
        done
        echo ""
    else
        echo "No $description processes found"
        echo ""
    fi
}

# Kill dynamo.frontend with router-mode kv
kill_processes "dynamo\.frontend.*--router-mode kv" "dynamo.frontend (router-mode kv)"

# Kill decode workers
kill_processes "dynamo\..*--is-decode-worker" "decode worker"

# Kill prefill workers  
kill_processes "dynamo\..*--is-prefill-worker" "prefill worker"

echo "Done!"
echo ""
echo "Remaining dynamo processes (if any):"
ps ax | grep -E "dynamo\.(frontend|vllm|sglang)" | grep -v grep | grep -v "$0" || echo "None found"

