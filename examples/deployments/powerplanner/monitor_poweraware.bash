#!/usr/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Real-time monitoring script for power-aware autoscaling

export NAMESPACE=${NAMESPACE:-dynamo-system}
export DEPLOYMENT_NAME=${DEPLOYMENT_NAME:-vllm-disagg-power-test}

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_status() {
    clear
    echo "========================================"
    echo "Power-Aware Autoscaling Monitor"
    echo "========================================"
    echo "Time: $(date)"
    echo ""

    # Find planner pod
    PLANNER_POD=$(kubectl get pods -n ${NAMESPACE} 2>/dev/null | grep "${DEPLOYMENT_NAME}.*planner" | grep Running | head -1 | awk '{print $1}')

    if [ -z "$PLANNER_POD" ]; then
        echo -e "${YELLOW}Planner pod not found or not running${NC}"
        echo ""
        return
    fi

    echo -e "${GREEN}Planner Pod: ${PLANNER_POD}${NC}"
    echo ""

    # Show recent planner decisions
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Recent Planner Decisions (last 5 lines):${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    kubectl logs ${PLANNER_POD} -n ${NAMESPACE} --tail=100 2>/dev/null | grep -E "Predicted number of engine replicas|Power budget" | tail -5 || echo "No decisions yet"
    echo ""

    # Show power budget status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Power Budget Status:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    POWER_LINE=$(kubectl logs ${PLANNER_POD} -n ${NAMESPACE} --tail=50 2>/dev/null | grep "Power budget" | tail -1)
    if [ ! -z "$POWER_LINE" ]; then
        if echo "$POWER_LINE" | grep -q "EXCEEDED"; then
            echo -e "${YELLOW}⚠️  $POWER_LINE${NC}"
        else
            echo -e "${GREEN}✓  $POWER_LINE${NC}"
        fi
    else
        echo "No power budget info yet"
    fi
    echo ""

    # Show worker pod annotations
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Worker Pod Power Annotations:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    kubectl get pods -n ${NAMESPACE} 2>/dev/null | grep -E "(prefill|decode)" | grep Running | while read line; do
        POD_NAME=$(echo $line | awk '{print $1}')
        ANNOTATION=$(kubectl get pod $POD_NAME -n ${NAMESPACE} -o jsonpath='{.metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}' 2>/dev/null || echo "not set")
        STATUS=$(echo $line | awk '{print $3}')
        if [ "$ANNOTATION" != "not set" ]; then
            echo -e "${GREEN}✓${NC} $POD_NAME: ${ANNOTATION}W"
        else
            echo -e "${YELLOW}○${NC} $POD_NAME: $ANNOTATION"
        fi
    done
    echo ""

    # Show Power Agent status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Power Agent Activity:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    AGENT_ACTIVITY=$(kubectl logs -n ${NAMESPACE} -l app=power-agent --tail=10 --since=30s 2>/dev/null | grep -E "Setting power limit|Enforcing limits" | tail -3)
    if [ ! -z "$AGENT_ACTIVITY" ]; then
        echo "$AGENT_ACTIVITY"
    else
        echo "No recent activity (pods may not have GPU processes yet)"
    fi
    echo ""

    # Show replica counts
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Current Replica Counts:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    PREFILL_COUNT=$(kubectl get pods -n ${NAMESPACE} 2>/dev/null | grep -c "prefill.*Running" || echo 0)
    DECODE_COUNT=$(kubectl get pods -n ${NAMESPACE} 2>/dev/null | grep -c "decode.*Running" || echo 0)
    echo "Prefill Workers: $PREFILL_COUNT"
    echo "Decode Workers:  $DECODE_COUNT"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Press Ctrl+C to exit | Refreshing every 5s"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

show_live_logs() {
    echo "========================================"
    echo "Live Planner Logs (Power-Aware)"
    echo "========================================"
    echo ""

    PLANNER_POD=$(kubectl get pods -n ${NAMESPACE} 2>/dev/null | grep "${DEPLOYMENT_NAME}.*planner" | grep Running | head -1 | awk '{print $1}')

    if [ -z "$PLANNER_POD" ]; then
        echo "Planner pod not found"
        exit 1
    fi

    echo "Following logs for: $PLANNER_POD"
    echo "Filtering for power-related messages..."
    echo ""

    kubectl logs -f ${PLANNER_POD} -n ${NAMESPACE} 2>/dev/null | grep --line-buffered -E "Power|power|Applied power limits|Predicted number"
}

show_agent_logs() {
    echo "========================================"
    echo "Live Power Agent Logs"
    echo "========================================"
    echo ""

    kubectl logs -f -n ${NAMESPACE} -l app=power-agent 2>/dev/null
}

show_help() {
    echo "Power-Aware Autoscaling Monitor"
    echo ""
    echo "Usage:"
    echo "  $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status    - Show dashboard with current status (default, refreshes every 5s)"
    echo "  planner   - Stream planner logs (power-related only)"
    echo "  agent     - Stream Power Agent logs"
    echo "  help      - Show this help"
    echo ""
    echo "Environment variables:"
    echo "  NAMESPACE        - Kubernetes namespace (default: dynamo-system)"
    echo "  DEPLOYMENT_NAME  - Deployment name prefix (default: vllm-disagg-power-test)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Show status dashboard"
    echo "  $0 planner           # Stream planner logs"
    echo "  $0 agent             # Stream agent logs"
    echo "  NAMESPACE=prod $0    # Monitor production namespace"
}

# Main
case "${1:-status}" in
    status)
        while true; do
            show_status
            sleep 5
        done
        ;;
    planner)
        show_live_logs
        ;;
    agent)
        show_agent_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac

