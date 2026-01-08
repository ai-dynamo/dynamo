#!/usr/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Ultimate clean test - full deployment from scratch with verification
# This script performs:
#   1. Complete cleanup
#   2. Deploy base infrastructure
#   3. Deploy power-aware features
#   4. Run verification suite

set +e  # Don't exit on error

# Dynamically determine the repository root (parent of examples/deployments/powerplanner)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PATH=${DEV_REPO}/bin_bin:$PATH
export MINIKUBE_HOME=${DEV_REPO}/minikube_home

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     ULTIMATE CLEAN TEST - FULL AUTOMATION VERIFICATION      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  1. Complete cleanup (deploy_poweraware_baseinfra.bash 0)"
echo "  2. Deploy base infrastructure (deploy_poweraware_baseinfra.bash 1)"
echo "  3. Deploy power-aware features (deploy_poweraware.bash)"
echo "  4. Run full verification (verify_poweraware.bash)"
echo ""

# Phase 1: Cleanup
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1/4: CLEANUP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash ${DEV_REPO}/examples/deployments/powerplanner/deploy_poweraware_baseinfra.bash 0 2>&1 | tail -10
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Phase 1: Cleanup complete${NC}"
else
    echo -e "${RED}✗ Phase 1: Cleanup failed${NC}"
    exit 1
fi

# Phase 2: Deploy base infrastructure
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2/4: DEPLOY BASE INFRASTRUCTURE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This will take several minutes..."
bash ${DEV_REPO}/examples/deployments/powerplanner/deploy_poweraware_baseinfra.bash 1 > /tmp/deploy_base_clean.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Phase 2: Base infrastructure deployed${NC}"
    echo "  Log: /tmp/deploy_base_clean.log"
    tail -20 /tmp/deploy_base_clean.log
else
    echo -e "${RED}✗ Phase 2: Base infrastructure failed${NC}"
    echo "  Log: /tmp/deploy_base_clean.log"
    tail -50 /tmp/deploy_base_clean.log
    exit 1
fi

# Phase 3: Deploy power-aware features
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3/4: DEPLOY POWER-AWARE FEATURES (AUTOMATED)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This will take several minutes (including profiling)..."
bash ${DEV_REPO}/examples/deployments/powerplanner/deploy_poweraware.bash > /tmp/deploy_power_clean.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Phase 3: Power-aware features deployed${NC}"
    echo "  Log: /tmp/deploy_power_clean.log"
    tail -30 /tmp/deploy_power_clean.log
else
    echo -e "${RED}✗ Phase 3: Power-aware deployment failed${NC}"
    echo "  Log: /tmp/deploy_power_clean.log"
    tail -50 /tmp/deploy_power_clean.log
    exit 1
fi

sleep 300
# Phase 4: Verification
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 4/4: VERIFICATION SUITE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash ${DEV_REPO}/examples/deployments/powerplanner/verify_poweraware.bash 2>&1
VERIFY_EXIT=$?

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "║              ULTIMATE CLEAN TEST: SUCCESS ✓                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}All phases completed successfully!${NC}"
    echo ""
    echo "Deployment logs:"
    echo "  - Base infrastructure: /tmp/deploy_base_clean.log"
    echo "  - Power-aware features: /tmp/deploy_power_clean.log"
    exit 0
else
    echo "║           ULTIMATE CLEAN TEST: PARTIAL SUCCESS              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${YELLOW}Deployment completed but some verification tests failed/warned${NC}"
    echo ""
    echo "Deployment logs:"
    echo "  - Base infrastructure: /tmp/deploy_base_clean.log"
    echo "  - Power-aware features: /tmp/deploy_power_clean.log"
    exit 1
fi

