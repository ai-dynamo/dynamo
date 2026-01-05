#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
trap 'echo "Error at line $LINENO. Exiting."' ERR

read -p "Are you sure you want to delete ALL Dynamo CRDs and their instances? (y/N): " confirm
if [[ "$confirm" != "y" ]]; then
  echo "Aborting."
  exit 1
fi

# Step 1: Get all CRDs with the prefix
DYNAMO_CRDS="$(kubectl get crds -o name | grep 'nvidia.com' | grep 'dynamo' | cut -d'/' -f2)"

if [ -z "${DYNAMO_CRDS}" ]; then
  echo "Dynamo CRDs not found"
  exit 1
fi

# Step 2: Delete all custom resource instances for each CRD
for CRD in ${DYNAMO_CRDS}; do
  SCOPE=$(kubectl get crd "${CRD}" -o jsonpath='{.spec.scope}')

  if [ "$SCOPE" == "Namespaced" ]; then
    echo "Deleting all namespaced instances of ${CRD}..."
    kubectl get "${CRD}" --all-namespaces -o name | xargs -r kubectl delete --wait=false
  else
    echo "Skipping cluster-scoped CRD: ${CRD}"
  fi
done


# Step 3: Wait for the Operator to handle finalizer removal
echo "Waiting for Dynamo Operator to handle the finalizer removal (30 seconds)..."
sleep 30

# Step 4: Verify all Custom Resources have been removed
for CRD in ${DYNAMO_CRDS}; do
  # Check CRs

  echo "Checking instances of ${CRD}"
  kubectl get "${CRD}" --all-namespaces -o name
done

# Step 5: Delete the CRDs themselves
echo "Deleting CRDs..."

for CRD in ${DYNAMO_CRDS}; do
  # Delete all CRD's

  echo "Deleting CRD: ${CRD}..."
  kubectl delete crd "${CRD}"
done


