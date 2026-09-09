#!/bin/bash
set -euo pipefail

# DRANET PCIe Topology Viewer
# Shows GPU ↔ EFA co-location by PCIe root complex, derived from DRA ResourceSlices.
# This reflects the topology that the Kubernetes scheduler uses for pcieRoot constraints.
#
# Usage: topoDranet.sh [node-name]
#   If node-name is omitted, shows topology for all GPU nodes.

NODE="${1:-}"

show_node_topology() {
  local node="$1"
  echo "======================================================================"
  echo "PCIe Topology: ${node}"
  echo "======================================================================"

  kubectl get resourceslices -o json | jq -r --arg node "$node" '
    [.items[] | select(.spec.driver=="gpu.nvidia.com" and .spec.nodeName==$node) |
     .spec.devices[] |
     {root: .attributes."resource.kubernetes.io/pcieRoot".string,
      gpu: .name,
      bus: .attributes."resource.kubernetes.io/pciBusID".string,
      product: .attributes.productName.string,
      uuid: .attributes.uuid.string}
    ] as $gpus |

    [.items[] | select(.spec.driver=="dra.net" and .spec.nodeName==$node) |
     .spec.devices[] |
     {root: .attributes."resource.kubernetes.io/pcieRoot".string,
      efa: .name,
      pci: .attributes."dra.net/pciAddress".string,
      rdma: .attributes."dra.net/rdmaDevice".string,
      numa: .attributes."dra.net/numaNode".int}
    ] as $efas |

    ($gpus | map(.root) | unique | sort)[] as $root |
    "PCIe Root: \($root)",
    ($gpus[] | select(.root==$root) |
     "  GPU  \(.gpu)  \(.bus)  \(.product)"),
    ($efas[] | select(.root==$root) |
     "  EFA  \(.efa)  \(.pci)  \(.rdma)  NUMA:\(.numa)"),
    ""
  ' 2>/dev/null || echo "  (no DRA ResourceSlices found for this node)"
}

if [ -n "${NODE}" ]; then
  show_node_topology "${NODE}"
else
  # All nodes that have GPU ResourceSlices
  NODES=$(kubectl get resourceslices -o json | \
    jq -r '.items[] | select(.spec.driver=="gpu.nvidia.com") | .spec.nodeName' | \
    sort -u)

  if [ -z "${NODES}" ]; then
    echo "No GPU ResourceSlices found. Is nvidia-dra-driver-gpu installed?"
    exit 1
  fi

  GPU_COUNT=$(kubectl get resourceslices -o json | \
    jq '[.items[] | select(.spec.driver=="gpu.nvidia.com")] | length')
  EFA_COUNT=$(kubectl get resourceslices -o json | \
    jq '[.items[] | select(.spec.driver=="dra.net")] | length')
  NODE_COUNT=$(echo "${NODES}" | wc -l | tr -d ' ')

  echo "DRANET Topology Summary"
  echo "  GPU nodes:        ${NODE_COUNT}"
  echo "  GPU ResourceSlices: ${GPU_COUNT}"
  echo "  EFA ResourceSlices: ${EFA_COUNT}"
  echo ""

  for node in ${NODES}; do
    show_node_topology "${node}"
  done
fi
