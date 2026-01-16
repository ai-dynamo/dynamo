#!/usr/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

export CONFIG_TYPE=disagg

# Dynamically determine the repository root (parent of examples/deployments/powerplanner)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEV_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PATH=${DEV_REPO}/bin_bin:$PATH
export MINIKUBE_HOME=${DEV_REPO}/minikube_home
export NAMESPACE=dynamo-system
export RELEASE_VERSION=0.7.0
export DOCKER_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:${RELEASE_VERSION}.post2
export MODEL_CONFIG_FILE=${SCRIPT_DIR}/${CONFIG_TYPE}.yaml
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B, Qwen/Qwen3-0.6B
export MODEL_NAME="Qwen/Qwen3-0.6B"

RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check prerequisites
check_prerequisites() {
  local MISSING_PREREQS=0

  # Check for HF_TOKEN
  if [ -z "${HF_TOKEN}" ]; then
    echo -e "${RED}ERROR: HF_TOKEN environment variable is not set${NC}"
    echo "Please set your Hugging Face token:"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    MISSING_PREREQS=1
  fi

  # Check for kubectl
  if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}ERROR: kubectl not found in PATH${NC}"
    echo "Please ensure kubectl is available in ${DEV_REPO}/bin_bin/"
    echo "or install it in your system PATH"
    MISSING_PREREQS=1
  fi

  # Check for minikube
  if ! command -v minikube &> /dev/null; then
    echo -e "${RED}ERROR: minikube not found in PATH${NC}"
    echo "Please ensure minikube is available in ${DEV_REPO}/bin_bin/"
    echo "or install it in your system PATH"
    MISSING_PREREQS=1
  fi

  # Check for helm
  if ! command -v helm &> /dev/null; then
    echo -e "${RED}ERROR: helm not found in PATH${NC}"
    echo "Please ensure helm is available in ${DEV_REPO}/bin_bin/"
    echo "or install it in your system PATH"
    MISSING_PREREQS=1
  fi

  # Check for docker
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: docker not found in PATH${NC}"
    echo "Please install Docker"
    MISSING_PREREQS=1
  fi

  if [ $MISSING_PREREQS -eq 1 ]; then
    echo ""
    echo -e "${RED}Please resolve the above prerequisites before continuing.${NC}"
    echo "See examples/deployments/powerplanner/README.md for setup instructions."
    exit 1
  fi
}

start_minikube () {
  minikube start --driver docker --mount --mount-string="/proc:/host/proc" --container-runtime docker --gpus all --memory=500gb --cpus=32
  sleep 5
  minikube addons enable istio-provisioner
  minikube addons enable istio
  minikube addons enable storage-provisioner-rancher

  # Pre-cache the vLLM runtime image to speed up deployments
  if minikube cache list | grep -q "${DOCKER_IMAGE}"; then
    echo "✓ vLLM runtime image already cached in Minikube"
  else
    echo "Caching vLLM runtime image in Minikube..."
    minikube cache add ${DOCKER_IMAGE}
  fi
}

check_minikube () {
  minikube status
  kubectl get pods -n istio-system
  kubectl get storageclass
}

install_crds () {
  helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
  helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}
}

install_platform () {
  helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
  helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace
}

verify_installation () {
  kubectl get crd | grep dynamo
  kubectl get pods -n ${NAMESPACE}
  cd ${DEV_REPO}/deploy/helm/charts
  helm uninstall dynamo-crds -n ${NAMESPACE}
  helm install dynamo-crds ./crds/ --namespace ${NAMESPACE}
  kubectl get crd dynamographdeployments.nvidia.com -o yaml | grep -i "subcomponenttype" -A 2 -B 2
  cd -
}

deploy_helloworld () {
  # Create HF token secret (needed for model downloads and power-aware deployment)
  # HF_TOKEN must be set in environment
  kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=${HF_TOKEN} -n ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

  kubectl apply -f ${MODEL_CONFIG_FILE} -n ${NAMESPACE}
}

delete_helloworld () {
  kubectl delete -f ${MODEL_CONFIG_FILE} -n ${NAMESPACE}
}

check_helloworld () {
  kubectl port-forward svc/vllm-${CONFIG_TYPE}-frontend 8000:8000 -n ${NAMESPACE} > /dev/null 2>&1 &
  local PF_PID=$!
  sleep 3
  curl http://localhost:8000/v1/models
  curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "'${MODEL_NAME}'", "messages": [{"role": "user", "content": "Tell me a story about a brave cat."}], "stream": false, "max_tokens": 100}'
  kill $PF_PID
}

install_kubeprometheusstack () {
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  helm repo update
  helm install prometheus -n monitoring --create-namespace -f ${DEV_REPO}/examples/deployments/powerplanner/prometheus-values.yaml prometheus-community/kube-prometheus-stack
}

main() {

# Check prerequisites first
check_prerequisites

if [ $1 -eq 1 ]; then
  start_minikube
  check_minikube

  # Install Prometheus FIRST so PodMonitors will be created when platform is installed
  install_kubeprometheusstack
  sleep 10

  install_platform
  sleep 20
  install_crds
  sleep 40

  verify_installation

  # Upgrade platform to regenerate PodMonitors now that Prometheus CRDs exist
  echo "Upgrading dynamo-platform to create PodMonitors..."
  helm upgrade dynamo-platform ./dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --reuse-values

  deploy_helloworld
  sleep 90
  check_helloworld
  delete_helloworld
else
  minikube stop
  minikube delete --all
fi

}

# Default to deploy (1) if no argument provided
# Usage: bash deploy_poweraware_baseinfra.bash [0|1]
#   0 = cleanup (stop and delete minikube)
#   1 = deploy (default)
main ${1:-1}

