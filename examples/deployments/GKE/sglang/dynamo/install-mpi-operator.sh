#!/bin/bash
# Install Kubeflow training-operator (MPI) v1.7.0 - required before running nccl-test-b200-3node.yaml
set -e
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
echo "Wait for MPI operator to be ready, then: kubectl apply -f nccl-test-b200-3node.yaml -n dynamo-system"
