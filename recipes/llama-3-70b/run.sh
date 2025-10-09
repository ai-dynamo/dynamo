#!/bin/bash

NAMESPACE=test-bis
CUR_DIR="$( cd "$( dirname "$0" )" && pwd )"

# create the pvc for model cache and download the model
kubectl apply -n $NAMESPACE -f $CUR_DIR/model/model-cache.yaml
kubectl apply -n $NAMESPACE -f $CUR_DIR/model/model-download.yaml
# Wait for the model download to complete
echo "Waiting for the model download to complete..."
kubectl wait --for=condition=Complete job/model-download-llama-3-70b -n $NAMESPACE --timeout=6000s

# deploy the agg example
kubectl apply -n $NAMESPACE -f $CUR_DIR/agg/llama3-70b-agg.yaml

# launch the benchmark job
kubectl apply -n $NAMESPACE -f $CUR_DIR/agg/llama3-70b-agg-benchmark.yaml
kubectl wait --for=condition=Complete job/llama3-70b-agg-benchmark -n $NAMESPACE --timeout=6000s

# print logs from the benchmark job
kubectl logs job/llama3-70b-agg-benchmark  -n $NAMESPACE