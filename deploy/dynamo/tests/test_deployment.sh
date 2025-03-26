#!/bin/bash

set -euo pipefail

export DYNAMO_SEREVR="${DYNAMO_SEREVR:-http://dynamo-server}"
export DYNAMO_IMAGE="${DYNAMO_IMAGE:-dynamo-base:latest}"
export DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-ci-hw}"

cd /workspace/examples/hello_world

# Step.1: Login to  dynamo server
dynamo server login --api-token TEST-TOKEN --endpoint $DYNAMO_SEREVR

# Step.2:  build a dynamo nim with framework-less base
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk -F"\"" '{ print $2 }')

# Step.3: Deploy!
echo $DYNAMO_TAG
dynamo deployment create $DYNAMO_TAG --no-wait -n $DEPLOYMENT_NAME
