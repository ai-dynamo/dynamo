#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Starts and stops a dynamodb-local container for the DynamoDB registration
# store's integration tests (tests/dynamodb_store.rs, tests/http_dynamodb.rs).
# Those tests skip unless GDS_TEST_DYNAMODB_ENDPOINT is set.
#
#   scripts/dynamodb-local.sh up && \
#     GDS_TEST_DYNAMODB_ENDPOINT=http://localhost:8000 \
#     cargo test -p dynamo-global-discovery-service
#   scripts/dynamodb-local.sh down
#
# `up` starts amazon/dynamodb-local in memory on port 8000 (override with
# GDS_DYNAMODB_LOCAL_PORT) and waits until it answers. `connect`, which one
# test uses, reads credentials from the environment: dynamodb-local accepts any
# value, so `export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test`.
# Every test makes its own table, so tests do not share state. `down` removes
# the container and with it every table.

set -euo pipefail

NAME="${GDS_DYNAMODB_LOCAL_NAME:-gds-dynamodb-local}"
PORT="${GDS_DYNAMODB_LOCAL_PORT:-8000}"
IMAGE="${GDS_DYNAMODB_LOCAL_IMAGE:-amazon/dynamodb-local}"
ENDPOINT="http://localhost:${PORT}"

up() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "removing the existing $NAME container" >&2
    docker rm -f "$NAME" >/dev/null
  fi
  docker run -d --name "$NAME" -p "${PORT}:8000" "$IMAGE" \
    -jar DynamoDBLocal.jar -inMemory -sharedDb >/dev/null

  # Any HTTP answer, even the 400 DynamoDB gives a bare GET, means it is up.
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$ENDPOINT"; then
      echo "dynamodb-local is up at $ENDPOINT" >&2
      echo "export GDS_TEST_DYNAMODB_ENDPOINT=$ENDPOINT"
      echo "export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_REGION=us-east-1"
      return 0
    fi
    sleep 0.5
  done
  echo "dynamodb-local did not answer at $ENDPOINT within 30 s" >&2
  docker logs "$NAME" >&2 || true
  exit 1
}

down() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "removed $NAME" >&2
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  *)
    echo "usage: $0 up|down" >&2
    exit 2
    ;;
esac
