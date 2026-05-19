#!/bin/bash
# Launch the Dynamo HTTP frontend with the embedded KV-aware router.
# The frontend serves OpenAI-compatible HTTP and routes to workers
# discovered via etcd (here: dynamo.backend.generate, two instances).
#
# Env vars:
#   KVBM_VENV         (default: ryan-velo-messenger/.sandbox)
#   KVR_HTTP_PORT     (default: 8080)
#   KVR_NAMESPACE     (default: dynamo)
set -eu

KVBM_VENV=${KVBM_VENV:-/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox}
KVR_HTTP_PORT=${KVR_HTTP_PORT:-8080}
KVR_NAMESPACE=${KVR_NAMESPACE:-dynamo}

echo "[frontend] launching dynamo.frontend --router-mode kv --http-port $KVR_HTTP_PORT"

# Frontend auto-derives KV block size from worker registration —
# don't pass --kv-cache-block-size unless overriding.
#
# --router-kv-events is the default; spelling it out documents intent.
# --router-min-initial-workers 2 makes the frontend wait until both
# workers are registered before serving (avoids R1 racing the second
# worker into etcd).
exec "$KVBM_VENV/bin/python" -m dynamo.frontend \
  --namespace "$KVR_NAMESPACE" \
  --router-mode kv \
  --router-kv-events \
  --router-min-initial-workers 2 \
  --http-port "$KVR_HTTP_PORT"
