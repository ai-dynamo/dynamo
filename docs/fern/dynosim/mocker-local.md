---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Simulate a Local Deployment with Mocker
subtitle: Run a GPU-free frontend and simulated worker from the command line
---

Mocker is a simulated Dynamo backend. Use it locally to test frontend discovery, routing, KV events,
and worker behavior without loading a model or using GPUs. This is a **live simulation**: you start a
frontend and workers, then send HTTP requests through the Dynamo runtime.

For a faster **offline replay** that drives simulated engines directly without a frontend, worker
registration, or runtime services, see [Run a DynoSim Simulation](runs.md).

This tutorial uses file-based discovery so that the frontend and worker can run without etcd or
NATS. For Kubernetes, see [Simulate a Kubernetes Deployment](mocker.md). For every available flag,
see the [Mocker CLI Reference](../components/mocker/mocker-cli-reference.mdx).

## Prerequisites

Install Dynamo by following [Install Dynamo](../getting-started/local-installation.mdx). Verify that
the local environment can import the frontend and Mocker modules:

```bash
python -m dynamo.frontend --help >/dev/null
python -m dynamo.mocker --help >/dev/null
```

## Start the frontend

In the first terminal, start the OpenAI-compatible frontend:

```bash
python -m dynamo.frontend --http-port 8000 --discovery-backend file
```

File-based discovery keeps this tutorial on one machine and selects the local event path.

## Start a Mocker worker

In a second terminal, start one simulated worker with the same discovery backend:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file
```

Wait for the worker to register with the frontend.

## Send a request

In a third terminal, send a request to the frontend:

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

A successful response confirms that local discovery and request routing are working.

## Run multiple simulated workers

Stop the Mocker process and restart it with four workers:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --num-workers 4
```

Send several requests and compare the frontend and worker logs. For local scale tests, prefer
`--num-workers` over launching many separate Mocker processes because the workers share one runtime
and thread pool.

## Simulate disaggregated serving

Stop the aggregated Mocker process. Start a prefill worker:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --disaggregation-mode prefill \
  --bootstrap-ports 50100
```

In another terminal, start a decode worker:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --disaggregation-mode decode
```

Send another request and inspect both worker logs to confirm that the request moved through prefill
and decode.

## Tune the simulation

Restart the worker with settings that match the experiment you want to run. For example:

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --num-gpu-blocks-override 8192 \
  --block-size 64 \
  --max-num-seqs 256 \
  --speedup-ratio 10.0
```

Use the [Mocker CLI Reference](../components/mocker/mocker-cli-reference.mdx) for scheduling,
KV-cache, timing, AIC, and transport settings. Use [Run a DynoSim Simulation](runs.md) when you want
to replay a complete trace without managing live frontend and worker processes.
