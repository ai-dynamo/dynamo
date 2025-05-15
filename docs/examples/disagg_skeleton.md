# Deployment Examples

This directory contains a hello world example which implements a simplified disaggregated serving architecture used for deploying Large Language Models (LLMs). It removes the LLM related inference code and focuses on how Dynamo handles routing, task queue and metadata communication between prefill and decode workers.

## Components

- frontend: A simple http server handles incoming requests
- processor: A pre/post processing server and invokes router server
- router: Handles API requests and routes them to appropriate workers based on specified strategy
- worker: A dummy decode worker
- prefill-worker: A dummy prefill worker

## Deployment Architectures

This figure shows an overview of the major components to deploy:

```
                                                 +----------------+
                                                 | prefill worker |-------+
                                                 |                |       |
                                                 +----------------+       | pull
                                                                          v
+------+      +-----------+      +------------------+    push     +---------------+
| HTTP |----->| processor |----->|  decode/monolith |------------>| prefill queue |
|      |<-----|           |<-----|      worker      |             |               |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^
       query best |    | return
           worker |    | worker_id
                  |    |         +------------------+
                  |    +---------|      router      |
                  +------------->|                  |
                                 +------------------+

```

## The Aggregated Deployment

In this example, we will use 2 nodes to demo the disagg serving.
- Node 1
  - Runs NATS and etcd services
  - Deploys Frontend, Processor and Router
  - Deploys DummyWorker as the monolith worker
- Node 2
  - Deploys DummyWorker as the monolith worker

### Prerequisites
On Node 1, start required services (etcd and NATS) using [Docker Compose](https://github.com/ai-dynamo/dynamo/blob/main/deploy/docker-compose.yml)
```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Run the Deployment

1. Set environment variables for NATS and etcd services

```bash
export NATS_SERVER="nats://Node_1_IP_ADDRESS:4222"
export ETCD_ENDPOINTS="http://Node_1_IP_ADDRESS:2379"
```

2. Launch Frontend, Processor and Router services:
```
cd dynamo/examples/hello_world/disagg_skeleton
dynamo serve components.graph:Frontend
```

3. Open a new terminal on Node 1 and deploy Worker service
```
export NATS_SERVER="nats://Node_1_IP_ADDRESS:4222"
export ETCD_ENDPOINTS="http://Node_1_IP_ADDRESS:2379"
cd dynamo/examples/hello_world/disagg_skeleton
dynamo serve components.worker:DummyWorker
```

4. Go to Node 2 and start Worker service as in step 3.
Now you should see both workers are ready in Node 1's terminal.

5. Query the Frontend with following two prompts. The router would assign different workers for each prompt and you can observe it from the responses.
- `Response: {"worker_output":"Tell me a joke_GeneratedBy_NODE1HOSTNAME","request_id":"id_number"}`
- `Response: {"worker_output":"Which team won 2020 World Series_GeneratedBy_NODE2HOSTNAME","request_id":"id_number"}`
```
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Tell me a joke",
  "request_id":"id_number"
}'
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Which team won 2020 World Series",
  "request_id":"id_number"
}'
```
6. Then modify the prompt and you will notice prompts with similar prefix will be routed to the same worker due to the simply routing algorithm used in this demo. For example, following query will be routed to the worker proceesed "Tell me a joke" prompt.
```
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Tell me a fact",
  "request_id":"id_number"
}'
```
-`Response: {"worker_output":"Tell me a fact_GeneratedBy_NODE1HOSTNAME","request_id":"id_number"}`

## The Disaggregated Deployment

In this example, we will use 3 nodes to demo the disagg serving.
- Node 1
  - Runs NATS and etcd services
  - Deploys Frontend and Processor
  - Deploys DummyWorker as the decode worker
- Node 2
  - Deploys DummyWorker as the decode worker
- Node 3
  - Deploys Prefill as the prefill worker

### Run the Deployment
1. Repeat step 1 to 4 to deploy Frontend, Processor, Router and 2 Workers as decode worker
2. Go to Node 3 and start the prefill worker.
```
export NATS_SERVER="nats://Node_1_IP_ADDRESS:4222"
export ETCD_ENDPOINTS="http://Node_1_IP_ADDRESS:2379"
cd dynamo/examples/hello_world/disagg_skeleton
dynamo serve components.prefill_worker:PrefillWorker
```
3. Query the Frontend. This time decode workers push requests to the prefill queue, and prefill worker pulles task from the queue to simulate the prefill task. The actual prefill is skipped in this demo.
```
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "This is prefill disagg serving example",
  "request_id":"12345"
}'
```
