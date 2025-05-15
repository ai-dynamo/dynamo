# Multinode Examples

Table of Contents
- [Single node sized models](#single-node-sized-models)
- [Multi-node sized models](#multi-node-sized-models)

## Single node sized models
You can deploy dynamo on multiple nodes via NATS/ETCD based discovery and communication. Here's an example of deploying disaggregated serving on 3 nodes using `nvidia/Llama-3.1-405B-Instruct-FP8`. Each node will need to be properly configured with Infiniband and/or RoCE for communication between decode and prefill workers.

##### Disaggregated Deployment with KV Routing
- Node 1: Frontend, Processor, Router, Decode Worker
- Node 2: Prefill Worker
- Node 3: Prefill Worker

Note that this can be easily extended to more nodes. You can also run the Frontend, Processor, and Router on a separate CPU only node if you'd like as long as all nodes have access to the NATS/ETCD endpoints!

**Step 1**: Start NATS/ETCD on your head node. Ensure you have the correct firewall rules to allow communication between the nodes as you will need the NATS/ETCD endpoints to be accessible by all other nodes.
```bash
# node 1
docker compose -f deploy/metrics/docker-compose.yml up -d
```

**Step 2**: Create the inference graph for this node. Here we will use the `agg_router.py` (even though we are doing disaggregated serving) graph because we want the `Frontend`, `Processor`, `Router`, and `VllmWorker` to spin up (we will spin up the other decode worker and prefill worker separately on different nodes later).

```python
# graphs/agg_router.py
Frontend.link(Processor).link(Router).link(VllmWorker)
```

**Step 3**: Create a configuration file for this node. We've provided a sample one for you in `configs/multinode-405b.yaml` for the 405B model. Note that we still include the `PrefillWorker` component in the configuration file even though we are not using it on node 1. This is because we can reuse the same configuration file on all nodes and just spin up individual workers on the other ones.

**Step 4**: Start the frontend, processor, router, and VllmWorker on node 1.
```bash
# node 1
cd $DYNAMO_HOME/examples/llm
dynamo serve graphs.agg_router:Frontend -f ./configs/multinode-405b.yaml
```

**Step 5**: Start the first prefill worker on node 2.
Since we only want to start the `PrefillWorker` on node 2, you can simply run just the PrefillWorker component directly with the configuration file from before.

```bash
# node 2
export NATS_SERVER = '<your-nats-server-address>' # note this should start with nats://...
export ETCD_ENDPOINTS = '<your-etcd-endpoints-address>'

cd $DYNAMO_HOME/examples/llm
dynamo serve components.prefill_worker:PrefillWorker -f ./configs/multinode-405b.yaml
```

**Step 6**: Start the second prefill worker on node 3.
```bash
# node 3
export NATS_SERVER = '<your-nats-server-address>' # note this should start with nats://...
export ETCD_ENDPOINTS = '<your-etcd-endpoints-address>'

cd $DYNAMO_HOME/examples/llm
dynamo serve components.prefill_worker:PrefillWorker -f ./configs/multinode-405b.yaml
```

**Step 7**: [Optional] Start more decode workers on other nodes
This example can be extended to more nodes as well. For example, if you'd like to spin up another decode worker, you can use
```bash
# node X
export NATS_SERVER = '<your-nats-server-address>' # note this should start with nats://...
export ETCD_ENDPOINTS = '<your-etcd-endpoints-address>'

cd $DYNAMO_HOME/examples/llm
dynamo serve components.worker:VllmWorker -f ./configs/multinode-405b.yaml --service-name VllmWorker
```

Note the use of `--service-name`. This will only spin up the worker that you are requesting and ignore any `depends` statements.

### Client

In another terminal:
```bash
# this test request has around 200 tokens isl

curl <node1-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "nvidia/Llama-3.1-405B-Instruct-FP8",
    "messages": [
      {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
      }
    ],
    "stream": true,
    "max_tokens": 300
  }'
```

#### Multi-node sized models

Multinode model support is coming soon. You can track progress [here](https://github.com/ai-dynamo/dynamo/issues/513)!

##### Aggregated Deployment

The steps for aggregated deployment of multi-node sized models is similar to
single-node sized models, except that you need to first configure the nodes
to be interconnected according to the framework's multi-node deployment guide.
In the below example, vLLM will be used as the framework to serve `DeepSeek-R1` model
using tensor parallel 16 on two H100x8 nodes.

**Step 1**: On each of the nodes, set up Ray cluster so that vLLM can access the resource
collectively:
```bash
# head node
ray start --head --port=6379

# example output and keep note of the IP address of the head node
# Local node IP: <head-node-address>

# set vLLM env arg
export VLLM_HOST_IP=<head-node-address>

# other node
ray start  --address=<head-node-address>:6379
export VLLM_HOST_IP=<current-node-address>

# verify the accessibility by checking aggregated GPU count shown in ray status
ray status

# Expected/Sample output for 2 nodes:
# ```bash
# ======== Autoscaler status: 2025-04-16 15:35:42.751688 ========
# Node status
# ---------------------------------------------------------------
# Active:
#  1 node_<hash_1>
#  1 node_<hash_2>
# Pending:
#  (no pending nodes)
# Recent failures:
#  (no failures)
# Resources
# ---------------------------------------------------------------
# Usage:
# XXX CPU
# XXX GPU
# XXX memory
# XXX object_store_memory
# Demands:
#  (no resource demands)
```

**Step 2**: On the head node, follow [LLM Deployment Guide](./README.md#getting-started) to
setup dynamo deployment for aggregated serving, using the configuration file,
`configs/multinode_agg_r1.yaml`, for DeepSeek-R1:
```bash
cd $DYNAMO_HOME/examples/llm
dynamo serve graphs.agg:Frontend -f ./configs/multinode_agg_r1.yaml
```

### Client

In another terminal, you can send the same curl request as described above but
with `"model": "deepseek-ai/DeepSeek-R1"`
```bash
# this test request has around 200 tokens isl

curl <node1-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [
      {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
      }
    ],
    "stream": true,
    "max_tokens": 300
  }'
```

##### Disaggregated Deployment

In this example, we will be deploying two replicas of the model (one prefill worker
and one decode worker). We will be using 4 H100x8 nodes and group every two of them
into one Ray cluster in the same way as described in aggregated deployment.
However, for etcd and nats server, we will only run them in
one node and let's consider that node to be the head node of the whole deployment.

Note that if you are starting etcd server directly instead of using `docker compose`,
you should add additional arguments to be discoverable in other node.
```bash
etcd --advertise-client-urls http://<head-node-ip>:2379 --listen-client-urls http://<head-node-ip>:2379,http://127.0.0.1:2379
```

**Step 1**: On every two nodes, set up Ray cluster as described in
[aggregated deployment](#aggregated-deployment). After that, you should have
two independent Ray cluster, each has access to 16 GPUs.

**Step 2** start the deployment by running different flavors of `dynamo serve`
on one of the node for each Ray cluster, using the configuration file,
`configs/mutinode_disagg_r1.yaml`.

For decode, below command will be used and the node will be the entry point of
the whole deployment. In other words, the ip of the node should be used to send
requests to.
```bash
# if not head node
export NATS_SERVER='nats://<nats-server-ip>:4222'
export ETCD_ENDPOINTS='<etcd-endpoints-ip>:2379'

cd $DYNAMO_HOME/examples/llm
dynamo serve graphs.agg:Frontend -f ./configs/mutinode_disagg_r1.yaml
```

For prefill:
```bash
# if not head node
export NATS_SERVER='nats://<nats-server-ip>:4222'
export ETCD_ENDPOINTS='<etcd-endpoints-ip>:2379'

cd $DYNAMO_HOME/examples/llm
dynamo serve components.prefill_worker:PrefillWorker -f ./configs/mutinode_disagg_r1.yaml
```

### Client

In another terminal, you can send the same curl request as described in
[aggregated deployment](#aggregated-deployment), addressing to the ip of
the decode node.
