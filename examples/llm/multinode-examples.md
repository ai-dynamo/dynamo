# Multinode Examples

Table of Contents
- [Single node sized models](#single-node-sized-models)

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
docker compose -f deploy/docker-compose.yml up -d
```

**Step 2**: Create the inference graph for this node. Here we will use the `agg_router.py` (even though we are doing disaggregated serving) graph because we want the `Frontend`, `Processor`, `Router`, and `VllmWorker` to spin up (we will spin up the other decode worker and prefill worker separately on different nodes later).

```python
# graphs/agg_router.py
Frontend.link(Processor).link(Router).link(VllmWorker)
```

**Step 3**: Create a configuration file for this node. We've provided a sample one for you in `configs/multinode-405b.yaml` for the 405B model. Note that we still include the `PrefillWorker` component in the configuration file even though we are not using it on node 1. This is because we can reuse the same configuration file on all nodes and just spin up individual workers on the other ones.

**Step 3**: Start the frontend, processor, router, and VllmWorker on node 1.
```bash
# node 1
cd $DYNAMO_HOME/examples/llm
dynamo serve graphs.agg_router:Frontend -f ./configs/multinode-405b.yaml
```

**Step 4**: Start the first prefill worker on node 2.
Since we only want to start the `PrefillWorker` on node 2, you can simply run just the PrefillWorker component directly with the configuration file from before.

```bash
# node 2
export NATS_SERVER = '<your-nats-server-address>' # note this should start with nats://...
export ETCD_ENDPOINTS = '<your-etcd-endpoints-address>'

cd $DYNAMO_HOME/examples/llm
dynamo serve components.prefill_worker:PrefillWorker -f ./configs/multinode-405b.yaml
```

**Step 5**: Start the second prefill worker on node 3.
```bash
# node 3
export NATS_SERVER = '<your-nats-server-address>' # note this should start with nats://...
export ETCD_ENDPOINTS = '<your-etcd-endpoints-address>'

cd $DYNAMO_HOME/examples/llm
dynamo serve components.prefill_worker:PrefillWorker -f ./configs/multinode-405b.yaml
```

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