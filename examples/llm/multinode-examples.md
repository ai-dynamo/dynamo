# Multinode Examples

Table of Contents
- [Single node sized models](#single-node-sized-models)
- [Multi-node sized models](#multi-node-sized-models)

## Single node sized models
You can deploy our example architectures on multiple nodes via NATS/ETCD based discovery and communication. Here's an example of deploying disaggregated serving on 2 nodes

##### Disaggregated Deployment with KV Routing
Node 1: Frontend, Processor, Router, 8 Decode
Node 2: 8 Prefill

**Step 1**: Start NATS/ETCD on your head node. Ensure you have the correct firewall rules to allow communication between the nodes as you will need the NATS/ETCD endpoints to be accessible by node 2.
```bash
# node 1
docker compose -f deploy/docker-compose.yml up -d
```

**Step 2**: Create the inference graph for this deployment. The easiest way to do this is to remove the `.link(PrefillWorker)` from the `disagg_router.py` file.

```python
# graphs/disagg_router.py
# imports...
Frontend.link(Processor).link(Router).link(VllmWorker)
```

**Step 3**: Start the frontend, processor, router, and 8 VllmWorkers on node 1.
```bash
# node 1
cd $DYNAMO_HOME/examples/llm
dynamo serve graphs.disagg_router:Frontend -f ./configs/disagg_router.yaml --VllmWorker.ServiceArgs.workers=8
```

**Step 4**: Start 8 PrefillWorkers on node 2.
Since we only want to start the `PrefillWorker` on node 2, you can simply run just the PrefillWorker component directly.

```bash
# node 2
export NATS_SERVER = '<your-nats-server-address>' # note this should start with nats://...
export ETCD_ENDPOINTS = '<your-etcd-endpoints-address>'

cd /workspace/examples/llm
dynamo serve components.prefill_worker:PrefillWorker -f ./configs/disagg_router.yaml --PrefillWorker.ServiceArgs.workers=8
```

You can now use the same curl request from above to interact with your deployment!

#### Multi-node sized models 
We support deploying models that require multiple nodes to serve. As our components are based on VLLM, we can use the same ray cluster flow to serve each model. Below is an example that lets you deploy Llama 3.1 405B FP8 with our disaggregated serving example. Please ensure that these nodes are correctly configured with Infiniband and/or RoCE.

In this example, we will deploy 2 Llama 3.1 405B models on 4 nodes each at tp16. This can be extended and configured as needed via the configuration yaml file

##### Disaggregated Deployment with KV Routing
Node 1: Frontend, Processor, Router, Decode Worker
Node 2: Decode Worker
Node 3: Prefill Worker
Node 4: Prefill Worker

**Step 1**: Start NATS/ETCD on your head node (node 1). Ensure you have the correct firewall rules to allow communication between the nodes as you will need the NATS/ETCD endpoints to be accessible by node 3
```bash
# node 1
docker compose -f deploy/docker-compose.yml up -d
```

**Step 2**: Set the neccesary environment variables on your head nodes (node 1 and node 3)
```bash
# node 1
export GLOO_SOCKET_IFNAME=...
export UCX_NET_DEVICES=...
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=...
export NCCL_SOCKET_IFNAME=...


# node 3
export NATS_SERVER = '<your-nats-server-address>' # note this should start with nats://...
export ETCD_ENDPOINTS = '<your-etcd-endpoints-address>'

export GLOO_SOCKET_IFNAME=...
export UCX_NET_DEVICES=...
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=...
export NCCL_SOCKET_IFNAME=...
```

**Step 3**: Start a ray cluster on your head nodes (node 1 and node 3)
```bash
# node 1
export 
ray start --head

# node 3
ray start --head
```

**Step 4**: Add ray worker nodes (node 2 for node 1 and node 4 for node 3). `6379` is the default ray port
```bash
# node 2
ray start --address='<node-1-ray-address>:6379'

# node 4
ray start --address='<node-3-ray-address>:6379'
```

You can run ray status on any of the nodes to ensure that each cluster has 2 nodes

**Step 5**: Create the inference graph for this deployment. The easiest way to do this is to remove the `.link(PrefillWorker)` from the `disagg_router.py` file.

```python
# graphs/disagg_router.py
# imports...
Frontend.link(Processor).link(Router).link(VllmWorker)
```

**Step 6**: Edit the `disagg_router.yaml` file to configure Llama 3.1 405B. You will have to change the model name on the components to `nvidia/Llama-3.1-405B-Instruct-FP8`, `VllmWorker.ServiceArgs.resources.gpu` to `16`, `VllmWorker.tensor-parallel-size` to `16`, `PrefillWorker.ServiceArgs.resources.gpu` to `16`, and `PrefillWorker.tensor-parallel-size` to `16`.

**Step 7**: Start the frontend, processor, router, and VllmWorker on node 1.
```bash
# node 1
cd $DYNAMO_HOME/examples/llm
dynamo serve graphs.disagg_router:Frontend -f ./configs/disagg_router.yaml --VllmWorker.enforce-eager=true
```

Because we setup the ray cluster on nodes 1 and 2, we can specify the `gpu` and `tensor-parallel-size` to be `16` for the VllmWorkers.

**Step 8**: Start the PrefillWorker on node 3.
```bash
# node 3
dynamo serve components.prefill_worker:PrefillWorker -f ./configs/disagg_router.yaml --PrefillWorker.enforce-eager=true
```

Because we setup the ray cluster on nodes 3 and 4, we can specify the `gpu` and `tensor-parallel-size` to be `16` for the PrefillWorkers.
