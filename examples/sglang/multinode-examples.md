# Multinode Examples

## Multi-node sized models

SGLang allows you to deploy multi-node sized models by adding in the `dist-init-addr`, `nnodes`, and `node-rank` arguments. Below we demonstrate and example of deploying DeepSeek R1 for disaggregated serving across 4 nodes. This example requires
4 nodes of 8xH100 GPUs.

**Step 1**: Start NATS/ETCD on your head node. Ensure you have the correct firewall rules to allow communication between the nodes as you will need the NATS/ETCD endpoints to be accessible by all other nodes.
```bash
# node 1
docker compose -f lib/runtime/docker-compose.yml up -d
```

**Step 2**: Ensure that your configuration file has the required arguments. Here's an example configuration that runs prefill and the model in TP16:

Node 1: Run HTTP ingress, processor, and 8 shards of the prefill worker
```yaml
# configs/prefill-1.yaml
Frontend:
  served_model_name: deepseek-ai/DeepSeek-R1
  endpoint: dynamo.SGLangWorker.generate
  port: 8000

SGLangWorker:
  model-path: deepseek-ai/DeepSeek-R1
  served-model-name: deepseek-ai/DeepSeek-R1
  tp: 16
  trust-remote-code: true
  skip-tokenizer-init: true
  dist-init-addr: <node-1-ip>:29500
  disaggregation-bootstrap-port: 30001
  disaggregation-mode: prefill
  disaggregation-transfer-backend: nixl
  nnodes: 2
  node-rank: 0
  mem-fraction-static: 0.82
  ServiceArgs:
    workers: 1
    resources:
      gpu: 8
```

Run this with:
```bash
cd examples/sglang
dynamo serve graphs.agg:Frontend -f configs/prefill-1.yaml
```

Node 2: Run the remaining 8 shards of the prefill worker and the decode worker
```yaml
# configs/prefill-2.yaml
SGLangWorker:
  model-path: deepseek-ai/DeepSeek-R1
  served-model-name: deepseek-ai/DeepSeek-R1
  tp: 16
  trust-remote-code: true
  skip-tokenizer-init: true
  mem-fraction-static: 0.82
  dist-init-addr: <node-1-ip>:29500
  disaggregation-bootstrap-port: 30001
  disaggregation-mode: prefill
  disaggregation-transfer-backend: nixl
  nnodes: 2
  node-rank: 1
  ServiceArgs:
    workers: 1
    resources:
      gpu: 8
```

On all other nodes, we need to export the NATS and ETCD endpoints. Run this with with:
```bash
export NATS_SERVER="nats://<node-1-ip>"
export ETCD_ENDPOINTS="<node-1-ip>:2379"

cd examples/sglang
dynamo serve graphs.disagg:Frontend -f configs/prefill-2.yaml --service-name SGLangWorker
```

Node 3: Run the first 8 shards of the decode worker
```yaml
# configs/decode-1.yaml
SGLangDecodeWorker:
  model-path: deepseek-ai/DeepSeek-R1
  served-model-name: deepseek-ai/DeepSeek-R1
  tp: 16
  trust-remote-code: true
  skip-tokenizer-init: true
  mem-fraction-static: 0.80
  dist-init-addr: 2:29500
  disaggregation-mode: decode
  disaggregation-transfer-backend: nixl
  disaggregation-bootstrap-port: 30001
  nnodes: 2
  node-rank: 0
  ServiceArgs:
    workers: 1
    resources:
      gpu: 8
```

Run this with:
```bash
export NATS_SERVER="nats://<node-1-ip>"
export ETCD_ENDPOINTS="<node-1-ip>:2379"

cd examples/sglang
dynamo serve graphs.disagg:Frontend -f configs/decode-1.yaml --service-name SGLangDecodeWorker
```

Node 4: Run the remaining 8 shards of the decode worker
```yaml
# configs/decode-2.yaml
SGLangDecodeWorker:
  model-path: deepseek-ai/DeepSeek-R1
  served-model-name: deepseek-ai/DeepSeek-R1
  tp: 16
  trust-remote-code: true
  skip-tokenizer-init: true
  mem-fraction-static: 0.80
  dist-init-addr: 2:29500
  disaggregation-mode: decode
  disaggregation-transfer-backend: nixl
  disaggregation-bootstrap-port: 30001
  disable-cuda-graph: true
  nnodes: 2
  node-rank: 1
  ServiceArgs:
    workers: 1
    resources:
      gpu: 8
```

Run this with:
```bash
export NATS_SERVER="nats://<node-1-ip>"
export ETCD_ENDPOINTS="<node-1-ip>:2379"

cd examples/sglang
dynamo serve graphs.disagg:Frontend -f configs/decode-2.yaml --service-name SGLangDecodeWorker
```

**Step 3**: Run inference
SGLang typically requires a warmup period to ensure the DeepGEMM kernels are loaded. We recommend running a few warmup requests and ensuring that the DeepGEMM kernels load in.

```bash
curl <node-1-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of the tennis world, where champions rise and fall with each Grand Slam, lies the legend of the Golden Racket of Wimbledon. Once wielded by the greatest players of antiquity, this mythical racket is said to bestow unparalleled precision, grace, and longevity upon its rightful owner. For centuries, it remained hidden, its location lost to all but the most dedicated scholars of the sport. You are Roger Federer, the Swiss maestro whose elegant play and sportsmanship have already cemented your place among the legends, but whose quest for perfection remains unquenched even as time marches on. Recent dreams have brought you visions of this ancient artifact, along with fragments of a map that seems to lead to its resting place. Your journey will take you through the hallowed grounds of tennis history, from the clay courts of Roland Garros to the hidden training grounds of forgotten champions, and finally to a secret chamber beneath Centre Court itself. Character Background: Develop a detailed background for Roger Federer in this quest. Describe his motivations for seeking the Golden Racket, his tennis skills and personal weaknesses, and any connections to the legends of the sport that came before him. Is he driven by a desire to extend his career, to secure his legacy as the greatest of all time, or perhaps by something more personal? What price might he be willing to pay to claim this artifact, and what challenges from rivals past and present might stand in his way?"
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```

