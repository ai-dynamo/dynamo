# llm-d Router on-ramp (raw vLLM workers, no extra control plane)

## Example progression from vLLM to vLLM + llm-d + GAIE

### Initial Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-qwen
  template:
    metadata:
      labels:
        app: vllm-qwen
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "Qwen/Qwen3-32B"
        ports:
        - name: http
          containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-qwen
spec:
  selector:
    app: vllm-qwen
  ports:
  - name: http
    port: 8000
    targetPort: 8000
```

### 1. Prerequisites

Set up a Gateway-API gateway + Inference Extension (GAIE) if you don't have one yet. Follow the instructions for your gateway (e.g. AgentGateway) and Inference Extension.

Create the HuggingFace token secret.

```bash
kubectl create secret generic llm-d-hf-token --from-literal=HF_TOKEN=<your-token>
```

### 2. Create / Wire the InferencePool and HTTPRoute

Point the `InferencePool` at your vLLM workers:

- `spec.selector` must match worker pod labels.
- `spec.targetPorts[].number` must match vLLM serving port (usually `8000`).
- `spec.endpointPickerRef` must point to the llm-d EPP service and ext-proc port (`9002`).

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: qwen-pool
spec:
  selector:
    matchLabels:
      app: vllm-qwen
  targetPorts:
  - number: 8000
  endpointPickerRef:
    kind: Service
    name: qwen-router-epp
    port:
      number: 9002
```

Attach your `HTTPRoute` to the gateway and target the pool:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: qwen-route
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: qwen-pool
```

### 3. Update vLLM deployment

Ensure worker labels match `InferencePool.spec.selector`:

```bash
kubectl -n <ns> label deployment <vllm-deployment> app=vllm-qwen --overwrite
```

Patch workers to publish KV events and align block size:

```yaml
args:
  - "--served-model-name"
  - "Qwen/Qwen3-32B"
  - "--port"
  - "8000"
  - "--enable-prefix-caching"
  - "--block-size"
  - "64"
  - "--kv-events-config"
  - '{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557","topic":"kv@$(POD_IP):$(POD_PORT)@Qwen/Qwen3-32B"}'
ports:
  - name: http
    containerPort: 8000
  - name: kv-events
    containerPort: 5557
```

Required alignment:

- vLLM `--block-size` must match `tokenProcessorConfig.blockSize`.
- vLLM KV topic should match `kvEventsConfig.topicFilter` (default prefix `kv@`).

### 4. Add llm-d EPP to your deployment

Deploy llm-d Router in gateway mode with a precise-prefix config.

Create `values-llmd-gateway.yaml`:

```yaml
router:
  modelServers:
    matchLabels:
      app: vllm-qwen
    targetPorts:
      - number: 8000

  epp:
    replicas: 1
    pluginsConfigFile: precise-prefix-config.yaml
    pluginsCustomConfig:
      precise-prefix-config.yaml: |
        apiVersion: llm-d.ai/v1alpha1
        kind: EndpointPickerConfig
        plugins:
        - type: token-producer
          parameters:
            modelName: "Qwen/Qwen3-32B"
            vllm:
              url: "http://localhost:8000"
        - type: endpoint-notification-source
        - type: metrics-data-source
        - type: core-metrics-extractor
        - type: decode-filter
        - type: precise-prefix-cache-producer
          parameters:
            tokenProcessorConfig:
              blockSize: 64
            kvEventsConfig:
              topicFilter: "kv@"
              discoverPods: true
              podDiscoveryConfig:
                socketPort: 5557
        - type: prefix-cache-scorer
          parameters:
            prefixMatchInfoProducerName: precise-prefix-cache-producer
        - type: kv-cache-utilization-scorer
        - type: queue-scorer
        - type: max-score-picker
        dataLayer:
          sources:
          - pluginRef: metrics-data-source
            extractors:
            - pluginRef: core-metrics-extractor
          - pluginRef: endpoint-notification-source
            extractors:
            - pluginRef: precise-prefix-cache-producer
        schedulingProfiles:
        - name: default
          plugins:
          - pluginRef: decode-filter
          - pluginRef: prefix-cache-scorer
            weight: 2
          - pluginRef: kv-cache-utilization-scorer
            weight: 1
          - pluginRef: queue-scorer
            weight: 1
          - pluginRef: max-score-picker

  tokenizer:
    enabled: true
    modelName: "Qwen/Qwen3-32B"

httpRoute:
  create: true
  inferenceGatewayName: inference-gateway

provider:
  name: none
```

Install:

```bash
helm upgrade -i qwen-router \
  /home/atchernych/code/gaie/llm-d-router/config/charts/llm-d-router-gateway \
  -n <ns> \
  -f values-llmd-gateway.yaml
```

If `InferencePool` and `HTTPRoute` are already managed separately, set:

- `httpRoute.create=false`
- keep `endpointPickerRef` pointing at the deployed EPP service on port `9002`

## Test

```bash
# terminal 1
kubectl -n <gateway-ns> port-forward svc/inference-gateway 8000:80

# terminal 2
curl --max-time 20 -sS "http://localhost:8000/v1/models" | jq .

curl --max-time 120 -sS "http://localhost:8000/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role":"user","content":"hello"}]
  }' | jq .
```

## Configuration checklist

- `InferencePool.spec.selector.matchLabels` == worker pod labels.
- `InferencePool.spec.targetPorts[].number` == worker serve port (`8000`).
- `InferencePool.spec.endpointPickerRef` == llm-d EPP service + ext-proc port (`9002`).
- worker `--kv-events-config.endpoint` == `kvEventsConfig.podDiscoveryConfig.socketPort`.
- worker KV topic matches `kvEventsConfig.topicFilter`.
- worker `--block-size` == `tokenProcessorConfig.blockSize`.
- `token-producer.parameters.modelName` matches the served model/tokenizer.

## Final vLLM workers deployment

After applying step 3 changes, a full worker deployment looks like this:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-qwen
  template:
    metadata:
      labels:
        app: vllm-qwen
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "--model"
          - "Qwen/Qwen3-32B"
          - "--served-model-name"      # NEW: pin the OpenAI model id
          - "Qwen/Qwen3-32B"
          - "--port"                   # NEW: explicit worker serving port
          - "8000"
          - "--enable-prefix-caching"  # NEW: required to emit prefix-aware KV events
          - "--block-size"             # NEW: must match llm-d tokenProcessorConfig.blockSize
          - "64"
          - "--kv-events-config"       # NEW: publish KV events on a ZMQ socket
          - '{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557","topic":"kv@$(POD_IP):$(POD_PORT)@Qwen/Qwen3-32B"}'
        ports:
          - name: http
            containerPort: 8000
          - name: kv-events            # NEW: KV event port the llm-d EPP subscribes to
            containerPort: 5557
        env:
          - name: HF_TOKEN             # NEW: lets worker pull gated/private models
            valueFrom:
              secretKeyRef:
                name: llm-d-hf-token
                key: HF_TOKEN
        resources:
          limits:
            nvidia.com/gpu: "1"
      tolerations:
        - effect: NoSchedule
          key: nvidia.com/gpu
          operator: Exists
```
