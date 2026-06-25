# llm-d Router on-ramp

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

Choose tokenization mode first:

- **Option A (EPP-local tokenizer sidecar):** create `llm-d-hf-token`.
- **Option B (tokenize through existing vLLM/render endpoint):** no EPP HF secret required.

Create the secret only for Option A:

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
  name: qwen-router
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
  name: qwen-router
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  rules:
  - backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: qwen-router
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

llm-d provides NO guide how to upgrade existing vLLM workers. They need to consult many docs:

- [Chart defaults (gateway chart)](https://github.com/llm-d/llm-d-router/blob/main/config/charts/llm-d-router-gateway/values.yaml)
- [Shared routerlib defaults](https://github.com/llm-d/llm-d-router/blob/main/config/charts/routerlib/values.yaml)
- [Precise prefix plugin example](https://github.com/llm-d/llm-d-router/blob/main/deploy/config/epp-precise-prefix-cache-config.yaml)
- [Tokenizer + vLLM URL example](https://github.com/llm-d/llm-d-router/blob/main/deploy/config/sim-epp-tokenizer-vllm-http-config.yaml)

If I were to add this guide I would do the following:

Create `values-llmd-gateway.yaml` in your real deployment repo (for example in
the same repo where your Helm release values live). The YAML below is a
template example you should copy and customize.

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

The config above is **Option A** (EPP-local tokenizer sidecar).
For **Option B** (reuse an existing render endpoint), apply these changes:

```yaml
router:
  tokenizer:
    enabled: false
  epp:
    pluginsCustomConfig:
      precise-prefix-config.yaml: |
        apiVersion: llm-d.ai/v1alpha1
        kind: EndpointPickerConfig
        plugins:
        - type: token-producer
          parameters:
            modelName: "Qwen/Qwen3-32B"
            vllm:
              url: "http://vllm-render.<ns>.svc.cluster.local:8000"
        # keep the rest of the plugin config unchanged
```

Customize this template for your deployment:

- `router.modelServers.matchLabels`: set to labels on your vLLM worker pods.
- `router.modelServers.targetPorts[].number`: set to your worker serving port.
- `httpRoute.inferenceGatewayName`: set to your gateway name.
- `token-producer.parameters.modelName`: set to the exact served model id.
- `tokenProcessorConfig.blockSize`: must match worker `--block-size`.
- `kvEventsConfig.podDiscoveryConfig.socketPort`: must match worker KV event port.
- Keep `decode-filter` out for aggregated serving (this example is aggregated).

Install (local chart path):

```bash
helm upgrade -i qwen-router \
  llm-d-router/config/charts/llm-d-router-gateway \
  -n <ns> \
  -f values-llmd-gateway.yaml
```

Install (OCI chart):

```bash
helm upgrade -i qwen-router \
  oci://ghcr.io/llm-d/charts/llm-d-router-gateway-dev \
  -n <ns> \
  -f values-llmd-gateway.yaml \
  --version <router-chart-version>
```

Rendered YAMLs from the Helm chart (example, precise-prefix KV-aware + load-aware config):

```bash
helm template qwen-router \
  llm-d-router/config/charts/llm-d-router-gateway \
  --namespace default \
  -f values-llmd-gateway.yaml \
  -s templates/epp.yaml \
  -s templates/httproute.yaml
```

The only differences in Option A vs B are in two resources:

- `ConfigMap`: `token-producer.parameters.vllm.url`
- `Deployment`: sidecar presence (`vllm-render`), HF token env (Option A), and related sidecar cache volume

Rendered `ConfigMap` (shows precise-prefix KV-aware scorer + load scorers):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: qwen-router-epp
  namespace: default
data:
  precise-prefix-config.yaml: |
    apiVersion: llm-d.ai/v1alpha1
    kind: EndpointPickerConfig
    plugins:
    - type: token-producer
      parameters:
        modelName: "Qwen/Qwen3-32B"
        vllm:
          url: "http://localhost:8000" # option A; for option B use http://vllm-render.<ns>.svc.cluster.local:8000
    - type: endpoint-notification-source
    - type: metrics-data-source
    - type: core-metrics-extractor
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
      - pluginRef: prefix-cache-scorer
        weight: 2
      - pluginRef: kv-cache-utilization-scorer
        weight: 1
      - pluginRef: queue-scorer
        weight: 1
      - pluginRef: max-score-picker
```

Rendered `Service`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: qwen-router-epp
  namespace: default
  labels:
    app.kubernetes.io/name: qwen-router-epp
    app.kubernetes.io/version: "0.0.0"
spec:
  selector:
    llm-d-router-gateway: qwen-router-epp
  ports:
    - name: grpc-ext-proc
      protocol: TCP
      port: 9002
    - name: http-metrics
      protocol: TCP
      port: 9090
  type: ClusterIP
```

Rendered `Deployment` (Option A shown: EPP-local tokenizer sidecar):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen-router-epp
  namespace: default
  labels:
    app.kubernetes.io/name: qwen-router-epp
    app.kubernetes.io/version: "0.0.0"
    llm-d.ai/igw-mode: llm-d-router-gateway
spec:
  replicas: 1
  selector:
    matchLabels:
      llm-d-router-gateway: qwen-router-epp
  template:
    metadata:
      labels:
        llm-d-router-gateway: qwen-router-epp
        llm-d.ai/igw-mode: llm-d-router-gateway
    spec:
      serviceAccountName: qwen-router-epp
      containers:
        - name: epp
          image: ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:main
          args:
            - --pool-name
            - qwen-router
            - --pool-namespace
            - default
            - --pool-group
            - "inference.networking.k8s.io"
            - --zap-encoder
            - "json"
            - --config-file
            - "/config/precise-prefix-config.yaml"
            - --grpc-health-port
            - "9003"
            - --tracing=false
          volumeMounts:
            - name: plugins-config-volume
              mountPath: "/config"
        - name: vllm-render # This is for option A, remove for option B
          image: docker.io/vllm/vllm-openai-cpu:v0.19.1
          command:
            - vllm
            - launch
            - render
          args:
            - "Qwen/Qwen3-32B"
            - "--port=8000"
          env:
            - name: HF_TOKEN # This is for option A, remove for option B
              valueFrom:
                secretKeyRef:
                  name: llm-d-hf-token
                  key: HF_TOKEN
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          emptyDir: {}
        - name: plugins-config-volume
          configMap:
            name: qwen-router-epp
```

Rendered `InferencePool`:

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: qwen-router
  namespace: default
spec:
  targetPorts:
    - number: 8000
  appProtocol: "http"
  selector:
    matchLabels:
      app: "vllm-qwen"
  endpointPickerRef:
    name: qwen-router-epp
    port:
      number: 9002
    failureMode: FailOpen
```

Rendered `HTTPRoute`:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: qwen-router
  namespace: default
spec:
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: inference-gateway
  rules:
    - backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: qwen-router
      matches:
        - path:
            type: PathPrefix
            value: /
      timeouts:
        request: 300s
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
- Aggregated serving: do not include `decode-filter` in the plugin chain.
- Option A: create `llm-d-hf-token` for tokenizer sidecar.
- Option B: set `router.tokenizer.enabled=false` and point `token-producer.parameters.vllm.url` at your existing render endpoint.

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
          - name: HF_TOKEN             # NEW (optional): needed only for gated/private models
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
