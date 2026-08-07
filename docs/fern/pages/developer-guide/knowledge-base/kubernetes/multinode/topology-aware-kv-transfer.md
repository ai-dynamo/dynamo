---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Topology-Aware KV Transfer
subtitle: Keep disaggregated prefill and decode KV-cache transfers within a selected topology domain
---

**Experimental.** Topology-aware KV transfer lets a disaggregated NVIDIA Dynamo deployment route decode requests toward workers that share the selected prefill worker's topology domain, such as zone or rack. This reduces slow cross-domain KV-cache transfers when prefill and decode workers exchange KV data over NIXL.

Use this feature when:

- Your deployment uses separate prefill and decode workers.
- Your cluster exposes useful node labels, such as `topology.kubernetes.io/zone` or a rack/block label.
- Same-domain KV transfer is required for correctness or strongly preferred for latency and bandwidth.

This page covers the Kubernetes operator path. For router and runtime behavior, see [Router Topology-Aware KV Transfer](../../modular-components/router/topology-aware-kv-transfer.md).
For RDMA/NIXL transport setup, see [Disagg Communication](../kubernetes-operator/disagg-communication.md).

## How It Works

```mermaid
flowchart LR
    DGD["DGD spec.experimental.kvTransferPolicy"] --> Operator["Operator configures workers"]
    Operator --> Source{"Topology source"}
    Source --> Label["labelKey"]
    Source --> Grove["clusterTopologyName"]
    Label --> Controller["Topology label controller"]
    Grove --> Controller
    Node["Node topology labels"] --> Controller
    Controller --> Pod["Worker pod label"]
    Pod --> Volume["/etc/dynamo/topology/<domain> files"]
    Volume --> Runtime["Worker publishes ModelRuntimeConfig topology metadata"]
    Runtime --> Prefill["Prefill router derives decode constraints"]
    Prefill --> Decode["Decode router selects same or preferred topology"]
```

Set exactly one topology source in `spec.experimental.kvTransferPolicy`:

- `labelKey` copies one Kubernetes node label onto the worker pod under the same key.
- `clusterTopologyName` uses the domain-to-node-label mappings from a Grove topology resource. The controller copies every topology level onto the worker pod under `nvidia.com/dynamo-topology.<domain>` labels.

For either source, the operator:

- Annotates worker pods with the selected topology source.
- Runs a topology-label controller that copies node topology values onto the worker pod after scheduling.
- Projects the copied pod labels into `/etc/dynamo/topology/<domain>` files with a Downward API volume.
- Injects worker environment variables that tell the backend runtime which topology domain and enforcement policy to publish.

The frontend does not read this policy from its own environment. Workers publish the topology metadata in their `ModelRuntimeConfig`; the router reads it from runtime discovery.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Disaggregated serving | Separate prefill and decode worker services. |
| KV router | The frontend should use `DYN_ROUTER_MODE=kv`. |
| Topology source | Set exactly one of `labelKey` or `clusterTopologyName`. |
| Node topology labels | Every worker node must carry the configured `labelKey`, or every source label defined by the Grove topology resource. |
| Grove pathway | Required only for `clusterTopologyName`. Install and enable Grove, and do not set `nvidia.com/enable-grove: "false"` on the DGD. See [Grove](grove.md). |
| Dynamo operator | The operator must include topology-label controller and node-read RBAC. |
| KV transfer transport | RDMA, EFA, or another NIXL-compatible transport should already be configured for production disaggregated deployments. |

Confirm that the label you plan to use exists on worker nodes:

```bash
kubectl get nodes -L topology.kubernetes.io/zone
```

## Required Same-Domain Routing

`enforcement: required` constrains decode worker selection to workers whose topology value matches the selected prefill worker for the configured domain. If no decode worker satisfies the generated constraint, the router fails the request instead of silently crossing the domain.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-disagg-zone
spec:
  experimental:
    kvTransferPolicy:
      labelKey: topology.kubernetes.io/zone
      domain: zone
      enforcement: required
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          env:
          - name: DYN_ROUTER_MODE
            value: kv
  - name: VllmPrefillWorker
    type: worker
    replicas: 2
    podTemplate:
      spec:
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "prefill"]
          envFrom:
          - secretRef:
              name: hf-token-secret
          resources:
            limits:
              nvidia.com/gpu: "1"
  - name: VllmDecodeWorker
    type: worker
    replicas: 2
    podTemplate:
      spec:
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "decode"]
          envFrom:
          - secretRef:
              name: hf-token-secret
          resources:
            limits:
              nvidia.com/gpu: "1"
```

`enforcement` defaults to `required` when omitted.

> [!IMPORTANT]
> `required` is a decode-routing constraint, not a capacity planner. The `DynamoGraphDeployment` author or cluster administrator must ensure that every topology domain that can receive prefill workers also has sufficient same-domain decode capacity. If a domain has prefill workers but no matching decode workers, or too little decode capacity, the router cannot spill to another domain without violating the policy.

### Use a Grove Topology Source

To use topology levels defined by Grove, set `clusterTopologyName` instead of `labelKey`. The selected `domain` must exist in the referenced topology resource.

```yaml
spec:
  experimental:
    kvTransferPolicy:
      clusterTopologyName: my-cluster-topology
      domain: rack
      enforcement: required
```

This path requires Grove to be enabled in the operator and for the DGD. The operator projects every topology level from the referenced resource, while the router uses only the selected `domain` for the KV-transfer constraint.

### Capacity Planning Across Domains

Plan prefill and decode capacity per topology domain before enabling `enforcement: required`. For example, assume:

- Two availability zones: `az-1` and `az-2`.
- The target fleet is 60 prefill workers and 120 decode workers.
- The fleet should be split evenly across the two zones.
- The target prefill-to-decode ratio is 1:2 in each zone.

That means each zone should run 30 prefill workers and 60 decode workers:

| Zone | Prefill workers | Decode workers | Ratio |
|------|-----------------|----------------|-------|
| `az-1` | 30 | 60 | 1:2 |
| `az-2` | 30 | 60 | 1:2 |

In a `DynamoGraphDeployment`, express this as separate prefill and decode components per zone. Pin each component to its zone and set `kvTransferPolicy.enforcement` to `required` so the router refuses cross-zone decode selection. The DGD author or cluster administrator must ensure each zone has enough schedulable capacity for its pinned replicas. Worker command and args are omitted here; configure each worker for prefill or decode mode as in the base disaggregated serving manifest:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-disagg-zone-capacity
spec:
  experimental:
    kvTransferPolicy:
      labelKey: topology.kubernetes.io/zone
      domain: zone
      enforcement: required
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          env:
          - name: DYN_ROUTER_MODE
            value: kv
  - name: VllmPrefillWorkerAz1
    type: worker
    replicas: 30
    podTemplate:
      spec:
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
              - matchExpressions:
                - key: topology.kubernetes.io/zone
                  operator: In
                  values: ["az-1"]
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          envFrom:
          - secretRef:
              name: hf-token-secret
  - name: VllmDecodeWorkerAz1
    type: worker
    replicas: 60
    podTemplate:
      spec:
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
              - matchExpressions:
                - key: topology.kubernetes.io/zone
                  operator: In
                  values: ["az-1"]
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          envFrom:
          - secretRef:
              name: hf-token-secret
  - name: VllmPrefillWorkerAz2
    type: worker
    replicas: 30
    podTemplate:
      spec:
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
              - matchExpressions:
                - key: topology.kubernetes.io/zone
                  operator: In
                  values: ["az-2"]
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          envFrom:
          - secretRef:
              name: hf-token-secret
  - name: VllmDecodeWorkerAz2
    type: worker
    replicas: 60
    podTemplate:
      spec:
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
              - matchExpressions:
                - key: topology.kubernetes.io/zone
                  operator: In
                  values: ["az-2"]
        containers:
        - name: main
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
          envFrom:
          - secretRef:
              name: hf-token-secret
```

## Preferred Same-Domain Routing

`enforcement: preferred` keeps all decode workers eligible but biases worker selection toward the same topology domain.

```yaml
spec:
  experimental:
    kvTransferPolicy:
      labelKey: topology.kubernetes.io/zone
      domain: zone
      enforcement: preferred
      preferredWeight: 0.85
```

`preferredWeight` is required with `enforcement: preferred`. It must be between `0` and `1`. A higher value creates a stronger same-domain preference, but it is not a probability and does not guarantee same-domain selection.

## Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `labelKey` | One topology source | Kubernetes node label key to copy onto worker pods, for example `topology.kubernetes.io/zone`. Mutually exclusive with `clusterTopologyName`. |
| `clusterTopologyName` | One topology source | Name of a Grove topology resource. Requires the Grove pathway and is mutually exclusive with `labelKey`. |
| `domain` | Yes | Logical topology domain published by workers, for example `zone` or `rack`. Must match `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`; with `clusterTopologyName`, it must name a level in the referenced resource. |
| `enforcement` | No | `required` or `preferred`. Defaults to `required`. |
| `preferredWeight` | Only with `preferred` | Bias weight from `0` to `1`; only valid with `enforcement: preferred`. |

> [!IMPORTANT]
> `kvTransferPolicy` is immutable after the DGD is created. To add, remove, or change the policy, delete and recreate the DGD.

The runtime uses `domain`, not the Kubernetes label key, when creating routing constraints. For example, `labelKey: topology.kubernetes.io/zone` and `domain: zone` produce worker topology metadata like:

```json
{
  "topology_domains": {
    "zone": "us-east-1a"
  },
  "kv_transfer_domain": "zone",
  "kv_transfer_enforcement": "required"
}
```

## Verify the Deployment

After the DGD creates worker pods, verify the operator pipeline from the selected topology source to the runtime topology files.

```bash
export NAMESPACE=<namespace>
export POD=<worker-pod>
```

For a `labelKey` source, verify the source annotation and copied label:

```bash
export LABEL_KEY=<label-key>

kubectl get pod "$POD" -n "$NAMESPACE" \
  -o jsonpath='{.metadata.annotations.nvidia\.com/topology-label-key}{"\n"}'

kubectl get pod "$POD" -n "$NAMESPACE" \
  -o go-template='{{ index .metadata.labels "'"$LABEL_KEY"'" }}{{ "\n" }}'
```

For a `clusterTopologyName` source, verify the source annotation and the canonical label for the selected domain:

```bash
export DOMAIN=<domain>
TOPOLOGY_LABEL="nvidia.com/dynamo-topology.$DOMAIN"

kubectl get pod "$POD" -n "$NAMESPACE" \
  -o jsonpath='{.metadata.annotations.nvidia\.com/topology-cluster-topology-name}{"\n"}'

kubectl get pod "$POD" -n "$NAMESPACE" \
  -o go-template='{{ index .metadata.labels "'"$TOPOLOGY_LABEL"'" }}{{ "\n" }}'
```

For either source, inspect the projected topology files:

```bash
kubectl exec "$POD" -n "$NAMESPACE" -- \
  sh -c 'find /etc/dynamo/topology -maxdepth 1 -type f -print -exec cat {} \;'
```

Expected results:

- The source annotation contains the configured `labelKey` or `clusterTopologyName`.
- The worker pod has the copied topology label or canonical Grove topology labels.
- `/etc/dynamo/topology/<domain>` exists for the selected domain and contains the topology value. The Grove path also projects files for the other topology levels.

Worker logs should include topology config during startup:

```bash
kubectl logs "$POD" -n "$NAMESPACE" | grep -i "Topology config"
```

## Troubleshooting

### Pod Has No Copied Topology Label

For a `labelKey` source, check whether the node has the configured label:

```bash
export LABEL_KEY=<label-key>
NODE=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')
kubectl get node "$NODE" \
  -o go-template='{{ index .metadata.labels "'"$LABEL_KEY"'" }}{{ "\n" }}'
```

For a `clusterTopologyName` source, verify that the referenced Grove topology resource exists, contains the selected `domain`, and maps each domain to a label present on the node. Also verify that the DGD does not set `nvidia.com/enable-grove: "false"`.

If the label is missing, the topology-label controller emits a warning event with reason `TopologyLabelMissing` and leaves topology metadata unavailable for that worker.

```bash
kubectl get events -n "$NAMESPACE" \
  --field-selector involvedObject.name="$POD",reason=TopologyLabelMissing
```

### Worker Exits While Waiting for Topology

When topology is enabled, the worker waits for the transfer-domain file to appear and contain data. If it stays empty, check:

- `spec.experimental.kvTransferPolicy.domain` matches the projected file name.
- The configured `labelKey`, or the source label for the selected Grove topology domain, exists on the worker's node.
- The worker pod has the source annotation for `labelKey` or `clusterTopologyName`.
- The topology-label controller is running and has node `get` RBAC.

### Required Policy Fails Requests

With `enforcement: required`, decode routing fails if no decode worker has the same generated topology taint as the selected prefill worker. Verify both prefill and decode workers publish the same `domain`, and that each domain where prefill workers can be selected has enough matching decode workers for the expected p/d ratio.

Use `preferred` while validating a heterogeneous rollout if cross-domain routing is acceptable during partial capacity.

## Relationship to Topology Aware Scheduling

[Topology Aware Scheduling](topology-aware-scheduling.md) controls where Kubernetes places pods. Topology-aware KV transfer controls how Dynamo routes between already-running prefill and decode workers.

Use them together when possible:

- Topology Aware Scheduling keeps workers placed inside useful topology boundaries.
- Topology-aware KV transfer prevents the router from choosing a decode worker outside the selected prefill worker's transfer domain.
