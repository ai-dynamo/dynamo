# Grove: Advanced Kubernetes Scheduling

Grove is an advanced Kubernetes scheduler and batch workload manager built on top of the Dynamo Kubernetes Platform. It enables sophisticated scheduling policies for multi-node GPU workloads, with special support for large-scale LLM inference deployments.

## Overview

Grove extends Kubernetes' default scheduling capabilities with:
- **Gang scheduling**: Ensures all pods in a workload start together or not at all
- **Topology-aware placement**: Optimizes pod placement based on network topology
- **Resource-aware scheduling**: Makes intelligent decisions based on GPU memory, compute capacity, and network bandwidth
- **Priority-based queueing**: Manages workload priorities and preemption policies

## Key Features

### PodGangSet
PodGangSet is Grove's primary scheduling primitive that groups related pods that must be scheduled together.

```yaml
apiVersion: grove.dynamo.ai/v1
kind: PodGangSet
metadata:
  name: llm-inference-gang
  namespace: default
spec:
  template:
    spec:
      containers:
      - name: worker
        image: dynamo/worker:latest
        resources:
          requests:
            nvidia.com/gpu: 1
  replicas: 8
  minAvailable: 8  # All pods must be schedulable
  scheduling:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: node-type
            operator: In
            values: ["gpu-compute"]
```

### PodClique
PodClique provides fine-grained control over pod co-location and anti-affinity rules within a gang.

```yaml
apiVersion: grove.dynamo.ai/v1
kind: PodClique
metadata:
  name: prefill-decode-clique
spec:
  selector:
    matchLabels:
      app: dynamo-worker
  topology:
    # Prefer pods to be co-located on the same rack
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            component: prefill
        topologyKey: topology.kubernetes.io/rack
```

## Deployment

### Prerequisites
- Kubernetes cluster with GPU nodes
- NVIDIA GPU Operator installed
- Node topology labels configured

### Install Grove Scheduler

```bash
# Install Grove CRDs and scheduler
kubectl apply -f https://github.com/ai-dynamo/grove/releases/latest/download/grove-crds.yaml
kubectl apply -f https://github.com/ai-dynamo/grove/releases/latest/download/grove-scheduler.yaml
```

### Configure Node Topology

Label your nodes with topology information:

```bash
# Label nodes with rack information
kubectl label node gpu-node-01 topology.kubernetes.io/rack=rack-1
kubectl label node gpu-node-02 topology.kubernetes.io/rack=rack-1
kubectl label node gpu-node-03 topology.kubernetes.io/rack=rack-2

# Label nodes with GPU types
kubectl label node gpu-node-01 accelerator=h100
kubectl label node gpu-node-02 accelerator=h100
kubectl label node gpu-node-03 accelerator=a100
```

## Integration with Dynamo

Grove integrates seamlessly with Dynamo's disaggregated serving architecture:

### Multi-Node Prefill/Decode Scheduling

```yaml
apiVersion: grove.dynamo.ai/v1
kind: PodGangSet
metadata:
  name: dynamo-multinode-serving
spec:
  template:
    metadata:
      labels:
        app: dynamo-worker
    spec:
      schedulerName: grove-scheduler
      containers:
      - name: dynamo-worker
        image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:latest
        env:
        - name: WORKER_TYPE
          value: "prefill"  # or "decode"
  replicas: 16
  minAvailable: 16
  scheduling:
    # Ensure all workers can communicate efficiently
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: network-tier
            operator: In
            values: ["high-bandwidth"]
```

## Best Practices

### Resource Planning
- Use `minAvailable: replicas` for strict gang scheduling
- Set appropriate resource requests and limits
- Consider network bandwidth requirements for multi-node workloads

### Topology Awareness
- Label nodes with rack, zone, and network topology information
- Use PodClique for fine-grained placement control
- Test different affinity rules to optimize for your workload

### Monitoring
Grove provides metrics for scheduling decisions:

```bash
# View Grove scheduler metrics
kubectl port-forward -n grove-system svc/grove-scheduler-metrics 8080:8080
curl localhost:8080/metrics | grep grove_
```

## Troubleshooting

### Common Issues

**Pods stuck in Pending state:**
- Check if sufficient resources are available across required nodes
- Verify node labels match gang affinity requirements
- Review Grove scheduler logs: `kubectl logs -n grove-system deployment/grove-scheduler`

**Gang scheduling not working:**
- Ensure `schedulerName: grove-scheduler` is set in pod specs
- Verify PodGangSet controller is running
- Check for resource conflicts with other scheduled workloads

For more detailed troubleshooting, see the [Grove Documentation](https://grove.dynamo.ai/docs).