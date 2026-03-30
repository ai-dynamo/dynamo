# Platform Deployment Details

## Helm release

```
Name:        dynamo-platform
Namespace:   failover-e2e-test
Chart:       dynamo-platform-1.0.0
Revision:    5
Status:      deployed
```

## Helm values

```yaml
dynamo-operator:
  controllerManager:
    manager:
      image:
        repository: dynamoci.azurecr.io/ai-dynamo/kubernetes-operator
        tag: multinode-failover-0b8cb8fc1
  etcdAddr: http://etcd.failover-e2e-test.svc.cluster.local:2379
  grove:
    enabled: true
  namespaceRestriction:
    enabled: true
global:
  grove:
    enabled: true
```

## Install/upgrade command

```bash
helm dependency build deploy/helm/charts/platform

helm upgrade dynamo-platform deploy/helm/charts/platform \
  -n failover-e2e-test --install \
  --set dynamo-operator.controllerManager.manager.image.repository=dynamoci.azurecr.io/ai-dynamo/kubernetes-operator \
  --set dynamo-operator.controllerManager.manager.image.tag=<OPERATOR_TAG> \
  --set dynamo-operator.controllerManager.manager.image.pullPolicy=Always \
  --set dynamo-operator.etcdAddr=http://etcd.failover-e2e-test.svc.cluster.local:2379 \
  --set dynamo-operator.namespaceRestriction.enabled=true \
  --set dynamo-operator.grove.enabled=true \
  --set global.grove.enabled=true \
  --wait --timeout 5m
```

## Images

| Component | Image | Tag | Notes |
|-----------|-------|-----|-------|
| Operator | dynamoci.azurecr.io/ai-dynamo/kubernetes-operator | multinode-failover-0b8cb8fc1 | Current. Needs rebuild for watch-based harness. |
| Engine | dynamoci.azurecr.io/ai-dynamo/dynamo | multinode-failover-650234f660-vllm-runtime | Contains KV cache oversizing fix. |

## Cluster dependencies

| Component | Namespace | Version | Notes |
|-----------|-----------|---------|-------|
| Grove | grove | v0.1.0-alpha.6 | Required for multinode (PodCliqueSet orchestration) |
| KAI Scheduler | kai-scheduler | v0.13.0-rc1 | Required for multinode (GPU-aware scheduling) |
| etcd | failover-e2e-test | v3.5.21 | Deployed as standalone pod, used for harness coordination + operator |
| DRA GPU driver | nvidia-dra-driver-gpu | - | Required for shared GPU access (failover) |

## Cluster

- AKS cluster: `dynamo-exp` (context: `dynamo-exp-6f8d9a`)
- GPU nodes: 2x `aks-a100exp-11297970-vmss000{000,001}` (8x A100-SXM4-80GB each)
- Tests target vmss000001
