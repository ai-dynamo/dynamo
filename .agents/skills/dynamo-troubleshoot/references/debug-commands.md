# Debug Commands Reference

Reference commands the skill uses, grouped by surface. Match against
[`dynamo-deploy/SKILL.md`](../../dynamo-deploy/SKILL.md) Command Safety
section before running any of these — DESTRUCTIVE commands require
explicit confirmation per.

---

## CRD-level

```bash
kubectl get dgd -A
kubectl get dgdr -A
kubectl get dgdsa -A
kubectl get dcd -A
kubectl get dynamomodel -A
kubectl get dynamocheckpoint -A
kubectl get dynamoworkermetadata -A
```

## Single resource

```bash
kubectl get dgd <name> -n <ns> -o yaml
kubectl describe dgd <name> -n <ns>
kubectl get dgd <name> -n <ns> -o jsonpath='{.status.conditions}'
```

## DGDR profiling Job

```bash
kubectl get dgdr <name> -n <ns> -o jsonpath='{.status.phase}'
kubectl get dgdr <name> -n <ns> -o jsonpath='{.status.profilingPhase}'

# Profiling pod (created by the Job).
kubectl get pods -n <ns> -l nvidia.com/dgdr-name=<name>
kubectl logs -f <profiling-pod> -n <ns>

# Generated DGD spec (autoApply: false).
kubectl get dgdr <name> -n <ns> -o jsonpath='{.status.profilingResults.selectedConfig}'
```

## Pods

```bash
kubectl get pods -n <ns> -l nvidia.com/dgd-name=<name> -o wide
kubectl describe pod <pod> -n <ns>
kubectl logs <pod> -n <ns>
kubectl logs <pod> -n <ns> --previous
kubectl logs <pod> -n <ns> --since=10m
kubectl exec <pod> -n <ns> -- nvidia-smi
kubectl exec <pod> -n <ns> -- env | grep -E "HF_TOKEN|UCX|NIXL"
```

## Events

```bash
kubectl get events -n <ns> --sort-by='.lastTimestamp' | tail -50
kubectl get events -n <ns> --field-selector involvedObject.name=<resource-name>
kubectl get events -n <ns> --field-selector involvedObject.kind=DynamoGraphDeploymentRequest
```

## Operator

```bash
kubectl get deploy -n <ns> -l app.kubernetes.io/name=dynamo-operator
kubectl logs -n <ns> deploy/dynamo-operator
kubectl logs -n <ns> deploy/dynamo-operator --since=15m
kubectl logs -n <ns> deploy/dynamo-operator | grep -i "<resource-name>"
```

## Webhook health

```bash
kubectl get apiservice | grep nvidia.com
kubectl get validatingwebhookconfigurations | grep dynamo
kubectl get mutatingwebhookconfigurations | grep dynamo
# Cluster admin: test the webhook with a dry-run apply.
kubectl apply --dry-run=server -f <known-good-dgdr.yaml> 2>&1 | head -3
```

## etcd / NATS (platform deps)

```bash
kubectl get pods -n <ns> -l app.kubernetes.io/name=etcd
kubectl get pods -n <ns> -l app.kubernetes.io/name=nats
kubectl exec -n <ns> etcd-0 -- etcdctl endpoint status -w table 2>/dev/null
kubectl exec -n <ns> <nats-box-pod> -- nats stream report
```

## Frontend /v1/models probe

```bash
kubectl port-forward -n <ns> svc/<frontend-svc> 8000:8000 &
PF_PID=$!
trap "kill $PF_PID 2>/dev/null" EXIT
sleep 2
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## Metrics

```bash
# Frontend metrics endpoint.
kubectl port-forward -n <ns> svc/<frontend-svc> 8001:8001 &
curl -s http://localhost:8001/metrics | grep -E "ttft|itl|request_total"

# KV-router metrics (when a router is in the graph).
kubectl port-forward -n <ns> svc/<router-svc> 8002:8002 &
curl -s http://localhost:8002/metrics | grep -i "kv"
```

## Helm

```bash
helm list -A | grep dynamo
helm status dynamo-platform -n <ns>
helm get values dynamo-platform -n <ns>
helm get manifest dynamo-platform -n <ns>
```
