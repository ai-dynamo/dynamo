# vLLM v0 Deployment

## Available Deployments

- **`agg.yaml`** - Aggregated (monolithic) deployment
- **`disagg.yaml`** - Disaggregated deployment
- **`disagg_planner.yaml`** - Disaggregated deployment with SLA-based autoscaling planner

## Quick Start

Set your environment variables:
```bash
export NAMESPACE=your-namespace
```

### Simple Deployment
```bash
envsubst < agg.yaml | kubectl apply -f -
```

### SLA Planner Deployment

**Step 1: Run profiling (required)**
```bash
envsubst < profiling_pvc.yaml | kubectl apply -f -
envsubst < profile_sla_job_disagg.yaml | kubectl apply -f -
```

**Step 2: Wait for profiling to complete**
```bash
kubectl get jobs -n $NAMESPACE
kubectl logs job/profile_sla_disagg -n $NAMESPACE
```

**Step 3: Deploy planner**
```bash
envsubst < disagg_planner.yaml | kubectl apply -f -
```

## Monitoring

```bash
kubectl get pods -n $NAMESPACE
kubectl logs -n $NAMESPACE deployment/disagg-planner-planner
```

## Documentation

For detailed configuration and architecture information, see:
- [Load Planner Documentation](../../../docs/architecture/load_planner.md)
- [SLA Planner Documentation](../../../docs/architecture/sla_planner.md)
- [Planner Benchmark Examples](../../../docs/guides/planner_benchmark/README.md)
