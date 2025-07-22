# vLLM Deployment

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
kubectl apply -f agg.yaml
```

### SLA Planner Deployment

**Optional Step 0: add a kubernetes secret**

```bash
kubectl create secret docker-registry nvcr-imagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=<nvapi key> \
  -n $NAMESPACE
```

**Step 1: Build your own vLLM image for profiling**

```bash
# in the project's root folder
./container/build.sh --framework VLLM
# Tag and push to your container registry
```

Replace the `image` within `profile_sla_job.yaml` with the tag of the image you pushed.

**Step 2: Run profiling (required)**
```bash
cd examples/vllm/deploy
envsubst < profiling_pvc.yaml | kubectl apply -f -
envsubst < profile_sla_sa.yaml | kubectl apply -f -
envsubst < profile_sla_rbac.yaml | kubectl apply -f -
envsubst < profile_sla_binding.yaml | kubectl apply -f -
envsubst < profile_sla_job.yaml | kubectl apply -f -
```

**Step 3: Wait for profiling to complete**
```bash
kubectl get jobs -n $NAMESPACE
kubectl logs job/profile_sla -n $NAMESPACE
```

**Step 4: Deploy planner (work in progress)**
```bash
envsubst < disagg_planner.yaml | kubectl apply -f -
```

## Monitoring

```bash
kubectl get pods -n $NAMESPACE
kubectl logs -n $NAMESPACE deployment/disagg-planner-planner
```

## RBAC Configuration

The SLA profiling job requires specific Kubernetes permissions to manage DynamoGraphDeployment resources and access cluster information. The RBAC setup consists of:

- **`profile_sla_sa.yaml`** - Service account with image pull secret for NVIDIA Container Registry access
- **`profile_sla_rbac.yaml`** - ClusterRole defining required permissions for managing deployments and accessing cluster resources
- **`profile_sla_binding.yaml`** - ClusterRoleBinding that associates the ClusterRole with the service account

All three files are necessary:
1. The service account provides identity and image pull credentials
2. The ClusterRole defines what operations are allowed
3. The ClusterRoleBinding connects the permissions to the service account

## Troubleshooting

### Image Pull Authentication Errors

If you see `ErrImagePull` or `ImagePullBackOff` errors with 401 unauthorized messages:

1. Ensure the `nvcr-imagepullsecret` exists in your namespace:
   ```bash
   kubectl get secret nvcr-imagepullsecret -n $NAMESPACE
   ```

2. Verify the service account was created with the image pull secret:
   ```bash
   kubectl get serviceaccount profile-sla-sa -n $NAMESPACE -o yaml
   ```

3. The service account should show `imagePullSecrets` containing `nvcr-imagepullsecret`.

## Documentation

For detailed configuration and architecture information, see:
- [SLA Planner Documentation](../../../docs/architecture/sla_planner.md)
- [Planner Benchmark Examples](../../../docs/guides/planner_benchmark/README.md)
