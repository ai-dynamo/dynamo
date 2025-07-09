Steps I followed:

Pre-requisite:

Deploy an LLM example (agg first). [LLM deployment guide](../../examples/llm/README.md#kubernetes-deployment) has detailed instructions.

1. Deploy Prometheus:

```bash
kubectl apply -f prometheus-config.yaml
kubectl apply -f prometheus-deployment.yaml
kubectl apply -f prometheus-service.yaml
```

2. Profiling PVC:

```bash
kubectl apply -f profiling-pvc.yaml
```

3. Run the job:

```bash
kubectl apply -f profile-sla-job.yaml
```

To get the image mentioned in line 13:

```bash
# in the main dynamo directory
export DOCKER_SERVER=nvcr.io/nvidian/nim-llm-dev
export IMAGE_TAG=dep-178.1 # or whatever tag
./container/build.sh --target runtime
docker tag dynamo:latest-vllm-runtime $DOCKER_SERVER/dynamo-base-docker-llm:$IMAGE_TAG
docker push $DOCKER_SERVER/dynamo-base-docker-llm:$IMAGE_TAG
```

4. [Not tested yet] Deploy planner-sla and replace the existing LLM planner with it. Not sure whether we can directly apply a DynamoComponentDeployment (ideal) or whether we need to do planner-sla-config.yaml and planner-sla-deployment.yaml (but then how would we attach it to the existing graph?)
