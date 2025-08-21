# Examples of using Dynamo Platform

## Serving examples locally

Follow individual examples under components/backends/ to serve models locally.

For example follow the [vLLM Backend Example](../../components/backends/vllm/README.md)

For a basic GPU - unaware example see the [Hello World Example](../../examples/runtime/hello_world/README.md)

## Deploying Examples to Kubernetes

First you need to install the Dynamo Cloud Platform. Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you.
Before you can deploy your graphs, you need to deploy the Dynamo Runtime and Dynamo Cloud images. This is a one-time action, only necessary the first time you deploy a DynamoGraph.

### Instructions for Dynamo User
If you are a **👤 Dynamo User** first follow the [Quickstart Guide](../guides/dynamo_deploy/quickstart.md) first.

### Instructions for Dynamo Contributor
If you are a **🧑‍💻 Dynamo Contributor** you may have to rebuild the dynamo platform images as the code evolves.
For more details read the [Cloud Guide](../guides/dynamo_deploy/dynamo_cloud.md)
Read more on deploying Dynamo Cloud read [deploy/cloud/helm/README.md](../../deploy/cloud/helm/README.md).


### Deploying a particular example

```bash
# Set your dynamo root directory
cd <root-dynamo-folder>
export PROJECT_ROOT=$(pwd)
export NAMESPACE=<your-namespace> # the namespace you used to deploy Dynamo cloud to.
```

Deploying an example consists of the simple `kubectl apply -f ... -n ${NAMESPACE}` command. For example:

```bash
kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}
```

You can use `kubectl get dynamoGraphDeployment -n ${NAMESPACE}` to view your deployment.
You can use `kubectl delete dynamoGraphDeployment <your-dep-name> -n ${NAMESPACE}` to delete the deployment.

We provide a Custom Resource yaml file for many examples under the `components/backends/<backend-name>/deploy/`folder.
Consult the examples below for the CRs for your specific inference backend.

[View SGLang k8s](../../components/backends/sglang/deploy/README.md)

[View vLLM K8s](../../components/backends/vllm/deploy/README.md)

[View TRTLLM k8s](../../components/backends/trtllm/deploy/README.md)

**Note 1** Example Image

The examples use a prebuilt image from the `nvcr.io` registry.
You can build your own image and update the image location in your CR file prior to applying.
You could build your own image using

```bash
./container/build.sh --framework <your-inference-framework>
```

For example for the `sglang` run
```bash
./container/build.sh --framework sglang
```

Then you would need to overwrite the image in the examples.

```bash
extraPodSpec:
        mainContainer:
          image: <image-in-your-$DYNAMO_IMAGE>
```

**Note 2**
Setup port forward if needed when deploying to Kubernetes.

List the services in your namespace:

```bash
kubectl get svc -n ${NAMESPACE}
```
Look for one that ends in `-frontend` and use it for port forward.

```bash
SERVICE_NAME=$(kubectl get svc -n ${NAMESPACE} -o name | grep frontend | sed 's|.*/||' | sed 's|-frontend||' | head -n1)
kubectl port-forward svc/${SERVICE_NAME}-frontend 8080:8080 -n ${NAMESPACE}
```

Consult the [Port Forward Documentation](https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/)


