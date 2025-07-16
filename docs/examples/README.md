# Examples of using Dynamo Platform

## Serving examples locally

TODO: Follow individual examples to serve models locally.


## Deploying Examples to Kubernetes

First you need to install the Dynamo Cloud Platform. Dynamo Cloud acts as an orchestration layer between the end user and Kubernetes, handling the complexity of deploying your graphs for you.
Before you can deploy your graphs, you need to deploy the Dynamo Runtime and Dynamo Cloud images. This is a one-time action, only necessary the first time you deploy a DynamoGraph.

### Instructions for Dynamo User
If you are a **👤 Dynamo User** first follow the [Quickstart Guide](../guides/dynamo_deploy/quickstart.md) first.

### Instructions for Dynamo Contributor
If you are a **🧑‍💻 Dynamo Contributor** first follow the instructions in [deploy/cloud/helm/README.md](../../deploy/cloud/helm/README.md) to create your Dynamo Cloud deployment.


You would have to rebuild the dynamo platform images as the code evolves. For more details please look at the [Cloud Guide](../guides/dynamo_deploy/dynamo_cloud.md)

Export the [Dynamo Base Image](../get_started.md#building-the-dynamo-base-image) you want to use (or built during the prerequisites step) as the `DYNAMO_IMAGE` environment variable.

```bash
export DYNAMO_IMAGE=<your-registry>/<your-image-name>:<your-tag>
```


### Deploying a particular example

```bash
# Set your dynamo root directory
cd <root-dynamo-folder>
export PROJECT_ROOT=$(pwd)
export NAMESPACE=<your-namespace> # the namespace you used to deploy Dynamo cloud to.
```

Deploying an example consists of the simple `kubectl apply -f ... -n ${NAMESPACE}` command. For example:

```bash
kubectl apply -f  examples/vllm/deploy/agg.yaml -n ${NAMESPACE}
```

You can use `kubectl get dynamoGraphDeployment -n ${NAMESPACE}` to view your deployment.
You can use `kubectl delete dynamoGraphDeployment <your-dep-name> -n ${NAMESPACE}` to delete the deployment.


**Note 1** Example Image

The examples use a prebuilt image from the `nvcr.io/nvidian/nim-llm-dev registry`.
You can build your own image and update the image location in your CR file prior to applying.
See [Building the Dynamo Base Image](../../README.md#building-the-dynamo-base-image)

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
kubectl port-forward svc/${SERVICE_NAME}-frontend 8000:8000 -n ${NAMESPACE}
```

Consult the [Port Forward Documentation](https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/)

More on [LLM examples](llm_deployment.md)