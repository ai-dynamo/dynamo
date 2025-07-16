# Examples of using Dynamo Platform

## Serving examples locally

Follow individual examples to serve models locally.


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

Pick your deployment destination.


If kubernetes
```bash
export DYNAMO_CLOUD=https://dynamo-cloud.nvidia.com
```

Setup port forward if needed when deploying to Kubernetes.

```bash
export DEPLOYMENT_NAME=vllm-v1-agg # example
# Forward the pod's port to localhost
kubectl port-forward svc/$DEPLOYMENT_NAME-frontend 8000:8000 -n ${NAMESPACE}
```

Deploying an example consists of the simple `kubectl apply -f ... -n ${NAMESPACE}` command. For example:

```bash
kubectl apply -f  examples/vllm/deploy/agg.yaml -n ${NAMESPACE}
```

**Note** You can change the image location in your CR file prior to applying

```bash
extraPodSpec:
        mainContainer:
          image: <image-in-your-$DYNAMO_IMAGE>
```


More on [LLM examples](llm_deployment.md)