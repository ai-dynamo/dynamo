# Deploying Dynamo Inference Graphs to Kubernetes using Helm

This guide will walk you through the process of deploying an inference graph created using the Dynamo SDK onto a Kubernetes cluster. Note that this is currently an experimental feature.

## Dynamo Kubernetes Operator Coming Soon!

![Dynamo Deploy](../images/dynamo-deploy.png)

While this guide covers deployment of Dynamo inference graphs using Helm, the preferred method to deploy an inference graph in the future will be via the Dynamo Kubernetes Operator. Dynamo Kubernetes Operator is a soon to be released cloud platform that will simplify the deployment and management of Dynamo inference graphs. It includes a set of components (Operator, UIs, Kubernetes Custom Resources, etc.) to simplify the deployment and management of Dynamo inference graphs.

 Once an inference graph is defined using the Dynamo SDK, it can be deployed onto a Kubernetes cluster using a simple `dynamo deploy` command that orchestrates the following deployment steps:

1. Building docker images from inference graph components on the cluster
2. Intelligently composing the encoded inference graph into a complete deployment on Kubernetes
3. Enabling autoscaling, monitoring, and observability for the inference graph
4. Easy administration of deployments via UI

The Dynamo Kubernetes Operator will be released soon.

## Helm Deployment Guide

### Setting up MicroK8s

Follow these steps to set up a local Kubernetes cluster using MicroK8s:

1. Install MicroK8s:
```bash
sudo snap install microk8s --classic
```

2. Configure user permissions:
```bash
sudo usermod -a -G microk8s $USER
sudo chown -R $USER ~/.kube
```

3. **Important**: Log out and log back in for the permissions to take effect

4. Start MicroK8s:
```bash
microk8s start
```

5. Enable required addons:
```bash
# Enable GPU support
microk8s enable gpu

# Enable storage support
# See: https://microk8s.io/docs/addon-hostpath-storage
microk8s enable storage
```

6. Configure kubectl:
```bash
mkdir -p ~/.kube
microk8s config >> ~/.kube/config
```

After completing these steps, you should be able to use the `kubectl` command to interact with your cluster.

### Installing Required Dependencies

Follow these steps to set up the namespace and install required components:

1. Set environment variables:
```bash
export NAMESPACE=dynamo-playground
export RELEASE_NAME=dynamo-platform
```

2. Install NATS messaging system:
```bash
# Navigate to dependencies directory
cd deploy/Kubernetes/pipeline/dependencies

# Add and update NATS Helm repository
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update

# Install NATS with custom values
helm install --namespace ${NAMESPACE} dynamo-platform-nats nats/nats \
    --create-namespace \
    --values nats-values.yaml
```

3. Install etcd key-value store:
```bash
# Install etcd using Bitnami chart
helm install --namespace ${NAMESPACE} dynamo-platform-etcd \
    oci://registry-1.docker.io/bitnamicharts/etcd \
    --values etcd-values.yaml
```

After completing these steps, your cluster will have the necessary messaging and storage infrastructure for running Dynamo inference graphs.

### Building and Deploying the Pipeline

Follow these steps to containerize and deploy your inference pipeline:

1. Build and containerize the pipeline:
```bash
# Navigate to example directory
cd examples/hello_world

# Set runtime image name
export DYNAMO_IMAGE=<dynamo_runtime_image_name>

# Build and containerize the Frontend service
dynamo build --containerize hello_world:Frontend
```

2. Push container to registry:
```bash
# Tag the built image for your registry
docker tag <BUILT_IMAGE_TAG> <TAG>

# Push to your container registry
docker push <TAG>
```

3. Deploy using Helm:
```bash
# Set release name for Helm
export HELM_RELEASE=helloworld

# Generate Helm values file from Frontend service
dynamo get frontend > pipeline-values.yaml

# Install/upgrade Helm release
helm upgrade -i "$HELM_RELEASE" ./chart \
    -f pipeline-values.yaml \
    --set image=<TAG> \
    --set dynamoIdentifier="hello_world:Frontend" \
    -n "$NAMESPACE"
```

4. Test the deployment:
```bash
# Forward the service port to localhost
kubectl -n ${NAMESPACE} port-forward svc/helloworld-frontend 3000:80

# Test the API endpoint
curl -X 'POST' 'http://localhost:3000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

For convenience, you can find a complete deployment script at `deploy/Kubernetes/pipeline/deploy.sh` that automates all of these steps.
