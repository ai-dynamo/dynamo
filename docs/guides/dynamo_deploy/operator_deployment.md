# Deploying Dynamo Inference Graphs to Kubernetes using the Dynamo Cloud Platform

This guide walks you through deploying an inference graph created with the Dynamo SDK onto a Kubernetes cluster using the Dynamo cloud platform and the Dynamo deploy CLI. The Dynamo cloud platform provides a streamlined experience for deploying and managing your inference services.

## Prerequisites

Before proceeding with deployment, ensure you have:

- [Dynamo CLI](../README.md#installation) installed
- A Kubernetes cluster with the [Dynamo cloud platform](../dynamo_cloud.md) installed
- Ubuntu 24.04 as the base image for your services
- Required dependencies:
  - Helm package manager
  - Dynamo SDK and CLI tools
  - Rust packages and toolchain

## Understanding the Deployment Process

The deployment process involves two main steps:

1. **Local Build (`dynamo build`)**
   - Creates a Dynamo service archive containing:
     - Service code and dependencies
     - Service configuration and metadata
     - Runtime requirements
     - Service graph definition
   - This archive is used as input for the remote build process

2. **Remote Image Build**
   - A `yatai-dynamonim-image-builder` pod is created in your cluster
   - This pod:
     - Takes the Dynamo service archive
     - Containerizes it using the specified base image
     - Pushes the final container image to your cluster's registry
   - The build process is managed by the Dynamo operator

## Deployment Steps

### 1. Login to Dynamo Server

First, configure your environment and login to the Dynamo server:

```bash
# Set your project root directory
export PROJECT_ROOT=$(pwd)

# Set your Kubernetes namespace (must match the namespace where Dynamo cloud is installed)
export KUBE_NS=hello-world

# Set the Dynamo server endpoint
export DYNAMO_SERVER=https://${KUBE_NS}.dev.aire.nvidia.com

# Login to the Dynamo server
dynamo server login --api-token TEST-TOKEN --endpoint $DYNAMO_SERVER
```

### 2. Build the Dynamo Image

> [!NOTE]
> For instructions on building the Dynamo base image, see the [Building the Dynamo Base Image](../../README.md#building-the-dynamo-base-image) section in the main README.

```bash
# Set your runtime image name
export DYNAMO_IMAGE=<dynamo_docker_image_name>

# Navigate to your project directory
cd $PROJECT_ROOT/examples/hello_world

# Build the service and capture the tag
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk -F"\"" '{ print $2 }')
```

### 3. Deploy to Kubernetes

Deploy your service using the Dynamo deployment command:

```bash
# Set your Helm release name
export HELM_RELEASE=ci-hw

# Create the deployment
dynamo deployment create $DYNAMO_TAG --no-wait -n $HELM_RELEASE
```

To delete an existing deployment:
```bash
kubectl delete dynamodeployment $HELM_RELEASE
```

### 4. Test the Deployment

The deployment process creates several pods:
1. A `yatai-dynamonim-image-builder` pod for building the container image
2. Service pods prefixed with `$HELM_RELEASE` once the build is complete

To test your deployment:

```bash
# Forward the service port to localhost
kubectl -n ${KUBE_NS} port-forward svc/${HELM_RELEASE}-frontend 3000:3000

# Test the API endpoint
curl -X 'POST' 'http://localhost:3000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```

## Expected Output

When you send a request with "test" as input, you'll see how the text flows through each service:

```
Frontend: Middle: Backend: test-mid-back
```

This demonstrates the service pipeline:
1. The Frontend receives "test"
2. The Middle service adds "-mid" to create "test-mid"
3. The Backend service adds "-back" to create "test-mid-back"
