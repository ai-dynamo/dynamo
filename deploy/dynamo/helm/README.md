# Deploy Dynamo Cloud

## Building Docker images for Dynamo Cloud components

You can build and push Docker images for the Dynamo cloud components (API server, API store, and operator) to any container registry of your choice. Here's how to build each component:

### Prerequisites
- [Earthly](https://earthly.dev/) installed
- Docker installed and running
- Access to a container registry of your choice

### Building and Pushing Images

First, set the required environment variables:
```bash
export CI_REGISTRY_IMAGE=<CONTAINER_REGISTRY>/<ORGANIZATION>
export CI_COMMIT_SHA=<TAG>
```

As a description of the placeholders:
- `<CONTAINER_REGISTRY>/<ORGANIZATION>`: Your container registry and organization name (e.g., `nvcr.io/myorg`, `docker.io/myorg`, etc.)
- `<TAG>`: The tag you want to use for the image (e.g., `latest`, `0.0.1`, etc.)

Note: Make sure you're logged in to your container registry before pushing images. For example:
```bash
docker login <CONTAINER_REGISTRY>
```

You can build each component individually or build all components at once:

#### Option 1: Build All Components at Once
```bash
earthly --push +all-docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
```

#### Option 2: Build Components Individually

1. **API Store**
```bash
cd deploy/dynamo/api-store
earthly --push +docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
```

2. **Operator**
```bash
cd deploy/dynamo/operator
earthly --push +docker --CI_REGISTRY_IMAGE=$CI_REGISTRY_IMAGE --CI_COMMIT_SHA=$CI_COMMIT_SHA
```

## Deploy Dynamo Cloud Platform

Pre-requisite: make sure your terminal is set in the `deploy/dynamo/helm/` directory.

```bash
cd deploy/dynamo/helm
export KUBE_NS=hello-world    # change this to whatever you want!
```

1. [One-time Action] Create a new kubernetes namespace and set it as your default. Create image pull secrets if needed.

```bash
kubectl create namespace $KUBE_NS
kubectl config set-context --current --namespace=$KUBE_NS

# [Optional] if needed, create image pull secrets
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=<your-registry> \
  --docker-username=<your-username> \
  --docker-password=<your-password> \
  --namespace=$KUBE_NS
```

2. Deploy the helm chart using the deploy script:

```bash
export NGC_TOKEN=$NGC_API_TOKEN
export NAMESPACE=$KUBE_NS
export CI_COMMIT_SHA=<TAG>  # Use the same tag you used when building the images
export CI_REGISTRY_IMAGE=<CONTAINER_REGISTRY>/<ORGANIZATION>  # Use the same registry/org you used when building the images
export RELEASE_NAME=$KUBE_NS

./deploy.sh
```