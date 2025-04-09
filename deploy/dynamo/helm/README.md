# Deploy Dynamo Cloud

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