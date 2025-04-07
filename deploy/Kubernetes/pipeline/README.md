# deploy Dynamo pipeline on Kubernetes

This is a proof of concept for a Helm chart to deploy services defined in a bento.yaml configuration.

## Usage

### Prerequisites

- make sure dynamo cli is installed
- make sure you have a docker image registry to which you can push and pull from k8s cluster
- set the imagePullSecrets in the values.yaml file
- navigate to the pipeline deployment directory by running:
  ```bash
  cd deploy/Kubernetes/pipeline
  ```
- build and set up the DYNAMO_IMAGE as described in the [main README](../../README.md#building-the-dynamo_image-base-image)
- make sure the `nats` and `etcd` dependencies are installed (under the `dependencies` subdirectory). For more details, see [Installing Required Dependencies](../../../docs/guides/dynamo_deploy.md#installing-required-dependencies)

### Install the Helm chart

```bash
export DYNAMO_IMAGE=<dynamo_docker_image_name>
./deploy.sh <docker_registry> <k8s_namespace> <path_to_dynamo_directory> <dynamo_identifier> [<dynamo_config_file>]

# example: export DYNAMO_IMAGE=nvcr.io/nvidian/nim-llm-dev/dynamo-base-worker:0.0.1
# example: ./deploy.sh nvcr.io/nvidian/nim-llm-dev my-namespace ../../../examples/hello_world/ hello_world:Frontend
# example: ./deploy.sh nvcr.io/nvidian/nim-llm-dev my-namespace ../../../examples/llm graphs.disagg_router:Frontend ../../../examples/llm/configs/disagg_router.yaml
```

### Test the deployment

```bash
# Forward the service port to localhost
kubectl -n <k8s_namespace> port-forward svc/hello-world-frontend 3000:80

# Test the API endpoint
curl -X 'POST' 'http://localhost:3000/generate' \
    -H 'accept: text/event-stream' \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}'
```