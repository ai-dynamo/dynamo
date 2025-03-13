# deploy Dynamo pipeline on Kubernetes

This is a proof of concept for a Helm chart to deploy services defined in a bento.yaml configuration.

## Usage

### Prerequisites

- make sure dynamo cli is installed

### Install the Helm chart

```bash
./deploy.sh <docker_registry> <k8s_namespace> <bento_name> <path_to_bento_directory>
# example./deploy.sh nvcr.io/nvidian/nim-llm-dev my-namespace my-helm-poc ../deploy/compoundai/sdk/examples/basic_service
```