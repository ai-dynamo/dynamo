# deploy Dynamo pipeline on Kubernetes

This is a proof of concept for a Helm chart to deploy services defined in a bento.yaml configuration.

## Usage

### Prerequisites

- make sure dynamo cli is installed
- make sure you have a docker image registry to which you can push and pull from k8s cluster
- set the imagePullSecrets in the values.yaml file

### Install the Helm chart

```bash
export DYNAMO_IMAGE=<dynamo_docker_image_name>
./deploy.sh <docker_registry> <k8s_namespace> <path_to_dynamo_directory> <dynamo_identifier>
# example :  ./deploy.sh nvcr.io/nvidian/nim-llm-dev my-namespace ../deploy/dynamo/sdk/examples/hello_world/ hello_world:Frontend
```