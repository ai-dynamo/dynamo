First install microk8s:

```bash
sudo snap install microk8s --classic
```

add user permissions
```bash
sudo usermod -a -G microk8s $USER
sudo chown -R $USER ~/.kube
```

Log out and log in

start microk8s
```
microk8s start
```

Add GPU support
```
microk8s enable gpu
```

Add storage support (follow https://microk8s.io/docs/addon-hostpath-storage)
```
microk8s enable storage
```

Kube config
```
mkdir -p ~/.kube && microk8s config >> ~/.kube/config
```

Now one can use `kubectl` command.

2. create a namespace

```
export NAMESPACE=dynamo-playground
export RELEASE_NAME=dynamo-platform

#install nats and etcd
cd deploy/Kubernetes/pipeline/dependencies

#install nats
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update
helm install --namespace ${NAMESPACE} dynamo-platform-nats nats/nats --create-namespace --values nats-values.yaml

#install etcd
helm install --namespace ${NAMESPACE} dynamo-platform-etcd oci://registry-1.docker.io/bitnamicharts/etcd --values etcd-values.yaml
```

Now let's containerize a hello world pipeline:
1. Build container image for `container/Dockerfile.vllm`
2. Containerize hello world pipeline
```
cd examples/hello_world
export DYNAMO_IMAGE=<dynamo_runtime_image_name>
dynamo build --containerize hello_world:Frontend
```

Once the container is built, it has to be tagged and pushed to container registry:
```
docker tag <BUILT_IMAGE_TAG> <TAG>
docker push <TAG>
```

Now one can deploy the pipeline onto k8s using helm
- get values.yaml for helm chart:
- install chart
```
export HELM_RELEASE=helloworld
dynamo get frontend > pipeline-values.yaml

helm upgrade -i "$HELM_RELEASE" ./chart -f pipeline-values.yaml --set image=<TAG> --set dynamoIdentifier="hello_world:Frontend" -n "$NAMESPACE"
```

Once the deployments are running, one can port-forward to localhost and make API calls to the frontend component:
```
kubectl -n ${NAMESPACE} port-forward svc/helloworld-frontend 3000:80
curl -X 'POST'   'http://localhost:3000/generate'   -H 'accept: text/event-stream'   -H 'Content-Type: application/json'   -d '{"text": "test"}'
```

Full script to build a container, push it to registry and deploya helm chart: `deploy/Kubernetes/pipeline/deploy.sh`
