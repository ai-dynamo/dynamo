# Deploy CompoundAI API server and Operator

### Manually install etcd and nats

Pre-requisite: make sure your terminal is set in the `deploy/dynamo/helm/` directory.

1. [Optional] Create a new kubernetes namespace and set it as your default

```bash
export KUBE_NS=cai    # change this to whatever you want!
kubectl create namespace $KUBE_NS
kubectl config set-context --current --namespace=$KUBE_NS
```

2. Install bitnami/etcd:

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

3. Install nats:

```bash
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update
```

4. Install etcd and nats to your kubernetes namespace:

```bash
helm install etcd bitnami/etcd -n $KUBE_NS -f etcd.yaml
helm install my-nats nats/nats --version 1.2.9 -f nats.yaml -n $KUBE_NS
```

5. Deploy the helm charts:

```bash
export NGC_TOKEN=$NGC_API_TOKEN
export NAMESPACE=$KUBE_NS
export CI_COMMIT_SHA=6083324a0a5f310dcec38c6863f043cd9070ffcc
export RELEASE_NAME=$KUBE_NS

./deploy.sh
```

6. Make an example cluster POST:

```bash
./post-cluster.sh
```

As a bonus, the CompoundAI Deployments UI is also deployed alongside, so you can access it at https://${NAMESPACE}.dev.aire.nvidia.com/