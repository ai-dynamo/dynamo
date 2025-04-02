# Deploy CompoundAI API server and Operator

### Manually install etcd and nats

Pre-requisite: make sure your terminal is set in the `deploy/dynamo/helm/` directory.

```bash
cd deploy/dynamo/helm
export KUBE_NS=hello-world    # change this to whatever you want!
```

1. [One-time Action] Create a new kubernetes namespace and set it as your default. Then create image pull secrets using the following commands. Update the docker registry, username, and password values according to your environment if you are using private registry to store images. **Note: the images may soon be published in a public repository, which will eliminate the need for image pull secrets.**

```bash
kubectl create namespace $KUBE_NS
kubectl config set-context --current --namespace=$KUBE_NS

kubectl create secret docker-registry nvcrimagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_TOKEN \
  --namespace=$KUBE_NS

kubectl create secret docker-registry gitlab-imagepull \
  --docker-server=gitlab-master.nvidia.com:5005 \
  --docker-username=<your-gitlab-username> \
  --docker-password=<your-gitlab-token> \
  --namespace=$KUBE_NS
```

2. Deploy the helm chart using the deploy script:

```bash
export NGC_TOKEN=$NGC_API_TOKEN
export NAMESPACE=$KUBE_NS
export CI_COMMIT_SHA=0774daf72d0629052b9ae30e43cfc0751b8fa744
export RELEASE_NAME=$KUBE_NS

./deploy.sh
```

3. [Optional] Make an example cluster POST:

```bash
./post-cluster.sh
```
