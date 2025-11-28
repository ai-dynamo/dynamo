# Steps to install Dynamo Cloud from Source

## 1. Build Dynamo Base Image

Create 1 ECR repositoriy

```
aws configure
aws ecr create-repository --repository-name <ECR_REPOSITORY>
```

Build Image

```
export NAMESPACE=dynamo-cloud
export DOCKER_SERVER=<ECR_REGISTRY>
export DOCKER_USERNAME=AWS
export DOCKER_PASSWORD="$(aws ecr get-login-password --region <ECR_REGION>)"
export IMAGE_TAG=0.3.2.1
./container/build.sh
```

Push Image

```
docker tag dynamo:latest-vllm <ECR_REGISTRY>/<ECR_REPOSITORY>:$IMAGE_TAG
aws ecr get-login-password | docker login --username AWS --password-stdin <ECR_REGISTRY>
docker push <ECR_REGISTRY>/<ECR_REPOSITORY>:$IMAGE_TAG
```

## 2. Install Dynamo Cloud

Build and Push Operator Image

```
cd deploy/cloud/operator
vim Earthfile # change ARG IMAGE_SUFFIX=<ECR_REPOSITORY>
earthly --push +docker --DOCKER_SERVER=$DOCKER_SERVER --IMAGE_TAG=$IMAGE_TAG
```

Create secrets

```
kubectl create namespace ${NAMESPACE}
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}
export HF_TOKEN=<HF_TOKEN>
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Install Dynamo Cloud

```
cd dynamo/cloud/helm
helm install dynamo-crds ./crds/ \
  --namespace default \
  --wait \
  --atomic
```

```
helm dep build ./platform/

kubectl create namespace ${NAMESPACE}

# Create docker registry secret
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}

# Install platform
helm install dynamo-platform ./platform/ \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set "dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret"
```

Your pods should be running like below

```
ubuntu@ip-192-168-83-157:~/dynamo/examples/backends/vllm/deploy$ kubectl get pods -A
NAMESPACE      NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-cloud   dynamo-platform-dynamo-operator-controller-manager-86795c5f4j4k   2/2     Running   0          4h17m
dynamo-cloud   dynamo-platform-etcd-0                                            1/1     Running   0          4h17m
dynamo-cloud   dynamo-platform-nats-0                                            2/2     Running   0          4h17m
dynamo-cloud   dynamo-platform-nats-box-5dbf45c748-bxqj7                         1/1     Running   0          4h17m
```
