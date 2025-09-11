# Deploying Dynamo Cloud on Kubernetes (Minikube)

This document covers the process of deploying Dynamo Cloud and running inference in a vLLM distributed runtime within a Kubernetes environment. The Dynamo Cloud Platform provides a managed deployment experience:

- Contains the infrastructure components required for the Dynamo cloud platform
- Leverage the Dynamo Operator and it's exposed CRD's to deploy Dynamo inference graphs
- Provides a managed deployment experience

This overview covers the setup process on a Minikube instance, including:

- Deploying the Dynamo Operator and creating Dynamo CRDs
- Deploying an inference graph built in vLLM Dynamo Runtime
- Running inference

---

## Prerequisites

Please refer to the general prerequisites required for running Dynamo. The cluster setup process will be covered in this document. In addition to the Dynamo prerequisites, the following are required for setting up the Minikube cluster:

- kubectl
- Helm
- Minikube

---

## Pull Dynamo GitHub

The Dynamo GitHub repository will be leveraged extensively throughout this walkthrough. Pull the repository using:

```bash
# clone Dynamo GitHub repo
git clone https://github.com/ai-dynamo/dynamo.git

# go to root of Dynamo repo
cd dynamo
```

---

## Set Up Minikube Cluster

### Start Kubernetes using Minikube

To run Minikube, you'll need:

- 2 CPUs or more
- 2GB of free memory
- 20GB of free disk space

If your machine has NVIDIA drivers, run the optional command below. Start the cluster:

```bash
# Start Minikube cluster with GPUs, NOTE: Potentially add --force flag to force minikube to use all available gpus
minikube start --driver=docker --container-runtime=docker --gpus=all

# Optional: Unmount /proc/driver/nvidia if machine has preinstalled drivers
ssh -o "StrictHostKeyChecking no" -i $(minikube ssh-key) docker@$(minikube ip) "sudo umount -R /proc/driver/nvidia"
```

---

### Accessing GPU Resources In Kubernetes

In the event that NVIDIA drivers are preinstalled on the target compute instance we'll be running Dynamo related workloads on, specifying the GPU flags in the `minikube start` command should typically bring up NVIDIA device plugin pods. The [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) lets Kubernetes detect and allocate NVIDIA GPUs to pods, enabling GPU-accelerated workloads. Without it, Kubernetes can't schedule GPUs for containers. If the device plugin pod is not enabled, you can install via the `minikube addon` function:

```bash
minikube addons enable nvidia-device-plugin
```

Once the device plugin pods are in a running state we can proceed with running GPU workloads in the minikube cluster. Please note depending on your cluster setup, you might manually have to install the NVIDIA device plugin, or the [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) which is preferred over just the NVIDIA device plugin especially for production or large-scale Kubernetes environments, as the GPU Operator automates the installation and management of GPU drivers, the device plugin, monitoring, and other GPU software on Kubernetes nodes. We can verify the device plugin pods are running by checking pod status in the `kube-system` namespace:

```bash
# check status of device plugin pod
kubectl get pods -n kube-system

# output is truncated to only show device plugin pods
NAME                                   READY   STATUS    RESTARTS      AGE
...
nvidia-device-plugin-daemonset-hvd5x   1/1     Running   0             1d
```

---

### Set Up Ingress Controller

Set up an Ingress controller to expose the Dynamo API Store service. In this case, we'll be leveraging NGINX and the Minikube addon to easily enable this ingress controller:

```bash
# enable ingress add on
minikube addons enable ingress

# verify pods are running
kubectl get pods -n ingress-nginx

# output should be similar
NAME                                        READY   STATUS      RESTARTS   AGE
ingress-nginx-admission-create-wnv5m        0/1     Completed   0          1d
ingress-nginx-admission-patch-977pp         0/1     Completed   0          1d
ingress-nginx-controller-768f948f8f-gg8vd   1/1     Running     0          1d
```

---

### Verify Default StorageClass

Ensure the cluster has access to a default storage class. Dynamo Cloud requires Persistent Volume Claim (PVC) support. Minikube should come preinstalled with a default `standard` storage class that can be leveraged for provisioning volumes:

```bash
# get storage class
kubectl get storageclass

# Output should show (default) flag next to storage class
NAME                 PROVISIONER                RECLAIMPOLICY   VOLUMEBINDINGMODE   ALLOWVOLUMEEXPANSION   AGE
standard (default)   k8s.io/minikube-hostpath   Delete          Immediate           false                  1d
```

---

## Dynamo Cloud

The Dynamo Cloud Platform consists of several key components:

- **Dynamo Operator**: Manages the lifecycle of Dynamo inference graphs.
- **Custom Resources**: Kubernetes custom resources for defining and managing Dynamo services.
- **Required Dynamo Services**: Deploys NATs & ETCD services that are leveraged by Dynamo

---

### Leveraging Dynamo Container Runtimes In Dynamo Cloud

Dynamo, is a high-throughput, low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments. Dynamo specializes in taking a given runtime (TRT-LLM, vLLM, SGLang, etc) and creating a highly scalable distributed runtime. It's important to verify which Dynamo runtime you'll want to leverage in your deployments, as each runtime will have slightly different implementations for optimizing a given workload with Dynamo.

To simplify this tutorial, we'll leverage prebuilt Dynamo containers targeting a vLLM runtime - these containers are available today as published artifacts in NGC catalog. Dynamo also supports buidling container runtimes from source and uploading them to a private registry. For tutorials on building images from source, please reference this [documentation](https://github.com/ai-dynamo/dynamo/tree/main/components/backends/vllm#pull-or-build-container). We'll make sure the container image we use is tied to the same release version as the Dynamo Cloud helm charts that will be in the next section:

```bash
# set release version
export RELEASE_VERSION=0.4.1

# configure dynamo image and corresponding tag
export DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:${RELEASE_VERSION}

# print dynamo image
echo ${DYNAMO_IMAGE}
```

Please make sure to take note of the resulting Dynamo container image that is defined - this will be leveraged later for deployment of Dynamo inference graphs.

---

### Create Secrets That Will Be Leveraged In Dynamo Cloud

Before deploying Dynamo cloud, we'll need to create the secrets that both Dynamo cloud, and the underlying inference graphs that will be deployed, will leverage. We'll create an image pull secret for pulling containers and assets from NGC, and a Huggingface secret that can be leveraged for pulling model specific weights and assets from Huggingface hub.

Before proceeding, please make sure you have access to both an [NGC API Key](https://org.ngc.nvidia.com/setup/api-key) and a [huggingface access token.](https://huggingface.co/docs/hub/en/security-tokens) We'll need to make sure that the created secrets are applied to the same namespace the underlying Dynamo Cloud service will be deployed in. This secret will be leveraged by the Dynamo platform deployment to pull secrets from NGC, this will be set in the `helm install` command further down in the document:

```bash
# export env for namespace dynamo platform will be deployed
export NAMESPACE=dynamo-cloud

# export environment variables needed to fetch helm chart
export NGC_API_KEY=<YOUR_NGC_API_KEY>

# create dynamo cloud namespace
kubectl create namespace ${NAMESPACE}

# create NGC image pull secrets
kubectl create secret docker-registry nvcrimagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=${NGC_API_KEY} \
  -n ${NAMESPACE}

# export huggingface token
export HF_TOKEN=<HF_TOKEN>

# create huggingface secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Once secrets have been provisioned, we can proceed with deploying Dynamo Cloud.

---

### Deploying the Dynamo Cloud Platform

In order to deploy the Dynamo cloud platform in kubernetes, we'll leverage helm charts. To be more precise, the following helm charts are needed:

- Dynamo CRD's Helm Chart
- Dynamo Platform Helm Chart

To simplify this tutorial, we'll leverage the helm charts that have already been pushed to NGC as public artifacts. If you're interested in developing or customizing Dynamo as a contributor, or using local helm charts from the source repository, please refer to this [guide here](https://github.com/ai-dynamo/dynamo/blob/main/docs/guides/dynamo_deploy/quickstart.md#2-installing-dynamo-cloud-from-source). You'll need to make sure you have [Earthly](https://earthly.dev/) installed for building components from source.

#### Fetch Dynamo Helm Charts

Now that we're aware of the corresponding helm charts we'll need to deploy to set up Dynamo Cloud, we'll start by fetching the helm charts. We'll leverage the exposed environment variables to target release components available on NGC. Once configured, we can use the code block below to pull the charts from NGC, we'll make sure the release version associated with the Dynamo images match the vLLM container tag that will be used to help facilitate deployments of inference graphs:

```bash
# Fetch the CRDs helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-0.4.0.tgz

# Fetch the platform helm chart
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz

# verify charts were fetched
ls -l *.tgz

# output should be similar
-rw-r--r-- 1 ubuntu ubuntu 13342 Jul 31 20:13 dynamo-crds-0.4.0.tgz
-rw-r--r-- 1 ubuntu ubuntu 89682 Jul 31 20:13 dynamo-platform-0.4.1.tgz
```

---

#### Deploy Dynamo CRD's Helm Chart

Once we've verified the charts have been pulled successfully, we'll install the Dynamo Cloud platform. First, we'll start by installing the chart that exposes Dynamo CRD's via helm:

```bash
# install dynamo crd's chart in default namespace (CRD's exposed from this chart aren't namespace scoped)
helm install dynamo-crds dynamo-crds-0.4.0.tgz \
  --namespace default \
  --wait \
  --atomic
```

Once we've installed the Dynamo CRD's chart, we can view the status of the deployment. Since this chart exposes CRD's we'll also verify they were created at the cluster level:

```bash
# verify helm status - should show `deployed`
helm list --filter dynamo-crds -n default

# output should be similar
NAME            NAMESPACE       REVISION        UPDATED                                 STATUS          CHART
dynamo-crds     default         1               2025-07-31 20:29:17.598324415 +0000 UTC deployed        dynamo-crds-0.4.0

# verify creation of CRD's
kubectl get crd | grep "dynamo"

# output should be similar
dynamocomponentdeployments.nvidia.com      2025-07-31T20:29:17Z
dynamographdeployments.nvidia.com          2025-07-31T20:29:17Z
```

---

#### Deploy Dynamo Platform Helm Chart

At this point, we've been able to deploy the Dynamo CRD's chart and verify creation of Dynamo custom resources. With these in place, we can deploy the Dynamo platform chart, this chart will be responsible for deploying the NATs & ETCD services that are needed by Dynamo, in addition to a controller manager resource:

```bash
# install dynamo platform helm chart in dynamo-cloud namespace
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --set "dynamo-operator.imagePullSecrets[0].name=nvcrimagepullsecret" \
  --namespace ${NAMESPACE}

# verify helm status - should show `deployed`
helm list --filter dynamo-platform -n ${NAMESPACE}

# output should be similar
NAME            NAMESPACE       REVISION        UPDATED                                 STATUS          CHART
dynamo-platform dynamo-cloud    1               2025-07-31 20:28:48.31568394 +0000 UTC  deployed        dynamo-platform-0.4.1

# verify dynamo platform pods are running
kubectl get pods -n ${NAMESPACE}

# output should be similar
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-549b5d5274nl   2/2     Running   0          15m
dynamo-platform-etcd-0                                            1/1     Running   0          15m
dynamo-platform-nats-0                                            2/2     Running   0          15m
dynamo-platform-nats-box-5dbf45c748-7nckv                         1/1     Running   0          15m
```

Once the CRD's have been exposed, and the platform pods are running, Dynamo Cloud has been deployed successfully.

---

## Deploying Dynamo Inference Graphs to Kubernetes

```bash
# cd into dynamo directory with CRD manifests
cd components/backends/vllm/deploy
```

Now that we're in the corresponding directory, we can view the manifests that are present here. For example, in the `agg_router.yaml` manifest, we'll see a `DynamoGraphDeployment` resource that is referenced - this will correspond to one of the CRD's that were exposed in the Dynamo helm deployment. This serves as a reference example we can leverage to facilitate the inference graph deployment.

Before we can apply this manifest to our cluster, we'll need to update the `extraPodSpec.mainContainer.image` path to point to the vLLM Dynamo container we configured in the earlier step (`echo ${DYNAMO_IMAGE}`). We'll also need to configure the `dynamoNamespace` field to point to the namespace we've deployed Dynamo cloud in (`echo ${NAMESPACE}`). You'll need to make sure these changes are in place for any `DynamoGraphDeployment` resource you want deployed.

By default, the Dynamo deployment will pull the Qwen3 0.6B's relevant model weights from huggingface, but the model invoked can be changed by updating the `args` commmand `python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B` under `spec.VllmDecodeWorker.extraPodSpec.mainContainer.args` . Once those updates have been made, we can apply the manifest in the Dynamo cloud namespace:

```bash
# apply CRD in dynamo-cloud namespace
kubectl apply -f agg_router.yaml -n ${NAMESPACE}
```

Once the CRD has been applied successfully, we can view the pods specific to our deployment start up:

```bash
# get status of vllm pods in dynamo cloud namespace
kubectl get pods -n ${NAMESPACE} | grep "vllm"

# output should be similar
vllm-agg-router-frontend-7f589c9fc9-scqxs                         1/1     Running   0               12m
vllm-agg-router-vllmdecodeworker-ffb797d87-lzjsk                  1/1     Running   0               12m
vllm-agg-router-vllmdecodeworker-ffb797d87-z2csm                  1/1     Running   0               12m
```

---

### Exposing the Frontend Service

Once all the pods are in a running state, verify the details regarding the Dynamo service that is spun up:

```bash
# get services running in dynamo-cloud namespace
kubectl get svc -n ${NAMESPACE}

# output should be similar
NAME                            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)     AGE
....
vllm-agg-router-frontend        ClusterIP   10.110.255.115   <none>        8000/TCP    10m
```

Once we've verified service details, we'll create an Ingress resource to expose the service and run inference on it. Use the following ingress configuration, which will be exposed via NGINX:

```yaml
cat <<EOF > vllm_agg_router_ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-agg-router-ingress
  namespace: $NAMESPACE
spec:
  ingressClassName: nginx
  rules:
  - host: dynamo-vllm-agg-router.test
    http:
      paths:
      - backend:
          service:
            name: vllm-agg-router-frontend
            port:
              number: 8000
        path: /
        pathType: Prefix
EOF
```

Apply the ingress resource:

```bash
kubectl apply -f vllm_agg_router_ingress.yaml -n ${NAMESPACE}
kubectl get ingress -n ${NAMESPACE}
```

Once the ingress resource has been created, make sure to add the entry along with it's address in your `/etc/hosts` file. Look up the external IP address as reported by Minikube by running the `minikube ip` command. Once found, update the hosts file with the following line:

`<YOUR_MINIKUBE_IP> dynamo-vllm-agg-router.test`

Once configured, we can make cURL requests to the Dynamo API endpoint:

```bash
curl http://dynamo-vllm-agg-router.test/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 100
  }'
```

---

## Clean Up Resources

In order to clean up any Dynamo related resources, we can simply delete the `DynamoGraphDeployment` resource we deployed earlier. This can be achieved by running the following command:

```bash
# delete dynamo deployment
kubectl delete -f agg_router.yaml -n ${NAMESPACE}
```

This will spin down the Dynamo deployment we configured and spin down all the resources that were leveraged for the deployment. As a final cleanup step, we can also delete the ingress resource that was created to expose the service:

```bash
# delete ingress resource created for dynamo service
kubectl delete -f vllm_agg_router_ingress.yaml -n ${NAMESPACE}
```

At this point, all deployment related resources will be spun down. If you want to tear down the Dynamo Cloud helm deployment or the Dynamo CRD's, please run the `helm uninstall` commands below:

```bash
# uninstall dynamo cloud
helm uninstall dynamo-platform -n ${NAMESPACE}

# uninstall dynamo CRD's
helm uninstall dynamo-crds -n default
```
