# Steps to create EKS cluster with EFS

## 1. Install CLIs

### a. Install AWS CLI (steps [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))

```
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### b. Install Kubernetes CLI (steps [here](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html))

```
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.30.0/2024-05-12/bin/linux/amd64/kubectl
chmod +x ./kubectl
mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
```

### c. Install EKS CLI (steps [here](https://eksctl.io/installation/))

```
ARCH=amd64
PLATFORM=$(uname -s)_$ARCH
curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"
curl -sL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt" | grep $PLATFORM | sha256sum --check
tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz
sudo mv /tmp/eksctl /usr/local/bin
```

### d. Install Helm CLI (steps [here](https://docs.aws.amazon.com/eks/latest/userguide/helm.html))

```
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 > get_helm.sh
chmod 700 get_helm.sh
./get_helm.sh
```

## 2. Create an EKS cluster

In this example we create an EKS cluster consisting of 1 `g6e.48xlarge` compute node, each with 8 NVIDIA L40S GPUs and 1 `c5.2xlarge` CPU node as control plane. We also setup EFA between the compute nodes.

### a. Configure AWS CLI

```
aws configure
```

### b. Create a config file for EKS cluster creation

```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: <CLUSTER_NAME>
  version: "1.32"
  region: <REGION_NAME>

iam:
  withOIDC: true

managedNodeGroups:
  - name: sys-ng
    instanceType: c5.2xlarge
    minSize: 1
    desiredCapacity: 1
    maxSize: 1
    iam:
      withAddonPolicies:
        imageBuilder: true
        autoScaler: true
        ebs: true
        efs: true
        awsLoadBalancerController: true
        cloudWatch: true
        albIngress: true

  - name: efa-compute-ng
    instanceType: g6e.48xlarge
    minSize: 1
    desiredCapacity: 1
    maxSize: 1
    volumeSize: 300
    efaEnabled: true
    privateNetworking: true
    iam:
      withAddonPolicies:
        imageBuilder: true
        autoScaler: true
        ebs: true
        efs: true
        awsLoadBalancerController: true
        cloudWatch: true
        albIngress: true
```

> [!NOTE]
> We set `minSize` and `desiredCapacity` to be 1 because AWS does not create your cluster successfully if no nodes are available. For example, if you specify `desiredCapacity` to be 2 but there are no available 2 nodes, your cluster creation will fail due to timeout even though there are no errors. The easiest way to avoid this is to create the cluster with 1 node and increase the number of nodes later in the EKS console. After you increase number of nodes in your node groups, make sure GPU nodes are in the same subnet. This is required for EFA to work.

### c. Create the EKS cluster

```
eksctl create cluster -f eks_cluster_config.yaml
```

## 3. Create an EFS file system

We'll need a common, shared storage location to enable pods deployed to multiple nodes to load shards of the same model. This way, they can be used in coordination to serve inference requests for models too large to loaded by GPUs on a single node. In Kubernetes, these common, shared storage locations are referred to as persistent volumes. Persistent volumes can be volume mapped in to any number of pods and then accessed by processes running inside of said pods as if they were part of the pod's file system. We will be using EFS as persistent volume.

Additionally, we will need to create a persistent-volume claim which can use to assign the persistent volume to a pod.
### a. Create an IAM role

Follow the steps to create an IAM role for your EFS file system: https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html#efs-create-iam-resources. This role will be used later when you install the EFS CSI Driver.

### b. Install EFS CSI driver

Install the EFS CSI Driver through the Amazon EKS add-on in AWS console: https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html#efs-install-driver. Once it's done, check the Add-ons section in EKS console, you should see the driver is showing `Active` under Status.

### c. Create EFS file system

Follow the steps to create an EFS file system: https://github.com/kubernetes-sigs/aws-efs-csi-driver/blob/master/docs/efs-create-filesystem.md. Make sure you mount subnets in the last step correctly. This will affect whether your nodes are able to access the created EFS file system.

## 4. Test

Follow the steps to check if your EFS file system is working properly with your nodes: https://github.com/kubernetes-sigs/aws-efs-csi-driver/tree/master/examples/kubernetes/multiple_pods. This test is going to mount your EFS file system on all of your available nodes and write a text file to the file system.

## 5. Create StorageClass

You can find your `fileSystemId` from AWS EFS. It usually start with `fs-`.

```
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: efs-sc
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: efs.csi.aws.com
parameters:
  fileSystemId: fs-01e72da3fcdbf8a4d
  provisioningMode: efs-ap
  directoryPerms: "777"
  uid: "1000"
  gid: "1000"
```

```
kubectl apply -f storageclass.yaml
```