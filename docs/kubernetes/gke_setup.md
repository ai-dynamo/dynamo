# GKE Workload Identity and Artifact Registry Setup Guide

This guide explains how to set up Workload Identity in GKE and configure access to Google Artifact Registry.

## Prerequisites

- Google Cloud SDK installed
- Access to a GKE cluster
- Required permissions to create and manage service accounts

## Project Setup

Set your project:
```bash
export NAMESPACE=your-k8s-namespace
export RELEASE=your-helm-release-name

export PROJECT=$(gcloud config get-value project)
# set the cluster related info (you can list cluster using gcloud container clusters list)
export CLUSTER_NAME=your-cluster-name
export CLUSTER_REGION=$(gcloud container clusters list --filter="name=${CLUSTER_NAME}" --format="get(location)")
gcloud config set project ${PROJECT}
# Retrieve the Workload Identifier Namespace associated with your cluster:
export CLUSTER_WIN=$(gcloud container clusters describe ${CLUSTER_NAME} \
  --region=${CLUSTER_REGION} \
  --format="value(workloadIdentityConfig.workloadPool)")
```

```{important}
Make sure Workload Identity is enabled in your cluster!
```


## Service Account Creation and Configuration

1. Create a service account for Workload Identity:

Go to the GCP console and create a new service account (or reuse an existing one)

```bash
gcloud iam service-accounts create workload-identity-sa\
    --display-name="workload identity service account" \
    --description="Service account to use for Workload Identity in GKE"
export SA=workload-identity-sa@${PROJECT}.iam.gserviceaccount.com
```

2. Configure Workload Identity bindings for Kubernetes service accounts:
```bash

gcloud iam service-accounts add-iam-policy-binding \
    ${SA} \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${CLUSTER_WIN}[${NAMESPACE}/${RELEASE}-dynamo-operator-controller-manager]"

gcloud iam service-accounts add-iam-policy-binding \
    ${SA} \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${CLUSTER_WIN}[${NAMESPACE}/${RELEASE}-dynamo-operator-image-builder]"

gcloud iam service-accounts add-iam-policy-binding \
    ${SA} \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${CLUSTER_WIN}[${NAMESPACE}/${RELEASE}-dynamo-operator-component]"
```

## Artifact Registry Access

### Option 1: Project-Level Access

Grant read and write access at the project level:
```bash
# Grant reader role
gcloud projects add-iam-policy-binding ${PROJECT} \
  --member="serviceAccount:${SA}" \
  --role="roles/artifactregistry.reader"

# Grant writer role
gcloud projects add-iam-policy-binding ${PROJECT} \
  --member="serviceAccount:${SA}" \
  --role="roles/artifactregistry.writer"
```

### Option 2: Repository-Level Access

Grant access to specific repository:
```bash
gcloud artifacts repositories add-iam-policy-binding your-artifact-repository \
  --location=${CLUSTER_REGION} \
  --project=${PROJECT} \
  --member="serviceAccount:${SA}" \
  --role="roles/artifactregistry.reader"
```

## GKE Node Access to Artifact Registry

This is needed to make sure pods can pull images from Artifact Registry without needing to specify an imagePullSecret

### For GKE Autopilot

```bash
# Get project number
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT} --format='value(projectNumber)')

# Grant access to the default compute service account
gcloud projects add-iam-policy-binding ${PROJECT} \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

### For Standard GKE

```bash
# Get node service account
export NODE_SERVICE_ACCOUNT=$(gcloud container clusters describe ${CLUSTER_NAME} \
  --region ${CLUSTER_REGION} \
  --format="get(nodeConfig.serviceAccount)")

# Grant access to node service account
gcloud projects add-iam-policy-binding ${PROJECT} \
  --member="serviceAccount:${NODE_SERVICE_ACCOUNT}" \
  --role="roles/artifactregistry.reader"
```

## Adding annotations to enable Workload Identity

This is an example of values.yaml used to deploy Dynamo Cloud using custom GCP annotations to enable Workload Identity.

```yaml

dynamo-operator:
  ...
  controllerManager:
    serviceAccount:
      create: true
      annotations:
        iam.gke.io/gcp-service-account: your-sa@your-gcp-project.iam.gserviceaccount.com
  ...
  dynamo:
    components:
      serviceAccount:
        annotations:
          iam.gke.io/gcp-service-account: your-sa@your-gcp-project.iam.gserviceaccount.com
    ...

....
```

You can use it during helm installation:

```bash
helm upgrade --install ${RELEASE} platform/ -f values.yaml --namespace ${NAMESPACE}
```

## Important Notes

1. **Prerequisites for Image Pulling**:
   - Workload Identity must be enabled on your GKE cluster
   - GKE nodes' service account must have the `artifactregistry.reader` role

2. **Troubleshooting**:
   - If pods can't pull images, verify both Workload Identity and node service account configurations
   - Check service account annotations on Kubernetes service accounts
   - Verify IAM bindings are correctly set up

## References

- [GKE Workload Identity Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)
- [Artifact Registry Authentication](https://cloud.google.com/artifact-registry/docs/docker/authentication)
- [IAM Roles for Artifact Registry](https://cloud.google.com/artifact-registry/docs/access-control)