# Log Aggregation in Dynamo on Kubernetes

This guide demonstrates how to set up logging for Dynamo in Kubernetes using Grafana Loki and Alloy. This setup provides a simple reference logging setup that can be followed in Kubernetes clusters including Minikube and MicroK8s.

> [!Note]
> This setup is intended for development and testing purposes. For production environments, please refer to the official documentation for high-availability configurations.

## Components Overview

- **[Grafana Loki](https://grafana.com/oss/loki/)**: Fast and cost-effective Kubernetes-native log aggregation system.

- **[Grafana Alloy](https://grafana.com/oss/alloy/)**: OpenTelemetry collector that replaces Promtail, gathering logs, metrics and traces from Kubernetes pods.

- **[Grafana](https://grafana.com/grafana/)**: Visualization platform for querying and exploring logs.

## Prerequisites

### 1. Dynamo Cloud Kubernetes Operator

This guide assumes you have installed Dynamo Cloud Kubernetes Operator. For more information, see [Dynamo Cloud Operator](./README.md).

### 2. Kube-prometheus

While this guide does not use Prometheus, it assumes Grafana is pre-installed with the kube-prometheus. For more information, see [kube-prometheus](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack).

### 3. Environment Variables

The following env variables are set:
- `MONITORING_NAMESPACE`: The namespace where Loki is installed
- `DYN_NAMESPACE`: The namespace where Dynamo Cloud Operator is installed

```bash
export MONITORING_NAMESPACE=monitoring
export DYN_NAMESPACE=dynamo-system
```

## Installation Steps

### 1. Install Loki

First, we'll install Loki in single binary mode, which is ideal for testing and development:

```bash
# Add the Grafana Helm repository
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Loki
helm install --values deploy/logging/values/loki-values.yaml loki grafana/loki -n $MONITORING_NAMESPACE
```

Our configuration (`loki-values.yaml`) sets up Loki in a simple configuration that is suitable for testing and development. It uses a local MinIO for storage. The installation pods can be viewed with:
```bash
kubectl get pods -n $MONITORING_NAMESPACE -l app=loki
```

### 2. Install Grafana Alloy

Next, install the Grafana Alloy collector to gather logs from your Kubernetes cluster and forward them to Loki. Here we use the Helm chart `k8s-monitoring` provided by Grafana to install the collector:

```bash
# Generate a custom values file with the namespace information
envsubst < deploy/logging/values/alloy-values.yaml > alloy-custom-values.yaml

# Install the collector
helm install --values alloy-custom-values.yaml alloy grafana/k8s-monitoring -n $MONITORING_NAMESPACE
```

The values file (`alloy-values.yaml`) includes the following configurations for the collector:
- Destination to forward logs to Loki
- Namespace to collect logs from
- Pod labels to be mapped to Loki labels
- Collection method (kubernetesApi or tailing `/var/log/containers/`)

```yaml
destinations:
- name: loki
  type: loki
  url: http://loki-gateway.$MONITORING_NAMESPACE.svc.cluster.local/loki/api/v1/push
podLogs:
  enabled: true
  gatherMethod: kubernetesApi # collect logs from the kubernetes api, rather than /var/log/containers/; friendly for testing and development
  collector: alloy-logs
  labels:
    app_kubernetes_io_name: app.kubernetes.io/name
    nvidia_com_dynamo_component_type: nvidia.com/dynamo-component-type
    nvidia_com_dynamo_graph_deployment_name: nvidia.com/dynamo-graph-deployment-name
  labelsToKeep:
  - "app_kubernetes_io_name"
  - "container"
  - "instance"
  - "job"
  - "level"
  - "namespace"
  - "service_name"
  - "service_namespace"
  - "deployment_environment"
  - "deployment_environment_name"
  - "nvidia_com_dynamo_component_type" # extract this label from the dynamo graph deployment
  - "nvidia_com_dynamo_graph_deployment_name" # extract this label from the dynamo graph deployment
  namespaces:
  - $DYN_NAMESPACE
```

### 3. Configure Grafana with the Loki datasource and Dynamo Logs dashboard

We will be viewing the logs associated with our DynamoGraphDeployment in Grafana. To do this, we need to configure Grafana with the Loki datasource and Dynamo Logs dashboard.

Since we are using Grafana with the Prometheus Operator, we can simply apply the following ConfigMaps to quickly achieve this configuration.

```bash
# Configure Grafana with the Loki datasource
envsubst < deploy/logging/grafana/loki-datasource.yaml | kubectl apply -n $MONITORING_NAMESPACE -f -

# Configure Grafana with the Dynamo Logs dashboard
envsubst < deploy/logging/grafana/logging-dashboard.yaml | kubectl apply -n $MONITORING_NAMESPACE -f -
```

> [!Note]
> If using Grafana installed without the Prometheus Operator, you can manually import the Loki datasource and Dynamo Logs dashboard using the Grafana UI.

### 4. Deploy a DynamoGraphDeployment with JSONL Logging

At this point, we should have everything in place to collect and view logs in our Grafana instance. All that is left is to deploy a DynamoGraphDeployment to collect logs from.

To enable structured logs in a DynamoGraphDeployment, we need to set the `DYN_LOGGING_JSONL` environment variable to `1`. This is done for us in the `agg_logging.yaml` setup for the Sglang backend. We can now deploy the DynamoGraphDeployment with:

```bash
kubectl apply -n $DYN_NAMESPACE -f components/backends/sglang/deploy/agg_logging.yaml
```

Send a few chat completions requests to generate structured logs across the frontend and worker pods across the DynamoGraphDeployment. We are now all set to view the logs in Grafana.

## Viewing Logs in Grafana

Port-forward the Grafana service to access the UI:

```bash
kubectl port-forward svc/prometheus-grafana 3000:80 -n $MONITORING_NAMESPACE
```

If everything is working, under Home > Dashboards > Dynamo Logs, you should see a dashboard that can be used to view the logs associated with our DynamoGraphDeployments

The dashboard enables filtering by DynamoGraphDeployment, namespace, and component type (e.g frontend, worker, etc).