<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# dynamo-platform

A Helm chart for NVIDIA Dynamo Platform.

![Version: 0.7.1](https://img.shields.io/badge/Version-0.7.1-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square)

## 🚀 Overview

The Dynamo Platform Helm chart deploys the complete Dynamo Cloud infrastructure on Kubernetes, including:

- **Dynamo Operator**: Kubernetes operator for managing Dynamo deployments
- **NATS**: High-performance messaging system for component communication
- **etcd**: Distributed key-value store for operator state management
- **Grove**: Multi-node inference orchestration (optional)
- **Kai Scheduler**: Advanced workload scheduling (optional)

## 📋 Prerequisites

- Kubernetes cluster (v1.20+)
- Helm 3.8+
- Sufficient cluster resources for your deployment scale
- Container registry access (if using private images)

## ⚠️ Important: Cluster-Wide vs Namespace-Scoped Deployment

### Single Cluster-Wide Operator (Recommended)

**By default, the Dynamo operator runs with cluster-wide permissions and should only be deployed ONCE per cluster.**

- ✅ **Recommended**: Deploy one cluster-wide operator per cluster
- ❌ **Not Recommended**: Multiple cluster-wide operators in the same cluster

### Multiple Namespace-Scoped Operators (Advanced)

If you need multiple operator instances (e.g., for multi-tenancy), use namespace-scoped deployment:

```yaml
# values.yaml
dynamo-operator:
  namespaceRestriction:
    enabled: true
    targetNamespace: "my-tenant-namespace"  # Optional, defaults to release namespace
```

### Validation and Safety

The chart includes built-in validation to prevent all operator conflicts:

- **Automatic Detection**: Scans for existing operators (both cluster-wide and namespace-restricted) during installation
- **Prevents Multiple Cluster-Wide**: Installation will fail if another cluster-wide operator exists
- **Prevents Mixed Deployments (Type 1)**: Installation will fail if trying to install namespace-restricted operator when cluster-wide exists
- **Prevents Mixed Deployments (Type 2)**: Installation will fail if trying to install cluster-wide operator when namespace-restricted operators exist
- **Safe Defaults**: Leader election uses shared ID for proper coordination

#### 🚫 **Blocked Conflict Scenarios**

| Existing Operator | New Operator | Status | Reason |
|-------------------|--------------|---------|--------|
| None | Cluster-wide | ✅ **Allowed** | No conflicts |
| None | Namespace-restricted | ✅ **Allowed** | No conflicts |
| Cluster-wide | Cluster-wide | ❌ **Blocked** | Multiple cluster managers |
| Cluster-wide | Namespace-restricted | ❌ **Blocked** | Cluster-wide already manages target namespace |
| Namespace-restricted | Cluster-wide | ❌ **Blocked** | Would conflict with existing namespace operators |
| Namespace-restricted A | Namespace-restricted B (diff ns) | ✅ **Allowed** | Different scopes |

## 🔧 Configuration

## Requirements

| Repository | Name | Version |
|------------|------|---------|
| file://components/operator | dynamo-operator | 0.6.1 |
| https://charts.bitnami.com/bitnami | etcd | 12.0.18 |
| https://nats-io.github.io/k8s/helm/charts/ | nats | 1.3.2 |
| oci://ghcr.io/nvidia/grove | grove(grove-charts) | v0.1.0-alpha.3 |
| oci://ghcr.io/nvidia/kai-scheduler | kai-scheduler | v0.9.4 |

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dynamo-operator.enabled | bool | `true` | Whether to enable the Dynamo Kubernetes operator deployment |
| dynamo-operator.natsAddr | string | `""` | NATS server address for operator communication (leave empty to use the bundled NATS chart). Format: "nats://hostname:port" |
| dynamo-operator.etcdAddr | string | `""` | etcd server address for operator state storage (leave empty to use the bundled etcd chart). Format: "http://hostname:port" or "https://hostname:port" |
| dynamo-operator.modelExpressURL | string | `""` | URL for the Model Express server if not deployed by this helm chart. This is ignored if Model Express server is installed by this helm chart (global.model-express.enabled is true). |
| dynamo-operator.namespaceRestriction | object | `{"enabled":false,"lease":{"duration":"30s","renewInterval":"10s"},"targetNamespace":null}` | Namespace access controls for the operator |
| dynamo-operator.namespaceRestriction.enabled | bool | `false` | Whether to restrict operator to specific namespaces. By default, the operator will run with cluster-wide permissions. Only 1 instance of the operator should be deployed in the cluster. If you want to deploy multiple operator instances, you can set this to true and specify the target namespace (by default, the target namespace is the helm release namespace). |
| dynamo-operator.namespaceRestriction.targetNamespace | string | `nil` | Target namespace for operator deployment (leave empty for current namespace) |
| dynamo-operator.controllerManager.tolerations | list | `[]` | Node tolerations for controller manager pods |
| dynamo-operator.controllerManager.affinity | list | `[]` | Affinity for controller manager pods |
| dynamo-operator.controllerManager.leaderElection.id | string | `""` | Leader election ID for cluster-wide coordination. WARNING: All cluster-wide operators must use the SAME ID to prevent split-brain. Different IDs would allow multiple leaders simultaneously. |
| dynamo-operator.controllerManager.leaderElection.namespace | string | `""` | Namespace for leader election leases (only used in cluster-wide mode). If empty, defaults to kube-system for cluster-wide coordination. All cluster-wide operators should use the SAME namespace for proper leader election. |
| dynamo-operator.controllerManager.manager.image.repository | string | `"nvcr.io/nvidia/ai-dynamo/kubernetes-operator"` | Official NVIDIA Dynamo operator image repository |
| dynamo-operator.controllerManager.manager.image.tag | string | `""` | Image tag (leave empty to use chart default) |
| dynamo-operator.controllerManager.manager.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy - when to pull the image |
| dynamo-operator.controllerManager.manager.args[0] | string | `"--health-probe-bind-address=:8081"` | Health probe endpoint for Kubernetes health checks |
| dynamo-operator.controllerManager.manager.args[1] | string | `"--metrics-bind-address=127.0.0.1:8080"` | Metrics endpoint for Prometheus scraping (localhost only for security) |
| dynamo-operator.imagePullSecrets | list | `[]` | Secrets for pulling private container images |
| dynamo-operator.dynamo.groveTerminationDelay | string | `"4h"` | How long to wait before forcefully terminating Grove instances |
| dynamo-operator.dynamo.internalImages.debugger | string | `"python:3.12-slim"` | Debugger image for troubleshooting deployments |
| dynamo-operator.dynamo.enableRestrictedSecurityContext | bool | `false` | Whether to enable restricted security contexts for enhanced security |
| dynamo-operator.dynamo.dockerRegistry.useKubernetesSecret | bool | `false` | Whether to use Kubernetes secrets for registry authentication |
| dynamo-operator.dynamo.dockerRegistry.server | string | `nil` | Docker registry server URL |
| dynamo-operator.dynamo.dockerRegistry.username | string | `nil` | Registry username |
| dynamo-operator.dynamo.dockerRegistry.password | string | `nil` | Registry password (consider using existingSecretName instead) |
| dynamo-operator.dynamo.dockerRegistry.existingSecretName | string | `nil` | Name of existing Kubernetes secret containing registry credentials |
| dynamo-operator.dynamo.dockerRegistry.secure | bool | `true` | Whether the registry uses HTTPS |
| dynamo-operator.dynamo.ingress.enabled | bool | `false` | Whether to create ingress resources |
| dynamo-operator.dynamo.ingress.className | string | `nil` | Ingress class name (e.g., "nginx", "traefik") |
| dynamo-operator.dynamo.ingress.tlsSecretName | string | `"my-tls-secret"` | Secret name containing TLS certificates |
| dynamo-operator.dynamo.istio.enabled | bool | `false` | Whether to enable Istio integration |
| dynamo-operator.dynamo.istio.gateway | string | `nil` | Istio gateway name for routing |
| dynamo-operator.dynamo.ingressHostSuffix | string | `""` | Host suffix for generated ingress hostnames |
| dynamo-operator.dynamo.virtualServiceSupportsHTTPS | bool | `false` | Whether VirtualServices should support HTTPS routing |
| dynamo-operator.dynamo.metrics.prometheusEndpoint | string | `""` | Endpoint that services can use to retrieve metrics. If set, dynamo operator will automatically inject the PROMETHEUS_ENDPOINT environment variable into services it manages. Users can override the value of the PROMETHEUS_ENDPOINT environment variable by modifying the corresponding deployment's environment variables |
| dynamo-operator.dynamo.mpiRun.secretName | string | `"mpi-run-ssh-secret"` | Name of the secret containing the SSH key for MPI Run |
| dynamo-operator.dynamo.mpiRun.sshKeygen.enabled | bool | `true` | Whether to enable SSH key generation for MPI Run |
| dynamo-operator.webhook.enabled | bool | `true` | Whether to enable admission webhooks for resource validation. When enabled, the operator will validate DynamoComponentDeployment and DynamoGraphDeployment resources before they are created or updated in the cluster. Enabled by default for production-ready validation and better error reporting. |
| dynamo-operator.webhook.certificateSecret.name | string | `"webhook-server-cert"` | Name of the Kubernetes secret containing webhook TLS certificates. The secret must contain three keys: tls.crt (server certificate), tls.key (server private key), and ca.crt (Certificate Authority certificate). |
| dynamo-operator.webhook.certificateSecret.external | bool | `false` | Whether to manage the certificate secret externally. When false (default), certificates are automatically generated via Helm hooks during installation. When true, you must create the secret manually before installing the chart. |
| dynamo-operator.webhook.certificateValidity | int | `365` | Certificate validity duration in days for auto-generated certificates. Only used when certManager.enabled=false and certificateSecret.external=false. After this duration, certificates will expire and need to be regenerated. |
| dynamo-operator.webhook.certGenerator.image.repository | string | `"bitnami/kubectl"` | Container image repository for certificate generation jobs. This image must contain both openssl and kubectl commands. |
| dynamo-operator.webhook.certGenerator.image.tag | string | `"latest"` | Container image tag for certificate generation jobs |
| dynamo-operator.webhook.certGenerator.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy for certificate generation jobs |
| dynamo-operator.webhook.caBundle | string | `""` | CA bundle (base64 encoded) for webhook validation. Only used when certificateSecret.external=true. For automatic certificate generation or cert-manager integration, leave this empty as it will be injected automatically. |
| dynamo-operator.webhook.failurePolicy | string | `"Fail"` | Webhook failure policy controls how Kubernetes handles requests when the webhook is unavailable. 'Fail' (recommended for production) rejects requests if the webhook cannot be reached, ensuring strict validation. 'Ignore' allows requests through if the webhook is unavailable, providing availability over validation guarantees. |
| dynamo-operator.webhook.timeoutSeconds | int | `10` | Timeout in seconds for webhook validation calls. If the webhook doesn't respond within this time, the request will be handled according to the failurePolicy. |
| dynamo-operator.webhook.namespaceSelector | object | `{}` | Custom namespace selector for webhook validation. Use this to include or exclude specific namespaces from webhook validation. For CLUSTER-WIDE operators, you can exclude namespaces managed by namespace-restricted operators by using: matchExpressions: [{ key: "dynamo-operator", operator: "NotIn", values: ["namespace-restricted"] }]. For NAMESPACE-RESTRICTED operators, leave empty as it will be auto-configured to match only the operator's namespace. |
| dynamo-operator.webhook.certManager.enabled | bool | `false` | Whether to use cert-manager for automatic certificate management. Requires cert-manager to be installed in the cluster. When enabled, cert-manager will automatically generate, renew, and rotate certificates, and the automatic certificate generation via Helm hooks will be disabled. |
| dynamo-operator.webhook.certManager.certificate.duration | string | `"8760h"` | Certificate duration for webhook certificates managed by cert-manager (e.g., "8760h" for 1 year). cert-manager will automatically renew the certificate before it expires. |
| dynamo-operator.webhook.certManager.certificate.renewBefore | string | `"360h"` | Time before certificate expiration to trigger renewal (e.g., "360h" for 15 days). cert-manager will attempt to renew the certificate when this threshold is reached. |
| dynamo-operator.webhook.certManager.certificate.rootCA.duration | string | `"87600h"` | Duration for the root CA certificate (e.g., "87600h" for 10 years). The root CA typically has a much longer lifetime than the leaf certificates it signs. |
| dynamo-operator.webhook.certManager.certificate.rootCA.renewBefore | string | `"720h"` | Time before root CA expiration to trigger renewal (e.g., "720h" for 30 days). Renewing a CA can be disruptive as all signed certificates must be reissued. |
| grove.enabled | bool | `false` | Whether to enable Grove for multi-node inference coordination, if enabled, the Grove operator will be deployed cluster-wide |
| kai-scheduler.enabled | bool | `false` | Whether to enable Kai Scheduler for intelligent resource allocation, if enabled, the Kai Scheduler operator will be deployed cluster-wide |
| etcd.enabled | bool | `true` | Whether to enable etcd deployment, disable if you want to use an external etcd instance. For complete configuration options, see: https://github.com/bitnami/charts/tree/main/bitnami/etcd , all etcd settings should be prefixed with "etcd." |
| etcd.image.repository | string | `"bitnamilegacy/etcd"` | following bitnami announcement for brownout - https://github.com/bitnami/charts/tree/main/bitnami/etcd#%EF%B8%8F-important-notice-upcoming-changes-to-the-bitnami-catalog, we need to use the legacy repository until we migrate to the new "secure" repository |
| nats.enabled | bool | `true` | Whether to enable NATS deployment, disable if you want to use an external NATS instance. For complete configuration options, see: https://github.com/nats-io/k8s/tree/main/helm/charts/nats , all nats settings should be prefixed with "nats." |

### NATS Configuration

For detailed NATS configuration options beyond `nats.enabled`, please refer to the official NATS Helm chart documentation:
**[NATS Helm Chart Documentation](https://github.com/nats-io/k8s/tree/main/helm/charts/nats)**

### etcd Configuration

For detailed etcd configuration options beyond `etcd.enabled`, please refer to the official Bitnami etcd Helm chart documentation:
**[etcd Helm Chart Documentation](https://github.com/bitnami/charts/tree/main/bitnami/etcd)**

## 📚 Additional Resources

- [Dynamo Cloud Deployment Installation Guide](../../../../docs/kubernetes/installation_guide.md)
- [NATS Documentation](https://docs.nats.io/)
- [etcd Documentation](https://etcd.io/docs/)
- [Kubernetes Operator Pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)
