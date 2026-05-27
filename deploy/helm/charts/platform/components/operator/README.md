# dynamo-operator

![Version: 1.3.0](https://img.shields.io/badge/Version-1.3.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.3.0](https://img.shields.io/badge/AppVersion-1.3.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| checkpoint.enabled | bool | `false` |  |
| checkpoint.seccomp.disabled | bool | `false` |  |
| checkpoint.seccomp.profile | string | `"profiles/block-iouring.json"` |  |
| checkpoint.storage | object | `{}` |  |
| controllerManager.affinity | object | `{}` |  |
| controllerManager.leaderElection.id | string | `""` |  |
| controllerManager.leaderElection.namespace | string | `""` |  |
| controllerManager.manager.containerSecurityContext.allowPrivilegeEscalation | bool | `false` |  |
| controllerManager.manager.containerSecurityContext.capabilities.drop[0] | string | `"ALL"` |  |
| controllerManager.manager.image.repository | string | `"nvcr.io/nvidia/ai-dynamo/kubernetes-operator"` |  |
| controllerManager.manager.image.tag | string | `""` |  |
| controllerManager.manager.resources.limits.cpu | string | `"1024m"` |  |
| controllerManager.manager.resources.limits.memory | string | `"2Gi"` |  |
| controllerManager.manager.resources.requests.cpu | string | `"512m"` |  |
| controllerManager.manager.resources.requests.memory | string | `"1Gi"` |  |
| controllerManager.podAnnotations | object | `{}` | Extra annotations to add to the controller-manager pod template. Useful for autodiscovery integrations that key off pod annotations (e.g. Datadog `ad.datadoghq.com/manager.checks`, Prometheus `prometheus.io/scrape`) without forking the chart. |
| controllerManager.podLabels | object | `{}` | Extra labels to add to the controller-manager pod template. Useful for unified service tagging conventions (e.g. Datadog `tags.datadoghq.com/manager.{service,env,version}`) and external selectors that target the controller pods directly. |
| controllerManager.replicas | int | `1` |  |
| controllerManager.serviceAccount.annotations | object | `{}` |  |
| controllerManager.tolerations | list | `[]` |  |
| dynamo.components.serviceAccount.annotations | object | `{}` |  |
| dynamo.dockerRegistry.existingSecretName | string | `""` |  |
| dynamo.dockerRegistry.password | string | `""` |  |
| dynamo.dockerRegistry.secure | bool | `true` |  |
| dynamo.dockerRegistry.server | string | `""` |  |
| dynamo.dockerRegistry.useKubernetesSecret | bool | `false` |  |
| dynamo.dockerRegistry.username | string | `"$oauthtoken"` |  |
| dynamo.groveTerminationDelay | string | `"15m"` |  |
| dynamo.metrics.podMonitors.enabled | string | `nil` | Whether to create PodMonitor resources for Prometheus scraping of dynamo components. null=auto-detect (creates PodMonitors if prometheus-operator CRDs exist in cluster), true=always create (use for helm template / GitOps workflows), false=never create |
| dynamo.metrics.prometheusEndpoint | string | `""` |  |
| dynamo.mpiRun.secretName | string | `"mpi-run-ssh-secret"` |  |
| env | list | `[]` |  |
| etcdAddr | string | `""` |  |
| featureGates.gmsSnapshot | bool | `false` | Temporary internal gate for GMS + Snapshot. |
| gpuDiscovery | object | `{"enabled":true}` | DEPRECATED: GPU discovery for namespace-scoped operators is deprecated along with namespace-restricted mode. |
| gpuDiscovery.enabled | bool | `true` | DEPRECATED: Only relevant when namespaceRestriction is enabled, which is deprecated. |
| kubernetesClusterDomain | string | `"cluster.local"` |  |
| metricsService.enabled | string | `nil` | Whether to create the operator metrics Service and ServiceMonitor. null=auto-detect (Service always created; ServiceMonitor created if prometheus-operator CRDs exist), true=always create both (use for helm template / GitOps workflows), false=never create either |
| metricsService.ports[0].name | string | `"metrics"` |  |
| metricsService.ports[0].port | int | `8080` |  |
| metricsService.ports[0].protocol | string | `"TCP"` |  |
| metricsService.ports[0].targetPort | string | `"metrics"` |  |
| metricsService.type | string | `"ClusterIP"` |  |
| modelExpressURL | string | `""` |  |
| namespaceRestriction | object | `{"enabled":false,"lease":{"duration":"30s","renewInterval":"10s"},"targetNamespace":""}` | DEPRECATED: Namespace-restricted mode is deprecated and will be removed in a future release. Use cluster-wide mode (the default) instead. Do not enable this for new deployments. |
| namespaceRestriction.enabled | bool | `false` | DEPRECATED: Do not enable for new deployments. Namespace-restricted mode is deprecated. |
| namespaceRestriction.lease | object | `{"duration":"30s","renewInterval":"10s"}` | DEPRECATED: Only used in namespace-restricted mode, which is deprecated. |
| namespaceRestriction.lease.duration | string | `"30s"` | DEPRECATED: Lease duration for namespace-restricted mode, which is deprecated. |
| namespaceRestriction.lease.renewInterval | string | `"10s"` | DEPRECATED: Lease renew interval for namespace-restricted mode, which is deprecated. |
| namespaceRestriction.targetNamespace | string | `""` | DEPRECATED: Only used in namespace-restricted mode, which is deprecated. |
| natsAddr | string | `""` |  |
| upgradeCRD | bool | `true` |  |
| webhook.caBundle | string | `""` |  |
| webhook.certManager.certificate.duration | string | `"8760h"` |  |
| webhook.certManager.certificate.renewBefore | string | `"360h"` |  |
| webhook.certManager.certificate.rootCA.duration | string | `"87600h"` |  |
| webhook.certManager.certificate.rootCA.renewBefore | string | `"720h"` |  |
| webhook.certManager.enabled | bool | `false` |  |
| webhook.certificateSecret.external | bool | `false` |  |
| webhook.certificateSecret.name | string | `"webhook-server-cert"` |  |
| webhook.failurePolicy | string | `"Fail"` |  |
| webhook.namespaceSelector | object | `{}` |  |
| webhook.timeoutSeconds | int | `10` |  |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)
