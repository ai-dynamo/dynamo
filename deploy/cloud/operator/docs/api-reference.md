// Generated documentation. Please do not edit.
:anchor_prefix: k8s-api

[id="{p}-api-reference"]
== API Reference

.Packages
- xref:{anchor_prefix}-nvidia-com-v1alpha1[$$nvidia.com/v1alpha1$$]


[id="{anchor_prefix}-nvidia-com-v1alpha1"]
=== nvidia.com/v1alpha1

Package v1alpha1 contains API Schema definitions for the nvidia.com v1alpha1 API group

.Resource Types
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeployment[$$DynamoComponentDeployment$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeployment[$$DynamoGraphDeployment$$]



[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-autoscaling"]
==== Autoscaling







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec[$$DynamoComponentDeploymentOverridesSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentsharedspec[$$DynamoComponentDeploymentSharedSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec[$$DynamoComponentDeploymentSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`enabled`* __boolean__ |  |  | 
| *`minReplicas`* __integer__ |  |  | 
| *`maxReplicas`* __integer__ |  |  | 
| *`behavior`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#horizontalpodautoscalerbehavior-v2-autoscaling[$$HorizontalPodAutoscalerBehavior$$]__ |  |  | 
| *`metrics`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#metricspec-v2-autoscaling[$$MetricSpec$$] array__ |  |  | 
|===




[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-basestatus"]
==== BaseStatus







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-basecrd[$$BaseCRD$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`version`* __string__ |  |  | 
| *`state`* __string__ |  |  | 
| *`conditions`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#condition-v1-meta[$$Condition$$] array__ |  |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeployment"]
==== DynamoComponentDeployment



DynamoComponentDeployment is the Schema for the dynamocomponentdeployments API





[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`apiVersion`* __string__ | `nvidia.com/v1alpha1` | |
| *`kind`* __string__ | `DynamoComponentDeployment` | |
| *`metadata`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#objectmeta-v1-meta[$$ObjectMeta$$]__ | Refer to Kubernetes API documentation for fields of `metadata`.
 |  | 
| *`spec`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec[$$DynamoComponentDeploymentSpec$$]__ | Spec defines the desired state for this Dynamo component deployment. + |  | 
| *`status`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentstatus[$$DynamoComponentDeploymentStatus$$]__ | Status reflects the current observed state of the component deployment. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec"]
==== DynamoComponentDeploymentOverridesSpec







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeploymentspec[$$DynamoGraphDeploymentSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`annotations`* __object (keys:string, values:string)__ | Annotations to add to generated Kubernetes resources for this component +
(such as Pod, Service, and Ingress when applicable). + |  | 
| *`labels`* __object (keys:string, values:string)__ | Labels to add to generated Kubernetes resources for this component. + |  | 
| *`serviceName`* __string__ | contains the name of the component + |  | 
| *`componentType`* __string__ | ComponentType indicates the role of this component (for example, "main"). + |  | 
| *`dynamoNamespace`* __string__ | dynamo namespace of the service (allows to override the dynamo namespace of the service defined in annotations inside the dynamo archive) + |  | 
| *`resources`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-resources[$$Resources$$]__ | Resources requested and limits for this component, including CPU, memory, +
GPUs/devices, and any runtime-specific resources. + |  | 
| *`autoscaling`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-autoscaling[$$Autoscaling$$]__ | Autoscaling config for this component (replica range, target utilization, etc.). + |  | 
| *`envs`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#envvar-v1-core[$$EnvVar$$] array__ | Envs defines additional environment variables to inject into the component containers. + |  | 
| *`envFromSecret`* __string__ | EnvFromSecret references a Secret whose key/value pairs will be exposed as +
environment variables in the component containers. + |  | 
| *`pvc`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-pvc[$$PVC$$]__ | PVC config describing volumes to be mounted by the component. + |  | 
| *`ingress`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-ingressspec[$$IngressSpec$$]__ | Ingress config to expose the component outside the cluster (or through a service mesh). + |  | 
| *`sharedMemory`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-sharedmemoryspec[$$SharedMemorySpec$$]__ | SharedMemory controls the tmpfs mounted at /dev/shm (enable/disable and size). + |  | 
| *`extraPodMetadata`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-extrapodmetadata[$$ExtraPodMetadata$$]__ | ExtraPodMetadata adds labels/annotations to the created Pods. + |  | 
| *`extraPodSpec`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-extrapodspec[$$ExtraPodSpec$$]__ | ExtraPodSpec allows to override the main pod spec configuration. +
It is a k8s standard PodSpec. It also contains a MainContainer (standard k8s Container) field +
that allows overriding the main container configuration. + |  | 
| *`livenessProbe`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core[$$Probe$$]__ | LivenessProbe to detect and restart unhealthy containers. + |  | 
| *`readinessProbe`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core[$$Probe$$]__ | ReadinessProbe to signal when the container is ready to receive traffic. + |  | 
| *`replicas`* __integer__ | Replicas is the desired number of Pods for this component when autoscaling is not used. + |  | 
| *`multinode`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-multinodespec[$$MultinodeSpec$$]__ | Multinode is the configuration for multinode components. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentsharedspec"]
==== DynamoComponentDeploymentSharedSpec







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec[$$DynamoComponentDeploymentOverridesSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec[$$DynamoComponentDeploymentSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`annotations`* __object (keys:string, values:string)__ | Annotations to add to generated Kubernetes resources for this component +
(such as Pod, Service, and Ingress when applicable). + |  | 
| *`labels`* __object (keys:string, values:string)__ | Labels to add to generated Kubernetes resources for this component. + |  | 
| *`serviceName`* __string__ | contains the name of the component + |  | 
| *`componentType`* __string__ | ComponentType indicates the role of this component (for example, "main"). + |  | 
| *`dynamoNamespace`* __string__ | dynamo namespace of the service (allows to override the dynamo namespace of the service defined in annotations inside the dynamo archive) + |  | 
| *`resources`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-resources[$$Resources$$]__ | Resources requested and limits for this component, including CPU, memory, +
GPUs/devices, and any runtime-specific resources. + |  | 
| *`autoscaling`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-autoscaling[$$Autoscaling$$]__ | Autoscaling config for this component (replica range, target utilization, etc.). + |  | 
| *`envs`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#envvar-v1-core[$$EnvVar$$] array__ | Envs defines additional environment variables to inject into the component containers. + |  | 
| *`envFromSecret`* __string__ | EnvFromSecret references a Secret whose key/value pairs will be exposed as +
environment variables in the component containers. + |  | 
| *`pvc`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-pvc[$$PVC$$]__ | PVC config describing volumes to be mounted by the component. + |  | 
| *`ingress`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-ingressspec[$$IngressSpec$$]__ | Ingress config to expose the component outside the cluster (or through a service mesh). + |  | 
| *`sharedMemory`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-sharedmemoryspec[$$SharedMemorySpec$$]__ | SharedMemory controls the tmpfs mounted at /dev/shm (enable/disable and size). + |  | 
| *`extraPodMetadata`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-extrapodmetadata[$$ExtraPodMetadata$$]__ | ExtraPodMetadata adds labels/annotations to the created Pods. + |  | 
| *`extraPodSpec`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-extrapodspec[$$ExtraPodSpec$$]__ | ExtraPodSpec allows to override the main pod spec configuration. +
It is a k8s standard PodSpec. It also contains a MainContainer (standard k8s Container) field +
that allows overriding the main container configuration. + |  | 
| *`livenessProbe`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core[$$Probe$$]__ | LivenessProbe to detect and restart unhealthy containers. + |  | 
| *`readinessProbe`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core[$$Probe$$]__ | ReadinessProbe to signal when the container is ready to receive traffic. + |  | 
| *`replicas`* __integer__ | Replicas is the desired number of Pods for this component when autoscaling is not used. + |  | 
| *`multinode`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-multinodespec[$$MultinodeSpec$$]__ | Multinode is the configuration for multinode components. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec"]
==== DynamoComponentDeploymentSpec



DynamoComponentDeploymentSpec defines the desired state of DynamoComponentDeployment



.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeployment[$$DynamoComponentDeployment$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`dynamoComponent`* __string__ | DynamoComponent selects the Dynamo component from the archive to deploy. +
Typically corresponds to a component defined in the packaged Dynamo artifacts. + |  | 
| *`dynamoTag`* __string__ | contains the tag of the DynamoComponent: for example, "my_package:MyService" + |  | 
| *`backendFramework`* __string__ | BackendFramework specifies the backend framework (e.g., "sglang", "vllm", "trtllm") + |  | Enum: [sglang vllm trtllm] +

| *`annotations`* __object (keys:string, values:string)__ | Annotations to add to generated Kubernetes resources for this component +
(such as Pod, Service, and Ingress when applicable). + |  | 
| *`labels`* __object (keys:string, values:string)__ | Labels to add to generated Kubernetes resources for this component. + |  | 
| *`serviceName`* __string__ | contains the name of the component + |  | 
| *`componentType`* __string__ | ComponentType indicates the role of this component (for example, "main"). + |  | 
| *`dynamoNamespace`* __string__ | dynamo namespace of the service (allows to override the dynamo namespace of the service defined in annotations inside the dynamo archive) + |  | 
| *`resources`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-resources[$$Resources$$]__ | Resources requested and limits for this component, including CPU, memory, +
GPUs/devices, and any runtime-specific resources. + |  | 
| *`autoscaling`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-autoscaling[$$Autoscaling$$]__ | Autoscaling config for this component (replica range, target utilization, etc.). + |  | 
| *`envs`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#envvar-v1-core[$$EnvVar$$] array__ | Envs defines additional environment variables to inject into the component containers. + |  | 
| *`envFromSecret`* __string__ | EnvFromSecret references a Secret whose key/value pairs will be exposed as +
environment variables in the component containers. + |  | 
| *`pvc`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-pvc[$$PVC$$]__ | PVC config describing volumes to be mounted by the component. + |  | 
| *`ingress`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-ingressspec[$$IngressSpec$$]__ | Ingress config to expose the component outside the cluster (or through a service mesh). + |  | 
| *`sharedMemory`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-sharedmemoryspec[$$SharedMemorySpec$$]__ | SharedMemory controls the tmpfs mounted at /dev/shm (enable/disable and size). + |  | 
| *`extraPodMetadata`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-extrapodmetadata[$$ExtraPodMetadata$$]__ | ExtraPodMetadata adds labels/annotations to the created Pods. + |  | 
| *`extraPodSpec`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-dynamo-common-extrapodspec[$$ExtraPodSpec$$]__ | ExtraPodSpec allows to override the main pod spec configuration. +
It is a k8s standard PodSpec. It also contains a MainContainer (standard k8s Container) field +
that allows overriding the main container configuration. + |  | 
| *`livenessProbe`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core[$$Probe$$]__ | LivenessProbe to detect and restart unhealthy containers. + |  | 
| *`readinessProbe`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core[$$Probe$$]__ | ReadinessProbe to signal when the container is ready to receive traffic. + |  | 
| *`replicas`* __integer__ | Replicas is the desired number of Pods for this component when autoscaling is not used. + |  | 
| *`multinode`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-multinodespec[$$MultinodeSpec$$]__ | Multinode is the configuration for multinode components. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentstatus"]
==== DynamoComponentDeploymentStatus



DynamoComponentDeploymentStatus defines the observed state of DynamoComponentDeployment



.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeployment[$$DynamoComponentDeployment$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`conditions`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#condition-v1-meta[$$Condition$$] array__ | INSERT ADDITIONAL STATUS FIELD - define observed state of cluster +
Important: Run "make" to regenerate code after modifying this file +
Conditions captures the latest observed state of the component (including +
availability and readiness) using standard Kubernetes condition types. + |  | 
| *`podSelector`* __object (keys:string, values:string)__ | PodSelector contains the labels that can be used to select Pods belonging to +
this component deployment. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeployment"]
==== DynamoGraphDeployment



DynamoGraphDeployment is the Schema for the dynamographdeployments API.





[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`apiVersion`* __string__ | `nvidia.com/v1alpha1` | |
| *`kind`* __string__ | `DynamoGraphDeployment` | |
| *`metadata`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#objectmeta-v1-meta[$$ObjectMeta$$]__ | Refer to Kubernetes API documentation for fields of `metadata`.
 |  | 
| *`spec`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeploymentspec[$$DynamoGraphDeploymentSpec$$]__ | Spec defines the desired state for this graph deployment. + |  | 
| *`status`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeploymentstatus[$$DynamoGraphDeploymentStatus$$]__ | Status reflects the current observed state of this graph deployment. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeploymentspec"]
==== DynamoGraphDeploymentSpec



DynamoGraphDeploymentSpec defines the desired state of DynamoGraphDeployment.



.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeployment[$$DynamoGraphDeployment$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`dynamoGraph`* __string__ | DynamoGraph selects the graph (workflow/topology) to deploy. This must match +
a graph name packaged with the Dynamo archive. + |  | 
| *`services`* __object (keys:string, values:xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec[$$DynamoComponentDeploymentOverridesSpec$$])__ | Services allows per-service overrides of the component deployment settings. +
- key: name of the service defined by the DynamoComponent +
- value: overrides for that service +
If not set for a service, the default DynamoComponentDeployment values are used. + |  | Optional: {} +

| *`envs`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#envvar-v1-core[$$EnvVar$$] array__ | Envs are environment variables applied to all services in the graph unless +
overridden by service-specific configuration. + |  | Optional: {} +

| *`backendFramework`* __string__ | BackendFramework specifies the backend framework (e.g., "sglang", "vllm", "trtllm"). + |  | Enum: [sglang vllm trtllm] +

|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeploymentstatus"]
==== DynamoGraphDeploymentStatus



DynamoGraphDeploymentStatus defines the observed state of DynamoGraphDeployment.



.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamographdeployment[$$DynamoGraphDeployment$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`state`* __string__ | State is a high-level textual status of the graph deployment lifecycle. + |  | 
| *`conditions`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#condition-v1-meta[$$Condition$$] array__ | Conditions contains the latest observed conditions of the graph deployment. +
The slice is merged by type on patch updates. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-ingressspec"]
==== IngressSpec







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec[$$DynamoComponentDeploymentOverridesSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentsharedspec[$$DynamoComponentDeploymentSharedSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec[$$DynamoComponentDeploymentSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`enabled`* __boolean__ | Enabled exposes the component through an ingress or virtual service when true. + |  | 
| *`host`* __string__ | Host is the base host name to route external traffic to this component. + |  | 
| *`useVirtualService`* __boolean__ | UseVirtualService indicates whether to configure a service-mesh VirtualService instead of a standard Ingress. + |  | 
| *`virtualServiceGateway`* __string__ | VirtualServiceGateway optionally specifies the gateway name to attach the VirtualService to. + |  | 
| *`hostPrefix`* __string__ | HostPrefix is an optional prefix added before the host. + |  | 
| *`annotations`* __object (keys:string, values:string)__ | Annotations to set on the generated Ingress/VirtualService resources. + |  | 
| *`labels`* __object (keys:string, values:string)__ | Labels to set on the generated Ingress/VirtualService resources. + |  | 
| *`tls`* __xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-ingresstlsspec[$$IngressTLSSpec$$]__ | TLS holds the TLS configuration used by the Ingress/VirtualService. + |  | 
| *`hostSuffix`* __string__ | HostSuffix is an optional suffix appended after the host. + |  | 
| *`ingressControllerClassName`* __string__ | IngressControllerClassName selects the ingress controller class (e.g., "nginx"). + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-ingresstlsspec"]
==== IngressTLSSpec







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-ingressspec[$$IngressSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`secretName`* __string__ | SecretName is the name of a Kubernetes Secret containing the TLS certificate and key. + |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-multinodespec"]
==== MultinodeSpec







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec[$$DynamoComponentDeploymentOverridesSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentsharedspec[$$DynamoComponentDeploymentSharedSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec[$$DynamoComponentDeploymentSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`nodeCount`* __integer__ | Indicates the number of nodes to deploy for multinode components. +
Total number of GPUs is NumberOfNodes * GPU limit. +
Must be greater than 1. + | 2 | Minimum: 2 +

|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-pvc"]
==== PVC







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec[$$DynamoComponentDeploymentOverridesSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentsharedspec[$$DynamoComponentDeploymentSharedSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec[$$DynamoComponentDeploymentSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`create`* __boolean__ | Create indicates to create a new PVC + |  | 
| *`name`* __string__ | Name is the name of the PVC + |  | 
| *`storageClass`* __string__ | StorageClass to be used for PVC creation. Leave it as empty if the PVC is already created. + |  | 
| *`size`* __xref:{anchor_prefix}-k8s-io-apimachinery-pkg-api-resource-quantity[$$Quantity$$]__ | Size of the NIM cache in Gi, used during PVC creation + |  | 
| *`volumeAccessMode`* __link:https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#persistentvolumeaccessmode-v1-core[$$PersistentVolumeAccessMode$$]__ | VolumeAccessMode is the volume access mode of the PVC + |  | 
| *`mountPoint`* __string__ |  |  | 
|===


[id="{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-sharedmemoryspec"]
==== SharedMemorySpec







.Appears In:
****
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentoverridesspec[$$DynamoComponentDeploymentOverridesSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentsharedspec[$$DynamoComponentDeploymentSharedSpec$$]
- xref:{anchor_prefix}-github-com-ai-dynamo-dynamo-deploy-cloud-operator-api-v1alpha1-dynamocomponentdeploymentspec[$$DynamoComponentDeploymentSpec$$]
****

[cols="20a,50a,15a,15a", options="header"]
|===
| Field | Description | Default | Validation
| *`disabled`* __boolean__ |  |  | 
| *`size`* __xref:{anchor_prefix}-k8s-io-apimachinery-pkg-api-resource-quantity[$$Quantity$$]__ |  |  | 
|===


