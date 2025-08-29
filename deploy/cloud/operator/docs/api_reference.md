# API Reference

## Packages
- [nvidia.com/v1alpha1](#nvidiacomv1alpha1)


## nvidia.com/v1alpha1

Package v1alpha1 contains API Schema definitions for the nvidia.com v1alpha1 API group.

### Resource Types
- [DynamoComponentDeployment](#dynamocomponentdeployment)
- [DynamoGraphDeployment](#dynamographdeployment)



#### Autoscaling







_Appears in:_
- [DynamoComponentDeploymentSharedSpec](#dynamocomponentdeploymentsharedspec)
- [DynamoComponentDeploymentSpec](#dynamocomponentdeploymentspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `enabled` _boolean_ |  |  |  |
| `minReplicas` _integer_ |  |  |  |
| `maxReplicas` _integer_ |  |  |  |
| `behavior` _[HorizontalPodAutoscalerBehavior](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#horizontalpodautoscalerbehavior-v2-autoscaling)_ |  |  |  |
| `metrics` _[MetricSpec](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#metricspec-v2-autoscaling) array_ |  |  |  |




#### DynamoComponentDeployment



DynamoComponentDeployment is the Schema for the dynamocomponentdeployments API





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `nvidia.com/v1alpha1` | | |
| `kind` _string_ | `DynamoComponentDeployment` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[DynamoComponentDeploymentSpec](#dynamocomponentdeploymentspec)_ | Spec defines the desired state for this Dynamo component deployment. |  |  |


#### DynamoComponentDeploymentSharedSpec







_Appears in:_
- [DynamoComponentDeploymentSpec](#dynamocomponentdeploymentspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `annotations` _object (keys:string, values:string)_ | Annotations to add to generated Kubernetes resources for this component<br />(such as Pod, Service, and Ingress when applicable). |  |  |
| `labels` _object (keys:string, values:string)_ | Labels to add to generated Kubernetes resources for this component. |  |  |
| `serviceName` _string_ | The name of the component |  |  |
| `componentType` _string_ | ComponentType indicates the role of this component (for example, "main"). |  |  |
| `dynamoNamespace` _string_ | Dynamo namespace of the service (allows to override the Dynamo namespace of the service defined in annotations inside the Dynamo archive) |  |  |
| `resources` _[Resources](#resources)_ | Resources requested and limits for this component, including CPU, memory,<br />GPUs/devices, and any runtime-specific resources. |  |  |
| `autoscaling` _[Autoscaling](#autoscaling)_ | Autoscaling config for this component (replica range, target utilization, etc.). |  |  |
| `envs` _[EnvVar](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#envvar-v1-core) array_ | Envs defines additional environment variables to inject into the component containers. |  |  |
| `envFromSecret` _string_ | EnvFromSecret references a Secret whose key/value pairs will be exposed as<br />environment variables in the component containers. |  |  |
| `pvc` _[PVC](#pvc)_ | PVC config describing volumes to be mounted by the component. |  |  |
| `ingress` _[IngressSpec](#ingressspec)_ | Ingress config to expose the component outside the cluster (or through a service mesh). |  |  |
| `sharedMemory` _[SharedMemorySpec](#sharedmemoryspec)_ | SharedMemory controls the tmpfs mounted at /dev/shm (enable/disable and size). |  |  |
| `extraPodMetadata` _[ExtraPodMetadata](#extrapodmetadata)_ | ExtraPodMetadata adds labels/annotations to the created Pods. |  |  |
| `extraPodSpec` _[ExtraPodSpec](#extrapodspec)_ | ExtraPodSpec allows to override the main pod spec configuration.<br />It is a k8s standard PodSpec. It also contains a MainContainer (standard k8s Container) field<br />that allows overriding the main container configuration. |  |  |
| `livenessProbe` _[Probe](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core)_ | LivenessProbe to detect and restart unhealthy containers. |  |  |
| `readinessProbe` _[Probe](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core)_ | ReadinessProbe to signal when the container is ready to receive traffic. |  |  |
| `replicas` _integer_ | Replicas is the desired number of Pods for this component when autoscaling is not used. |  |  |
| `multinode` _[MultinodeSpec](#multinodespec)_ | Multinode is the configuration for multinode components. |  |  |


#### DynamoComponentDeploymentSpec



DynamoComponentDeploymentSpec defines the desired state of DynamoComponentDeployment



_Appears in:_
- [DynamoComponentDeployment](#dynamocomponentdeployment)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `dynamoComponent` _string_ | DynamoComponent selects the Dynamo component from the archive to deploy.<br />Typically corresponds to a component defined in the packaged Dynamo artifacts. |  |  |
| `dynamoTag` _string_ | contains the tag of the DynamoComponent: for example, "my_package:MyService" |  |  |
| `backendFramework` _string_ | BackendFramework specifies the backend framework (e.g., "sglang", "vllm", "trtllm") |  | Enum: [sglang vllm trtllm] <br /> |
| `annotations` _object (keys:string, values:string)_ | Annotations to add to generated Kubernetes resources for this component<br />(such as Pod, Service, and Ingress when applicable). |  |  |
| `labels` _object (keys:string, values:string)_ | Labels to add to generated Kubernetes resources for this component. |  |  |
| `serviceName` _string_ | The name of the component |  |  |
| `componentType` _string_ | ComponentType indicates the role of this component (for example, "main"). |  |  |
| `dynamoNamespace` _string_ | Dynamo namespace of the service (allows to override the Dynamo namespace of the service defined in annotations inside the Dynamo archive) |  |  |
| `resources` _[Resources](#resources)_ | Resources requested and limits for this component, including CPU, memory,<br />GPUs/devices, and any runtime-specific resources. |  |  |
| `autoscaling` _[Autoscaling](#autoscaling)_ | Autoscaling config for this component (replica range, target utilization, etc.). |  |  |
| `envs` _[EnvVar](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#envvar-v1-core) array_ | Envs defines additional environment variables to inject into the component containers. |  |  |
| `envFromSecret` _string_ | EnvFromSecret references a Secret whose key/value pairs will be exposed as<br />environment variables in the component containers. |  |  |
| `pvc` _[PVC](#pvc)_ | PVC config describing volumes to be mounted by the component. |  |  |
| `ingress` _[IngressSpec](#ingressspec)_ | Ingress config to expose the component outside the cluster (or through a service mesh). |  |  |
| `sharedMemory` _[SharedMemorySpec](#sharedmemoryspec)_ | SharedMemory controls the tmpfs mounted at /dev/shm (enable/disable and size). |  |  |
| `extraPodMetadata` _[ExtraPodMetadata](#extrapodmetadata)_ | ExtraPodMetadata adds labels/annotations to the created Pods. |  |  |
| `extraPodSpec` _[ExtraPodSpec](#extrapodspec)_ | ExtraPodSpec allows to override the main pod spec configuration.<br />It is a k8s standard PodSpec. It also contains a MainContainer (standard k8s Container) field<br />that allows overriding the main container configuration. |  |  |
| `livenessProbe` _[Probe](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core)_ | LivenessProbe to detect and restart unhealthy containers. |  |  |
| `readinessProbe` _[Probe](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#probe-v1-core)_ | ReadinessProbe to signal when the container is ready to receive traffic. |  |  |
| `replicas` _integer_ | Replicas is the desired number of Pods for this component when autoscaling is not used. |  |  |
| `multinode` _[MultinodeSpec](#multinodespec)_ | Multinode is the configuration for multinode components. |  |  |


#### DynamoGraphDeployment



DynamoGraphDeployment is the Schema for the dynamographdeployments API.





| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `apiVersion` _string_ | `nvidia.com/v1alpha1` | | |
| `kind` _string_ | `DynamoGraphDeployment` | | |
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |  |  |
| `spec` _[DynamoGraphDeploymentSpec](#dynamographdeploymentspec)_ | Spec defines the desired state for this graph deployment. |  |  |
| `status` _[DynamoGraphDeploymentStatus](#dynamographdeploymentstatus)_ | Status reflects the current observed state of this graph deployment. |  |  |


#### DynamoGraphDeploymentSpec



DynamoGraphDeploymentSpec defines the desired state of DynamoGraphDeployment.



_Appears in:_
- [DynamoGraphDeployment](#dynamographdeployment)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `dynamoGraph` _string_ | DynamoGraph selects the graph (workflow/topology) to deploy. This must match<br />a graph name packaged with the Dynamo archive. |  |  |
| `envs` _[EnvVar](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#envvar-v1-core) array_ | Envs are environment variables applied to all services in the graph unless<br />overridden by service-specific configuration. |  | Optional: {} <br /> |
| `backendFramework` _string_ | BackendFramework specifies the backend framework (e.g., "sglang", "vllm", "trtllm"). |  | Enum: [sglang vllm trtllm] <br /> |


#### DynamoGraphDeploymentStatus



DynamoGraphDeploymentStatus defines the observed state of DynamoGraphDeployment.



_Appears in:_
- [DynamoGraphDeployment](#dynamographdeployment)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `state` _string_ | State is a high-level textual status of the graph deployment lifecycle. |  |  |
| `conditions` _[Condition](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#condition-v1-meta) array_ | Conditions contains the latest observed conditions of the graph deployment.<br />The slice is merged by type on patch updates. |  |  |


#### IngressSpec







_Appears in:_
- [DynamoComponentDeploymentSharedSpec](#dynamocomponentdeploymentsharedspec)
- [DynamoComponentDeploymentSpec](#dynamocomponentdeploymentspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `enabled` _boolean_ | Enabled exposes the component through an ingress or virtual service when true. |  |  |
| `host` _string_ | Host is the base host name to route external traffic to this component. |  |  |
| `useVirtualService` _boolean_ | UseVirtualService indicates whether to configure a service-mesh VirtualService instead of a standard Ingress. |  |  |
| `virtualServiceGateway` _string_ | VirtualServiceGateway optionally specifies the gateway name to attach the VirtualService to. |  |  |
| `hostPrefix` _string_ | HostPrefix is an optional prefix added before the host. |  |  |
| `annotations` _object (keys:string, values:string)_ | Annotations to set on the generated Ingress/VirtualService resources. |  |  |
| `labels` _object (keys:string, values:string)_ | Labels to set on the generated Ingress/VirtualService resources. |  |  |
| `tls` _[IngressTLSSpec](#ingresstlsspec)_ | TLS holds the TLS configuration used by the Ingress/VirtualService. |  |  |
| `hostSuffix` _string_ | HostSuffix is an optional suffix appended after the host. |  |  |
| `ingressControllerClassName` _string_ | IngressControllerClassName selects the ingress controller class (e.g., "nginx"). |  |  |


#### IngressTLSSpec







_Appears in:_
- [IngressSpec](#ingressspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `secretName` _string_ | SecretName is the name of a Kubernetes Secret containing the TLS certificate and key. |  |  |


#### MultinodeSpec







_Appears in:_
- [DynamoComponentDeploymentSharedSpec](#dynamocomponentdeploymentsharedspec)
- [DynamoComponentDeploymentSpec](#dynamocomponentdeploymentspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `nodeCount` _integer_ | Indicates the number of nodes to deploy for multinode components.<br />Total number of GPUs is NumberOfNodes * GPU limit.<br />Must be greater than 1. | 2 | Minimum: 2 <br /> |


#### PVC







_Appears in:_
- [DynamoComponentDeploymentSharedSpec](#dynamocomponentdeploymentsharedspec)
- [DynamoComponentDeploymentSpec](#dynamocomponentdeploymentspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `create` _boolean_ | Create indicates to create a new PVC |  |  |
| `name` _string_ | Name is the name of the PVC |  |  |
| `storageClass` _string_ | StorageClass to be used for PVC creation. Leave it as empty if the PVC is already created. |  |  |
| `size` _[Quantity](#quantity)_ | Size of the NIM cache in Gi, used during PVC creation |  |  |
| `volumeAccessMode` _[PersistentVolumeAccessMode](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/#persistentvolumeaccessmode-v1-core)_ | VolumeAccessMode is the volume access mode of the PVC |  |  |
| `mountPoint` _string_ |  |  |  |


#### SharedMemorySpec







_Appears in:_
- [DynamoComponentDeploymentSharedSpec](#dynamocomponentdeploymentsharedspec)
- [DynamoComponentDeploymentSpec](#dynamocomponentdeploymentspec)

| Field | Description | Default | Validation |
| --- | --- | --- | --- |
| `disabled` _boolean_ |  |  |  |
| `size` _[Quantity](#quantity)_ |  |  |  |


