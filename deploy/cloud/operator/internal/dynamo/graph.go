/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	istioNetworking "istio.io/api/networking/v1beta1"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/imdario/mergo"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
)

// ServiceConfig represents the YAML configuration structure for a service
type DynamoConfig struct {
	Enabled       bool   `yaml:"enabled"`
	Namespace     string `yaml:"namespace"`
	Name          string `yaml:"name"`
	ComponentType string `yaml:"component_type,omitempty"`
}

type Resources struct {
	CPU    *string           `yaml:"cpu,omitempty" json:"cpu,omitempty"`
	Memory *string           `yaml:"memory,omitempty" json:"memory,omitempty"`
	GPU    *string           `yaml:"gpu,omitempty" json:"gpu,omitempty"`
	Custom map[string]string `yaml:"custom,omitempty" json:"custom,omitempty"`
}

type Traffic struct {
	Timeout int `yaml:"timeout"`
}

type Autoscaling struct {
	MinReplicas int `yaml:"min_replicas"`
	MaxReplicas int `yaml:"max_replicas"`
}

type Config struct {
	Dynamo       *DynamoConfig        `yaml:"dynamo,omitempty"`
	Resources    *Resources           `yaml:"resources,omitempty"`
	Traffic      *Traffic             `yaml:"traffic,omitempty"`
	Autoscaling  *Autoscaling         `yaml:"autoscaling,omitempty"`
	HttpExposed  bool                 `yaml:"http_exposed,omitempty"`
	ApiEndpoints []string             `yaml:"api_endpoints,omitempty"`
	Workers      *int32               `yaml:"workers,omitempty"`
	TotalGpus    *int32               `yaml:"total_gpus,omitempty"`
	ExtraPodSpec *common.ExtraPodSpec `yaml:"extraPodSpec,omitempty"`
}

type ServiceConfig struct {
	Name         string              `yaml:"name"`
	Dependencies []map[string]string `yaml:"dependencies,omitempty"`
	Config       Config              `yaml:"config"`
}

type DynDeploymentConfig = map[string]*DynDeploymentServiceConfig

// ServiceConfig represents the configuration for a specific service
type DynDeploymentServiceConfig struct {
	ServiceArgs *ServiceArgs `json:"ServiceArgs,omitempty"`
}

// ServiceArgs represents the arguments that can be passed to any service
type ServiceArgs struct {
	Workers   *int32     `json:"workers,omitempty"`
	Resources *Resources `json:"resources,omitempty"`
	TotalGpus *int32     `json:"total_gpus,omitempty"`
}

func (s ServiceConfig) GetNamespace() *string {
	if s.Config.Dynamo == nil || s.Config.Dynamo.Namespace == "" {
		return nil
	}
	return &s.Config.Dynamo.Namespace
}

func GetDefaultDynamoNamespace(ctx context.Context, dynamoDeployment *v1alpha1.DynamoGraphDeployment) string {
	return fmt.Sprintf("dynamo-%s", dynamoDeployment.Name)
}

func ParseDynDeploymentConfig(ctx context.Context, jsonContent []byte) (DynDeploymentConfig, error) {
	var config DynDeploymentConfig
	err := json.Unmarshal(jsonContent, &config)
	return config, err
}

func SetLwsAnnotations(serviceArgs *ServiceArgs, deployment *v1alpha1.DynamoComponentDeployment) error {
	if serviceArgs.Resources != nil &&
		serviceArgs.Resources.GPU != nil && *serviceArgs.Resources.GPU != "" && *serviceArgs.Resources.GPU != "0" &&
		serviceArgs.TotalGpus != nil && *serviceArgs.TotalGpus > 0 {

		gpusPerNodeStr := *serviceArgs.Resources.GPU
		gpusPerNode, errGpusPerNode := strconv.Atoi(gpusPerNodeStr)

		if errGpusPerNode != nil {
			return fmt.Errorf("failed to parse GPUs per node value '%s' for service %s: %w", gpusPerNodeStr, deployment.Spec.ServiceName, errGpusPerNode)
		}

		// Calculate lwsSize using ceiling division to ensure enough nodes for all GPUs
		lwsSize := (int(*serviceArgs.TotalGpus) + gpusPerNode - 1) / gpusPerNode
		if lwsSize > 1 {
			if deployment.Spec.Annotations == nil {
				deployment.Spec.Annotations = make(map[string]string)
			}
			deployment.Spec.Annotations["nvidia.com/lws-size"] = strconv.Itoa(lwsSize)
			deployment.Spec.Annotations["nvidia.com/deployment-type"] = "leader-worker"
		}
	}
	return nil
}

// GenerateDynamoComponentsDeployments generates a map of DynamoComponentDeployments from a DynamoGraphConfig
func GenerateDynamoComponentsDeployments(ctx context.Context, parentDynamoGraphDeployment *v1alpha1.DynamoGraphDeployment, defaultIngressSpec *v1alpha1.IngressSpec) (map[string]*v1alpha1.DynamoComponentDeployment, error) {
	deployments := make(map[string]*v1alpha1.DynamoComponentDeployment)
	graphDynamoNamespace := ""
	for componentName, component := range parentDynamoGraphDeployment.Spec.Services {
		deployment := &v1alpha1.DynamoComponentDeployment{}
		deployment.Spec.DynamoComponentDeploymentSharedSpec = component.DynamoComponentDeploymentSharedSpec
		deployment.Name = GetDynamoComponentName(parentDynamoGraphDeployment, componentName)
		deployment.Spec.BackendFramework = parentDynamoGraphDeployment.Spec.BackendFramework
		deployment.Namespace = parentDynamoGraphDeployment.Namespace
		deployment.Spec.ServiceName = componentName
		dynamoNamespace := GetDefaultDynamoNamespace(ctx, parentDynamoGraphDeployment)
		if component.DynamoNamespace != nil && *component.DynamoNamespace != "" {
			dynamoNamespace = *component.DynamoNamespace
		}
		if graphDynamoNamespace != "" && graphDynamoNamespace != dynamoNamespace {
			return nil, fmt.Errorf("namespace mismatch for component %s: graph uses namespace %s but component specifies %s", componentName, graphDynamoNamespace, dynamoNamespace)
		}
		graphDynamoNamespace = dynamoNamespace
		deployment.Spec.DynamoNamespace = &dynamoNamespace
		labels := make(map[string]string)
		// add the labels in the spec in order to label all sub-resources
		deployment.Spec.Labels = labels
		// and add the labels to the deployment itself
		deployment.Labels = labels
		labels[commonconsts.KubeLabelDynamoComponent] = componentName
		labels[commonconsts.KubeLabelDynamoNamespace] = dynamoNamespace
		if component.ComponentType == commonconsts.ComponentTypePlanner {
			if deployment.Spec.ExtraPodSpec == nil {
				deployment.Spec.ExtraPodSpec = &common.ExtraPodSpec{}
			}
			deployment.Spec.ExtraPodSpec.ServiceAccountName = commonconsts.PlannerServiceAccountName
		}
		if deployment.IsMainComponent() && defaultIngressSpec != nil && deployment.Spec.Ingress == nil {
			deployment.Spec.Ingress = defaultIngressSpec
		}
		// merge the envs from the parent deployment with the envs from the service
		if len(parentDynamoGraphDeployment.Spec.Envs) > 0 {
			deployment.Spec.Envs = MergeEnvs(parentDynamoGraphDeployment.Spec.Envs, deployment.Spec.Envs)
		}
		err := updateDynDeploymentConfig(deployment, commonconsts.DynamoServicePort)
		if err != nil {
			return nil, err
		}
		err = overrideWithDynDeploymentConfig(ctx, deployment)
		if err != nil {
			return nil, err
		}
		// we only override the replicas if it is not set in the CRD.
		// replicas, if set in the CRD must always be the source of truth.
		if component.Replicas != nil {
			deployment.Spec.Replicas = component.Replicas
		}
		deployments[componentName] = deployment
	}
	return deployments, nil
}

// updateDynDeploymentConfig updates the runtime config object for the given dynamoDeploymentComponent
// It updates the port for the given service (if it is the main component)
func updateDynDeploymentConfig(dynamoDeploymentComponent *v1alpha1.DynamoComponentDeployment, newPort int) error {
	if dynamoDeploymentComponent.IsMainComponent() {
		dynamoDeploymentConfig := dynamoDeploymentComponent.GetDynamoDeploymentConfig()
		if dynamoDeploymentConfig != nil {
			var config map[string]any
			if err := json.Unmarshal(dynamoDeploymentConfig, &config); err != nil {
				return fmt.Errorf("failed to unmarshal %v: %w", commonconsts.DynamoDeploymentConfigEnvVar, err)
			}
			// Safely navigate and update the config
			if serviceConfig, ok := config[dynamoDeploymentComponent.Spec.ServiceName].(map[string]any); ok {
				serviceConfig["port"] = newPort
			}
			// Marshal back to JSON string
			updated, err := json.Marshal(config)
			if err != nil {
				return fmt.Errorf("failed to marshal updated config: %w", err)
			}
			dynamoDeploymentComponent.SetDynamoDeploymentConfig(updated)
		}
	}
	return nil
}

func overrideWithDynDeploymentConfig(ctx context.Context, dynamoDeploymentComponent *v1alpha1.DynamoComponentDeployment) error {
	dynamoDeploymentConfig := dynamoDeploymentComponent.GetDynamoDeploymentConfig()
	if dynamoDeploymentConfig == nil {
		return nil
	}
	dynDeploymentConfig, err := ParseDynDeploymentConfig(ctx, dynamoDeploymentConfig)
	if err != nil {
		return fmt.Errorf("failed to parse %v: %w", commonconsts.DynamoDeploymentConfigEnvVar, err)
	}
	componentDynConfig := dynDeploymentConfig[dynamoDeploymentComponent.Spec.ServiceName]
	if componentDynConfig != nil {
		if componentDynConfig.ServiceArgs != nil && componentDynConfig.ServiceArgs.Workers != nil {
			dynamoDeploymentComponent.Spec.Replicas = componentDynConfig.ServiceArgs.Workers
		}
		if componentDynConfig.ServiceArgs != nil && componentDynConfig.ServiceArgs.Resources != nil {
			requests := &common.ResourceItem{}
			limits := &common.ResourceItem{}
			if dynamoDeploymentComponent.Spec.Resources == nil {
				dynamoDeploymentComponent.Spec.Resources = &common.Resources{
					Requests: requests,
					Limits:   limits,
				}
			} else {
				if dynamoDeploymentComponent.Spec.Resources.Requests != nil {
					requests = dynamoDeploymentComponent.Spec.Resources.Requests
				} else {
					dynamoDeploymentComponent.Spec.Resources.Requests = requests
				}
				if dynamoDeploymentComponent.Spec.Resources.Limits != nil {
					limits = dynamoDeploymentComponent.Spec.Resources.Limits
				} else {
					dynamoDeploymentComponent.Spec.Resources.Limits = limits
				}
			}
			if componentDynConfig.ServiceArgs.Resources.GPU != nil {
				requests.GPU = *componentDynConfig.ServiceArgs.Resources.GPU
				limits.GPU = *componentDynConfig.ServiceArgs.Resources.GPU
			}
			if componentDynConfig.ServiceArgs.Resources.CPU != nil {
				requests.CPU = *componentDynConfig.ServiceArgs.Resources.CPU
				limits.CPU = *componentDynConfig.ServiceArgs.Resources.CPU
			}
			if componentDynConfig.ServiceArgs.Resources.Memory != nil {
				requests.Memory = *componentDynConfig.ServiceArgs.Resources.Memory
				limits.Memory = *componentDynConfig.ServiceArgs.Resources.Memory
			}
			if componentDynConfig.ServiceArgs.Resources.Custom != nil {
				requests.Custom = componentDynConfig.ServiceArgs.Resources.Custom
				limits.Custom = componentDynConfig.ServiceArgs.Resources.Custom
			}
			if err := SetLwsAnnotations(componentDynConfig.ServiceArgs, dynamoDeploymentComponent); err != nil {
				return err
			}
		}
	}
	return nil
}

func MergeEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
	envMap := make(map[string]corev1.EnvVar)

	// Add all common environment variables.
	for _, env := range common {
		envMap[env.Name] = env
	}

	// Override or add with service-specific environment variables.
	for _, env := range specific {
		envMap[env.Name] = env
	}

	// Convert the map back to a slice.
	merged := make([]corev1.EnvVar, 0, len(envMap))
	for _, env := range envMap {
		merged = append(merged, env)
	}
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Name < merged[j].Name
	})
	return merged
}

func GetDynamoComponentName(dynamoDeployment *v1alpha1.DynamoGraphDeployment, component string) string {
	return fmt.Sprintf("%s-%s", dynamoDeployment.Name, strings.ToLower(component))
}

type SecretsRetriever interface {
	GetSecrets(namespace, registry string) ([]string, error)
}

// getNumberOfNodes extracts the NumberOfNodes from DynamoConfig
func getNumberOfNodes(dynamoConfig *v1alpha1.DynamoConfig) int32 {
	if dynamoConfig != nil && dynamoConfig.NumberOfNodes != nil {
		return *dynamoConfig.NumberOfNodes
	}
	return 1 // Default to single node
}

func GenerateComponentService(ctx context.Context, componentName, componentNamespace string) (*corev1.Service, error) {
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      componentName,
			Namespace: componentNamespace,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				commonconsts.KubeLabelDynamoSelector: componentName,
			},
			Ports: []corev1.ServicePort{
				{
					Name:       commonconsts.DynamoServicePortName,
					Port:       commonconsts.DynamoServicePort,
					TargetPort: intstr.FromString(commonconsts.DynamoContainerPortName),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
	return service, nil
}

func GenerateComponentIngress(ctx context.Context, componentName, componentNamespace string, ingressSpec v1alpha1.IngressSpec) *networkingv1.Ingress {
	resourceName := componentName
	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      resourceName,
			Namespace: componentNamespace,
		},
	}
	host := getIngressHost(ingressSpec)
	ingress.Spec = networkingv1.IngressSpec{
		IngressClassName: ingressSpec.IngressControllerClassName,
		Rules: []networkingv1.IngressRule{
			{
				Host: host,
				IngressRuleValue: networkingv1.IngressRuleValue{
					HTTP: &networkingv1.HTTPIngressRuleValue{
						Paths: []networkingv1.HTTPIngressPath{
							{
								Path:     "/",
								PathType: &[]networkingv1.PathType{networkingv1.PathTypePrefix}[0],
								Backend: networkingv1.IngressBackend{
									Service: &networkingv1.IngressServiceBackend{
										Name: resourceName,
										Port: networkingv1.ServiceBackendPort{
											Number: commonconsts.DynamoServicePort,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if ingressSpec.TLS != nil {
		ingress.Spec.TLS = []networkingv1.IngressTLS{
			{
				Hosts:      []string{host},
				SecretName: ingressSpec.TLS.SecretName,
			},
		}
	}
	return ingress
}

func getIngressHost(ingressSpec v1alpha1.IngressSpec) string {
	host := ingressSpec.Host
	if ingressSpec.HostPrefix != nil {
		host = *ingressSpec.HostPrefix + host
	}
	ingressSuffix := commonconsts.DefaultIngressSuffix
	if ingressSpec.HostSuffix != nil {
		ingressSuffix = *ingressSpec.HostSuffix
	}
	return fmt.Sprintf("%s.%s", host, ingressSuffix)
}

func GenerateComponentVirtualService(ctx context.Context, componentName, componentNamespace string, ingressSpec v1alpha1.IngressSpec) *networkingv1beta1.VirtualService {
	vs := &networkingv1beta1.VirtualService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      componentName,
			Namespace: componentNamespace,
		},
	}
	vs.Spec = istioNetworking.VirtualService{
		Hosts: []string{
			getIngressHost(ingressSpec),
		},
		Gateways: []string{*ingressSpec.VirtualServiceGateway},
		Http: []*istioNetworking.HTTPRoute{
			{
				Match: []*istioNetworking.HTTPMatchRequest{
					{
						Uri: &istioNetworking.StringMatch{
							MatchType: &istioNetworking.StringMatch_Prefix{Prefix: "/"},
						},
					},
				},
				Route: []*istioNetworking.HTTPRouteDestination{
					{
						Destination: &istioNetworking.Destination{
							Host: componentName,
							Port: &istioNetworking.PortSelector{
								Number: commonconsts.DynamoServicePort,
							},
						},
					},
				},
			},
		},
	}
	return vs
}

func GenerateDefaultIngressSpec(dynamoDeployment *v1alpha1.DynamoGraphDeployment, ingressConfig controller_common.IngressConfig) v1alpha1.IngressSpec {
	res := v1alpha1.IngressSpec{
		Enabled:           ingressConfig.VirtualServiceGateway != "" || ingressConfig.IngressControllerClassName != "",
		Host:              dynamoDeployment.Name,
		UseVirtualService: ingressConfig.VirtualServiceGateway != "",
	}
	if ingressConfig.IngressControllerClassName != "" {
		res.IngressControllerClassName = &ingressConfig.IngressControllerClassName
	}
	if ingressConfig.IngressControllerTLSSecret != "" {
		res.TLS = &v1alpha1.IngressTLSSpec{
			SecretName: ingressConfig.IngressControllerTLSSecret,
		}
	}
	if ingressConfig.IngressHostSuffix != "" {
		res.HostSuffix = &ingressConfig.IngressHostSuffix
	}
	if ingressConfig.VirtualServiceGateway != "" {
		res.VirtualServiceGateway = &ingressConfig.VirtualServiceGateway
	}
	return res
}

// Helper: mergeContainerCommand returns userCmd if specified, else defaultCmd
func mergeContainerCommand(defaultCmd, userCmd []string) []string {
	if len(userCmd) > 0 {
		return userCmd
	}
	return defaultCmd
}

// Define Role enum for leader/worker/main
// Use this type everywhere instead of string for role

type Role string

const (
	RoleLeader Role = "leader"
	RoleWorker Role = "worker"
	RoleMain   Role = "main"
)

// Update ServiceRole struct for expandRolesForService

type ServiceRole struct {
	Name     string
	Role     Role
	Replicas int32
}

// Update expandRolesForService to use Role
func expandRolesForService(serviceName string, serviceReplicas *int32, numberOfNodes int32) []ServiceRole {
	var roles []ServiceRole
	if numberOfNodes > 1 {
		roles = append(roles, ServiceRole{Name: serviceName + "-ldr", Role: RoleLeader, Replicas: 1})
		roles = append(roles, ServiceRole{Name: serviceName + "-wkr", Role: RoleWorker, Replicas: numberOfNodes - 1})
	} else {
		roles = append(roles, ServiceRole{Name: serviceName, Role: RoleMain, Replicas: *serviceReplicas})
	}
	return roles
}

// Define BackendFramework enum for sglang, vllm, trtllm

type BackendFramework string

const (
	BackendFrameworkSGLang BackendFramework = "sglang"
	BackendFrameworkVLLM   BackendFramework = "vllm"
	BackendFrameworkTRTLLM BackendFramework = "trtllm"
)

// Backend interface for modular backend logic
// Each backend (SGLang, VLLM, etc.) implements this interface
type Backend interface {
	GenerateCommandAndArgs(componentType string, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType) ([]string, []string)
	MergeArgs(defaultArgs, userArgs []string, multinode bool, role Role, componentType string, numberOfNodes int32, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType) []string
}

// BackendFactory creates backend instances based on the framework type
func BackendFactory(backendFramework BackendFramework) Backend {
	switch backendFramework {
	case BackendFrameworkSGLang:
		return &SGLangBackend{}
	case BackendFrameworkVLLM:
		return &VLLMBackend{}
	case BackendFrameworkTRTLLM:
		return nil // Not implemented yet
	default:
		return nil
	}
}

// addStandardEnvVars adds the standard environment variables that are common to both Grove and Controller
func addStandardEnvVars(container *corev1.Container, controllerConfig controller_common.Config) {
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  commonconsts.EnvDynamoServicePort,
		Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
	})

	if controllerConfig.NatsAddress != "" {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "NATS_SERVER",
			Value: controllerConfig.NatsAddress,
		})
	}

	if controllerConfig.EtcdAddress != "" {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "ETCD_ENDPOINTS",
			Value: controllerConfig.EtcdAddress,
		})
	}
}

// GenerateBasePodSpec creates a basic PodSpec with common logic shared between controller and grove
// Includes standard environment variables (DYNAMO_PORT, NATS_SERVER, ETCD_ENDPOINTS)
// Deployment-specific environment merging should be handled by the caller
func GenerateBasePodSpec(
	component *v1alpha1.DynamoComponentDeploymentOverridesSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	namespace string,
	role Role,
	numberOfNodes int32,
	controllerConfig controller_common.Config,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
) (corev1.PodSpec, error) {
	container := corev1.Container{
		Name:           "main",
		LivenessProbe:  component.LivenessProbe,
		ReadinessProbe: component.ReadinessProbe,
		Env:            component.Envs,
		Ports: []corev1.ContainerPort{
			{
				Protocol:      corev1.ProtocolTCP,
				Name:          commonconsts.DynamoContainerPortName,
				ContainerPort: int32(commonconsts.DynamoServicePort),
			},
			{
				Protocol:      corev1.ProtocolTCP,
				Name:          commonconsts.DynamoHealthPortName,
				ContainerPort: int32(commonconsts.DynamoHealthPort),
			},
		},
	}
	backend := BackendFactory(backendFramework)
	if backend == nil {
		return corev1.PodSpec{}, fmt.Errorf("unsupported backend framework: %s", backendFramework)
	}
	cmd, args := backend.GenerateCommandAndArgs(component.ComponentType, numberOfNodes, role, component, multinodeDeploymentType)

	if component.ExtraPodSpec != nil && component.ExtraPodSpec.MainContainer != nil {
		main := component.ExtraPodSpec.MainContainer.DeepCopy()
		if main != nil {
			// merge the extraPodSpec from the parent deployment with the extraPodSpec from the service
			err := mergo.Merge(&container, *main, mergo.WithOverride)
			if err != nil {
				return corev1.PodSpec{}, fmt.Errorf("failed to merge extraPodSpec: %w", err)
			}
			container.Env = MergeEnvs(component.Envs, container.Env)
		}
	}
	isMultinode := numberOfNodes > 1
	container.Command = mergeContainerCommand(cmd, container.Command)
	container.Args = backend.MergeArgs(args, container.Args, isMultinode, role, component.ComponentType, numberOfNodes, component, multinodeDeploymentType)

	resourcesConfig, err := controller_common.GetResourcesConfig(component.Resources)
	if err != nil {
		return corev1.PodSpec{}, fmt.Errorf("failed to get resources config: %w", err)
	}
	if resourcesConfig != nil {
		container.Resources = *resourcesConfig
	}
	imagePullSecrets := []corev1.LocalObjectReference{}
	if secretsRetriever != nil && component.ExtraPodSpec != nil && component.ExtraPodSpec.MainContainer != nil && component.ExtraPodSpec.MainContainer.Image != "" {
		secretsName, err := secretsRetriever.GetSecrets(namespace, component.ExtraPodSpec.MainContainer.Image)
		if err == nil {
			for _, secretName := range secretsName {
				imagePullSecrets = append(imagePullSecrets, corev1.LocalObjectReference{Name: secretName})
			}
		}
	}
	if component.EnvFromSecret != nil {
		container.EnvFrom = append(container.EnvFrom, corev1.EnvFromSource{
			SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: *component.EnvFromSecret},
			},
		})
	}

	// Add standard environment variables (shared between Grove and Controller)
	// This happens before any deployment-specific merging in the wrappers
	addStandardEnvVars(&container, controllerConfig)

	var volumes []corev1.Volume
	if component.PVC != nil {
		volumes = append(volumes, corev1.Volume{
			Name: *component.PVC.Name,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: *component.PVC.Name,
				},
			},
		})
		container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
			Name:      *component.PVC.Name,
			MountPath: *component.PVC.MountPoint,
		})
	}
	shmVolume, shmVolumeMount := generateSharedMemoryVolumeAndMount(isMultinode, &container.Resources)
	volumes = append(volumes, shmVolume)
	container.VolumeMounts = append(container.VolumeMounts, shmVolumeMount)
	var podSpec corev1.PodSpec
	if component.ExtraPodSpec != nil && component.ExtraPodSpec.PodSpec != nil {
		podSpec = *component.ExtraPodSpec.PodSpec.DeepCopy()
	}
	podSpec.Containers = append(podSpec.Containers, container)
	podSpec.Volumes = append(podSpec.Volumes, volumes...)
	podSpec.ImagePullSecrets = append(podSpec.ImagePullSecrets, imagePullSecrets...)
	return podSpec, nil
}

// GeneratePodSpecForComponent creates a PodSpec for Grove deployments (simplified wrapper)
func GeneratePodSpecForComponent(
	component *v1alpha1.DynamoComponentDeploymentOverridesSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	dynamoDeployment *v1alpha1.DynamoGraphDeployment,
	role Role,
	numberOfNodes int32,
	controllerConfig controller_common.Config,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
) (corev1.PodSpec, error) {
	if len(dynamoDeployment.Spec.Envs) > 0 {
		component.Envs = MergeEnvs(dynamoDeployment.Spec.Envs, component.Envs)
	}
	podSpec, err := GenerateBasePodSpec(component, backendFramework, secretsRetriever, dynamoDeployment.Namespace, role, numberOfNodes, controllerConfig, multinodeDeploymentType)
	if err != nil {
		return corev1.PodSpec{}, err
	}
	return podSpec, nil
}

// GenerateGrovePodGangSet generates a Grove PodGangSet for the given deployment, supporting both single-node and multinode cases.
func GenerateGrovePodGangSet(
	ctx context.Context,
	dynamoDeployment *v1alpha1.DynamoGraphDeployment,
	controllerConfig controller_common.Config,
	secretsRetriever SecretsRetriever,
) (*grovev1alpha1.PodGangSet, error) {
	gangSet := &grovev1alpha1.PodGangSet{}
	gangSet.Name = dynamoDeployment.Name
	gangSet.Namespace = dynamoDeployment.Namespace
	gangSet.Spec.Replicas = 1
	if controllerConfig.Grove.TerminationDelay > 0 {
		gangSet.Spec.Template.TerminationDelay = &metav1.Duration{Duration: controllerConfig.Grove.TerminationDelay}
	}
	backendFramework := getBackendFramework(dynamoDeployment)
	var scalingGroups []grovev1alpha1.PodCliqueScalingGroupConfig
	for serviceName, component := range dynamoDeployment.Spec.Services {
		numberOfNodes := getNumberOfNodes(component.DynamoConfig)
		isMultinode := numberOfNodes > 1
		roles := expandRolesForService(serviceName, component.Replicas, numberOfNodes)
		var cliqueNames []string
		for _, r := range roles {
			podSpec, err := GeneratePodSpecForComponent(
				component,
				backendFramework,
				secretsRetriever,
				dynamoDeployment,
				r.Role,
				numberOfNodes,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to generate podSpec for role %s: %w", r.Name, err)
			}
			clique := &grovev1alpha1.PodCliqueTemplateSpec{
				Name: strings.ToLower(string(r.Name)),
				Labels: map[string]string{
					commonconsts.KubeLabelDynamoSelector: GetDynamoComponentName(dynamoDeployment, string(r.Name)),
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					RoleName: strings.ToLower(string(r.Name)),
					Replicas: r.Replicas,
					PodSpec:  podSpec,
				},
			}
			if component.ExtraPodMetadata != nil {
				// merge the extraPodMetadata from the parent deployment with the extraPodMetadata from the service
				mergo.Merge(&clique.Annotations, component.ExtraPodMetadata.Annotations, mergo.WithOverride)
				mergo.Merge(&clique.Labels, component.ExtraPodMetadata.Labels, mergo.WithOverride)
			}
			gangSet.Spec.Template.Cliques = append(gangSet.Spec.Template.Cliques, clique)
			cliqueNames = append(cliqueNames, strings.ToLower(string(r.Name)))
		}
		if isMultinode {
			scalingGroups = append(scalingGroups, grovev1alpha1.PodCliqueScalingGroupConfig{
				Name:        strings.ToLower(serviceName) + "-sg",
				CliqueNames: cliqueNames,
				Replicas:    component.Replicas,
			})
		}
	}
	if len(scalingGroups) > 0 {
		gangSet.Spec.Template.PodCliqueScalingGroupConfigs = scalingGroups
	}
	return gangSet, nil
}

func getBackendFramework(dynamoDeployment *v1alpha1.DynamoGraphDeployment) BackendFramework {
	if dynamoDeployment.Spec.BackendFramework != "" {
		return BackendFramework(dynamoDeployment.Spec.BackendFramework)
	}
	return BackendFrameworkVLLM
}

// ConvertDynamoComponentDeploymentToSpec converts a DynamoComponentDeployment to our component spec interface
// This is a helper for the controller to use our backend logic
func ConvertDynamoComponentDeploymentToSpec(dynComponent *v1alpha1.DynamoComponentDeployment) *v1alpha1.DynamoComponentDeploymentOverridesSpec {
	return &v1alpha1.DynamoComponentDeploymentOverridesSpec{
		DynamoComponentDeploymentSharedSpec: *dynComponent.Spec.DynamoComponentDeploymentSharedSpec.DeepCopy(),
	}
}

// GenerateBasePodSpecForController generates a PodSpec using backend logic for controller usage
// This preserves the base pod generation while allowing controller-specific enhancements
func GenerateBasePodSpecForController(
	dynComponent *v1alpha1.DynamoComponentDeployment,
	secretsRetriever SecretsRetriever,
	controllerConfig controller_common.Config,
	role Role,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
) (corev1.PodSpec, error) {
	// Convert to our interface
	componentSpec := ConvertDynamoComponentDeploymentToSpec(dynComponent)

	numberOfNodes := getNumberOfNodes(dynComponent.Spec.DynamoComponentDeploymentSharedSpec.DynamoConfig)

	// Generate base PodSpec with standard env vars using merged component envs
	podSpec, err := GenerateBasePodSpec(
		componentSpec,
		BackendFramework(dynComponent.Spec.BackendFramework),
		secretsRetriever,
		dynComponent.Namespace,
		role,
		numberOfNodes,
		controllerConfig,
		multinodeDeploymentType,
	)
	if err != nil {
		return corev1.PodSpec{}, err
	}

	return podSpec, nil
}

func generateSharedMemoryVolumeAndMount(isMultinode bool, resources *corev1.ResourceRequirements) (corev1.Volume, corev1.VolumeMount) {
	sharedMemorySizeLimit := resource.MustParse("512Mi")
	// Check if we have memory limits to work with
	memoryLimit := resources.Limits[corev1.ResourceMemory]
	if !memoryLimit.IsZero() {
		// Use 1/4 of memory limit
		calculatedSize := resource.NewQuantity(memoryLimit.Value()/4, resource.BinarySI)
		// Apply bounds: minimum 512Mi, maximum 8Gi
		minSize := resource.MustParse("512Mi")
		maxSize := resource.MustParse("8Gi")

		if calculatedSize.Cmp(minSize) > 0 && calculatedSize.Cmp(maxSize) < 0 {
			sharedMemorySizeLimit = *calculatedSize
		} else if calculatedSize.Cmp(maxSize) >= 0 {
			sharedMemorySizeLimit = maxSize // Cap at maximum
		}
		// If calculatedSize < minSize, keep the 512Mi base
	}
	// For multinode, ensure we have at least 1Gi
	if isMultinode {
		minMultinodeSize := resource.MustParse("1Gi")
		if sharedMemorySizeLimit.Cmp(minMultinodeSize) < 0 {
			sharedMemorySizeLimit = minMultinodeSize
		}
	}
	volume := corev1.Volume{
		Name: commonconsts.KubeValueNameSharedMemory,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{
				Medium:    corev1.StorageMediumMemory,
				SizeLimit: &sharedMemorySizeLimit,
			},
		},
	}
	volumeMount := corev1.VolumeMount{
		Name:      commonconsts.KubeValueNameSharedMemory,
		MountPath: "/dev/shm",
	}
	return volume, volumeMount
}
