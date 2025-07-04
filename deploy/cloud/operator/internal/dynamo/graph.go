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
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
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
func GenerateDynamoComponentsDeployments(ctx context.Context, parentDynamoGraphDeployment *v1alpha1.DynamoGraphDeployment, ingressSpec *v1alpha1.IngressSpec) (map[string]*v1alpha1.DynamoComponentDeployment, error) {
	deployments := make(map[string]*v1alpha1.DynamoComponentDeployment)
	graphDynamoNamespace := ""
	for componentName, component := range parentDynamoGraphDeployment.Spec.Services {
		deployment := &v1alpha1.DynamoComponentDeployment{}
		deployment.Spec.DynamoComponentDeploymentSharedSpec = component.DynamoComponentDeploymentSharedSpec
		deployment.Name = getDynamoComponentName(parentDynamoGraphDeployment, componentName)
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
		if deployment.IsMainComponent() && ingressSpec != nil {
			deployment.Spec.Ingress = *ingressSpec
		}
		// merge the envs from the parent deployment with the envs from the service
		if len(parentDynamoGraphDeployment.Spec.Envs) > 0 {
			deployment.Spec.Envs = mergeEnvs(parentDynamoGraphDeployment.Spec.Envs, deployment.Spec.Envs)
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

func mergeEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
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
	return merged
}

func getDynamoComponentName(dynamoDeployment *v1alpha1.DynamoGraphDeployment, component string) string {
	return fmt.Sprintf("%s-%s", dynamoDeployment.Name, strings.ToLower(component))
}
