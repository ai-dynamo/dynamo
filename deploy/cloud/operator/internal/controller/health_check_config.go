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
 * Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
 */

package controller

import (
	"encoding/json"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// HealthCheckConfig holds the custom health check paths for a component
type HealthCheckConfig struct {
	LivenessPath  string `json:"livenessPath,omitempty"`
	ReadinessPath string `json:"readinessPath,omitempty"`
}

// DefaultHealthCheckConfig returns the default health check paths
func DefaultHealthCheckConfig() HealthCheckConfig {
	return HealthCheckConfig{
		LivenessPath:  "/healthz",
		ReadinessPath: "/readyz",
	}
}

// GetHealthCheckConfig extracts health check configuration from component deployment config
func GetHealthCheckConfig(dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment, ctrlConfig *commonController.Config) HealthCheckConfig {
	// Start with default paths
	config := DefaultHealthCheckConfig()

	// If controller config has custom paths, use them as base paths
	if ctrlConfig != nil {
		if ctrlConfig.CustomLivenessPath != "" {
			config.LivenessPath = ctrlConfig.CustomLivenessPath
		}
		if ctrlConfig.CustomReadinessPath != "" {
			config.ReadinessPath = ctrlConfig.CustomReadinessPath
		}
	}

	// Try to extract custom paths from deployment config
	deploymentConfig := dynamoComponentDeployment.GetDynamoDeploymentConfig()
	if deploymentConfig != nil {
		var configMap map[string]interface{}
		if err := json.Unmarshal(deploymentConfig, &configMap); err == nil {
			// Look for health check configuration
			if healthConfig, ok := configMap["healthCheck"].(map[string]interface{}); ok {
				if livenessPath, ok := healthConfig["livenessPath"].(string); ok && livenessPath != "" {
					config.LivenessPath = livenessPath
				}
				if readinessPath, ok := healthConfig["readinessPath"].(string); ok && readinessPath != "" {
					config.ReadinessPath = readinessPath
				}
			}
		}
	}

	// Check annotations for health check paths (highest priority)
	annotations := dynamoComponentDeployment.Spec.Annotations
	if annotations != nil {
		if path, ok := annotations["nvidia.com/liveness-path"]; ok && path != "" {
			config.LivenessPath = path
		}
		if path, ok := annotations["nvidia.com/readiness-path"]; ok && path != "" {
			config.ReadinessPath = path
		}
	}

	return config
}

// CreateDefaultLivenessProbe creates a default liveness probe with the specified path
func CreateDefaultLivenessProbe(path string) *corev1.Probe {
	return &corev1.Probe{
		InitialDelaySeconds: 60, // 1 minute
		PeriodSeconds:       60, // Check every 1 minute
		TimeoutSeconds:      5,  // 5 second timeout
		FailureThreshold:    10, // Allow 10 failures before declaring unhealthy
		SuccessThreshold:    1,  // Need 1 success to be considered healthy
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: path,
				Port: intstr.FromString(commonconsts.DynamoHealthPortName),
			},
		},
	}
}

// CreateDefaultReadinessProbe creates a default readiness probe with the specified path
func CreateDefaultReadinessProbe(path string) *corev1.Probe {
	return &corev1.Probe{
		InitialDelaySeconds: 60, // 1 minute
		PeriodSeconds:       60, // Check every 1 minute
		TimeoutSeconds:      5,  // 5 second timeout
		FailureThreshold:    10, // Allow 10 failures before declaring not ready
		SuccessThreshold:    1,  // Need 1 success to be considered ready
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: path,
				Port: intstr.FromString(commonconsts.DynamoHealthPortName),
			},
		},
	}
}

// UpdateProbeWithCustomPath updates an existing probe with a custom path if it's an HTTP probe
func UpdateProbeWithCustomPath(probe *corev1.Probe, path string) *corev1.Probe {
	if probe == nil {
		return nil
	}

	// Only update HTTP probes
	if probe.HTTPGet != nil {
		// Create a deep copy to avoid modifying the original
		newProbe := probe.DeepCopy()
		newProbe.HTTPGet.Path = path
		return newProbe
	}

	return probe
}

// GetProbesWithCustomPaths returns liveness and readiness probes with custom paths
func GetProbesWithCustomPaths(
	dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment,
	ctrlConfig *commonController.Config,
) (*corev1.Probe, *corev1.Probe) {
	// Get health check configuration
	healthConfig := GetHealthCheckConfig(dynamoComponentDeployment, ctrlConfig)

	// Get existing probes from the deployment spec
	livenessProbe := dynamoComponentDeployment.Spec.LivenessProbe
	readinessProbe := dynamoComponentDeployment.Spec.ReadinessProbe

	// If probes are defined, update them with custom paths if they're HTTP probes
	if livenessProbe != nil {
		livenessProbe = UpdateProbeWithCustomPath(livenessProbe, healthConfig.LivenessPath)
	} else {
		// Create default probe with custom path
		livenessProbe = CreateDefaultLivenessProbe(healthConfig.LivenessPath)
	}

	if readinessProbe != nil {
		readinessProbe = UpdateProbeWithCustomPath(readinessProbe, healthConfig.ReadinessPath)
	} else {
		// Create default probe with custom path
		readinessProbe = CreateDefaultReadinessProbe(healthConfig.ReadinessPath)
	}

	return livenessProbe, readinessProbe
}
