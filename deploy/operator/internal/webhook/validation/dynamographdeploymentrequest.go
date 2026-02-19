/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package validation

import (
	"errors"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// toFloat64 converts a numeric value (int or float64) to float64.
// Returns 0 if the value is neither int nor float64.
func toFloat64(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	default:
		return 0
	}
}

// DynamoGraphDeploymentRequestValidator validates DynamoGraphDeploymentRequest resources.
// This validator can be used by both webhooks and controllers for consistent validation.
type DynamoGraphDeploymentRequestValidator struct {
	request               *nvidiacomv1alpha1.DynamoGraphDeploymentRequest
	isClusterWideOperator bool
	gpuDiscoveryEnabled   bool
}

// NewDynamoGraphDeploymentRequestValidator creates a new validator for DynamoGraphDeploymentRequest.
// isClusterWide indicates whether the operator has cluster-wide permissions.
// gpuDiscoveryEnabled indicates whether Helm provisioned node read access for the operator.
func NewDynamoGraphDeploymentRequestValidator(request *nvidiacomv1alpha1.DynamoGraphDeploymentRequest, isClusterWide bool, gpuDiscoveryEnabled bool) *DynamoGraphDeploymentRequestValidator {
	return &DynamoGraphDeploymentRequestValidator{
		request:               request,
		isClusterWideOperator: isClusterWide,
		gpuDiscoveryEnabled:   gpuDiscoveryEnabled,
	}
}

// Validate performs stateless validation on the DynamoGraphDeploymentRequest.
// Returns warnings and error.
func (v *DynamoGraphDeploymentRequestValidator) Validate() (admission.Warnings, error) {
	var warnings admission.Warnings
	var err error

	// Warn about deprecated enableGpuDiscovery field
	if v.request.Spec.EnableGPUDiscovery != nil {
		warnings = append(warnings, "spec.enableGpuDiscovery is deprecated and will be removed in v1beta1. GPU discovery is now always attempted automatically. This field has no effect.")
	}

	// Validate profiler image is specified
	if v.request.Spec.ProfilingConfig.ProfilerImage == "" {
		err = errors.Join(err, errors.New("spec.profilingConfig.profilerImage is required"))
	}

	// Validate that profilingConfig.config is provided
	if v.request.Spec.ProfilingConfig.Config == nil || len(v.request.Spec.ProfilingConfig.Config.Raw) == 0 {
		err = errors.Join(err, errors.New("spec.profilingConfig.config is required and must not be empty"))
	}

	// Note: GPU discovery is now automatic for cluster-wide operators
	// Namespace-restricted operators automatically skip GPU discovery and require manual hardware config

	// Parse config to validate structure (only if config is present)
	if v.request.Spec.ProfilingConfig.Config != nil && len(v.request.Spec.ProfilingConfig.Config.Raw) > 0 {
		var config map[string]any
		if parseErr := yaml.Unmarshal(v.request.Spec.ProfilingConfig.Config.Raw, &config); parseErr != nil {
			err = errors.Join(err, fmt.Errorf("failed to parse spec.profilingConfig.config: %w", parseErr))
		} else {
			// Warn if deployment.model or engine.backend are specified in config (they will be overwritten by spec fields)
			if engineConfig, ok := config["engine"].(map[string]any); ok {
				if backend, ok := engineConfig["backend"].(string); ok && backend != "" && backend != v.request.Spec.Backend {
					warnings = append(warnings, fmt.Sprintf("spec.profilingConfig.config.engine.backend (%s) will be overwritten by spec.backend (%s)", backend, v.request.Spec.Backend))
				}
			}
			if deployment, ok := config["deployment"].(map[string]any); ok {
				if model, ok := deployment["model"].(string); ok && model != "" && model != v.request.Spec.Model {
					warnings = append(warnings, fmt.Sprintf("spec.profilingConfig.config.deployment.model (%s) will be overwritten by spec.model (%s)", model, v.request.Spec.Model))
				}
			}
		}
	}

	// Validate GPU hardware information is available (last, so other errors are collected first)
	if gpuErr := v.validateGPUHardwareInfo(); gpuErr != nil {
		err = errors.Join(err, gpuErr)
	}

	return warnings, err
}

// validateGPUHardwareInfo ensures GPU hardware information will be available for profiling.
// Returns an error at admission time if GPU discovery is disabled and no manual hardware config is provided.
func (v *DynamoGraphDeploymentRequestValidator) validateGPUHardwareInfo() error {
	// Parse profiling config
	var config map[string]any
	if v.request.Spec.ProfilingConfig.Config != nil {
		if err := yaml.Unmarshal(v.request.Spec.ProfilingConfig.Config.Raw, &config); err != nil {
			// Config parse errors will be caught by other validators
			return nil
		}
	} else {
		config = make(map[string]any)
	}

	// Check if manual hardware config is provided
	hardwareVal, hasHardware := config["hardware"]
	var hasManualHardwareConfig bool
	if hasHardware && hardwareVal != nil {
		if hardwareConfig, ok := hardwareVal.(map[string]any); ok {
			// Check if essential hardware fields are provided
			_, hasGPUModel := hardwareConfig["gpuModel"]
			_, hasGPUVram := hardwareConfig["gpuVramMib"]
			_, hasNumGPUs := hardwareConfig["numGpusPerNode"]
			hasManualHardwareConfig = hasGPUModel || hasGPUVram || hasNumGPUs
		}
	}

	// Check if explicit GPU ranges are provided
	var hasExplicitGPURanges bool
	if engineVal, hasEngine := config["engine"]; hasEngine && engineVal != nil {
		if engineConfig, ok := engineVal.(map[string]any); ok {
			minGPUs, hasMin := engineConfig["minNumGpusPerEngine"]
			maxGPUs, hasMax := engineConfig["maxNumGpusPerEngine"]
			// Validate explicit GPU ranges
			if hasMin && hasMax {
				minVal := toFloat64(minGPUs)
				maxVal := toFloat64(maxGPUs)

				// Validate that min <= max
				if minVal > maxVal {
					return fmt.Errorf("invalid GPU range: minNumGpusPerEngine (%v) cannot be greater than maxNumGpusPerEngine (%v)",
						minVal, maxVal)
				}

				hasExplicitGPURanges = minVal > 0 && maxVal > 0
			}
		}
	}

	if hasManualHardwareConfig || hasExplicitGPURanges {
		return nil
	}

	// No manual hardware config provided. Cluster-wide operators always have GPU discovery via node
	// permissions. Namespace-scoped operators rely on Helm-provisioned GPU discovery (gpuDiscovery.enabled).
	if v.isClusterWideOperator || v.gpuDiscoveryEnabled {
		return nil
	}

	return errors.New("GPU hardware configuration required: GPU discovery is disabled (set dynamo-operator.gpuDiscovery.enabled=true in Helm values, or provide hardware config in spec.profilingConfig.config)")
}

// ValidateUpdate performs stateful validation comparing old and new DynamoGraphDeploymentRequest.
// Returns warnings and error.
func (v *DynamoGraphDeploymentRequestValidator) ValidateUpdate(old *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	// TODO: Add update validation logic for DynamoGraphDeploymentRequest
	// Placeholder for future immutability checks
	return nil, nil
}
