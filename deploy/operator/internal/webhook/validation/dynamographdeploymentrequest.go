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

// DynamoGraphDeploymentRequestValidator validates DynamoGraphDeploymentRequest resources.
// This validator can be used by both webhooks and controllers for consistent validation.
type DynamoGraphDeploymentRequestValidator struct {
	request               *nvidiacomv1alpha1.DynamoGraphDeploymentRequest
	isClusterWideOperator bool
}

// NewDynamoGraphDeploymentRequestValidator creates a new validator for DynamoGraphDeploymentRequest.
// The isClusterWide parameter indicates whether the operator is running in cluster-wide or namespace-restricted mode.
func NewDynamoGraphDeploymentRequestValidator(request *nvidiacomv1alpha1.DynamoGraphDeploymentRequest, isClusterWide bool) *DynamoGraphDeploymentRequestValidator {
	return &DynamoGraphDeploymentRequestValidator{
		request:               request,
		isClusterWideOperator: isClusterWide,
	}
}

// Validate performs stateless validation on the DynamoGraphDeploymentRequest.
// Returns warnings and error.
func (v *DynamoGraphDeploymentRequestValidator) Validate() (admission.Warnings, error) {
	var warnings admission.Warnings
	var err error

	// Validate profiler image is specified
	if v.request.Spec.ProfilingConfig.ProfilerImage == "" {
		err = errors.Join(err, errors.New("spec.profilingConfig.profilerImage is required"))
	}

	// Validate that profilingConfig.config is provided
	if v.request.Spec.ProfilingConfig.Config == nil || len(v.request.Spec.ProfilingConfig.Config.Raw) == 0 {
		err = errors.Join(err, errors.New("spec.profilingConfig.config is required and must not be empty"))
	}

	// Validate enableGpuDiscovery is only true for cluster-wide operators
	if v.request.Spec.EnableGpuDiscovery && !v.isClusterWideOperator {
		err = errors.Join(err, errors.New("spec.enableGpuDiscovery can only be set to true for cluster-wide operators. Namespace-restricted operators cannot access cluster nodes for GPU discovery. Please set enableGpuDiscovery to false and provide hardware configuration (hardware.min_num_gpus_per_engine, hardware.max_num_gpus_per_engine, hardware.num_gpus_per_node) in spec.profilingConfig.config"))
	}

	// Parse config to validate structure (only if config is present)
	if v.request.Spec.ProfilingConfig.Config != nil && len(v.request.Spec.ProfilingConfig.Config.Raw) > 0 {
		var config map[string]interface{}
		if parseErr := yaml.Unmarshal(v.request.Spec.ProfilingConfig.Config.Raw, &config); parseErr != nil {
			err = errors.Join(err, fmt.Errorf("failed to parse spec.profilingConfig.config: %w", parseErr))
		} else {
			// Warn if deployment.model or engine.backend are specified in config (they will be overwritten by spec fields)
			if engineConfig, ok := config["engine"].(map[string]interface{}); ok {
				if backend, ok := engineConfig["backend"].(string); ok && backend != "" && backend != v.request.Spec.Backend {
					warnings = append(warnings, fmt.Sprintf("spec.profilingConfig.config.engine.backend (%s) will be overwritten by spec.backend (%s)", backend, v.request.Spec.Backend))
				}
			}
			if deployment, ok := config["deployment"].(map[string]interface{}); ok {
				if model, ok := deployment["model"].(string); ok && model != "" && model != v.request.Spec.Model {
					warnings = append(warnings, fmt.Sprintf("spec.profilingConfig.config.deployment.model (%s) will be overwritten by spec.model (%s)", model, v.request.Spec.Model))
				}
			}
		}
	}

	return warnings, err
}

// ValidateUpdate performs stateful validation comparing old and new DynamoGraphDeploymentRequest.
// Returns warnings and error.
func (v *DynamoGraphDeploymentRequestValidator) ValidateUpdate(old *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (admission.Warnings, error) {
	// TODO: Add update validation logic for DynamoGraphDeploymentRequest
	// Placeholder for future immutability checks
	return nil, nil
}
