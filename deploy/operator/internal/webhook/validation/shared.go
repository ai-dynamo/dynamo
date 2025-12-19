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

package validation

import (
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// SharedSpecValidator validates DynamoComponentDeploymentSharedSpec fields.
// This validator is used by both DynamoComponentDeploymentValidator and DynamoGraphDeploymentValidator
// to provide consistent validation logic for shared spec fields.
type SharedSpecValidator struct {
	spec      *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
	fieldPath string // e.g., "spec" for DCD, "spec.services[foo]" for DGD
}

// NewSharedSpecValidator creates a new validator for DynamoComponentDeploymentSharedSpec.
// fieldPath is used to provide context in error messages (e.g., "spec" or "spec.services[main]").
func NewSharedSpecValidator(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, fieldPath string) *SharedSpecValidator {
	return &SharedSpecValidator{
		spec:      spec,
		fieldPath: fieldPath,
	}
}

// Validate performs validation on the shared spec fields.
// Returns warnings (e.g., deprecation notices) and error if validation fails.
func (v *SharedSpecValidator) Validate() (admission.Warnings, error) {
	// Validate replicas if specified
	if v.spec.Replicas != nil && *v.spec.Replicas < 0 {
		return nil, fmt.Errorf("%s.replicas must be non-negative", v.fieldPath)
	}

	// Validate ingress configuration if enabled
	if v.spec.Ingress != nil && v.spec.Ingress.Enabled {
		if err := v.validateIngress(); err != nil {
			return nil, err
		}
	}

	// Validate volume mounts
	if err := v.validateVolumeMounts(); err != nil {
		return nil, err
	}

	// Validate shared memory
	if v.spec.SharedMemory != nil {
		if err := v.validateSharedMemory(); err != nil {
			return nil, err
		}
	}

	// Collect warnings (e.g., deprecation notices)
	var warnings admission.Warnings

	// Check for deprecated autoscaling field
	//nolint:staticcheck // SA1019: Intentionally checking deprecated field to warn users
	if v.spec.Autoscaling != nil {
		warnings = append(warnings, fmt.Sprintf(
			"%s.autoscaling is deprecated and ignored. Use DynamoGraphDeploymentScalingAdapter "+
				"with HPA, KEDA, or Planner for autoscaling instead. See docs/kubernetes/autoscaling.md",
			v.fieldPath))
	}

	return warnings, nil
}

// validateIngress validates the ingress configuration.
func (v *SharedSpecValidator) validateIngress() error {
	if v.spec.Ingress.Host == "" {
		return fmt.Errorf("%s.ingress.host is required when ingress is enabled", v.fieldPath)
	}
	return nil
}

// validateVolumeMounts validates the volume mount configurations.
func (v *SharedSpecValidator) validateVolumeMounts() error {
	for i, volumeMount := range v.spec.VolumeMounts {
		if err := v.validateVolumeMount(i, &volumeMount); err != nil {
			return err
		}
	}
	return nil
}

// validateVolumeMount validates a single volume mount configuration.
func (v *SharedSpecValidator) validateVolumeMount(index int, volumeMount *nvidiacomv1alpha1.VolumeMount) error {
	// If useAsCompilationCache is false, mountPoint is required
	if !volumeMount.UseAsCompilationCache && volumeMount.MountPoint == "" {
		return fmt.Errorf("%s.volumeMounts[%d].mountPoint is required when useAsCompilationCache is false", v.fieldPath, index)
	}
	return nil
}

// validateSharedMemory validates the shared memory configuration.
func (v *SharedSpecValidator) validateSharedMemory() error {
	// If disabled is false (i.e., shared memory is enabled), size is required
	if !v.spec.SharedMemory.Disabled && v.spec.SharedMemory.Size.IsZero() {
		return fmt.Errorf("%s.sharedMemory.size is required when disabled is false", v.fieldPath)
	}
	return nil
}
