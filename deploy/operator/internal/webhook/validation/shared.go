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
	"context"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controllercommon "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// SharedSpecValidator validates DynamoComponentDeploymentSharedSpec fields.
// This validator is used by both DynamoComponentDeploymentValidator and DynamoGraphDeploymentValidator
// to provide consistent validation logic for shared spec fields.
type SharedSpecValidator struct {
	spec                *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
	fieldPath           string       // e.g., "spec" for DCD, "spec.services[foo]" for DGD
	calculatedNamespace string       // The namespace that will be used: {k8s_namespace}-{dgd_name}
	mgr                 ctrl.Manager // Optional: for API group detection via discovery client
}

// NewSharedSpecValidator creates a new validator for DynamoComponentDeploymentSharedSpec.
// fieldPath is used to provide context in error messages (e.g., "spec" or "spec.services[main]").
// calculatedNamespace is the namespace the operator will use:
//   - If GlobalDynamoNamespace is true: "dynamo" (global constant)
//   - Otherwise: {k8s_namespace}-{dgd_name}
func NewSharedSpecValidator(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, fieldPath string, calculatedNamespace string) *SharedSpecValidator {
	return &SharedSpecValidator{
		spec:                spec,
		fieldPath:           fieldPath,
		calculatedNamespace: calculatedNamespace,
		mgr:                 nil,
	}
}

// NewSharedSpecValidatorWithManager creates a validator with a manager for API group detection.
// This allows the validator to check for API group availability (e.g., inference.networking.k8s.io) when validating EPP components.
func NewSharedSpecValidatorWithManager(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, fieldPath string, calculatedNamespace string, mgr ctrl.Manager) *SharedSpecValidator {
	return &SharedSpecValidator{
		spec:                spec,
		fieldPath:           fieldPath,
		calculatedNamespace: calculatedNamespace,
		mgr:                 mgr,
	}
}

// Validate performs validation on the shared spec fields.
// Context is required for any operations that may need to query the cluster (e.g., CRD checks).
// Returns warnings (e.g., deprecation notices) and error if validation fails.
func (v *SharedSpecValidator) Validate(ctx context.Context) (admission.Warnings, error) {
	// Collect warnings (e.g., deprecation notices)
	var warnings admission.Warnings

	// Warn about deprecated dynamoNamespace field
	if v.spec.DynamoNamespace != nil && *v.spec.DynamoNamespace != "" {
		warnings = append(warnings, fmt.Sprintf(
			"%s.dynamoNamespace is deprecated and ignored. Value '%s' will be replaced with '%s'. "+
				"Remove this field from your configuration",
			v.fieldPath, *v.spec.DynamoNamespace, v.calculatedNamespace))
	}

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

	// Check for deprecated autoscaling field
	//nolint:staticcheck // SA1019: Intentionally checking deprecated field to warn users
	if v.spec.Autoscaling != nil {
		warnings = append(warnings, fmt.Sprintf(
			"%s.autoscaling is deprecated and ignored. Use DynamoGraphDeploymentScalingAdapter "+
				"with HPA, KEDA, or Planner for autoscaling instead. See docs/kubernetes/autoscaling.md",
			v.fieldPath))
	}

	// Validate EPP-specific constraints
	if err := v.validateEPPConfig(ctx); err != nil {
		return nil, err
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

// validateEPPConfig validates EPP-specific configuration constraints.
func (v *SharedSpecValidator) validateEPPConfig(ctx context.Context) error {
	// Only validate if this is an EPP component
	if v.spec.ComponentType != consts.ComponentTypeEPP {
		return nil
	}

	// Check if InferencePool API group is available in the cluster (if manager is provided)
	if v.mgr != nil {
		if err := v.checkInferencePoolAPIAvailability(ctx); err != nil {
			return fmt.Errorf("%s: cannot deploy EPP component: %w", v.fieldPath, err)
		}
	}

	// EPP must be single-node (cannot be multinode)
	if v.spec.IsMultinode() {
		return fmt.Errorf("%s: EPP component cannot be multinode (multinode field must be nil or nodeCount must be 1)", v.fieldPath)
	}

	// EPP should have exactly 1 replica (optional constraint - can be relaxed if needed)
	if v.spec.Replicas != nil && *v.spec.Replicas != 1 {
		return fmt.Errorf("%s: EPP component must have exactly 1 replica (found %d replicas)", v.fieldPath, *v.spec.Replicas)
	}

	// EPP components MUST have EPPConfig
	if v.spec.EPPConfig == nil {
		return fmt.Errorf("%s.eppConfig is required for EPP components", v.fieldPath)
	}

	// Either ConfigMapRef or Config must be specified (no default)
	if v.spec.EPPConfig.ConfigMapRef == nil && v.spec.EPPConfig.Config == nil {
		return fmt.Errorf("%s.eppConfig: either configMapRef or config must be specified (no default configuration provided)", v.fieldPath)
	}

	// ConfigMapRef and Config are mutually exclusive
	if v.spec.EPPConfig.ConfigMapRef != nil && v.spec.EPPConfig.Config != nil {
		return fmt.Errorf("%s.eppConfig: configMapRef and config are mutually exclusive, only one can be specified", v.fieldPath)
	}

	// If ConfigMapRef is provided, validate it
	if v.spec.EPPConfig.ConfigMapRef != nil {
		if v.spec.EPPConfig.ConfigMapRef.Name == "" {
			return fmt.Errorf("%s.eppConfig.configMapRef.name is required", v.fieldPath)
		}
	}

	return nil
}

// checkInferencePoolAPIAvailability checks if the inference.networking.k8s.io API group is available in the cluster.
// Returns an error if the API group is not available, which prevents EPP deployment.
// This reuses the controller_common.DetectInferencePoolAvailability function.
func (v *SharedSpecValidator) checkInferencePoolAPIAvailability(ctx context.Context) error {
	if v.mgr == nil {
		// No manager provided, skip the check (e.g., in controller without webhooks)
		return nil
	}

	if !controllercommon.DetectInferencePoolAvailability(ctx, v.mgr) {
		return fmt.Errorf(
			"InferencePool API group (%s) is not available in the cluster. "+
				"EPP requires the Gateway API Inference Extension to be installed. "+
				"Please install the Gateway API Inference Extension before deploying EPP components",
			epp.InferencePoolGroup)
	}

	return nil
}
