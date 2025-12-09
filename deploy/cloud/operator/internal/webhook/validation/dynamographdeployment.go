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
	"errors"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/webhook"
	authenticationv1 "k8s.io/api/authentication/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoGraphDeploymentValidator validates DynamoGraphDeployment resources.
// This validator can be used by both webhooks and controllers for consistent validation.
type DynamoGraphDeploymentValidator struct {
	deployment *nvidiacomv1alpha1.DynamoGraphDeployment
}

// NewDynamoGraphDeploymentValidator creates a new validator for DynamoGraphDeployment.
func NewDynamoGraphDeploymentValidator(deployment *nvidiacomv1alpha1.DynamoGraphDeployment) *DynamoGraphDeploymentValidator {
	return &DynamoGraphDeploymentValidator{
		deployment: deployment,
	}
}

// Validate performs stateless validation on the DynamoGraphDeployment.
// Returns warnings and error.
func (v *DynamoGraphDeploymentValidator) Validate() (admission.Warnings, error) {
	// Validate that at least one service is specified
	if len(v.deployment.Spec.Services) == 0 {
		return nil, fmt.Errorf("spec.services must have at least one service")
	}

	// Validate PVCs
	if err := v.validatePVCs(); err != nil {
		return nil, err
	}

	var allWarnings admission.Warnings

	// Validate each service
	for serviceName, service := range v.deployment.Spec.Services {
		warnings, err := v.validateService(serviceName, service)
		if err != nil {
			return nil, err
		}
		allWarnings = append(allWarnings, warnings...)
	}

	return allWarnings, nil
}

// ValidateUpdate performs stateful validation comparing old and new DynamoGraphDeployment.
// userInfo is used for identity-based validation (replica protection).
// If userInfo is nil, replica changes for DGDSA-enabled services are rejected (fail closed).
// Returns warnings and error.
func (v *DynamoGraphDeploymentValidator) ValidateUpdate(old *nvidiacomv1alpha1.DynamoGraphDeployment, userInfo *authenticationv1.UserInfo) (admission.Warnings, error) {
	var warnings admission.Warnings

	// Validate immutable fields
	if err := v.validateImmutableFields(old, &warnings); err != nil {
		return warnings, err
	}

	// Validate replicas changes for services with scaling adapter enabled
	// Pass userInfo (may be nil - will fail closed for DGDSA-enabled services)
	if err := v.validateReplicasChanges(old, userInfo); err != nil {
		return warnings, err
	}

	return warnings, nil
}

// validateImmutableFields checks that immutable fields have not been changed.
// Appends warnings to the provided slice.
func (v *DynamoGraphDeploymentValidator) validateImmutableFields(old *nvidiacomv1alpha1.DynamoGraphDeployment, warnings *admission.Warnings) error {
	if v.deployment.Spec.BackendFramework != old.Spec.BackendFramework {
		*warnings = append(*warnings, "Changing spec.backendFramework may cause unexpected behavior")
		return fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation")
	}
	return nil
}

// validateReplicasChanges checks if replicas were changed for services with scaling adapter enabled.
// Only authorized service accounts (operator controller, planner) can modify these fields.
// If userInfo is nil, all replica changes for DGDSA-enabled services are rejected (fail closed).
func (v *DynamoGraphDeploymentValidator) validateReplicasChanges(old *nvidiacomv1alpha1.DynamoGraphDeployment, userInfo *authenticationv1.UserInfo) error {
	// If the request comes from an authorized service account, allow the change
	if userInfo != nil && internalwebhook.CanModifyDGDReplicas(*userInfo) {
		return nil
	}

	var errs []error

	for serviceName, newService := range v.deployment.Spec.Services {
		// Check if scaling adapter is enabled for this service (enabled by default)
		scalingAdapterEnabled := true
		if newService.ScalingAdapter != nil && newService.ScalingAdapter.Disable {
			scalingAdapterEnabled = false
		}

		if !scalingAdapterEnabled {
			// Scaling adapter is disabled, users can modify replicas directly
			continue
		}

		// Get old service (if exists)
		oldService, exists := old.Spec.Services[serviceName]
		if !exists {
			// New service, no comparison needed
			continue
		}

		// Check if replicas changed
		oldReplicas := int32(1) // default
		if oldService.Replicas != nil {
			oldReplicas = *oldService.Replicas
		}

		newReplicas := int32(1) // default
		if newService.Replicas != nil {
			newReplicas = *newService.Replicas
		}

		if oldReplicas != newReplicas {
			errs = append(errs, fmt.Errorf(
				"spec.services[%s].replicas cannot be modified directly when scaling adapter is enabled; "+
					"scale or update the related DynamoGraphDeploymentScalingAdapter instead",
				serviceName))
		}
	}

	return errors.Join(errs...)
}

// validateService validates a single service configuration using SharedSpecValidator.
// Returns warnings and error.
func (v *DynamoGraphDeploymentValidator) validateService(serviceName string, service *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) (admission.Warnings, error) {
	// Use SharedSpecValidator to validate service spec (which is a DynamoComponentDeploymentSharedSpec)
	fieldPath := fmt.Sprintf("spec.services[%s]", serviceName)
	sharedValidator := NewSharedSpecValidator(service, fieldPath)
	return sharedValidator.Validate()
}

// validatePVCs validates the PVC configurations.
func (v *DynamoGraphDeploymentValidator) validatePVCs() error {
	for i, pvc := range v.deployment.Spec.PVCs {
		if err := v.validatePVC(i, &pvc); err != nil {
			return err
		}
	}
	return nil
}

// validatePVC validates a single PVC configuration.
func (v *DynamoGraphDeploymentValidator) validatePVC(index int, pvc *nvidiacomv1alpha1.PVC) error {
	var err error

	// Validate name is not nil
	if pvc.Name == nil || *pvc.Name == "" {
		err = errors.Join(err, fmt.Errorf("spec.pvcs[%d].name is required", index))
	}

	// Check if create is true
	if pvc.Create != nil && *pvc.Create {
		// Validate required fields when create is true
		if pvc.StorageClass == "" {
			err = errors.Join(err, fmt.Errorf("spec.pvcs[%d].storageClass is required when create is true", index))
		}

		if pvc.Size.IsZero() {
			err = errors.Join(err, fmt.Errorf("spec.pvcs[%d].size is required when create is true", index))
		}

		if pvc.VolumeAccessMode == "" {
			err = errors.Join(err, fmt.Errorf("spec.pvcs[%d].volumeAccessMode is required when create is true", index))
		}
	}

	return err
}
