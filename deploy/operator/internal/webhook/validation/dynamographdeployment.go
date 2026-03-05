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
	"errors"
	"fmt"
	"sort"
	"strings"

	semver "github.com/Masterminds/semver/v3"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	authenticationv1 "k8s.io/api/authentication/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	// maxCombinedResourceNameLength is the maximum allowed combined length for Grove resource names.
	// This constraint comes from Grove's PodCliqueSet webhook validation which enforces a 45-character
	// limit on the combined length of PodCliqueSet name + PodCliqueScalingGroup name + PodClique name.
	// Pod names follow formats like: <pcs-name>-<pcs-index>-<pcsg-name>-<pcsg-index>-<pclq-name>-<random>
	// The random string and hyphens consume additional characters, leaving 45 for the resource names.
	maxCombinedResourceNameLength = 45
)

// DynamoGraphDeploymentValidator validates DynamoGraphDeployment resources.
// This validator can be used by both webhooks and controllers for consistent validation.
type DynamoGraphDeploymentValidator struct {
	deployment *nvidiacomv1alpha1.DynamoGraphDeployment
	mgr        ctrl.Manager // Optional: for API group detection via discovery client
}

// NewDynamoGraphDeploymentValidator creates a new validator for DynamoGraphDeployment.
func NewDynamoGraphDeploymentValidator(deployment *nvidiacomv1alpha1.DynamoGraphDeployment) *DynamoGraphDeploymentValidator {
	return &DynamoGraphDeploymentValidator{
		deployment: deployment,
		mgr:        nil,
	}
}

// NewDynamoGraphDeploymentValidatorWithManager creates a validator with a manager for API group detection.
func NewDynamoGraphDeploymentValidatorWithManager(deployment *nvidiacomv1alpha1.DynamoGraphDeployment, mgr ctrl.Manager) *DynamoGraphDeploymentValidator {
	return &DynamoGraphDeploymentValidator{
		deployment: deployment,
		mgr:        mgr,
	}
}

// Validate performs stateless validation on the DynamoGraphDeployment.
// Context is required for operations that may need to query the cluster (e.g., CRD checks).
// Returns warnings and error.
func (v *DynamoGraphDeploymentValidator) Validate(ctx context.Context) (admission.Warnings, error) {
	// Validate that at least one service is specified
	if len(v.deployment.Spec.Services) == 0 {
		return nil, fmt.Errorf("spec.services must have at least one service")
	}

	// Validate annotations
	if err := v.validateAnnotations(); err != nil {
		return nil, err
	}

	// Validate PVCs
	if err := v.validatePVCs(); err != nil {
		return nil, err
	}

	// Validate restart
	if err := v.validateRestart(); err != nil {
		return nil, err
	}

	var allWarnings admission.Warnings

	// Validate each service
	for serviceName, service := range v.deployment.Spec.Services {
		warnings, err := v.validateService(ctx, serviceName, service)
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

	// Validate service topology is unchanged (service names must remain the same)
	if err := v.validateServiceTopology(old); err != nil {
		return warnings, err
	}

	// Validate replicas changes for services with scaling adapter enabled
	// Pass userInfo (may be nil - will fail closed for DGDSA-enabled services)
	if err := v.validateReplicasChanges(old, userInfo); err != nil {
		return warnings, err
	}

	// Validate no restart.id change during active rolling update
	if err := v.validateNoRestartDuringRollingUpdate(old); err != nil {
		return warnings, err
	}

	return warnings, nil
}

// validateImmutableFields checks that immutable fields have not been changed.
// Appends warnings to the provided slice.
func (v *DynamoGraphDeploymentValidator) validateImmutableFields(old *nvidiacomv1alpha1.DynamoGraphDeployment, warnings *admission.Warnings) error {
	var errs []error

	if v.deployment.Spec.BackendFramework != old.Spec.BackendFramework {
		*warnings = append(*warnings, "Changing spec.backendFramework may cause unexpected behavior")
		errs = append(errs, fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation"))
	}

	// Validate that node topology (single-node vs multi-node) is not changed for each service.
	for serviceName, newService := range v.deployment.Spec.Services {
		// Get old service (if exists)
		oldService, exists := old.Spec.Services[serviceName]
		if !exists {
			// New service, no comparison needed
			continue
		}

		if oldService.IsMultinode() != newService.IsMultinode() {
			errs = append(errs, fmt.Errorf(
				"spec.services[%s] cannot change node topology (between single-node and multi-node) after creation",
				serviceName,
			))
		}
	}

	return errors.Join(errs...)

}

// validateServiceTopology ensures the set of service names remains unchanged.
// Users can modify service specifications, but cannot add or remove services.
// This maintains graph topology immutability while allowing configuration updates.
func (v *DynamoGraphDeploymentValidator) validateServiceTopology(old *nvidiacomv1alpha1.DynamoGraphDeployment) error {
	oldServices := getServiceNames(old.Spec.Services)
	newServices := getServiceNames(v.deployment.Spec.Services)

	added := difference(newServices, oldServices)
	removed := difference(oldServices, newServices)

	// Fast path: no changes
	if len(added) == 0 && len(removed) == 0 {
		return nil
	}

	// Sort for deterministic error messages
	sort.Strings(added)
	sort.Strings(removed)

	// Build descriptive error message
	var errMsg string
	switch {
	case len(added) > 0 && len(removed) > 0:
		errMsg = fmt.Sprintf(
			"service topology is immutable and cannot be modified after creation: "+
				"services added: %v, services removed: %v",
			added, removed)
	case len(added) > 0:
		errMsg = fmt.Sprintf(
			"service topology is immutable and cannot be modified after creation: "+
				"services added: %v",
			added)
	case len(removed) > 0:
		errMsg = fmt.Sprintf(
			"service topology is immutable and cannot be modified after creation: "+
				"services removed: %v",
			removed)
	}

	return errors.New(errMsg)
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
		// Check if scaling adapter is enabled for this service (disabled by default)
		scalingAdapterEnabled := newService.ScalingAdapter != nil && newService.ScalingAdapter.Enabled

		if !scalingAdapterEnabled {
			// Scaling adapter is not enabled, users can modify replicas directly
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
func (v *DynamoGraphDeploymentValidator) validateService(ctx context.Context, serviceName string, service *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) (admission.Warnings, error) {
	// Validate service name length constraints for Grove PodCliqueSet naming
	// Only validate when Grove pathway may be in use
	if v.isGrovePathway() {
		if err := v.validateServiceNameLength(serviceName, service); err != nil {
			return nil, err
		}
	}

	// Use SharedSpecValidator to validate service spec (which is a DynamoComponentDeploymentSharedSpec)
	fieldPath := fmt.Sprintf("spec.services[%s]", serviceName)
	calculatedNamespace := v.deployment.GetDynamoNamespaceForService(service)

	var sharedValidator *SharedSpecValidator
	if v.mgr != nil {
		sharedValidator = NewSharedSpecValidatorWithManager(service, fieldPath, calculatedNamespace, v.mgr)
	} else {
		sharedValidator = NewSharedSpecValidator(service, fieldPath, calculatedNamespace)
	}

	return sharedValidator.Validate(ctx)
}

// validateServiceNameLength validates that the service name combined with the DGD name
// won't exceed Grove's 45-character limit for resource naming.
//
// Grove generates PodCliqueSet resources with the following naming patterns:
// - PodCliqueSet name: DGD name (e.g., "vllm-agg")
// - For multinode services:
//   - PodCliqueScalingGroup name: lowercase(serviceName) (e.g., "vllmprefillworker")
//   - PodClique names: lowercase(serviceName + "-ldr") and lowercase(serviceName + "-wkr")
//
// - For single-node services:
//   - PodClique name: lowercase(serviceName)
//
// The combined length of these names must not exceed 45 characters.
func (v *DynamoGraphDeploymentValidator) validateServiceNameLength(serviceName string, service *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) error {
	dgdName := v.deployment.Name
	lowerServiceName := strings.ToLower(serviceName)

	// Check if this is a multinode service
	isMultinode := service.GetNumberOfNodes() > 1

	if isMultinode {
		// For multinode: PodCliqueSet name + PodCliqueScalingGroup name + PodClique name (with leader suffix)
		// The PodClique name is serviceName + "-ldr" (using GroveRoleSuffixLeader)
		leaderPodCliqueName := lowerServiceName + "-" + consts.GroveRoleSuffixLeader
		combinedLength := len(dgdName) + len(lowerServiceName) + len(leaderPodCliqueName)

		if combinedLength > maxCombinedResourceNameLength {
			return fmt.Errorf("spec.services[%s]: combined resource name length %d exceeds %d-character limit required for pod naming. "+
				"Consider shortening the DynamoGraphDeployment name '%s' (length %d) or service name '%s' (length %d). "+
				"For multinode services, the combined length of DGD name + service name + service name with role suffix (e.g., '%s-ldr') must not exceed %d characters",
				serviceName, combinedLength, maxCombinedResourceNameLength,
				dgdName, len(dgdName), serviceName, len(serviceName),
				lowerServiceName, maxCombinedResourceNameLength)
		}
	} else {
		// For single-node: PodCliqueSet name + PodClique name
		combinedLength := len(dgdName) + len(lowerServiceName)

		if combinedLength > maxCombinedResourceNameLength {
			return fmt.Errorf("spec.services[%s]: combined resource name length %d exceeds %d-character limit required for pod naming. "+
				"Consider shortening the DynamoGraphDeployment name '%s' (length %d) or service name '%s' (length %d). "+
				"The combined length of DGD name + service name must not exceed %d characters",
				serviceName, combinedLength, maxCombinedResourceNameLength,
				dgdName, len(dgdName), serviceName, len(serviceName),
				maxCombinedResourceNameLength)
		}
	}

	return nil
}

// isGrovePathway determines if Grove pathway may be used for this deployment.
// Grove is used when the nvidia.com/enable-grove annotation is NOT explicitly set to "false".
// This is a conservative check - if Grove might be used, we validate the name length constraints.
func (v *DynamoGraphDeploymentValidator) isGrovePathway() bool {
	return v.deployment.Annotations == nil ||
		strings.ToLower(v.deployment.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse
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

func (v *DynamoGraphDeploymentValidator) validateRestart() error {
	if v.deployment.Spec.Restart == nil {
		return nil
	}

	restart := v.deployment.Spec.Restart

	var err error
	if restart.ID == "" {
		err = errors.Join(err, fmt.Errorf("spec.restart.id is required"))
	}

	return errors.Join(err, v.validateRestartStrategyOrder())
}

func (v *DynamoGraphDeploymentValidator) validateRestartStrategyOrder() error {
	restart := v.deployment.Spec.Restart
	if restart.Strategy == nil || len(restart.Strategy.Order) == 0 {
		return nil
	}

	if restart.Strategy.Type == nvidiacomv1alpha1.RestartStrategyTypeParallel {
		return errors.New("spec.restart.strategy.order cannot be specified when strategy is parallel")
	}

	var err error

	uniqueOrder := getUnique(restart.Strategy.Order)

	if len(uniqueOrder) != len(restart.Strategy.Order) {
		err = errors.Join(err, fmt.Errorf("spec.restart.strategy.order must be unique"))
	}

	if len(uniqueOrder) != len(v.deployment.Spec.Services) {
		err = errors.Join(err, fmt.Errorf("spec.restart.strategy.order must have the same number of unique services as the deployment"))
	}

	for _, serviceName := range uniqueOrder {
		if _, exists := v.deployment.Spec.Services[serviceName]; !exists {
			err = errors.Join(err, fmt.Errorf("spec.restart.strategy.order contains unknown service: %s", serviceName))
		}
	}

	return err
}

// validateAnnotations validates known DGD annotations have valid values.
func (v *DynamoGraphDeploymentValidator) validateAnnotations() error {
	annotations := v.deployment.GetAnnotations()
	if annotations == nil {
		return nil
	}

	var errs []error

	// Validate operator origin version is valid semver (if present)
	if value, exists := annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; exists {
		if _, err := semver.NewVersion(value); err != nil {
			errs = append(errs, fmt.Errorf("annotation %s has invalid value %q: must be valid semver",
				consts.KubeAnnotationDynamoOperatorOriginVersion, value))
		}
	}

	// Validate vLLM distributed executor backend override
	if value, exists := annotations[consts.KubeAnnotationVLLMDistributedExecutorBackend]; exists {
		switch strings.ToLower(value) {
		case "mp", "ray":
			// valid
		default:
			errs = append(errs, fmt.Errorf("annotation %s has invalid value %q: must be \"mp\" or \"ray\"",
				consts.KubeAnnotationVLLMDistributedExecutorBackend, value))
		}
	}

	return errors.Join(errs...)
}

func getUnique[T comparable](slice []T) []T {
	seen := make(map[T]struct{}, len(slice))
	uniqueSlice := make([]T, 0, len(slice))
	for _, element := range slice {
		if _, exists := seen[element]; !exists {
			seen[element] = struct{}{}
			uniqueSlice = append(uniqueSlice, element)
		}
	}
	return uniqueSlice
}

// getServiceNames extracts service names from a services map.
// Returns a set-like map for efficient lookup and comparison.
func getServiceNames(services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) map[string]struct{} {
	names := make(map[string]struct{}, len(services))
	for name := range services {
		names[name] = struct{}{}
	}
	return names
}

// difference returns elements in set a that are not in set b (a - b).
// This is used to find added or removed services.
func difference(a, b map[string]struct{}) []string {
	var result []string
	for name := range a {
		if _, exists := b[name]; !exists {
			result = append(result, name)
		}
	}
	return result
}

// validateNoRestartDuringRollingUpdate rejects restart.id changes while a rolling update is active.
func (v *DynamoGraphDeploymentValidator) validateNoRestartDuringRollingUpdate(old *nvidiacomv1alpha1.DynamoGraphDeployment) error {
	// Check if a rolling update is active (Pending or InProgress)
	if old.Status.RollingUpdate == nil {
		return nil
	}
	phase := old.Status.RollingUpdate.Phase
	if phase != nvidiacomv1alpha1.RollingUpdatePhasePending && phase != nvidiacomv1alpha1.RollingUpdatePhaseInProgress {
		return nil
	}

	// Compare restart IDs
	oldID := ""
	if old.Spec.Restart != nil {
		oldID = old.Spec.Restart.ID
	}
	newID := ""
	if v.deployment.Spec.Restart != nil {
		newID = v.deployment.Spec.Restart.ID
	}

	if oldID != newID {
		return fmt.Errorf("spec.restart.id cannot be changed while a rolling update is %s", phase)
	}

	return nil
}
