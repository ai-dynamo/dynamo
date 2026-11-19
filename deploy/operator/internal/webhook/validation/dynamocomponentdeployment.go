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
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	componentdeployment "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/component"
	leaderworkerset "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/leaderworkerset"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoComponentDeploymentValidator validates v1beta1 DynamoComponentDeployment resources.
type DynamoComponentDeploymentValidator struct {
	discoveryBackend configv1alpha1.DiscoveryBackend
}

// NewDynamoComponentDeploymentValidator creates a validator for v1beta1 DynamoComponentDeployment.
// discoveryBackend is the operator's default discovery backend.
func NewDynamoComponentDeploymentValidator(discoveryBackend configv1alpha1.DiscoveryBackend) *DynamoComponentDeploymentValidator {
	return &DynamoComponentDeploymentValidator{discoveryBackend: discoveryBackend}
}

// dynamoComponentDeploymentValidation carries DCD-specific request state.
// API values and derived traversal state remain explicit validator arguments.
type dynamoComponentDeploymentValidation struct {
	sharedValidation
}

// Validate performs stateless validation on the v1beta1 DynamoComponentDeployment.
// ctx and dcd must not be nil.
func (v *DynamoComponentDeploymentValidator) Validate(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (admission.Warnings, error) {
	return v.validate(ctx, dcd, runtimeVersionSourceV1Beta1)
}

func (v *DynamoComponentDeploymentValidator) validate(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	runtimeVersionSource runtimeVersionValidationSource,
) (admission.Warnings, error) {
	validation := &dynamoComponentDeploymentValidation{
		ctx:                                ctx,
		runtimeVersionSource:               runtimeVersionSource,
		discoveryBackend:                   v.discoveryBackend,
		allowMissingRuntimeVersionOverride: true,
	}

	allErrs := validation.validateDynamoComponentDeployment(dcd, nil)
	allErrs = append(allErrs, validation.validateWorkerClassCheckpointRefOwnership(dcd)...)
	alpha, err := alphaDynamoComponentDeploymentForValidation(dcd)
	if err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoComponentDeployment fields: %w", err)
	}
	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentV1alpha1(alpha, nil, dynamo.DCDUsesOrderedEnvironment(dcd))...)

	return validation.warnings, invalidDynamoComponentDeploymentError(dcd, allErrs)
}

// ValidateUpdate performs complete validation of an updated v1beta1 DCD and
// compares its state with the previous object.
// ctx, oldDCD, and newDCD must not be nil. runtimeVersionSource identifies the request's source API.
func (v *DynamoComponentDeploymentValidator) ValidateUpdate(
	ctx context.Context,
	oldDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	newDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	runtimeVersionSource runtimeVersionValidationSource,
) (admission.Warnings, error) {
	validation := &dynamoComponentDeploymentValidation{
		ctx:                                ctx,
		runtimeVersionSource:               runtimeVersionSource,
		discoveryBackend:                   v.discoveryBackend,
		ratchetRuntimeVersion:              true,
		allowMissingRuntimeVersionOverride: true,
	}

	allErrs := validation.validateDynamoComponentDeployment(newDCD, oldDCD)
	allErrs = append(allErrs, validation.validateWorkerClassCheckpointRefOwnership(newDCD)...)
	newAlpha, err := alphaDynamoComponentDeploymentForValidation(newDCD)
	if err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoComponentDeployment fields: %w", err)
	}
	oldAlpha, err := alphaDynamoComponentDeploymentForValidation(oldDCD)
	if err != nil {
		if validation.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
			return nil, fmt.Errorf("cannot validate old v1alpha1 DynamoComponentDeployment fields: %w", err)
		}
		oldAlpha = nil
	}
	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentV1alpha1(newAlpha, oldAlpha, dynamo.DCDUsesOrderedEnvironment(newDCD))...)

	// Run the source-version old/new ratchet after the stateless traversal.
	if validation.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		allErrs = append(allErrs, validation.validateDynamoComponentDeploymentSharedSpecUpdateV1alpha1(
			&newAlpha.Spec.DynamoComponentDeploymentSharedSpec,
			&oldAlpha.Spec.DynamoComponentDeploymentSharedSpec,
			field.NewPath("spec"),
		)...)
	}

	allErrs = append(allErrs, validation.validateDynamoComponentDeploymentUpdate(newDCD, oldDCD)...)
	return validation.warnings, invalidDynamoComponentDeploymentError(newDCD, allErrs)
}

// validateWorkerClassCheckpointRefOwnership limits worker checkpointRef to
// DCDs marked as DGD-controller-owned, the supported source for worker
// compatibility. The owner reference selects behavior; it is not an
// authorization boundary for Snapshot access.
func (v *dynamoComponentDeploymentValidation) validateWorkerClassCheckpointRefOwnership(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) field.ErrorList {
	switch dcd.Spec.ComponentType {
	case nvidiacomv1beta1.ComponentTypeWorker,
		nvidiacomv1beta1.ComponentTypePrefill,
		nvidiacomv1beta1.ComponentTypeDecode:
	default:
		return nil
	}
	if dcd.Spec.Experimental == nil ||
		dcd.Spec.Experimental.Checkpoint == nil ||
		dcd.Spec.Experimental.Checkpoint.CheckpointRef == nil ||
		strings.TrimSpace(*dcd.Spec.Experimental.Checkpoint.CheckpointRef) == "" {
		return nil
	}

	if dcd.GetGraphDeploymentController() != nil {
		return nil
	}

	checkpointRefPath := field.NewPath("spec", "experimental", "checkpoint", "checkpointRef")
	if v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		checkpointRefPath = field.NewPath("spec", "checkpoint", "checkpointRef")
	}
	return field.ErrorList{field.Forbidden(
		checkpointRefPath,
		"worker-class checkpointRef is supported only on DynamoGraphDeployment-managed components",
	)}
}

// validateDynamoComponentDeployment validates dcd. dcd must not be nil; oldDCD is nil on create.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeployment(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	oldDCD *nvidiacomv1beta1.DynamoComponentDeployment,
) field.ErrorList {
	var oldSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec
	var oldMeta *metav1.ObjectMeta
	if oldDCD != nil {
		oldSpec = &oldDCD.Spec
		oldMeta = &oldDCD.ObjectMeta
	}
	allErrs := v.validateObjectMeta(&dcd.ObjectMeta, oldMeta, field.NewPath("metadata"))

	// Conversion preserves the component label identity, including omitted alpha serviceName defaults.
	if oldDCD == nil || dcd.Spec.ComponentName != oldDCD.Spec.ComponentName {
		namePath := field.NewPath("spec", "name")
		if v.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
			namePath = field.NewPath("spec", "serviceName")
		}
		if messages := k8svalidation.IsValidLabelValue(dcd.Spec.ComponentName); len(messages) > 0 {
			allErrs = append(allErrs, field.Invalid(namePath, dcd.Spec.ComponentName, "generated "+consts.KubeLabelDynamoComponent+" label value is invalid: "+strings.Join(messages, "; ")))
		}
	}
	allErrs = append(allErrs, v.validateDynamoComponentDeploymentRendered(dcd, oldDCD, field.NewPath("metadata", "name"))...)
	return append(allErrs, v.validateDynamoComponentDeploymentSpec(
		&dcd.Spec, oldSpec, field.NewPath("spec"),
		dynamo.DCDUsesOrderedEnvironment(dcd), oldDCD != nil && dynamo.DCDUsesOrderedEnvironment(oldDCD),
	)...)
}

// validateDynamoComponentDeploymentSpec validates spec. spec and fldPath must not be nil.
// oldSpec is nil on create; the booleans select the new and old rendering contracts.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentSpec(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	oldSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	fldPath *field.Path,
	orderedEnvironment, oldOrderedEnvironment bool,
) field.ErrorList {
	// Standalone DCDs use neither Grove nor live InferencePool discovery.
	const (
		grovePathway                      = false
		validateInferencePoolAvailability = false
	)
	var oldSharedSpec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
	if oldSpec != nil {
		oldSharedSpec = &oldSpec.DynamoComponentDeploymentSharedSpec
	}

	// Entering the semantic contract does not grandfather an ineligible legacy list.
	var allErrs field.ErrorList
	if orderedEnvironment {
		var oldEnv []corev1.EnvVar
		if oldSpec != nil && oldOrderedEnvironment {
			oldEnv = oldSpec.Env
		}
		allErrs = append(allErrs, v.validateEnvVars(spec.Env, oldEnv, fldPath.Child("env"), orderedEnvironment)...)
	}
	allErrs = append(allErrs, validateElasticEPRequiresCommand(spec.BackendFramework, &spec.DynamoComponentDeploymentSharedSpec, fldPath)...)
	allErrs = append(allErrs, dynamo.ValidateSnapshotFailover(&spec.DynamoComponentDeploymentSharedSpec, fldPath, spec.BackendFramework)...)
	allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpec(
		&spec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		dynamoComponentDeploymentSharedSpecValidationOptions{
			grovePathway:                      grovePathway,
			validateInferencePoolAvailability: validateInferencePoolAvailability,
			oldComponent:                      oldSharedSpec,
		},
	)...)
	return allErrs
}

// validateDynamoComponentDeploymentUpdate validates an update. newDCD and oldDCD must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentUpdate(
	newDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	oldDCD *nvidiacomv1beta1.DynamoComponentDeployment,
) field.ErrorList {
	allErrs := v.validateObjectMetaUpdate(&newDCD.ObjectMeta, &oldDCD.ObjectMeta, field.NewPath("metadata"))
	return append(allErrs, v.validateDynamoComponentDeploymentSpecUpdate(
		&newDCD.Spec,
		&oldDCD.Spec,
		field.NewPath("spec"),
	)...)
}

// validateDynamoComponentDeploymentSpecUpdate validates a spec update.
// newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoComponentDeploymentValidation) validateDynamoComponentDeploymentSpecUpdate(
	newSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	oldSpec *nvidiacomv1beta1.DynamoComponentDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	// Standalone DCD updates preserve direct replica modification.
	const canModifyReplicas = true

	allErrs := field.ErrorList{}
	if newSpec.BackendFramework != oldSpec.BackendFramework {
		v.warn("Changing spec.backendFramework may cause unexpected behavior")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("backendFramework"),
			newSpec.BackendFramework,
			"is immutable and cannot be changed after creation",
		))
	}

	allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecUpdate(
		&newSpec.DynamoComponentDeploymentSharedSpec,
		&oldSpec.DynamoComponentDeploymentSharedSpec,
		fldPath,
		canModifyReplicas,
		nvidiacomv1beta1.DynamoComponentDeploymentGVK.GroupKind(),
	)...)
	return allErrs
}

// validateDynamoComponentDeploymentRendered validates identities produced by component rendering.
// newDCD and fldPath must be non-nil; oldDCD is nil on create.
func (v *sharedValidation) validateDynamoComponentDeploymentRendered(newDCD, oldDCD *nvidiacomv1beta1.DynamoComponentDeployment, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if oldDCD == nil || newDCD.Name != oldDCD.Name {
		if messages := k8svalidation.IsDNS1123Subdomain(newDCD.Name); len(messages) > 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, newDCD.Name, "generated DynamoComponentDeployment name is invalid: "+strings.Join(messages, "; ")))
		}
	}

	// Render each version's actual workload shape, including LWS selector omission.
	lwsEnabled := features.MustGateFrom(v.ctx).Enabled(features.LWS)
	if lwsEnabled && newDCD.IsMultinode() {
		var oldService *corev1.Service
		if oldDCD != nil && oldDCD.IsMultinode() {
			oldService = &corev1.Service{Name: leaderworkerset.Name(oldDCD.Name)}
		}
		allErrs = append(allErrs, v.validateService(&corev1.Service{Name: leaderworkerset.Name(newDCD.Name)}, oldService, fldPath)...)
	} else {
		workload := componentdeployment.Render(newDCD, metav1.ObjectMeta{}, &corev1.PodTemplateSpec{}, appsv1.DeploymentStrategy{})
		var oldMetadata *metav1.ObjectMeta
		if oldDCD != nil && !(lwsEnabled && oldDCD.IsMultinode()) {
			oldWorkload := componentdeployment.Render(oldDCD, metav1.ObjectMeta{}, &corev1.PodTemplateSpec{}, appsv1.DeploymentStrategy{})
			oldMetadata = &oldWorkload.Spec.Template.ObjectMeta
		}
		allErrs = append(allErrs, v.validateComponentPodMetadata(&workload.Spec.Template.ObjectMeta, oldMetadata, fldPath)...)
	}

	// Render optional discovery Services with the same annotations and operator default.
	service, _ := componentdeployment.RenderService(componentdeployment.ServiceParams{ServiceName: newDCD.Name, ComponentType: string(newDCD.Spec.ComponentType), IsK8sDiscovery: commoncontroller.IsK8sDiscoveryEnabled(v.discoveryBackend, dynamo.GetDCDKubeAnnotations(newDCD))})
	if service != nil {
		var oldService *corev1.Service
		if oldDCD != nil {
			oldService, _ = componentdeployment.RenderService(componentdeployment.ServiceParams{ServiceName: oldDCD.Name, ComponentType: string(oldDCD.Spec.ComponentType), IsK8sDiscovery: commoncontroller.IsK8sDiscoveryEnabled(v.discoveryBackend, dynamo.GetDCDKubeAnnotations(oldDCD))})
		}
		allErrs = append(allErrs, v.validateService(service, oldService, fldPath)...)
	}
	return allErrs
}
