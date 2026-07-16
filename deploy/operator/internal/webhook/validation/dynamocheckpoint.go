/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoCheckpointValidator validates DynamoCheckpoint resources.
type DynamoCheckpointValidator struct{}

// NewDynamoCheckpointValidator creates a DynamoCheckpoint validator.
func NewDynamoCheckpointValidator() *DynamoCheckpointValidator {
	return &DynamoCheckpointValidator{}
}

// dynamoCheckpointValidation carries DynamoCheckpoint-specific request state.
// API values, paths, and accumulated errors remain explicit validator arguments.
type dynamoCheckpointValidation struct {
	sharedValidation
}

// Validate performs stateless validation on checkpoint. ctx and checkpoint must not be nil.
func (v *DynamoCheckpointValidator) Validate(
	ctx context.Context,
	checkpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) (admission.Warnings, error) {
	validation := &dynamoCheckpointValidation{sharedValidation: sharedValidation{ctx: ctx}}
	allErrs := validation.validateDynamoCheckpoint(checkpoint)
	return validation.warnings, invalidDynamoCheckpointError(checkpoint, allErrs)
}

// ValidateUpdate validates newCheckpoint against oldCheckpoint.
// ctx, oldCheckpoint, and newCheckpoint must not be nil.
func (v *DynamoCheckpointValidator) ValidateUpdate(
	ctx context.Context,
	oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
	newCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) (admission.Warnings, error) {
	validation := &dynamoCheckpointValidation{sharedValidation: sharedValidation{ctx: ctx}}
	allErrs := validation.validateDynamoCheckpointUpdate(newCheckpoint, oldCheckpoint)
	return validation.warnings, invalidDynamoCheckpointError(newCheckpoint, allErrs)
}

// validateDynamoCheckpoint validates checkpoint. checkpoint must not be nil.
func (v *dynamoCheckpointValidation) validateDynamoCheckpoint(
	checkpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, v.validateObjectMeta(&checkpoint.ObjectMeta, field.NewPath("metadata"), false)...)
	allErrs = append(allErrs, v.validateDynamoCheckpointSpec(&checkpoint.Spec, field.NewPath("spec"))...)
	return allErrs
}

// validateDynamoCheckpointSpec validates spec. spec and fldPath must not be nil.
func (v *dynamoCheckpointValidation) validateDynamoCheckpointSpec(
	spec *nvidiacomv1alpha1.DynamoCheckpointSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	gpuMemoryServicePath := fldPath.Child("gpuMemoryService")

	if gpuMemoryService := spec.GPUMemoryService; gpuMemoryService != nil && gpuMemoryService.Enabled {
		if !features.MustGateFrom(v.ctx).Enabled(features.GMSSnapshot) {
			allErrs = append(allErrs, field.Forbidden(
				gpuMemoryServicePath,
				"GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate",
			))
		}

		switch gpuMemoryService.Mode {
		case "", nvidiacomv1alpha1.GMSModeIntraPod:
			containers := spec.Job.PodTemplateSpec.Spec.Containers
			for i, name := range gpuMemoryService.ExtraClientContainers {
				if containerIndexByName(containers, name) < 0 {
					allErrs = append(allErrs, field.Invalid(
						gpuMemoryServicePath.Child("extraClientContainers").Index(i),
						name,
						"does not name a container in spec.job.podTemplateSpec.spec.containers",
					))
				}
			}
		case nvidiacomv1alpha1.GMSModeInterPod:
			allErrs = append(allErrs, field.NotSupported(
				gpuMemoryServicePath.Child("mode"),
				gpuMemoryService.Mode,
				[]string{string(nvidiacomv1alpha1.GMSModeIntraPod)},
			))
		}
	}

	allErrs = append(allErrs, v.validateDynamoCheckpointJobConfig(
		&spec.Job,
		fldPath.Child("job"),
		spec.GPUMemoryService,
	)...)
	return allErrs
}

// validateDynamoCheckpointJobConfig validates job. job and fldPath must not be nil.
// gpuMemoryService comes from the owning DynamoCheckpointSpec and may be nil.
func (v *dynamoCheckpointValidation) validateDynamoCheckpointJobConfig(
	job *nvidiacomv1alpha1.DynamoCheckpointJobConfig,
	fldPath *field.Path,
	gpuMemoryService *nvidiacomv1alpha1.GPUMemoryServiceSpec,
) field.ErrorList {
	if gpuMemoryService == nil || !gpuMemoryService.Enabled ||
		(gpuMemoryService.Mode != "" && gpuMemoryService.Mode != nvidiacomv1alpha1.GMSModeIntraPod) {
		return nil
	}

	allErrs := field.ErrorList{}
	podSpec := &job.PodTemplateSpec.Spec
	podSpecPath := fldPath.Child("podTemplateSpec", "spec")

	if !common.HasVolume(podSpec.Volumes, gms.SharedVolumeName) {
		allErrs = append(allErrs, field.Required(
			podSpecPath.Child("volumes"),
			fmt.Sprintf("must contain the GMS shared volume %q", gms.SharedVolumeName),
		))
	}

	clientContainerErrors := func(containerIndex int, initContainer bool) {
		containersPath := podSpecPath.Child("containers")
		containers := podSpec.Containers
		if initContainer {
			containersPath = podSpecPath.Child("initContainers")
			containers = podSpec.InitContainers
		}
		container := &containers[containerIndex]
		containerPath := containersPath.Index(containerIndex)
		if !common.HasEnvValue(container.Env, gms.EnvSocketDir, gms.SharedMountPath) {
			allErrs = append(allErrs, field.Required(
				containerPath.Child("env"),
				fmt.Sprintf("must contain %s=%s for GMS", gms.EnvSocketDir, gms.SharedMountPath),
			))
		}
		if !common.HasContainerResourceClaim(container, dra.ClaimName) {
			allErrs = append(allErrs, field.Required(
				containerPath.Child("resources", "claims"),
				fmt.Sprintf("must contain the GMS resource claim %q", dra.ClaimName),
			))
		}
		if !common.HasVolumeMount(container.VolumeMounts, gms.SharedVolumeName, gms.SharedMountPath) {
			allErrs = append(allErrs, field.Required(
				containerPath.Child("volumeMounts"),
				fmt.Sprintf("must mount volume %q at %q for GMS", gms.SharedVolumeName, gms.SharedMountPath),
			))
		}
	}

	serverIndex := containerIndexByName(podSpec.InitContainers, gms.ServerContainerName)
	if serverIndex < 0 {
		allErrs = append(allErrs, field.Required(
			podSpecPath.Child("initContainers"),
			fmt.Sprintf("must contain the GMS init sidecar %q", gms.ServerContainerName),
		))
	} else {
		clientContainerErrors(serverIndex, true)
	}

	targetContainerName := job.TargetContainerName
	if targetContainerName == "" {
		targetContainerName = consts.MainContainerName
	}
	clientNames := map[string]bool{targetContainerName: true}
	for _, name := range gpuMemoryService.ExtraClientContainers {
		clientNames[name] = true
	}
	for i := range podSpec.Containers {
		if clientNames[podSpec.Containers[i].Name] {
			clientContainerErrors(i, false)
		}
	}

	if !common.HasPodResourceClaim(podSpec, dra.ClaimName) {
		allErrs = append(allErrs, field.Required(
			podSpecPath.Child("resourceClaims"),
			fmt.Sprintf("must contain the GMS pod resource claim %q", dra.ClaimName),
		))
	}

	if containerIndexByName(podSpec.Containers, targetContainerName) < 0 {
		if job.TargetContainerName == "" {
			allErrs = append(allErrs, field.Required(
				podSpecPath.Child("containers"),
				fmt.Sprintf("must contain the default target container %q", targetContainerName),
			))
		} else {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("targetContainerName"),
				job.TargetContainerName,
				"does not name a container in podTemplateSpec.spec.containers",
			))
		}
	}

	return allErrs
}

// validateDynamoCheckpointUpdate validates an update. newCheckpoint and oldCheckpoint must not be nil.
func (v *dynamoCheckpointValidation) validateDynamoCheckpointUpdate(
	newCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
	oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) field.ErrorList {
	// spec.identity immutability is enforced by source-version CEL before this traversal.
	_ = oldCheckpoint
	return v.validateDynamoCheckpoint(newCheckpoint)
}
