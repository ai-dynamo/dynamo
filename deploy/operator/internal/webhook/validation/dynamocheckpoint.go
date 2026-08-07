/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	checkpointinternal "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"k8s.io/apimachinery/pkg/api/equality"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	ctx context.Context
}

// Validate performs stateless validation on checkpoint. ctx and checkpoint must not be nil.
func (v *DynamoCheckpointValidator) Validate(
	ctx context.Context,
	checkpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) (admission.Warnings, error) {
	validation := &dynamoCheckpointValidation{ctx: ctx}
	allErrs := validation.validateDynamoCheckpoint(checkpoint)
	return nil, invalidDynamoCheckpointError(checkpoint, allErrs)
}

// ValidateUpdate validates newCheckpoint against oldCheckpoint.
// ctx, oldCheckpoint, and newCheckpoint must not be nil.
func (v *DynamoCheckpointValidator) ValidateUpdate(
	ctx context.Context,
	oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
	newCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) (admission.Warnings, error) {
	validation := &dynamoCheckpointValidation{ctx: ctx}
	allErrs := validation.validateDynamoCheckpointUpdate(newCheckpoint, oldCheckpoint)
	return nil, invalidDynamoCheckpointError(newCheckpoint, allErrs)
}

// validateDynamoCheckpoint validates checkpoint. checkpoint must not be nil.
func (v *dynamoCheckpointValidation) validateDynamoCheckpoint(
	checkpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) field.ErrorList {
	return v.validateDynamoCheckpointWithOld(checkpoint, nil)
}

func (v *dynamoCheckpointValidation) validateDynamoCheckpointWithOld(
	checkpoint *nvidiacomv1alpha1.DynamoCheckpoint,
	oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) field.ErrorList {
	metadataPath := field.NewPath("metadata")
	specPath := field.NewPath("spec")
	allErrs := field.ErrorList{}
	if !features.MustGateFrom(v.ctx).Enabled(features.Checkpoint) {
		allErrs = append(allErrs, field.Forbidden(
			specPath,
			"checkpoint functionality is disabled in the operator configuration",
		))
	}
	identityPath := specPath.Child("identity")
	automatic := checkpointinternal.IsAutomaticCheckpoint(checkpoint)
	claimsAutomaticAuthority := checkpointinternal.ClaimsAutomaticCheckpointAuthority(checkpoint)
	oldAutomatic := checkpointinternal.IsAutomaticCheckpoint(oldCheckpoint)

	// Reject malformed authority markers before classifying standalone objects.
	if claimsAutomaticAuthority && !automatic {
		allErrs = append(allErrs, field.Invalid(
			metadataPath.Child("annotations").Key(consts.CheckpointAutoAnnotation),
			checkpoint.Annotations[consts.CheckpointAutoAnnotation],
			fmt.Sprintf("must be %q", consts.KubeLabelValueTrue),
		))
	}

	// Require direct artifact authority on new automatic checkpoints.
	if automatic && oldCheckpoint == nil {
		if checkpoint.Labels == nil ||
			checkpoint.Labels[snapshotprotocol.CheckpointIDLabel] == "" {
			allErrs = append(allErrs, field.Required(
				metadataPath.Child("labels").Key(snapshotprotocol.CheckpointIDLabel),
				"is required for a new DGD-managed automatic checkpoint",
			))
		}
		if checkpoint.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation] == "" {
			allErrs = append(allErrs, field.Required(
				metadataPath.Child("annotations").Key(snapshotprotocol.CheckpointArtifactVersionAnnotation),
				"is required for a new DGD-managed automatic checkpoint",
			))
		}
	}

	// Read deprecated identity only for standalone and upgrade compatibility.
	//nolint:staticcheck // SA1019: Automatic validation preserves the legacy identity boundary.
	checkpointIdentity := checkpoint.Spec.Identity
	if oldCheckpoint == nil && automatic && checkpointIdentity != nil {
		allErrs = append(allErrs, field.Forbidden(
			identityPath,
			"must be omitted for a new DGD-managed automatic checkpoint",
		))
	}
	//nolint:staticcheck // SA1019: Standalone checkpoints still require the deprecated compatibility identity.
	if !claimsAutomaticAuthority && !oldAutomatic && checkpointIdentity == nil {
		allErrs = append(allErrs, field.Required(
			identityPath,
			"is required for a standalone checkpoint",
		))
	}
	allErrs = append(allErrs, v.validateDynamoCheckpointSpec(&checkpoint.Spec, specPath)...)
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
	allErrs := validateAutomaticCheckpointUpdate(newCheckpoint, oldCheckpoint)
	if !newCheckpoint.DeletionTimestamp.IsZero() {
		return allErrs
	}
	// spec.identity immutability is enforced by source-version CEL before this traversal.
	return append(
		allErrs,
		v.validateDynamoCheckpointWithOld(newCheckpoint, oldCheckpoint)...,
	)
}

func validateAutomaticCheckpointUpdate(
	newCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
	oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) field.ErrorList {
	metadataPath := field.NewPath("metadata")
	specPath := field.NewPath("spec")
	if !checkpointinternal.IsAutomaticCheckpoint(oldCheckpoint) {
		if checkpointinternal.IsAutomaticCheckpoint(newCheckpoint) {
			return field.ErrorList{field.Forbidden(
				metadataPath.Child("annotations").Key(consts.CheckpointAutoAnnotation),
				"cannot be added to an existing standalone checkpoint",
			)}
		}
		return nil
	}

	allErrs := field.ErrorList{}

	// Report immutable metadata without exposing unrelated object contents.
	immutableMetadata := func(path *field.Path, value string) {
		allErrs = append(allErrs, field.Invalid(
			path,
			value,
			apivalidation.FieldImmutableErrorMsg,
		))
	}

	if newCheckpoint.Annotations[consts.CheckpointAutoAnnotation] !=
		oldCheckpoint.Annotations[consts.CheckpointAutoAnnotation] {
		immutableMetadata(
			metadataPath.Child("annotations").Key(consts.CheckpointAutoAnnotation),
			newCheckpoint.Annotations[consts.CheckpointAutoAnnotation],
		)
	}
	if newCheckpoint.Labels[snapshotprotocol.CheckpointIDLabel] !=
		oldCheckpoint.Labels[snapshotprotocol.CheckpointIDLabel] {
		immutableMetadata(
			metadataPath.Child("labels").Key(snapshotprotocol.CheckpointIDLabel),
			newCheckpoint.Labels[snapshotprotocol.CheckpointIDLabel],
		)
	}
	if newCheckpoint.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation] !=
		oldCheckpoint.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation] {
		immutableMetadata(
			metadataPath.Child("annotations").Key(snapshotprotocol.CheckpointArtifactVersionAnnotation),
			newCheckpoint.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation],
		)
	}
	for _, key := range []string{
		consts.KubeLabelDynamoComponent,
		consts.KubeLabelDynamoWorkerHash,
	} {
		if newCheckpoint.Labels[key] != oldCheckpoint.Labels[key] {
			immutableMetadata(
				metadataPath.Child("labels").Key(key),
				newCheckpoint.Labels[key],
			)
		}
	}

	// Permit only the DGD reconciler's narrow Retain ownership detach.
	dgdNameChanged := newCheckpoint.Labels[consts.KubeLabelDynamoGraphDeploymentName] !=
		oldCheckpoint.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	ownerReferencesChanged := !ownerReferencesEqual(
		oldCheckpoint.OwnerReferences,
		newCheckpoint.OwnerReferences,
	)
	if (dgdNameChanged || ownerReferencesChanged) &&
		!isRetainDetachUpdate(oldCheckpoint, newCheckpoint) {
		if dgdNameChanged {
			immutableMetadata(
				metadataPath.Child("labels").Key(consts.KubeLabelDynamoGraphDeploymentName),
				newCheckpoint.Labels[consts.KubeLabelDynamoGraphDeploymentName],
			)
		}
		if ownerReferencesChanged {
			allErrs = append(allErrs, field.Invalid(
				metadataPath.Child("ownerReferences"),
				newCheckpoint.OwnerReferences,
				apivalidation.FieldImmutableErrorMsg,
			))
		}
	}

	// Preserve legacy identity exactly while keeping it outside authority.
	//nolint:staticcheck // SA1019: Update validation preserves persisted compatibility identity.
	oldIdentity, newIdentity := oldCheckpoint.Spec.Identity, newCheckpoint.Spec.Identity
	identityPath := specPath.Child("identity")
	switch {
	case oldIdentity == nil && newIdentity != nil:
		allErrs = append(allErrs, field.Forbidden(
			identityPath,
			"cannot be added to an existing DGD-managed automatic checkpoint",
		))
	case oldIdentity != nil && newIdentity == nil:
		allErrs = append(allErrs, field.Forbidden(
			identityPath,
			"legacy identity on a DGD-managed automatic checkpoint cannot be removed",
		))
	case oldIdentity != nil &&
		!equality.Semantic.DeepEqual(oldIdentity, newIdentity):
		allErrs = append(allErrs, field.Forbidden(
			identityPath,
			"legacy identity on a DGD-managed automatic checkpoint cannot be changed",
		))
	}

	// Freeze all automatic capture configuration after creation.
	if !equality.Semantic.DeepEqual(oldCheckpoint.Spec.Job, newCheckpoint.Spec.Job) {
		allErrs = append(allErrs, field.Forbidden(
			specPath.Child("job"),
			apivalidation.FieldImmutableErrorMsg,
		))
	}
	if !equality.Semantic.DeepEqual(
		oldCheckpoint.Spec.GPUMemoryService,
		newCheckpoint.Spec.GPUMemoryService,
	) {
		allErrs = append(allErrs, field.Forbidden(
			specPath.Child("gpuMemoryService"),
			apivalidation.FieldImmutableErrorMsg,
		))
	}
	return allErrs
}

func ownerReferencesEqual(oldReferences, newReferences []metav1.OwnerReference) bool {
	return equality.Semantic.DeepEqual(oldReferences, newReferences) ||
		(len(oldReferences) == 0 && len(newReferences) == 0)
}

func isRetainDetachUpdate(
	oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
	newCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) bool {
	oldPolicy := oldCheckpoint.Annotations[consts.CheckpointDeletionPolicyAnnotation]
	newPolicy := newCheckpoint.Annotations[consts.CheckpointDeletionPolicyAnnotation]
	retain := string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain)
	if oldPolicy != retain || newPolicy != retain {
		return false
	}

	// Ownership may only remain unchanged or be fully removed.
	ownerReferencesUnchanged := ownerReferencesEqual(
		oldCheckpoint.OwnerReferences,
		newCheckpoint.OwnerReferences,
	)
	ownerReferencesRemoved := len(oldCheckpoint.OwnerReferences) > 0 &&
		len(newCheckpoint.OwnerReferences) == 0
	if !ownerReferencesUnchanged && !ownerReferencesRemoved {
		return false
	}

	// Policy sync may remove ownership first; DGD finalization removes the DGD
	// label either in that same patch or after ownership is already detached.
	oldDGDName := oldCheckpoint.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	newDGDName := newCheckpoint.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	ownerDetachForRetain := ownerReferencesRemoved &&
		oldDGDName != "" &&
		(newDGDName == oldDGDName || newDGDName == "")
	dgdNameDetachForRetain := ownerReferencesUnchanged &&
		len(newCheckpoint.OwnerReferences) == 0 &&
		oldDGDName != "" && newDGDName == ""
	return ownerDetachForRetain || dgdNameDetachForRetain
}
