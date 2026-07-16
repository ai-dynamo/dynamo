/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// sharedValidation carries request-wide dependencies and accumulation used by
// validation for API types shared by multiple resources.
type sharedValidation struct {
	ctx      context.Context
	mgr      ctrl.Manager
	warnings admission.Warnings
}

func (v *sharedValidation) warn(message string) {
	v.warnings = append(v.warnings, message)
}

func (v *sharedValidation) warnf(format string, args ...any) {
	v.warn(fmt.Sprintf(format, args...))
}

// validateDynamoComponentDeploymentSharedSpec validates spec. spec and fldPath must not be nil.
// grovePathway and validateInferencePoolAvailability are supplied by the owning resource.
func (v *sharedValidation) validateDynamoComponentDeploymentSharedSpec(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
	grovePathway bool,
	validateInferencePoolAvailability bool,
) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.MinAvailable != nil && !grovePathway {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("minAvailable"),
			"is currently supported only for Grove-backed DynamoGraphDeployment components",
		))
	}
	if spec.SharedMemorySize != nil && spec.SharedMemorySize.Sign() < 0 {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("sharedMemorySize"),
			spec.SharedMemorySize.String(),
			"must be non-negative",
		))
	}

	if spec.ComponentType == nvidiacomv1beta1.ComponentTypeEPP {
		if validateInferencePoolAvailability {
			if err := inferencePoolAvailabilityError(v.ctx, v.mgr); err != nil {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("type"), fmt.Sprintf("cannot deploy EPP component: %v", err)))
			}
		}
		if spec.IsMultinode() {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("multinode"), "EPP component cannot be multinode"))
		}
		if spec.Replicas != nil && *spec.Replicas != 1 {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("replicas"),
				*spec.Replicas,
				"EPP component must have exactly 1 replica",
			))
		}
		if spec.EPPConfig == nil {
			allErrs = append(allErrs, field.Required(fldPath.Child("eppConfig"), "is required for EPP components"))
		}
	}
	if spec.EPPConfig != nil {
		allErrs = append(allErrs, v.validateEPPConfig(spec.EPPConfig, fldPath.Child("eppConfig"))...)
	}

	if spec.FrontendSidecar != nil {
		frontendSidecarPath := fldPath.Child("frontendSidecar")
		if spec.PodTemplate == nil {
			allErrs = append(allErrs, field.Required(
				fldPath.Child("podTemplate", "spec", "containers"),
				"is required when frontendSidecar is set",
			))
		} else if *spec.FrontendSidecar == "" {
			allErrs = append(allErrs, field.Invalid(frontendSidecarPath, *spec.FrontendSidecar, "must not be empty"))
		} else if !hasContainerNamed(spec.PodTemplate.Spec.Containers, *spec.FrontendSidecar) {
			allErrs = append(allErrs, field.Invalid(
				frontendSidecarPath,
				*spec.FrontendSidecar,
				"must match a podTemplate.spec.containers name",
			))
		}
	}

	if spec.Experimental != nil {
		allErrs = append(allErrs, v.validateExperimentalSpec(
			spec.Experimental,
			fldPath.Child("experimental"),
			spec.ComponentType,
			dynamo.GetMainContainerResources(spec),
		)...)
	}

	return allErrs
}

// validateEPPConfig validates config. config and fldPath must not be nil.
func (v *sharedValidation) validateEPPConfig(
	config *nvidiacomv1beta1.EPPConfig,
	fldPath *field.Path,
) field.ErrorList {
	if config.ConfigMapRef == nil || config.ConfigMapRef.Name != "" {
		return nil
	}
	return field.ErrorList{field.Required(fldPath.Child("configMapRef", "name"), "is required")}
}

// validateTopologyConstraint validates constraint. constraint, specConstraint, and fldPath must not be nil.
// topologyInfo may be nil when live topology validation is not applicable.
func (v *sharedValidation) validateTopologyConstraint(
	constraint *nvidiacomv1beta1.TopologyConstraint,
	fldPath *field.Path,
	specConstraint *nvidiacomv1beta1.SpecTopologyConstraint,
	topologyInfo *clusterTopologyInfo,
) field.ErrorList {
	if topologyInfo == nil {
		return nil
	}

	packDomainPath := fldPath.Child("packDomain")
	componentIndex, exists := topologyInfo.domainIndex[string(constraint.PackDomain)]
	if !exists {
		return field.ErrorList{field.Invalid(
			packDomainPath,
			constraint.PackDomain,
			fmt.Sprintf("does not exist in ClusterTopology %q; available domains: %v", topologyInfo.name, topologyInfo.domains),
		)}
	}
	if specConstraint.PackDomain == "" {
		return nil
	}
	specIndex, exists := topologyInfo.domainIndex[string(specConstraint.PackDomain)]
	if exists && componentIndex < specIndex {
		return field.ErrorList{field.Invalid(
			packDomainPath,
			constraint.PackDomain,
			fmt.Sprintf("must be equal to or narrower than the deployment-level domain %q", specConstraint.PackDomain),
		)}
	}
	return nil
}

// validateExperimentalSpec validates experimental. experimental and fldPath must not be nil.
func (v *sharedValidation) validateExperimentalSpec(
	experimental *nvidiacomv1beta1.ExperimentalSpec,
	fldPath *field.Path,
	componentType nvidiacomv1beta1.ComponentType,
	resources corev1.ResourceRequirements,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if experimental.GPUMemoryService != nil {
		gpuMemoryServicePath := fldPath.Child("gpuMemoryService")
		switch componentType {
		case nvidiacomv1beta1.ComponentTypeWorker,
			nvidiacomv1beta1.ComponentTypePrefill,
			nvidiacomv1beta1.ComponentTypeDecode:
		default:
			allErrs = append(allErrs, field.Forbidden(
				gpuMemoryServicePath,
				"GPU memory service is only supported for worker, prefill, or decode components",
			))
		}

		gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(resources)
		if err != nil || gpuCount < 1 {
			allErrs = append(allErrs, field.Forbidden(
				gpuMemoryServicePath,
				"GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
			))
		}
	}
	if experimental.Failover != nil {
		allErrs = append(allErrs, v.validateFailoverSpec(
			experimental.Failover,
			fldPath.Child("failover"),
			experimental.GPUMemoryService,
			componentType,
			resources,
		)...)
	}
	if experimental.Checkpoint != nil {
		allErrs = append(allErrs, v.validateComponentCheckpointConfig(
			experimental.Checkpoint,
			fldPath.Child("checkpoint"),
			experimental.GPUMemoryService,
		)...)
	}

	if experimental.Checkpoint != nil && experimental.Checkpoint.Enabled &&
		experimental.GPUMemoryService != nil && !features.MustGateFrom(v.ctx).Enabled(features.GMSSnapshot) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("checkpoint"),
			"GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate",
		))
	}
	return allErrs
}

// validateFailoverSpec validates failover. failover and fldPath must not be nil.
// gms may be nil because failover validates that sibling relationship.
func (v *sharedValidation) validateFailoverSpec(
	failover *nvidiacomv1beta1.FailoverSpec,
	fldPath *field.Path,
	gms *nvidiacomv1beta1.GPUMemoryServiceSpec,
	componentType nvidiacomv1beta1.ComponentType,
	resources corev1.ResourceRequirements,
) field.ErrorList {
	allErrs := field.ErrorList{}
	failoverMode := effectiveGMSMode(failover.Mode)
	if gms == nil {
		allErrs = append(allErrs, field.Forbidden(
			fldPath,
			fmt.Sprintf("gpuMemoryService is required when failover mode is %q", failoverMode),
		))
	} else if effectiveGMSMode(gms.Mode) != failoverMode {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("mode"),
			failover.Mode,
			fmt.Sprintf("must match gpuMemoryService.mode %q", gms.Mode),
		))
	}

	if failoverMode == nvidiacomv1beta1.GMSModeInterPod {
		gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(resources)
		if err != nil {
			allErrs = append(allErrs, field.Forbidden(
				fldPath,
				fmt.Sprintf("failed to read main-container GPU limit: %v", err),
			))
		} else if gpuCount < 1 {
			allErrs = append(allErrs, field.Forbidden(
				fldPath,
				"GMS failover requires at least 1 GPU in podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu",
			))
		}

		switch componentType {
		case nvidiacomv1beta1.ComponentTypeEPP,
			nvidiacomv1beta1.ComponentTypeFrontend,
			nvidiacomv1beta1.ComponentTypePlanner:
			allErrs = append(allErrs, field.Forbidden(
				fldPath,
				fmt.Sprintf("GMS failover is not supported for component type %q", componentType),
			))
		}
	}
	return allErrs
}

// validateComponentCheckpointConfig validates checkpoint. checkpoint and fldPath must not be nil.
// gms may be nil because checkpoint validates that sibling relationship.
func (v *sharedValidation) validateComponentCheckpointConfig(
	checkpointConfig *nvidiacomv1beta1.ComponentCheckpointConfig,
	fldPath *field.Path,
	gms *nvidiacomv1beta1.GPUMemoryServiceSpec,
) field.ErrorList {
	if checkpointConfig.Enabled && !features.MustGateFrom(v.ctx).Enabled(features.Checkpoint) {
		return field.ErrorList{field.Forbidden(fldPath, "checkpoint functionality is disabled in the operator configuration")}
	}
	if checkpointConfig.Job == nil {
		return nil
	}
	return v.validateComponentCheckpointJobConfig(checkpointConfig.Job, fldPath.Child("job"), gms)
}

// validateComponentCheckpointJobConfig validates job. job and fldPath must not be nil.
// gms may be nil because the job validates that sibling relationship.
func (v *sharedValidation) validateComponentCheckpointJobConfig(
	job *nvidiacomv1beta1.ComponentCheckpointJobConfig,
	fldPath *field.Path,
	gms *nvidiacomv1beta1.GPUMemoryServiceSpec,
) field.ErrorList {
	if len(job.GMSClientContainers) == 0 {
		return nil
	}
	if gms == nil {
		return field.ErrorList{field.Forbidden(
			fldPath.Child("gmsClientContainers"),
			"requires gpuMemoryService to be set",
		)}
	}
	if effectiveGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod {
		return field.ErrorList{field.Forbidden(
			fldPath.Child("gmsClientContainers"),
			"is only supported with gpuMemoryService.mode=IntraPod",
		)}
	}
	return nil
}

// validateDynamoComponentDeploymentSharedSpecUpdate validates a component update.
// newComponent, oldComponent, and fldPath must not be nil; ownerKind.Kind must not be empty.
func (v *sharedValidation) validateDynamoComponentDeploymentSharedSpecUpdate(
	newComponent *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	oldComponent *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
	canModifyReplicas bool,
	ownerKind schema.GroupKind,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if (newComponent.ScalingAdapter != nil || oldComponent.ScalingAdapter != nil) && !canModifyReplicas &&
		k8sptr.Deref(newComponent.Replicas, int32(1)) != k8sptr.Deref(oldComponent.Replicas, int32(1)) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("replicas"),
			"cannot be modified directly when scaling adapter is enabled; scale or update the related DynamoGraphDeploymentScalingAdapter instead",
		))
	}

	if newComponent.IsMultinode() != oldComponent.IsMultinode() {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("multinode"),
			newComponent.Multinode,
			"cannot change node topology between single-node and multi-node after creation",
		))
	}

	topologyPath := fldPath.Child("topologyConstraint")
	if newComponent.TopologyConstraint != nil {
		allErrs = append(allErrs, v.validateTopologyConstraintUpdate(
			newComponent.TopologyConstraint,
			oldComponent.TopologyConstraint,
			topologyPath,
			ownerKind,
		)...)
	} else if oldComponent.TopologyConstraint != nil {
		allErrs = append(allErrs, field.Invalid(
			topologyPath,
			newComponent.TopologyConstraint,
			fmt.Sprintf("is immutable and cannot be added, removed, or changed after creation; delete and recreate the %s to change topology constraints", ownerKind.Kind),
		))
	}

	if newComponent.Experimental != nil {
		allErrs = append(allErrs, v.validateExperimentalSpecUpdate(
			newComponent.Experimental,
			oldComponent.Experimental,
			fldPath.Child("experimental"),
			ownerKind,
		)...)
	} else if oldComponent.Experimental != nil {
		oldGMS := gpuMemoryServiceForExperimental(oldComponent.Experimental)
		if isInterPodGMS(oldGMS) {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("experimental", "gpuMemoryService", "mode"),
				nil,
				fmt.Sprintf("the inter-pod GMS layout cannot be toggled after creation; delete and recreate the %s", ownerKind.Kind),
			))
		}
		oldFailover := failoverForExperimental(oldComponent.Experimental)
		if isInterPodFailover(oldFailover) {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("experimental", "failover"),
				nil,
				fmt.Sprintf("inter-pod GMS failover cannot be toggled after creation; delete and recreate the %s", ownerKind.Kind),
			))
		}
	}
	return allErrs
}

// validateTopologyConstraintUpdate validates a topology constraint update.
// newConstraint and fldPath must not be nil; oldConstraint may be nil for an addition and ownerKind.Kind must not be empty.
func (v *sharedValidation) validateTopologyConstraintUpdate(
	newConstraint *nvidiacomv1beta1.TopologyConstraint,
	oldConstraint *nvidiacomv1beta1.TopologyConstraint,
	fldPath *field.Path,
	ownerKind schema.GroupKind,
) field.ErrorList {
	if oldConstraint != nil && newConstraint.PackDomain == oldConstraint.PackDomain {
		return nil
	}
	return field.ErrorList{field.Invalid(
		fldPath,
		newConstraint,
		fmt.Sprintf("is immutable and cannot be added, removed, or changed after creation; delete and recreate the %s to change topology constraints", ownerKind.Kind),
	)}
}

// validateExperimentalSpecUpdate validates an experimental spec update.
// newExperimental and fldPath must not be nil; oldExperimental may be nil for an addition and ownerKind.Kind must not be empty.
func (v *sharedValidation) validateExperimentalSpecUpdate(
	newExperimental *nvidiacomv1beta1.ExperimentalSpec,
	oldExperimental *nvidiacomv1beta1.ExperimentalSpec,
	fldPath *field.Path,
	ownerKind schema.GroupKind,
) field.ErrorList {
	allErrs := field.ErrorList{}
	newGMS := newExperimental.GPUMemoryService
	oldGMS := gpuMemoryServiceForExperimental(oldExperimental)
	if isInterPodGMS(newGMS) != isInterPodGMS(oldGMS) {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("gpuMemoryService", "mode"),
			k8sptr.Deref(newGMS, nvidiacomv1beta1.GPUMemoryServiceSpec{}).Mode,
			fmt.Sprintf("the inter-pod GMS layout cannot be toggled after creation; delete and recreate the %s", ownerKind.Kind),
		))
	}

	newFailover := newExperimental.Failover
	oldFailover := failoverForExperimental(oldExperimental)
	if isInterPodFailover(newFailover) != isInterPodFailover(oldFailover) {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("failover"),
			newFailover,
			fmt.Sprintf("inter-pod GMS failover cannot be toggled after creation; delete and recreate the %s", ownerKind.Kind),
		))
	}
	if isInterPodFailover(newFailover) && isInterPodFailover(oldFailover) &&
		effectiveNumShadows(newFailover) != effectiveNumShadows(oldFailover) {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("failover", "numShadows"),
			newFailover.NumShadows,
			fmt.Sprintf("is immutable for inter-pod GMS failover; delete and recreate the %s to change it", ownerKind.Kind),
		))
	}
	return allErrs
}
