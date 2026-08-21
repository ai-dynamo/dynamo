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
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	corev1 "k8s.io/api/core/v1"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// sharedValidation carries request-wide dependencies and accumulation used by
// validation for API types shared by multiple resources.
type sharedValidation struct {
	ctx                                context.Context
	mgr                                ctrl.Manager
	warnings                           admission.Warnings
	runtimeVersionSource               runtimeVersionValidationSource
	allowMissingRuntimeVersionOverride bool
}

func (v *sharedValidation) warn(message string) {
	v.warnings = append(v.warnings, message)
}

func (v *sharedValidation) warnf(format string, args ...any) {
	v.warn(fmt.Sprintf(format, args...))
}

type dynamoComponentDeploymentSharedSpecValidationOptions struct {
	grovePathway                      bool
	validateInferencePoolAvailability bool
	providerOverridesSupported        bool
	workloadProvider                  string
}

// validateDynamoComponentDeploymentSharedSpec validates spec. spec and fldPath must not be nil.
// Options are supplied by the owning resource.
func (v *sharedValidation) validateDynamoComponentDeploymentSharedSpec(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
	options dynamoComponentDeploymentSharedSpecValidationOptions,
) field.ErrorList {
	allErrs := field.ErrorList{}

	// Validate the provider-native fragment in this component context.
	if spec.ProviderOverride != nil {
		allErrs = append(allErrs, v.validateProviderOverride(
			spec.ProviderOverride,
			fldPath.Child("providerOverride"),
			providerOverrideValidationOptions{
				supported:        options.providerOverridesSupported,
				workloadProvider: options.workloadProvider,
				scope:            provideroverride.ScopeComponent,
				component:        spec,
			},
		)...)
	}

	// Enforce Grove-only availability semantics before validating later fields.
	if spec.MinAvailable != nil && !options.grovePathway {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("minAvailable"),
			"is currently supported only for Grove-backed DynamoGraphDeployment components",
		))
	}

	// Validate explicit multinode role configuration in API declaration order.
	if spec.Multinode != nil {
		allErrs = append(allErrs, v.validateMultinodeSpec(
			spec.Multinode,
			fldPath.Child("multinode"),
			options.providerOverridesSupported,
			options.workloadProvider,
			spec,
		)...)
	}

	// Reject invalid shared-memory quantities before resource-specific validation.
	if spec.SharedMemorySize != nil && spec.SharedMemorySize.Sign() < 0 {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("sharedMemorySize"),
			spec.SharedMemorySize.String(),
			"must be non-negative",
		))
	}

	if spec.ComponentType == nvidiacomv1beta1.ComponentTypeEPP {
		if options.validateInferencePoolAvailability {
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
			experimentalSpecValidationOptions{
				componentType: spec.ComponentType,
				resources:     dynamo.GetMainContainerResources(spec),
				containers:    podTemplateContainers(spec.PodTemplate),
				grovePathway:  options.grovePathway,
			},
		)...)
	}

	// Validate runtime compatibility against the source-version fields.
	if v.validatesRuntimeVersionFor(runtimeVersionSourceV1Beta1) {
		image, imagePath := runtimeVersionImageAndPath(spec, fldPath)
		if image == "" {
			allErrs = append(allErrs, field.Required(imagePath, "is required"))
		} else if !v.allowMissingRuntimeVersionOverride &&
			runtimeVersionOverrideRequired(image, spec.RuntimeVersionOverride) {
			allErrs = append(allErrs, field.Required(
				fldPath.Child("runtimeVersionOverride"),
				runtimeVersionOverrideRequiredMessage,
			))
		}
	}

	return allErrs
}

type providerOverrideValidationOptions struct {
	supported        bool
	workloadProvider string
	scope            provideroverride.Scope
	component        *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
}

// validateProviderOverride validates override. override and fldPath must not be nil.
// options.component is nil only for the root DGD provider context.
func (v *sharedValidation) validateProviderOverride(
	override *nvidiacomv1beta1.ProviderOverride,
	fldPath *field.Path,
	options providerOverrideValidationOptions,
) field.ErrorList {
	// Reject provider fragments in API contexts that cannot lower them.
	if !options.supported {
		return field.ErrorList{field.Forbidden(
			fldPath,
			"provider overrides are supported only for components embedded in a DynamoGraphDeployment",
		)}
	}

	// Require the durable provider selection before interpreting the fragment.
	if options.workloadProvider == "" {
		return field.ErrorList{field.Forbidden(
			fldPath,
			fmt.Sprintf(
				"requires controller-owned annotation %q to be materialized; wait for controller adoption and retry",
				consts.KubeAnnotationWorkloadProvider,
			),
		)}
	}

	// Provider-native overrides currently target only Grove schemas.
	if options.workloadProvider != consts.WorkloadProviderGrove {
		return field.ErrorList{field.Forbidden(
			fldPath,
			fmt.Sprintf("requires workload provider %q, but %q is selected", consts.WorkloadProviderGrove, options.workloadProvider),
		)}
	}

	// Validate the explicit provider schema version before resolving its target.
	if override.APIVersion == "" {
		return field.ErrorList{field.Required(fldPath.Child("apiVersion"), "is required")}
	}
	if override.APIVersion != provideroverride.GroveAPIVersion {
		return field.ErrorList{field.NotSupported(
			fldPath.Child("apiVersion"),
			override.APIVersion,
			[]string{provideroverride.GroveAPIVersion},
		)}
	}

	// Resolve the only target valid for this provider context and component shape.
	expectedTarget, err := provideroverride.ExpectedTarget(
		options.workloadProvider,
		override.APIVersion,
		options.scope,
		options.component,
	)
	if err != nil {
		return field.ErrorList{field.Forbidden(fldPath, err.Error())}
	}
	if override.Target == "" {
		return field.ErrorList{field.Required(
			fldPath.Child("target"),
			"must be defaulted from the provider context",
		)}
	}
	if override.Target != expectedTarget {
		return field.ErrorList{field.Invalid(
			fldPath.Child("target"),
			override.Target,
			fmt.Sprintf("must match the provider-context target %q", expectedTarget),
		)}
	}

	// Map provider ownership and shape errors to exact Kubernetes field paths.
	valuePath := fldPath.Child("value")
	allErrs := field.ErrorList{}
	for _, valueErr := range provideroverride.ValidateValue(override.Target, override.Value.Raw) {
		errPath := valuePath
		if valueErr.Path != "" {
			parts := strings.Split(valueErr.Path, ".")
			errPath = valuePath.Child(parts[0], parts[1:]...)
		}
		if valueErr.OwnershipViolation {
			allErrs = append(allErrs, field.Forbidden(errPath, valueErr.Detail))
			continue
		}
		allErrs = append(allErrs, field.Invalid(errPath, nil, valueErr.Detail))
	}
	return allErrs
}

// validateMultinodeSpec validates multinode. multinode, component, and fldPath must not be nil.
func (v *sharedValidation) validateMultinodeSpec(
	multinode *nvidiacomv1beta1.MultinodeSpec,
	fldPath *field.Path,
	providerOverridesSupported bool,
	workloadProvider string,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) field.ErrorList {
	allErrs := field.ErrorList{}

	// Validate the optional leader provider context.
	if multinode.Leader != nil {
		allErrs = append(allErrs, v.validateMultinodeRoleSpec(
			multinode.Leader,
			fldPath.Child("leader"),
			multinodeRoleSpecValidationOptions{
				providerOverridesSupported: providerOverridesSupported,
				workloadProvider:           workloadProvider,
				scope:                      provideroverride.ScopeMultinodeLeader,
				component:                  component,
			},
		)...)
	}

	// Validate the optional worker provider context.
	if multinode.Worker != nil {
		allErrs = append(allErrs, v.validateMultinodeRoleSpec(
			multinode.Worker,
			fldPath.Child("worker"),
			multinodeRoleSpecValidationOptions{
				providerOverridesSupported: providerOverridesSupported,
				workloadProvider:           workloadProvider,
				scope:                      provideroverride.ScopeMultinodeWorker,
				component:                  component,
			},
		)...)
	}
	return allErrs
}

type multinodeRoleSpecValidationOptions struct {
	providerOverridesSupported bool
	workloadProvider           string
	scope                      provideroverride.Scope
	component                  *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
}

// validateMultinodeRoleSpec validates role. role and fldPath must not be nil;
// the optional provider override may be nil.
func (v *sharedValidation) validateMultinodeRoleSpec(
	role *nvidiacomv1beta1.MultinodeRoleSpec,
	fldPath *field.Path,
	options multinodeRoleSpecValidationOptions,
) field.ErrorList {
	if role.ProviderOverride == nil {
		return nil
	}

	// Validate the provider fragment against this exact multinode role.
	return v.validateProviderOverride(
		role.ProviderOverride,
		fldPath.Child("providerOverride"),
		providerOverrideValidationOptions{
			supported:        options.providerOverridesSupported,
			workloadProvider: options.workloadProvider,
			scope:            options.scope,
			component:        options.component,
		},
	)
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

type experimentalSpecValidationOptions struct {
	componentType nvidiacomv1beta1.ComponentType
	resources     corev1.ResourceRequirements
	containers    []corev1.Container
	grovePathway  bool
}

// validateExperimentalSpec validates experimental. experimental and fldPath must not be nil.
func (v *sharedValidation) validateExperimentalSpec(
	experimental *nvidiacomv1beta1.ExperimentalSpec,
	fldPath *field.Path,
	options experimentalSpecValidationOptions,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if experimental.GPUMemoryService != nil {
		allErrs = append(allErrs, v.validateGPUMemoryServiceSpec(
			experimental.GPUMemoryService,
			fldPath.Child("gpuMemoryService"),
			options.componentType,
			options.resources,
			options.containers,
		)...)
	}
	if experimental.Failover != nil {
		allErrs = append(allErrs, v.validateFailoverSpec(
			experimental.Failover,
			fldPath.Child("failover"),
			experimental.GPUMemoryService,
			options.componentType,
			options.resources,
		)...)
	}
	if experimental.Grove != nil {
		allErrs = append(allErrs, v.validateGroveSpec(
			experimental.Grove,
			fldPath.Child("grove"),
			options.grovePathway,
		)...)
	}
	if experimental.Checkpoint != nil {
		allErrs = append(allErrs, v.validateComponentCheckpointConfig(
			experimental.Checkpoint,
			fldPath.Child("checkpoint"),
			experimental.GPUMemoryService,
		)...)
	}

	for _, err := range checkpoint.ValidateCheckpointCompatibility(experimental) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("checkpoint"),
			err.Error(),
		))
	}
	return allErrs
}

// validateGPUMemoryServiceSpec validates gpuMemoryService. gpuMemoryService and fldPath must not be nil.
func (v *sharedValidation) validateGPUMemoryServiceSpec(
	gpuMemoryService *nvidiacomv1beta1.GPUMemoryServiceSpec,
	fldPath *field.Path,
	componentType nvidiacomv1beta1.ComponentType,
	resources corev1.ResourceRequirements,
	containers []corev1.Container,
) field.ErrorList {
	allErrs := field.ErrorList{}

	// Restrict GMS to component types that own GPU-backed workloads.
	switch componentType {
	case nvidiacomv1beta1.ComponentTypeWorker,
		nvidiacomv1beta1.ComponentTypePrefill,
		nvidiacomv1beta1.ComponentTypeDecode:
	default:
		allErrs = append(allErrs, field.Forbidden(
			fldPath,
			"GPU memory service is only supported for worker, prefill, or decode components",
		))
	}

	// Require the main container to expose at least one GPU to GMS.
	gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(resources)
	if err != nil || gpuCount < 1 {
		allErrs = append(allErrs, field.Forbidden(
			fldPath,
			"GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
		))
	}

	// Skip container-client validation for the inter-pod topology.
	if effectiveGMSMode(gpuMemoryService.Mode) != nvidiacomv1beta1.GMSModeIntraPod {
		return allErrs
	}

	// Validate every GMS client reference at its indexed field path.
	seen := make(map[string]struct{}, len(gpuMemoryService.ExtraClientContainers))
	extraClientContainersPath := fldPath.Child("extraClientContainers")
	for i, name := range gpuMemoryService.ExtraClientContainers {
		clientPath := extraClientContainersPath.Index(i)
		if _, exists := seen[name]; exists {
			allErrs = append(allErrs, field.Duplicate(clientPath, name))
			continue
		}
		seen[name] = struct{}{}
		if !hasContainerNamed(containers, name) {
			allErrs = append(allErrs, field.Invalid(
				clientPath,
				name,
				"does not name a container in podTemplate.spec.containers",
			))
		}
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

// validateGroveSpec validates grove. grove and fldPath must not be nil.
// grovePathway is supplied by the owning resource.
func (v *sharedValidation) validateGroveSpec(
	grove *nvidiacomv1beta1.GroveSpec,
	fldPath *field.Path,
	grovePathway bool,
) field.ErrorList {
	if grove.ForceScalingGroup && !grovePathway {
		return field.ErrorList{field.Forbidden(
			fldPath.Child("forceScalingGroup"),
			"is currently supported only for Grove-backed DynamoGraphDeployment components",
		)}
	}
	return nil
}

// validateComponentCheckpointConfig validates checkpoint. checkpoint and fldPath must not be nil.
// gms may be nil because checkpoint validates that sibling relationship.
func (v *sharedValidation) validateComponentCheckpointConfig(
	checkpointConfig *nvidiacomv1beta1.ComponentCheckpointConfig,
	fldPath *field.Path,
	gms *nvidiacomv1beta1.GPUMemoryServiceSpec,
) field.ErrorList {
	var allErrs field.ErrorList
	if checkpointConfig.Enabled && !features.MustGateFrom(v.ctx).Enabled(features.Checkpoint) {
		allErrs = append(allErrs, field.Forbidden(fldPath, "checkpoint functionality is disabled in the operator configuration"))
	}
	if checkpointConfig.Job == nil {
		return allErrs
	}
	return append(allErrs, v.validateComponentCheckpointJobConfig(checkpointConfig.Job, fldPath.Child("job"), gms)...)
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
	validateGPUMemoryServiceNewState bool,
) field.ErrorList {
	allErrs := field.ErrorList{}

	// Keep an existing component-level provider identity stable across updates.
	if newComponent.ProviderOverride != nil && oldComponent.ProviderOverride != nil {
		allErrs = append(allErrs, validateProviderOverrideUpdate(
			newComponent.ProviderOverride,
			oldComponent.ProviderOverride,
			fldPath.Child("providerOverride"),
		)...)
	}

	// Keep the component's multinode shape and nested provider identities stable.
	if newComponent.IsMultinode() != oldComponent.IsMultinode() {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("multinode"),
			newComponent.Multinode,
			"cannot change node topology between single-node and multi-node after creation",
		))
	}

	// Keep existing multinode role provider identities stable across updates.
	if newComponent.Multinode != nil && oldComponent.Multinode != nil {
		allErrs = append(allErrs, validateMultinodeSpecUpdate(
			newComponent.Multinode,
			oldComponent.Multinode,
			fldPath.Child("multinode"),
		)...)
	}

	// Protect replica ownership when a scaling adapter is present in either state.
	if (newComponent.ScalingAdapter != nil || oldComponent.ScalingAdapter != nil) && !canModifyReplicas &&
		k8sptr.Deref(newComponent.Replicas, int32(1)) != k8sptr.Deref(oldComponent.Replicas, int32(1)) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("replicas"),
			"cannot be modified directly when scaling adapter is enabled; scale or update the related DynamoGraphDeploymentScalingAdapter instead",
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
			experimentalSpecUpdateValidationOptions{
				ownerKind:                        ownerKind,
				componentType:                    newComponent.ComponentType,
				resources:                        dynamo.GetMainContainerResources(newComponent),
				containers:                       podTemplateContainers(newComponent.PodTemplate),
				validateGPUMemoryServiceNewState: validateGPUMemoryServiceNewState,
			},
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
		if forceScalingGroupFor(oldComponent.Experimental) {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("experimental", "grove", "forceScalingGroup"),
				nil,
				fmt.Sprintf("cannot be toggled after creation; delete and recreate the %s to change it", ownerKind.Kind),
			))
		}
	}

	// Ratchet legacy image absence or an unchanged legacy tuple, but reject a newly invalid tuple.
	if v.validatesRuntimeVersionFor(runtimeVersionSourceV1Beta1) {
		newImage, imagePath := runtimeVersionImageAndPath(newComponent, fldPath)
		oldImage, _ := runtimeVersionImageAndPath(oldComponent, fldPath)
		if newImage == "" && oldImage != "" {
			allErrs = append(allErrs, field.Required(imagePath, "is required"))
		} else if !v.allowMissingRuntimeVersionOverride &&
			runtimeVersionOverrideRequired(newImage, newComponent.RuntimeVersionOverride) &&
			(newImage != oldImage || newComponent.RuntimeVersionOverride != oldComponent.RuntimeVersionOverride) {
			allErrs = append(allErrs, field.Required(
				fldPath.Child("runtimeVersionOverride"),
				runtimeVersionOverrideRequiredMessage,
			))
		}
	}
	return allErrs
}

// validateMultinodeSpecUpdate validates a multinode update. newMultinode, oldMultinode, and fldPath must not be nil.
func validateMultinodeSpecUpdate(
	newMultinode *nvidiacomv1beta1.MultinodeSpec,
	oldMultinode *nvidiacomv1beta1.MultinodeSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}

	// Keep an existing leader provider identity stable across updates.
	if newMultinode.Leader != nil && oldMultinode.Leader != nil {
		allErrs = append(allErrs, validateMultinodeRoleSpecUpdate(
			newMultinode.Leader,
			oldMultinode.Leader,
			fldPath.Child("leader"),
		)...)
	}

	// Keep an existing worker provider identity stable across updates.
	if newMultinode.Worker != nil && oldMultinode.Worker != nil {
		allErrs = append(allErrs, validateMultinodeRoleSpecUpdate(
			newMultinode.Worker,
			oldMultinode.Worker,
			fldPath.Child("worker"),
		)...)
	}
	return allErrs
}

// validateMultinodeRoleSpecUpdate validates a role update. newRole, oldRole,
// and fldPath must not be nil; either optional provider override may be nil.
func validateMultinodeRoleSpecUpdate(
	newRole *nvidiacomv1beta1.MultinodeRoleSpec,
	oldRole *nvidiacomv1beta1.MultinodeRoleSpec,
	fldPath *field.Path,
) field.ErrorList {
	if newRole.ProviderOverride == nil || oldRole.ProviderOverride == nil {
		return nil
	}

	// Keep an existing role-level provider identity stable across updates.
	return validateProviderOverrideUpdate(
		newRole.ProviderOverride,
		oldRole.ProviderOverride,
		fldPath.Child("providerOverride"),
	)
}

// validateProviderOverrideUpdate validates an override update. newOverride, oldOverride, and fldPath must not be nil.
func validateProviderOverrideUpdate(
	newOverride *nvidiacomv1beta1.ProviderOverride,
	oldOverride *nvidiacomv1beta1.ProviderOverride,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}

	// Keep the persisted provider schema version immutable.
	if newOverride.APIVersion != oldOverride.APIVersion {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("apiVersion"),
			newOverride.APIVersion,
			apivalidation.FieldImmutableErrorMsg,
		))
	}

	// Keep the persisted lowering target immutable.
	if newOverride.Target != oldOverride.Target {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("target"),
			newOverride.Target,
			apivalidation.FieldImmutableErrorMsg,
		))
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

type experimentalSpecUpdateValidationOptions struct {
	ownerKind                        schema.GroupKind
	componentType                    nvidiacomv1beta1.ComponentType
	resources                        corev1.ResourceRequirements
	containers                       []corev1.Container
	validateGPUMemoryServiceNewState bool
}

// validateExperimentalSpecUpdate validates an experimental spec update.
// newExperimental and fldPath must not be nil; oldExperimental may be nil for an addition and options.ownerKind.Kind must not be empty.
func (v *sharedValidation) validateExperimentalSpecUpdate(
	newExperimental *nvidiacomv1beta1.ExperimentalSpec,
	oldExperimental *nvidiacomv1beta1.ExperimentalSpec,
	fldPath *field.Path,
	options experimentalSpecUpdateValidationOptions,
) field.ErrorList {
	allErrs := field.ErrorList{}
	newGMS := newExperimental.GPUMemoryService
	if newGMS != nil && options.validateGPUMemoryServiceNewState {
		allErrs = append(allErrs, v.validateGPUMemoryServiceSpec(
			newGMS,
			fldPath.Child("gpuMemoryService"),
			options.componentType,
			options.resources,
			options.containers,
		)...)
	}

	oldGMS := gpuMemoryServiceForExperimental(oldExperimental)
	if isInterPodGMS(newGMS) != isInterPodGMS(oldGMS) {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("gpuMemoryService", "mode"),
			k8sptr.Deref(newGMS, nvidiacomv1beta1.GPUMemoryServiceSpec{}).Mode,
			fmt.Sprintf("the inter-pod GMS layout cannot be toggled after creation; delete and recreate the %s", options.ownerKind.Kind),
		))
	}

	newFailover := newExperimental.Failover
	oldFailover := failoverForExperimental(oldExperimental)
	if isInterPodFailover(newFailover) != isInterPodFailover(oldFailover) {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("failover"),
			newFailover,
			fmt.Sprintf("inter-pod GMS failover cannot be toggled after creation; delete and recreate the %s", options.ownerKind.Kind),
		))
	}
	if isInterPodFailover(newFailover) && isInterPodFailover(oldFailover) &&
		effectiveNumShadows(newFailover) != effectiveNumShadows(oldFailover) {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("failover", "numShadows"),
			newFailover.NumShadows,
			fmt.Sprintf("is immutable for inter-pod GMS failover; delete and recreate the %s to change it", options.ownerKind.Kind),
		))
	}

	oldGrove := groveForExperimental(oldExperimental)
	if newExperimental.Grove != nil {
		allErrs = append(allErrs, v.validateGroveSpecUpdate(
			newExperimental.Grove,
			oldGrove,
			fldPath.Child("grove"),
			options.ownerKind,
		)...)
	} else if oldGrove != nil && oldGrove.ForceScalingGroup {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("grove", "forceScalingGroup"),
			nil,
			fmt.Sprintf("cannot be toggled after creation; delete and recreate the %s to change it", options.ownerKind.Kind),
		))
	}
	return allErrs
}

// validateGroveSpecUpdate validates a grove update. newGrove and fldPath must
// not be nil; oldGrove may be nil for an addition. false and omitted both
// mean automatic selection, so only the effective opt-in is immutable.
func (v *sharedValidation) validateGroveSpecUpdate(
	newGrove *nvidiacomv1beta1.GroveSpec,
	oldGrove *nvidiacomv1beta1.GroveSpec,
	fldPath *field.Path,
	ownerKind schema.GroupKind,
) field.ErrorList {
	oldForced := oldGrove != nil && oldGrove.ForceScalingGroup
	if newGrove.ForceScalingGroup == oldForced {
		return nil
	}
	return field.ErrorList{field.Invalid(
		fldPath.Child("forceScalingGroup"),
		newGrove.ForceScalingGroup,
		fmt.Sprintf("cannot be toggled after creation; delete and recreate the %s to change it", ownerKind.Kind),
	)}
}
