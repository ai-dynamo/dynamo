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
	"sort"
	"strings"

	semver "github.com/Masterminds/semver/v3"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// DynamoGraphDeploymentValidator validates v1beta1 DynamoGraphDeployment resources.
type DynamoGraphDeploymentValidator struct {
	mgr ctrl.Manager
}

// NewDynamoGraphDeploymentValidator creates a validator for v1beta1 DynamoGraphDeployment.
// mgr must not be nil.
func NewDynamoGraphDeploymentValidator(
	mgr ctrl.Manager,
) *DynamoGraphDeploymentValidator {
	return &DynamoGraphDeploymentValidator{
		mgr: mgr,
	}
}

// dynamoGraphDeploymentValidation carries DGD-specific request state.
// API values and derived traversal state remain explicit validator arguments.
type dynamoGraphDeploymentValidation struct {
	sharedValidation
	userInfo          *authenticationv1.UserInfo
	operatorPrincipal string
}

type dynamoGraphDeploymentSpecValidationOptions struct {
	dgdName                 string
	generation              int64
	workloadProvider        string
	grovePathway            bool
	grovePathwayRequirement string
	// oldComponents is nil on create. On update it carries the stored components
	// by name so the net-new GPU-product power rules can ratchet an unchanged
	// pre-existing violation.
	oldComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
}

// Validate performs stateless validation on the v1beta1 DynamoGraphDeployment.
// ctx and deployment must not be nil.
func (v *DynamoGraphDeploymentValidator) Validate(
	ctx context.Context,
	deployment *nvidiacomv1beta1.DynamoGraphDeployment,
	runtimeVersionSource runtimeVersionValidationSource,
) (admission.Warnings, error) {
	return v.validate(ctx, deployment, nil, runtimeVersionSource, false)
}

// validate performs the stateless traversal. deployment must not be nil.
// oldDeployment is the stored object on update and nil on create; it is used
// only to ratchet net-new rules against an unchanged pre-existing violation.
func (v *DynamoGraphDeploymentValidator) validate(
	ctx context.Context,
	deployment *nvidiacomv1beta1.DynamoGraphDeployment,
	oldDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	runtimeVersionSource runtimeVersionValidationSource,
	ratchetRuntimeVersion bool,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentValidation{
		sharedValidation: sharedValidation{
			ctx:                   ctx,
			mgr:                   v.mgr,
			runtimeVersionSource:  runtimeVersionSource,
			ratchetRuntimeVersion: ratchetRuntimeVersion,
		},
	}

	allErrs := validation.validateDynamoGraphDeployment(deployment, oldDeployment)
	alpha, err := alphaDynamoGraphDeploymentForValidation(deployment)
	if err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoGraphDeployment fields: %w", err)
	}
	allErrs = append(allErrs, validation.validateDynamoGraphDeploymentV1alpha1(alpha)...)

	return validation.warnings, invalidDynamoGraphDeploymentError(deployment, allErrs)
}

// ValidateUpdate performs stateful validation comparing old and new v1beta1 DGD objects.
// ctx, oldDGD, and newDGD must not be nil. runtimeVersionSource identifies the request's source API.
// If userInfo is nil, replica changes for DGDSA-enabled components fail closed.
func (v *DynamoGraphDeploymentValidator) ValidateUpdate(
	ctx context.Context,
	oldDGD *nvidiacomv1beta1.DynamoGraphDeployment,
	newDGD *nvidiacomv1beta1.DynamoGraphDeployment,
	userInfo *authenticationv1.UserInfo,
	operatorPrincipal string,
	runtimeVersionSource runtimeVersionValidationSource,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentValidation{
		sharedValidation: sharedValidation{
			ctx:                   ctx,
			mgr:                   v.mgr,
			runtimeVersionSource:  runtimeVersionSource,
			ratchetRuntimeVersion: true,
		},
		userInfo:          userInfo,
		operatorPrincipal: operatorPrincipal,
	}

	allErrs := validation.validateDynamoGraphDeploymentUpdate(newDGD, oldDGD)
	if validation.hasRuntimeVersionSource(runtimeVersionSourceV1Alpha1) {
		newAlpha, err := alphaDynamoGraphDeploymentForValidation(newDGD)
		if err != nil {
			return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoGraphDeployment fields: %w", err)
		}
		oldAlpha, err := alphaDynamoGraphDeploymentForValidation(oldDGD)
		if err != nil {
			return nil, fmt.Errorf("cannot validate old preserved v1alpha1 DynamoGraphDeployment fields: %w", err)
		}
		allErrs = append(allErrs, validation.validateDynamoGraphDeploymentSpecUpdateV1alpha1(
			&newAlpha.Spec,
			&oldAlpha.Spec,
			field.NewPath("spec"),
		)...)
	}
	return validation.warnings, invalidDynamoGraphDeploymentError(newDGD, allErrs)
}

// validateDynamoGraphDeployment validates dgd. dgd must not be nil.
// oldDGD is the stored object on update and nil on create.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeployment(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	oldDGD *nvidiacomv1beta1.DynamoGraphDeployment,
) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, v.validateObjectMeta(
		&dgd.ObjectMeta,
		field.NewPath("metadata"),
		hasIntraPodFailover(&dgd.Spec),
	)...)

	groveEnabled := features.MustGateFrom(v.ctx).Enabled(features.Grove)
	grovePathway, grovePathwayRequirement := grovePathwayForDynamoGraphDeployment(groveEnabled, dgd)
	workloadProvider := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]

	// Match stored components by name, as the update loop does, so the net-new
	// GPU-product power rules can ratchet an unchanged pre-existing violation.
	var oldComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
	if oldDGD != nil {
		oldComponents = componentsByName(oldDGD.Spec.Components)
	}

	specOpts := dynamoGraphDeploymentSpecValidationOptions{
		dgdName:                 dgd.Name,
		generation:              dgd.Generation,
		workloadProvider:        workloadProvider,
		grovePathway:            grovePathway,
		grovePathwayRequirement: grovePathwayRequirement,
		oldComponents:           oldComponents,
	}
	allErrs = append(allErrs, v.validateDynamoGraphDeploymentSpec(&dgd.Spec, field.NewPath("spec"), specOpts)...)

	return allErrs
}

// validateObjectMeta validates DGD objectMeta. objectMeta and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateObjectMeta(
	objectMeta *metav1.ObjectMeta,
	fldPath *field.Path,
	hasIntraPodFailover bool,
) field.ErrorList {
	allErrs := field.ErrorList{}
	annotationsPath := fldPath.Child("annotations")
	if value, exists := objectMeta.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; exists {
		if _, err := semver.NewVersion(value); err != nil {
			allErrs = append(allErrs, field.Invalid(
				annotationsPath.Key(consts.KubeAnnotationDynamoOperatorOriginVersion),
				value,
				"must be valid semver",
			))
		}
	}
	if value, invalid := invalidVLLMDistributedExecutorBackendAnnotation(objectMeta.Annotations); invalid {
		allErrs = append(allErrs, field.Invalid(
			annotationsPath.Key(consts.KubeAnnotationVLLMDistributedExecutorBackend),
			value,
			`must be "mp" or "ray"`,
		))
	}
	if value, exists := objectMeta.Annotations[consts.KubeAnnotationGroveUpdateStrategy]; exists &&
		value != string(grovev1alpha1.RollingRecreateStrategy) &&
		value != string(grovev1alpha1.OnDeleteStrategy) {
		allErrs = append(allErrs, field.NotSupported(
			annotationsPath.Key(consts.KubeAnnotationGroveUpdateStrategy),
			value,
			[]string{
				string(grovev1alpha1.RollingRecreateStrategy),
				string(grovev1alpha1.OnDeleteStrategy),
			},
		))
	}
	if value, exists := objectMeta.Annotations[consts.KubeAnnotationDynamoKubeDiscoveryMode]; exists && value != "pod" && value != "container" {
		allErrs = append(allErrs, field.NotSupported(
			annotationsPath.Key(consts.KubeAnnotationDynamoKubeDiscoveryMode),
			value,
			[]string{"pod", "container"},
		))
	}

	// Restrict the durable workload provider to programs implemented by the controller.
	if value, exists := objectMeta.Annotations[consts.KubeAnnotationWorkloadProvider]; exists &&
		value != consts.WorkloadProviderComponent && value != consts.WorkloadProviderGrove {
		allErrs = append(allErrs, field.NotSupported(
			annotationsPath.Key(consts.KubeAnnotationWorkloadProvider),
			value,
			[]string{consts.WorkloadProviderComponent, consts.WorkloadProviderGrove},
		))
	}

	if hasIntraPodFailover && objectMeta.Annotations[consts.KubeAnnotationDynamoKubeDiscoveryMode] != "container" {
		allErrs = append(allErrs, field.Invalid(
			annotationsPath.Key(consts.KubeAnnotationDynamoKubeDiscoveryMode),
			objectMeta.Annotations[consts.KubeAnnotationDynamoKubeDiscoveryMode],
			`must be "container" when intra-pod failover is configured`,
		))
	}

	return allErrs
}

// validateDynamoGraphDeploymentSpec validates spec. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSpec(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
	opts dynamoGraphDeploymentSpecValidationOptions,
) field.ErrorList {
	const validateInferencePoolAvailability = true

	allErrs := field.ErrorList{}

	// Validate the root provider fragment against the selected graph workload program.
	if spec.ProviderOverride != nil {
		allErrs = append(allErrs, v.validateProviderOverride(
			spec.ProviderOverride,
			fldPath.Child("providerOverride"),
			providerOverrideValidationOptions{
				supported:        true,
				workloadProvider: opts.workloadProvider,
				scope:            provideroverride.ScopeRoot,
			},
		)...)
	}

	if spec.PriorityClassName != "" && !opts.grovePathway {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("priorityClassName"), opts.grovePathwayRequirement))
	}

	componentsPath := fldPath.Child("components")
	if len(spec.Components) == 0 {
		allErrs = append(allErrs, field.Required(componentsPath, "must have at least one component"))
	}
	components := componentsByName(spec.Components)
	for i := range spec.Components {
		component := &spec.Components[i]
		componentPath := componentsPath.Index(i)

		if opts.grovePathway {
			combinedLength, detail := dgdComponentResourceNameLength(opts.dgdName, spec.Components, component)
			if combinedLength > maxCombinedResourceNameLength {
				allErrs = append(allErrs, field.Invalid(
					componentPath.Child("name"),
					component.ComponentName,
					fmt.Sprintf(
						"combined resource name length %d exceeds the %d-character pod-name limit (%s); shorten DynamoGraphDeployment name %q or component name %q",
						combinedLength,
						maxCombinedResourceNameLength,
						detail,
						opts.dgdName,
						component.ComponentName,
					),
				))
			}
		}

		gms := gpuMemoryServiceFor(component)
		if gms != nil && effectiveGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod {
			modePath := componentPath.Child("experimental", "gpuMemoryService", "mode")
			if !opts.grovePathway {
				allErrs = append(allErrs, field.Forbidden(modePath, opts.grovePathwayRequirement))
			}
			if spec.BackendFramework != string(dynamo.BackendFrameworkVLLM) {
				detected := spec.BackendFramework
				if detected == "" {
					detected = unsetValue
				}
				allErrs = append(allErrs, field.Invalid(
					modePath,
					gms.Mode,
					fmt.Sprintf("the inter-pod GMS layout is currently supported only for vLLM (detected backend: %s)", detected),
				))
			}
		}

		// Phase-1 power accounting reads scalar GPU resources, so it cannot account
		// for DRA devices, and it needs the exact GPU product to range-check the cap.
		allErrs = append(allErrs, v.validateDGDComponentPowerAnnotation(
			component,
			opts.oldComponents[component.ComponentName],
			componentPath,
		)...)

		allErrs = append(allErrs, validateElasticEPRequiresCommand(spec.BackendFramework, component, componentPath)...)

		allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpec(
			component,
			componentPath,
			dynamoComponentDeploymentSharedSpecValidationOptions{
				grovePathway:                      opts.grovePathway,
				validateInferencePoolAvailability: validateInferencePoolAvailability,
				providerOverridesSupported:        true,
				workloadProvider:                  opts.workloadProvider,
			},
		)...)
	}

	if spec.Restart != nil {
		allErrs = append(allErrs, v.validateRestart(spec.Restart, fldPath.Child("restart"), components)...)
	}

	constraintPath := fldPath.Child("topologyConstraint")
	hasComponentConstraint := false
	for i := range spec.Components {
		if spec.Components[i].TopologyConstraint != nil {
			hasComponentConstraint = true
			break
		}
	}
	hasAnyConstraint := spec.TopologyConstraint != nil || hasComponentConstraint
	if hasAnyConstraint {
		allErrs = append(allErrs, groveTopologyOverrideCompositionErrors(
			spec,
			opts.workloadProvider,
			fldPath,
		)...)
		topologyErrs := field.ErrorList{}
		if spec.TopologyConstraint == nil {
			topologyErrs = append(topologyErrs, field.Required(
				constraintPath,
				"is required when any component topology constraint is set",
			))
		} else {
			if spec.TopologyConstraint.PackDomain == "" && !hasComponentConstraint {
				topologyErrs = append(topologyErrs, field.Required(
					constraintPath.Child("packDomain"),
					"is required when no component topologyConstraint is set",
				))
			}

			var topologyInfo *clusterTopologyInfo
			if spec.TopologyConstraint.ClusterTopologyName != "" &&
				opts.generation <= 1 && opts.grovePathway {
				var err error
				topologyInfo, err = readGroveClusterTopology(v.ctx, v.mgr, spec.TopologyConstraint.ClusterTopologyName)
				if err != nil {
					detail := fmt.Sprintf("failed to read ClusterTopologyBinding: %v", err)
					if k8serrors.IsNotFound(err) {
						detail = "references a ClusterTopologyBinding resource that was not found"
					}
					topologyErrs = append(topologyErrs, field.Invalid(
						constraintPath.Child("clusterTopologyName"),
						spec.TopologyConstraint.ClusterTopologyName,
						detail,
					))
				}
			}

			topologyErrs = append(topologyErrs, v.validateSpecTopologyConstraint(
				spec.TopologyConstraint,
				constraintPath,
				topologyInfo,
			)...)
			for i := range spec.Components {
				componentConstraint := spec.Components[i].TopologyConstraint
				if componentConstraint == nil {
					continue
				}
				topologyErrs = append(topologyErrs, v.validateTopologyConstraint(
					componentConstraint,
					componentsPath.Index(i).Child("topologyConstraint"),
					spec.TopologyConstraint,
					topologyInfo,
				)...)
			}
		}
		allErrs = append(allErrs, topologyErrs...)
	}

	if spec.Experimental != nil {
		allErrs = append(allErrs, v.validateDynamoGraphDeploymentExperimentalSpec(
			spec.Experimental,
			fldPath.Child("experimental"),
			opts.generation,
			opts.grovePathway,
			opts.grovePathwayRequirement,
		)...)
	}

	return allErrs
}

// groveTopologyOverrideCompositionErrors returns conflicts between the typed
// and provider-native topology APIs. spec and fldPath must not be nil.
func groveTopologyOverrideCompositionErrors(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	provider string,
	fldPath *field.Path,
) field.ErrorList {
	if provider != consts.WorkloadProviderGrove {
		return nil
	}
	detail := "cannot be combined with spec.topologyConstraint or components[].topologyConstraint; use either the typed topology API or provider-native Grove topology overrides"
	allErrs := field.ErrorList{}
	if providerOverrideWritesGroveTopology(spec.ProviderOverride, provider, provideroverride.ScopeRoot, nil) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("providerOverride", "value"), detail))
	}
	componentsPath := fldPath.Child("components")
	for i := range spec.Components {
		component := &spec.Components[i]
		componentPath := componentsPath.Index(i)
		if providerOverrideWritesGroveTopology(component.ProviderOverride, provider, provideroverride.ScopeComponent, component) {
			allErrs = append(allErrs, field.Forbidden(componentPath.Child("providerOverride", "value"), detail))
		}
		if component.Multinode == nil {
			continue
		}
		if component.Multinode.Leader != nil && providerOverrideWritesGroveTopology(
			component.Multinode.Leader.ProviderOverride,
			provider,
			provideroverride.ScopeMultinodeLeader,
			component,
		) {
			allErrs = append(allErrs, field.Forbidden(componentPath.Child("multinode", "leader", "providerOverride", "value"), detail))
		}
		if component.Multinode.Worker != nil && providerOverrideWritesGroveTopology(
			component.Multinode.Worker.ProviderOverride,
			provider,
			provideroverride.ScopeMultinodeWorker,
			component,
		) {
			allErrs = append(allErrs, field.Forbidden(componentPath.Child("multinode", "worker", "providerOverride", "value"), detail))
		}
	}
	return allErrs
}

// providerOverrideWritesGroveTopology reports whether an optional override
// writes topology. component may be nil only for ScopeRoot.
func providerOverrideWritesGroveTopology(
	override *nvidiacomv1beta1.ProviderOverride,
	provider string,
	scope provideroverride.Scope,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) bool {
	if override == nil {
		return false
	}
	target := override.Target
	if target == "" {
		resolved, err := provideroverride.ExpectedTarget(provider, override.APIVersion, scope, component)
		if err != nil {
			return false
		}
		target = resolved
	}
	return provideroverride.WritesGroveTopology(target, override.Value.Raw)
}

// validateRestart validates restart. restart and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateRestart(
	restart *nvidiacomv1beta1.Restart,
	fldPath *field.Path,
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) field.ErrorList {
	if restart.Strategy == nil {
		return nil
	}
	return v.validateRestartStrategy(restart.Strategy, fldPath.Child("strategy"), components)
}

// validateRestartStrategy validates strategy. strategy and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateRestartStrategy(
	strategy *nvidiacomv1beta1.RestartStrategy,
	fldPath *field.Path,
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) field.ErrorList {
	if len(strategy.Order) == 0 {
		return nil
	}

	orderPath := fldPath.Child("order")
	if strategy.Type == nvidiacomv1beta1.RestartStrategyTypeParallel {
		return field.ErrorList{field.Forbidden(orderPath, "cannot be specified when strategy is parallel")}
	}

	allErrs := field.ErrorList{}
	uniqueOrder := getUnique(strategy.Order)
	if len(uniqueOrder) != len(strategy.Order) {
		allErrs = append(allErrs, field.Invalid(orderPath, strategy.Order, "must be unique"))
	}
	if len(uniqueOrder) != len(components) {
		allErrs = append(allErrs, field.Invalid(
			orderPath,
			strategy.Order,
			"must have the same number of unique components as the deployment",
		))
	}
	for i, componentName := range strategy.Order {
		if _, exists := components[componentName]; !exists {
			allErrs = append(allErrs, field.NotSupported(orderPath.Index(i), componentName, sortedComponentNames(components)))
		}
	}
	return allErrs
}

// validateSpecTopologyConstraint validates constraint. constraint and fldPath must not be nil.
// topologyInfo may be nil when live topology validation is not applicable.
func (v *dynamoGraphDeploymentValidation) validateSpecTopologyConstraint(
	constraint *nvidiacomv1beta1.SpecTopologyConstraint,
	fldPath *field.Path,
	topologyInfo *clusterTopologyInfo,
) field.ErrorList {
	if topologyInfo == nil || constraint.PackDomain == "" {
		return nil
	}
	if _, exists := topologyInfo.domainIndex[string(constraint.PackDomain)]; exists {
		return nil
	}
	return field.ErrorList{field.Invalid(
		fldPath.Child("packDomain"),
		constraint.PackDomain,
		fmt.Sprintf("does not exist in ClusterTopologyBinding %q; available domains: %v", topologyInfo.name, topologyInfo.domains),
	)}
}

// validateDynamoGraphDeploymentExperimentalSpec validates experimental. experimental and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentExperimentalSpec(
	experimental *nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec,
	fldPath *field.Path,
	generation int64,
	grovePathway bool,
	grovePathwayRequirement string,
) field.ErrorList {
	if experimental.KvTransferPolicy == nil {
		return nil
	}
	return v.validateKvTransferPolicy(
		experimental.KvTransferPolicy,
		fldPath.Child("kvTransferPolicy"),
		generation,
		grovePathway,
		grovePathwayRequirement,
	)
}

// validateKvTransferPolicy validates policy. policy and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateKvTransferPolicy(
	policy *nvidiacomv1beta1.KvTransferPolicy,
	fldPath *field.Path,
	generation int64,
	grovePathway bool,
	grovePathwayRequirement string,
) field.ErrorList {
	if policy.ClusterTopologyName == "" {
		return nil
	}

	allErrs := field.ErrorList{}
	namePath := fldPath.Child("clusterTopologyName")
	if nameErrs := k8svalidation.IsDNS1123Subdomain(policy.ClusterTopologyName); len(nameErrs) > 0 {
		allErrs = append(allErrs, field.Invalid(
			namePath,
			policy.ClusterTopologyName,
			strings.Join(nameErrs, "; "),
		))
	}
	if !grovePathway {
		allErrs = append(allErrs, field.Forbidden(namePath, grovePathwayRequirement))
	}
	if len(allErrs) != 0 || generation > 1 {
		return allErrs
	}

	topologyInfo, err := readGroveClusterTopology(v.ctx, v.mgr, policy.ClusterTopologyName)
	if err != nil {
		detail := fmt.Sprintf("failed to read ClusterTopologyBinding: %v", err)
		if k8serrors.IsNotFound(err) {
			detail = "references a ClusterTopologyBinding resource that was not found"
		}
		return append(allErrs, field.Invalid(namePath, policy.ClusterTopologyName, detail))
	}
	if _, exists := topologyInfo.domainIndex[string(policy.Domain)]; !exists {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("domain"),
			policy.Domain,
			fmt.Sprintf("does not exist in ClusterTopologyBinding %q; available domains: %v", topologyInfo.name, topologyInfo.domains),
		))
	}
	return allErrs
}

// validateDynamoGraphDeploymentUpdate validates an update. newDGD and oldDGD must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentUpdate(
	newDGD *nvidiacomv1beta1.DynamoGraphDeployment,
	oldDGD *nvidiacomv1beta1.DynamoGraphDeployment,
) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, v.validateObjectMetaUpdate(
		&newDGD.ObjectMeta,
		&oldDGD.ObjectMeta,
		field.NewPath("metadata"),
	)...)
	allErrs = append(allErrs, v.validateDynamoGraphDeploymentSpecUpdate(
		&newDGD.Spec,
		&oldDGD.Spec,
		field.NewPath("spec"),
	)...)

	if oldDGD.Status.RollingUpdate != nil {
		phase := oldDGD.Status.RollingUpdate.Phase
		if phase == nvidiacomv1beta1.RollingUpdatePhasePending || phase == nvidiacomv1beta1.RollingUpdatePhaseInProgress {
			oldID := k8sptr.Deref(oldDGD.Spec.Restart, nvidiacomv1beta1.Restart{}).ID
			newID := k8sptr.Deref(newDGD.Spec.Restart, nvidiacomv1beta1.Restart{}).ID
			if oldID != newID {
				allErrs = append(allErrs, field.Invalid(
					field.NewPath("spec", "restart", "id"),
					newID,
					fmt.Sprintf("cannot be changed while a rolling update is %s", phase),
				))
			}
		}
	}
	return allErrs
}

// validateObjectMetaUpdate validates a DGD metadata update.
// newObjectMeta, oldObjectMeta, and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateObjectMetaUpdate(
	newObjectMeta *metav1.ObjectMeta,
	oldObjectMeta *metav1.ObjectMeta,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	annotationsPath := fldPath.Child("annotations")
	newProvider, newProviderExists := newObjectMeta.Annotations[consts.KubeAnnotationWorkloadProvider]
	oldProvider, oldProviderExists := oldObjectMeta.Annotations[consts.KubeAnnotationWorkloadProvider]

	// Reserve the initial legacy-provider materialization for the configured operator identity.
	if !oldProviderExists && newProviderExists && v.operatorPrincipal != "" &&
		(v.userInfo == nil || v.userInfo.Username != v.operatorPrincipal) {
		allErrs = append(allErrs, field.Forbidden(
			annotationsPath.Key(consts.KubeAnnotationWorkloadProvider),
			"may only be materialized by the Dynamo operator",
		))
	}

	// Once materialized, the workload provider cannot be replaced or removed.
	if oldProviderExists && (!newProviderExists || newProvider != oldProvider) {
		var invalidValue any
		if newProviderExists {
			invalidValue = newProvider
		}
		allErrs = append(allErrs, field.Invalid(
			annotationsPath.Key(consts.KubeAnnotationWorkloadProvider),
			invalidValue,
			apivalidation.FieldImmutableErrorMsg,
		))
	}

	return allErrs
}

// validateDynamoGraphDeploymentSpecUpdate validates a spec update. newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSpecUpdate(
	newSpec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	oldSpec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}

	// Keep an existing root provider identity stable across updates.
	if newSpec.ProviderOverride != nil && oldSpec.ProviderOverride != nil {
		allErrs = append(allErrs, validateProviderOverrideUpdate(
			newSpec.ProviderOverride,
			oldSpec.ProviderOverride,
			fldPath.Child("providerOverride"),
		)...)
	}

	newComponents := componentsByName(newSpec.Components)
	oldComponents := componentsByName(oldSpec.Components)

	added := difference(componentNameSet(newComponents), componentNameSet(oldComponents))
	removed := difference(componentNameSet(oldComponents), componentNameSet(newComponents))
	sort.Strings(added)
	sort.Strings(removed)
	if len(added) != 0 || len(removed) != 0 {
		detail := "component topology is immutable and cannot be modified after creation"
		switch {
		case len(added) != 0 && len(removed) != 0:
			detail = fmt.Sprintf("%s: components added: %v, components removed: %v", detail, added, removed)
		case len(added) != 0:
			detail = fmt.Sprintf("%s: components added: %v", detail, added)
		default:
			detail = fmt.Sprintf("%s: components removed: %v", detail, removed)
		}
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("components"), detail))
	}

	canModifyReplicas := v.userInfo != nil && internalwebhook.CanModifyDGDReplicas(v.operatorPrincipal, *v.userInfo)
	// The DGD handler runs the stateless new-state traversal before this validator,
	// so the GMS new-state rules are redundant on the production admission path.
	// They stay enabled because the exported ValidateUpdate is reachable on its own.
	const validateGPUMemoryServiceNewState = true
	componentsPath := fldPath.Child("components")
	for i := range newSpec.Components {
		newComponent := &newSpec.Components[i]
		oldComponent, exists := oldComponents[newComponent.ComponentName]
		if !exists {
			continue
		}
		allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpecUpdate(
			newComponent,
			oldComponent,
			componentsPath.Index(i),
			canModifyReplicas,
			nvidiacomv1beta1.DynamoGraphDeploymentGVK.GroupKind(),
			validateGPUMemoryServiceNewState,
		)...)
		allErrs = append(allErrs, v.validateDynamoGraphDeploymentSharedSpecUpdate(
			newComponent,
			oldComponent,
			componentsPath.Index(i),
		)...)
	}

	if newSpec.BackendFramework != oldSpec.BackendFramework {
		v.warn("Changing spec.backendFramework may cause unexpected behavior")
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("backendFramework"),
			newSpec.BackendFramework,
			"is immutable and cannot be changed after creation",
		))
	}

	topologyPath := fldPath.Child("topologyConstraint")
	if newSpec.TopologyConstraint != nil {
		allErrs = append(allErrs, v.validateSpecTopologyConstraintUpdate(
			newSpec.TopologyConstraint,
			oldSpec.TopologyConstraint,
			topologyPath,
		)...)
	} else if oldSpec.TopologyConstraint != nil {
		allErrs = append(allErrs, field.Invalid(
			topologyPath,
			newSpec.TopologyConstraint,
			"is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints",
		))
	}

	if newSpec.Experimental != nil {
		allErrs = append(allErrs, v.validateDynamoGraphDeploymentExperimentalSpecUpdate(
			newSpec.Experimental,
			oldSpec.Experimental,
			fldPath.Child("experimental"),
		)...)
	} else if oldPolicy := kvTransferPolicyFor(oldSpec.Experimental); oldPolicy != nil {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("experimental", "kvTransferPolicy"),
			newSpec.Experimental,
			"is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change the KV transfer policy",
		))
	}

	return allErrs
}

// validateDynamoGraphDeploymentSharedSpecUpdate validates DGD-specific component fields on update.
// newComponent, oldComponent, and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSharedSpecUpdate(
	newComponent *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	oldComponent *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	newPowerLimit, newHasPowerLimit := dgdPowerLimit(newComponent)
	oldPowerLimit, oldHasPowerLimit := dgdPowerLimit(oldComponent)

	// Reject transitions into, out of, or within the power-annotation contract.
	if newHasPowerLimit != oldHasPowerLimit ||
		(newHasPowerLimit && newPowerLimit != oldPowerLimit) {
		var invalidValue any
		if newHasPowerLimit {
			invalidValue = newPowerLimit
		}
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("podTemplate", "metadata", "annotations").Key(consts.KubeAnnotationGPUPowerLimit),
			invalidValue,
			apivalidation.FieldImmutableErrorMsg,
		))
	}

	if oldHasPowerLimit {
		// Keep the Planner's remaining cached per-replica power inputs stable.
		newNumberOfGPUs := effectiveNumberOfGPUsV1Beta1(newComponent, fldPath)
		oldNumberOfGPUs := effectiveNumberOfGPUsV1Beta1(oldComponent, fldPath)
		if !newNumberOfGPUs.equal(oldNumberOfGPUs) {
			allErrs = append(allErrs, field.Invalid(
				newNumberOfGPUs.path,
				newNumberOfGPUs.invalidValue(),
				apivalidation.FieldImmutableErrorMsg,
			))
		}

		// Keep the exact GPU product the power range was validated against stable.
		// The create rules already reject a non-empty nodeName and a missing or
		// unknown product on any request they are not ratcheted for, and the ratchet
		// requires an identical placement contract, so no additional nodeName rule
		// is reachable here.
		newGPUProduct := dgdGPUProductSelector(newComponent)
		if newGPUProduct != dgdGPUProductSelector(oldComponent) {
			var invalidValue any
			if newGPUProduct != "" {
				invalidValue = newGPUProduct
			}
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("podTemplate", "spec", "nodeSelector").Key(gpuProductNodeSelectorLabel),
				invalidValue,
				apivalidation.FieldImmutableErrorMsg,
			))
		}

		if newComponent.GetNumberOfNodes() != oldComponent.GetNumberOfNodes() {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("multinode", "nodeCount"),
				newComponent.GetNumberOfNodes(),
				apivalidation.FieldImmutableErrorMsg,
			))
		}
	}
	return allErrs
}

// validateSpecTopologyConstraintUpdate validates a topology constraint update.
// newConstraint and fldPath must not be nil; oldConstraint may be nil for an addition.
func (v *dynamoGraphDeploymentValidation) validateSpecTopologyConstraintUpdate(
	newConstraint *nvidiacomv1beta1.SpecTopologyConstraint,
	oldConstraint *nvidiacomv1beta1.SpecTopologyConstraint,
	fldPath *field.Path,
) field.ErrorList {
	if oldConstraint != nil &&
		newConstraint.ClusterTopologyName == oldConstraint.ClusterTopologyName &&
		newConstraint.PackDomain == oldConstraint.PackDomain {
		return nil
	}
	return field.ErrorList{field.Invalid(
		fldPath,
		newConstraint,
		"is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints",
	)}
}

// validateDynamoGraphDeploymentExperimentalSpecUpdate validates an experimental spec update.
// newExperimental and fldPath must not be nil; oldExperimental may be nil for an addition.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentExperimentalSpecUpdate(
	newExperimental *nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec,
	oldExperimental *nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec,
	fldPath *field.Path,
) field.ErrorList {
	newPolicy := newExperimental.KvTransferPolicy
	oldPolicy := kvTransferPolicyFor(oldExperimental)
	if newPolicy != nil {
		return v.validateKvTransferPolicyUpdate(newPolicy, oldPolicy, fldPath.Child("kvTransferPolicy"))
	}
	if oldPolicy == nil {
		return nil
	}
	return field.ErrorList{field.Invalid(
		fldPath.Child("kvTransferPolicy"),
		newPolicy,
		"is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change the KV transfer policy",
	)}
}

// validateKvTransferPolicyUpdate validates a policy update.
// newPolicy and fldPath must not be nil; oldPolicy may be nil for an addition.
func (v *dynamoGraphDeploymentValidation) validateKvTransferPolicyUpdate(
	newPolicy *nvidiacomv1beta1.KvTransferPolicy,
	oldPolicy *nvidiacomv1beta1.KvTransferPolicy,
	fldPath *field.Path,
) field.ErrorList {
	if kvTransferPoliciesEqual(newPolicy, oldPolicy) {
		return nil
	}
	return field.ErrorList{field.Invalid(
		fldPath,
		newPolicy,
		"is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change the KV transfer policy",
	)}
}

// validateDGDComponentPowerAnnotation validates the power-limit annotation value,
// its incompatibility with DRA-backed GPU allocation, and the GPU product the
// annotated component selects. component and componentPath must not be nil.
//
// oldComponent is the stored component of the same name on update and nil on
// create. It exists only to ratchet the net-new GPU-product rules: a
// power-annotated component created before those rules shipped has no product
// selector and would otherwise fail every subsequent write, including the
// Planner's own replica updates. Suppression is total for those rules and never
// introduces an error, so it cannot reach a component without the annotation,
// and it never reaches the positive-integer or DRA rules, which keep firing
// unconditionally.
func (v *dynamoGraphDeploymentValidation) validateDGDComponentPowerAnnotation(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	oldComponent *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	componentPath *field.Path,
) field.ErrorList {
	powerLimitValue, hasPowerLimit := dgdPowerLimit(component)
	if !hasPowerLimit {
		return nil
	}
	var allErrs field.ErrorList
	powerLimitPath := componentPath.Child("podTemplate", "metadata", "annotations").Key(consts.KubeAnnotationGPUPowerLimit)
	powerLimit, valueErrs := validateDGDPowerLimitValue(powerLimitValue, powerLimitPath)
	allErrs = append(allErrs, valueErrs...)

	if draPath := dgdDRAPath(component, componentPath); draPath != nil {
		allErrs = append(allErrs, field.Forbidden(
			draPath,
			fmt.Sprintf(
				"cannot be combined with annotation %q: power-aware planning does not support DRA-backed device allocation",
				consts.KubeAnnotationGPUPowerLimit,
			),
		))
	}

	// Ratchet the net-new GPU-product rules for an unchanged pre-existing violation.
	if oldComponent != nil && equality.Semantic.DeepEqual(
		dgdPowerProductContract(oldComponent),
		dgdPowerProductContract(component),
	) {
		return allErrs
	}
	return append(allErrs, validateDGDComponentGPUProduct(component, componentPath, powerLimit, powerLimitPath)...)
}

// validateDGDComponentGPUProduct validates the GPU product a power-annotated
// component selects, and the requested cap against that product's reviewed
// settable TGP range. component, componentPath, and powerLimitPath must not be
// nil. powerLimit carries its own presence boolean; an unparseable annotation is
// already reported by validateDGDPowerLimitValue and skips the range comparison.
//
// component.PodTemplate is non-nil for every caller. That is domain semantics
// rather than a defensive guard: this rule is reached only for a component
// carrying the power annotation, and the annotation lives on
// podTemplate.metadata, so a component without a pod template cannot carry it.
func validateDGDComponentGPUProduct(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	componentPath *field.Path,
	powerLimit dgdPowerLimitWatts,
	powerLimitPath *field.Path,
) field.ErrorList {
	var allErrs field.ErrorList
	podSpecPath := componentPath.Child("podTemplate", "spec")
	selectorPath := podSpecPath.Child("nodeSelector").Key(gpuProductNodeSelectorLabel)
	product := dgdGPUProductSelector(component)

	// Resolve the selected product, then range-check the cap against it. Report
	// an unknown product as Invalid rather than NotSupported: NotSupported would
	// render the whole catalog into the error and publish it as an apparent
	// hardware support matrix.
	productRange, known := powerRanges[product]
	switch {
	case product == "":
		allErrs = append(allErrs, field.Required(
			selectorPath,
			fmt.Sprintf(
				"is required when annotation %q is set, so the requested power limit can be validated against the selected GPU product",
				consts.KubeAnnotationGPUPowerLimit,
			),
		))
	case !known:
		allErrs = append(allErrs, field.Invalid(
			selectorPath,
			product,
			fmt.Sprintf(
				"has no reviewed GPU power range in this operator release; select a GPU product with a reviewed range or remove annotation %q",
				consts.KubeAnnotationGPUPowerLimit,
			),
		))
	case powerLimit.parsed && (powerLimit.watts < productRange.Min || powerLimit.watts > productRange.Max):
		allErrs = append(allErrs, field.Invalid(
			powerLimitPath,
			powerLimit.value,
			fmt.Sprintf(
				"must be between %d and %d watts inclusive for GPU product %q",
				productRange.Min,
				productRange.Max,
				product,
			),
		))
	}

	if component.PodTemplate.Spec.NodeName != "" {
		allErrs = append(allErrs, field.Forbidden(
			podSpecPath.Child("nodeName"),
			fmt.Sprintf(
				"cannot be combined with annotation %q: bypassing the scheduler invalidates the GPU product selected by %q",
				consts.KubeAnnotationGPUPowerLimit,
				gpuProductNodeSelectorLabel,
			),
		))
	}
	return allErrs
}
