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
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	authenticationv1 "k8s.io/api/authentication/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
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
	grovePathway            bool
	grovePathwayRequirement string
}

// Validate performs stateless validation on the v1beta1 DynamoGraphDeployment.
// ctx and deployment must not be nil.
func (v *DynamoGraphDeploymentValidator) Validate(
	ctx context.Context,
	deployment *nvidiacomv1beta1.DynamoGraphDeployment,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentValidation{
		sharedValidation: sharedValidation{ctx: ctx, mgr: v.mgr},
	}

	allErrs := validation.validateDynamoGraphDeployment(deployment)
	alpha, err := alphaDynamoGraphDeploymentForValidation(deployment)
	if err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoGraphDeployment fields: %w", err)
	}
	allErrs = append(allErrs, validation.validateDynamoGraphDeploymentV1alpha1(alpha)...)

	return validation.warnings, invalidDynamoGraphDeploymentError(deployment, allErrs)
}

// ValidateUpdate performs stateful validation comparing old and new v1beta1 DGD objects.
// ctx, oldDGD, and newDGD must not be nil.
// If userInfo is nil, replica changes for DGDSA-enabled components fail closed.
func (v *DynamoGraphDeploymentValidator) ValidateUpdate(
	ctx context.Context,
	oldDGD *nvidiacomv1beta1.DynamoGraphDeployment,
	newDGD *nvidiacomv1beta1.DynamoGraphDeployment,
	userInfo *authenticationv1.UserInfo,
	operatorPrincipal string,
) (admission.Warnings, error) {
	validation := &dynamoGraphDeploymentValidation{
		sharedValidation:  sharedValidation{ctx: ctx, mgr: v.mgr},
		userInfo:          userInfo,
		operatorPrincipal: operatorPrincipal,
	}

	allErrs := validation.validateDynamoGraphDeploymentUpdate(newDGD, oldDGD)
	return validation.warnings, invalidDynamoGraphDeploymentError(newDGD, allErrs)
}

// validateDynamoGraphDeployment validates dgd. dgd must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeployment(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, v.validateObjectMeta(
		&dgd.ObjectMeta,
		field.NewPath("metadata"),
		hasIntraPodFailover(&dgd.Spec),
	)...)

	groveEnabled := features.GateFromContext(v.ctx).Enabled(features.Grove)
	grovePathway, grovePathwayRequirement := grovePathwayForDynamoGraphDeployment(groveEnabled, dgd)
	specOpts := dynamoGraphDeploymentSpecValidationOptions{
		dgdName:                 dgd.Name,
		generation:              dgd.Generation,
		grovePathway:            grovePathway,
		grovePathwayRequirement: grovePathwayRequirement,
	}
	allErrs = append(allErrs, v.validateDynamoGraphDeploymentSpec(&dgd.Spec, field.NewPath("spec"), specOpts)...)

	return allErrs
}

// validateObjectMeta validates objectMeta. objectMeta and fldPath must not be nil.
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

		allErrs = append(allErrs, v.validateDynamoComponentDeploymentSharedSpec(
			component,
			componentPath,
			opts.grovePathway,
			validateInferencePoolAvailability,
		)...)
	}

	if spec.Restart != nil {
		allErrs = append(allErrs, v.validateRestart(spec.Restart, fldPath.Child("restart"), components)...)
	}

	constraintPath := fldPath.Child("topologyConstraint")
	hasAnyConstraint := spec.TopologyConstraint != nil
	for i := range spec.Components {
		if spec.Components[i].TopologyConstraint != nil {
			hasAnyConstraint = true
			break
		}
	}
	if hasAnyConstraint {
		topologyErrs := field.ErrorList{}
		if spec.TopologyConstraint == nil {
			topologyErrs = append(topologyErrs, field.Required(
				constraintPath,
				"is required when any component topology constraint is set",
			))
		} else {
			if spec.TopologyConstraint.PackDomain == "" {
				for i := range spec.Components {
					if spec.Components[i].TopologyConstraint == nil {
						topologyErrs = append(topologyErrs, field.Required(
							componentsPath.Index(i).Child("topologyConstraint"),
							"is required because spec.topologyConstraint.packDomain is not set",
						))
					}
				}
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

// validateDynamoGraphDeploymentSpecUpdate validates a spec update. newSpec, oldSpec, and fldPath must not be nil.
func (v *dynamoGraphDeploymentValidation) validateDynamoGraphDeploymentSpecUpdate(
	newSpec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	oldSpec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
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
