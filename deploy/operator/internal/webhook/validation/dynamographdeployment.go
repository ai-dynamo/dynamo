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
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	semver "github.com/Masterminds/semver/v3"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controllercommon "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	authenticationv1 "k8s.io/api/authentication/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

var betaTopologyDomainRegex = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`)

// DynamoGraphDeploymentBetaValidator validates v1beta1 DynamoGraphDeployment resources.
type DynamoGraphDeploymentBetaValidator struct {
	deployment   *nvidiacomv1beta1.DynamoGraphDeployment
	mgr          ctrl.Manager
	groveEnabled bool
}

// NewDynamoGraphDeploymentBetaValidator creates a validator for v1beta1 DynamoGraphDeployment.
func NewDynamoGraphDeploymentBetaValidator(
	deployment *nvidiacomv1beta1.DynamoGraphDeployment,
	mgr ctrl.Manager,
	groveEnabled bool,
) *DynamoGraphDeploymentBetaValidator {
	return &DynamoGraphDeploymentBetaValidator{
		deployment:   deployment,
		mgr:          mgr,
		groveEnabled: groveEnabled,
	}
}

// Validate performs stateless validation on the v1beta1 DynamoGraphDeployment.
func (v *DynamoGraphDeploymentBetaValidator) Validate(ctx context.Context) (admission.Warnings, error) {
	components, err := betaComponentsByName(v.deployment)
	if err != nil {
		return nil, err
	}
	if len(components) == 0 {
		return nil, fmt.Errorf("spec.components must have at least one component")
	}

	if err := v.validateAnnotations(); err != nil {
		return nil, err
	}
	if err := v.validateRestart(components); err != nil {
		return nil, err
	}
	if err := v.validatePriorityClassName(); err != nil {
		return nil, err
	}
	if err := v.validateTopologyConstraints(ctx, components); err != nil {
		return nil, err
	}
	if err := v.validateKvTransferPolicy(ctx); err != nil {
		return nil, err
	}
	if err := v.validateFailoverRequiresDiscoveryMode(); err != nil {
		return nil, err
	}

	var allWarnings admission.Warnings
	for i := range v.deployment.Spec.Components {
		component := &v.deployment.Spec.Components[i]
		warnings, err := v.validateComponent(ctx, component)
		if err != nil {
			return nil, err
		}
		allWarnings = append(allWarnings, warnings...)
	}

	return allWarnings, nil
}

// ValidateUpdate performs stateful validation comparing old and new v1beta1 DGD objects.
func (v *DynamoGraphDeploymentBetaValidator) ValidateUpdate(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
	userInfo *authenticationv1.UserInfo,
	operatorPrincipal string,
) (admission.Warnings, error) {
	var warnings admission.Warnings

	oldComponents, err := betaComponentsByName(old)
	if err != nil {
		return warnings, err
	}
	newComponents, err := betaComponentsByName(v.deployment)
	if err != nil {
		return warnings, err
	}

	if err := v.validateImmutableFields(old, oldComponents, newComponents, &warnings); err != nil {
		return warnings, err
	}
	if err := v.validateComponentTopology(oldComponents, newComponents); err != nil {
		return warnings, err
	}
	if err := v.validateReplicasChanges(oldComponents, newComponents, userInfo, operatorPrincipal); err != nil {
		return warnings, err
	}
	if err := v.validateNoRestartDuringRollingUpdate(old); err != nil {
		return warnings, err
	}

	return warnings, nil
}

func (v *DynamoGraphDeploymentBetaValidator) validateImmutableFields(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
	oldComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	newComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	warnings *admission.Warnings,
) error {
	var errs []error

	if v.deployment.Spec.BackendFramework != old.Spec.BackendFramework {
		*warnings = append(*warnings, "Changing spec.backendFramework may cause unexpected behavior")
		errs = append(errs, fmt.Errorf("spec.backendFramework is immutable and cannot be changed after creation"))
	}

	componentNames := sortedBetaComponentNames(newComponents)
	for _, componentName := range componentNames {
		newComponent := newComponents[componentName]
		oldComponent, exists := oldComponents[componentName]
		if !exists {
			continue
		}
		if oldComponent.IsMultinode() != newComponent.IsMultinode() {
			errs = append(errs, fmt.Errorf(
				"spec.components[%s] cannot change node topology (between single-node and multi-node) after creation",
				componentName,
			))
		}
		if oldComponent.MinAvailable != nil {
			if newComponent.MinAvailable == nil || *newComponent.MinAvailable != *oldComponent.MinAvailable {
				errs = append(errs, fmt.Errorf(
					"spec.components[%s].minAvailable is immutable after creation",
					componentName,
				))
			}
		}

		oldInterPodGMS := oldComponent.IsInterPodGMSEnabled()
		newInterPodGMS := newComponent.IsInterPodGMSEnabled()
		if oldInterPodGMS != newInterPodGMS {
			errs = append(errs, fmt.Errorf(
				"spec.components[%s].experimental.gpuMemoryService.mode: the inter-pod GMS layout cannot be toggled after creation; "+
					"delete and recreate the DynamoGraphDeployment",
				componentName,
			))
		}
		oldInterPodFailover := oldComponent.IsInterPodFailoverEnabled()
		newInterPodFailover := newComponent.IsInterPodFailoverEnabled()
		if oldInterPodFailover != newInterPodFailover {
			errs = append(errs, fmt.Errorf(
				"spec.components[%s].experimental.failover: inter-pod GMS failover cannot be toggled after creation; "+
					"delete and recreate the DynamoGraphDeployment",
				componentName,
			))
		}
		if oldInterPodFailover && newInterPodFailover && oldComponent.GetNumShadows() != newComponent.GetNumShadows() {
			errs = append(errs, fmt.Errorf(
				"spec.components[%s].experimental.failover.numShadows is immutable for inter-pod GMS failover; "+
					"delete and recreate the DynamoGraphDeployment to change it",
				componentName,
			))
		}
	}

	if err := v.validateTopologyConstraintImmutability(old, oldComponents, newComponents); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateKvTransferPolicyImmutability(old); err != nil {
		errs = append(errs, err)
	}

	return errors.Join(errs...)
}

func (v *DynamoGraphDeploymentBetaValidator) validateComponentTopology(
	oldComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	newComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	oldNames := betaComponentNameSet(oldComponents)
	newNames := betaComponentNameSet(newComponents)

	added := difference(newNames, oldNames)
	removed := difference(oldNames, newNames)
	if len(added) == 0 && len(removed) == 0 {
		return nil
	}

	sort.Strings(added)
	sort.Strings(removed)

	switch {
	case len(added) > 0 && len(removed) > 0:
		return fmt.Errorf("component topology is immutable and cannot be modified after creation: components added: %v, components removed: %v", added, removed)
	case len(added) > 0:
		return fmt.Errorf("component topology is immutable and cannot be modified after creation: components added: %v", added)
	default:
		return fmt.Errorf("component topology is immutable and cannot be modified after creation: components removed: %v", removed)
	}
}

func (v *DynamoGraphDeploymentBetaValidator) validateReplicasChanges(
	oldComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	newComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	userInfo *authenticationv1.UserInfo,
	operatorPrincipal string,
) error {
	if userInfo != nil && internalwebhook.CanModifyDGDReplicas(operatorPrincipal, *userInfo) {
		return nil
	}

	var errs []error
	for _, componentName := range sortedBetaComponentNames(newComponents) {
		newComponent := newComponents[componentName]
		if newComponent.ScalingAdapter == nil {
			continue
		}
		oldComponent, exists := oldComponents[componentName]
		if !exists {
			continue
		}

		oldReplicas := int32(1)
		if oldComponent.Replicas != nil {
			oldReplicas = *oldComponent.Replicas
		}
		newReplicas := int32(1)
		if newComponent.Replicas != nil {
			newReplicas = *newComponent.Replicas
		}

		if oldReplicas != newReplicas {
			errs = append(errs, fmt.Errorf(
				"spec.components[%s].replicas cannot be modified directly when scaling adapter is enabled; "+
					"scale or update the related DynamoGraphDeploymentScalingAdapter instead",
				componentName))
		}
	}

	return errors.Join(errs...)
}

func (v *DynamoGraphDeploymentBetaValidator) validateComponent(
	ctx context.Context,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (admission.Warnings, error) {
	componentName := component.ComponentName
	fieldPath := fmt.Sprintf("spec.components[%s]", componentName)

	if component.IsInterPodGMSEnabled() && !v.isGrovePathway() {
		return nil, v.grovePathwayRequiredError(fmt.Sprintf(
			"%s: experimental.gpuMemoryService.mode=%q",
			fieldPath, nvidiacomv1beta1.GMSModeInterPod))
	}

	if component.IsInterPodGMSEnabled() &&
		v.deployment.Spec.BackendFramework != backendFrameworkVLLM {
		detected := v.deployment.Spec.BackendFramework
		if detected == "" {
			detected = unsetValue
		}
		return nil, fmt.Errorf(
			"%s: the inter-pod GMS layout (experimental.gpuMemoryService.mode=%q) is currently supported only for vLLM (detected: %s); "+
				"set spec.backendFramework=%q",
			fieldPath, nvidiacomv1beta1.GMSModeInterPod, detected, backendFrameworkVLLM)
	}

	if v.isGrovePathway() {
		if err := v.validateComponentNameLength(componentName, component); err != nil {
			return nil, err
		}
	}

	sharedValidator := NewBetaSharedSpecValidator(component, fieldPath, v.isGrovePathway(), v.mgr)
	return sharedValidator.Validate(ctx)
}

func (v *DynamoGraphDeploymentBetaValidator) validateComponentNameLength(
	componentName string,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	pcsName := dynamo.PCSNameForDGD(v.deployment.Name, v.deployment.Spec.Components)
	lowerComponentName := strings.ToLower(componentName)

	hasPodCliqueScalingGroup := component.GetNumberOfNodes() > 1 || component.IsInterPodGMSEnabled()
	if !hasPodCliqueScalingGroup {
		combinedLength := len(pcsName) + len(lowerComponentName)
		if combinedLength > maxCombinedResourceNameLength {
			return fmt.Errorf("%s: combined resource name length %d exceeds %d-character limit required for pod naming. "+
				"Consider shortening the DynamoGraphDeployment name '%s' (length %d) or component name '%s' (length %d). "+
				"The combined length of PCS name + component name must not exceed %d characters. "+
				"The PCS name '%s' was auto-truncated from DGD name '%s'",
				fmt.Sprintf("spec.components[%s]", componentName), combinedLength, maxCombinedResourceNameLength,
				v.deployment.Name, len(v.deployment.Name), componentName, len(componentName),
				maxCombinedResourceNameLength,
				pcsName, v.deployment.Name)
		}
		return nil
	}

	longestPCLQName := dynamo.LongestPodCliqueNameForDGDComponent(componentName, component)
	combinedLength := len(pcsName) + len(lowerComponentName) + len(longestPCLQName)
	if combinedLength > maxCombinedResourceNameLength {
		return fmt.Errorf("spec.components[%s]: combined resource name length %d exceeds %d-character limit required for pod naming. "+
			"Consider shortening the DynamoGraphDeployment name '%s' (length %d) or component name '%s' (length %d). "+
			"The combined length of PCS name + PCSG name + longest PodClique name ('%s') must not exceed %d characters. "+
			"The PCS name '%s' was auto-truncated from DGD name '%s'",
			componentName, combinedLength, maxCombinedResourceNameLength,
			v.deployment.Name, len(v.deployment.Name), componentName, len(componentName),
			longestPCLQName, maxCombinedResourceNameLength,
			pcsName, v.deployment.Name)
	}

	return nil
}

func (v *DynamoGraphDeploymentBetaValidator) isGrovePathway() bool {
	if !v.groveEnabled {
		return false
	}
	return v.deployment.Annotations == nil ||
		strings.ToLower(v.deployment.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse
}

func (v *DynamoGraphDeploymentBetaValidator) grovePathwayRequiredError(subject string) error {
	if !v.groveEnabled {
		return fmt.Errorf("%s requires the Grove pathway, but Grove is disabled in the operator configuration", subject)
	}
	return fmt.Errorf("%s requires the Grove pathway; remove or unset the %q annotation (currently %q)",
		subject, consts.KubeAnnotationEnableGrove, v.deployment.Annotations[consts.KubeAnnotationEnableGrove])
}

func (v *DynamoGraphDeploymentBetaValidator) validateRestart(
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	if v.deployment.Spec.Restart == nil {
		return nil
	}

	restart := v.deployment.Spec.Restart
	var err error
	if restart.ID == "" {
		err = errors.Join(err, fmt.Errorf("spec.restart.id is required"))
	}
	return errors.Join(err, v.validateRestartStrategyOrder(components))
}

func (v *DynamoGraphDeploymentBetaValidator) validateRestartStrategyOrder(
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	restart := v.deployment.Spec.Restart
	if restart.Strategy == nil || len(restart.Strategy.Order) == 0 {
		return nil
	}
	if restart.Strategy.Type == nvidiacomv1beta1.RestartStrategyTypeParallel {
		return errors.New("spec.restart.strategy.order cannot be specified when strategy is parallel")
	}

	var err error
	uniqueOrder := getUnique(restart.Strategy.Order)
	if len(uniqueOrder) != len(restart.Strategy.Order) {
		err = errors.Join(err, fmt.Errorf("spec.restart.strategy.order must be unique"))
	}
	if len(uniqueOrder) != len(components) {
		err = errors.Join(err, fmt.Errorf("spec.restart.strategy.order must have the same number of unique components as the deployment"))
	}
	for _, componentName := range uniqueOrder {
		if _, exists := components[componentName]; !exists {
			err = errors.Join(err, fmt.Errorf("spec.restart.strategy.order contains unknown component: %s", componentName))
		}
	}
	return err
}

func (v *DynamoGraphDeploymentBetaValidator) validatePriorityClassName() error {
	if v.deployment.Spec.PriorityClassName == "" || v.isGrovePathway() {
		return nil
	}
	return v.grovePathwayRequiredError("spec.priorityClassName")
}

func (v *DynamoGraphDeploymentBetaValidator) validateAnnotations() error {
	annotations := v.deployment.GetAnnotations()
	if annotations == nil {
		return nil
	}

	var errs []error
	if value, exists := annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; exists {
		if _, err := semver.NewVersion(value); err != nil {
			errs = append(errs, fmt.Errorf("annotation %s has invalid value %q: must be valid semver",
				consts.KubeAnnotationDynamoOperatorOriginVersion, value))
		}
	}
	if value, exists := annotations[consts.KubeAnnotationVLLMDistributedExecutorBackend]; exists {
		switch strings.ToLower(value) {
		case vllmDistributedExecutorBackendMP, vllmDistributedExecutorBackendRay:
		default:
			errs = append(errs, fmt.Errorf("annotation %s has invalid value %q: must be \"mp\" or \"ray\"",
				consts.KubeAnnotationVLLMDistributedExecutorBackend, value))
		}
	}
	if value, exists := annotations[consts.KubeAnnotationDynamoKubeDiscoveryMode]; exists {
		switch value {
		case "pod", "container":
		default:
			errs = append(errs, fmt.Errorf("annotation %s has invalid value %q: must be \"pod\" or \"container\"",
				consts.KubeAnnotationDynamoKubeDiscoveryMode, value))
		}
	}
	return errors.Join(errs...)
}

func (v *DynamoGraphDeploymentBetaValidator) validateTopologyConstraints(
	ctx context.Context,
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	specConstraint := v.deployment.Spec.TopologyConstraint
	hasAnyConstraint := specConstraint != nil

	var errs []error
	if specConstraint != nil {
		if specConstraint.PackDomain != "" && !isValidBetaTopologyDomainFormat(specConstraint.PackDomain) {
			errs = append(errs, fmt.Errorf("spec.topologyConstraint.packDomain %q is not a valid topology domain; "+
				"must match ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", specConstraint.PackDomain))
		}
	}

	componentNames := sortedBetaComponentNames(components)
	for _, componentName := range componentNames {
		component := components[componentName]
		if component.TopologyConstraint == nil {
			continue
		}
		hasAnyConstraint = true
		fieldPath := fmt.Sprintf("spec.components[%s]", componentName)

		if component.TopologyConstraint.PackDomain == "" {
			errs = append(errs, fmt.Errorf("%s.topologyConstraint.packDomain is required", fieldPath))
			continue
		}
		if !isValidBetaTopologyDomainFormat(component.TopologyConstraint.PackDomain) {
			errs = append(errs, fmt.Errorf("%s.topologyConstraint.packDomain %q is not a valid topology domain; "+
				"must match ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", fieldPath, component.TopologyConstraint.PackDomain))
		}
	}

	if !hasAnyConstraint {
		return nil
	}
	if specConstraint == nil {
		errs = append(errs, fmt.Errorf("spec.topologyConstraint with clusterTopologyName is required "+
			"when any topology constraint is set (at spec or component level)"))
		return errors.Join(errs...)
	}
	if specConstraint.ClusterTopologyName == "" {
		errs = append(errs, fmt.Errorf("spec.topologyConstraint.clusterTopologyName is required "+
			"when any topology constraint is set"))
	}
	if specConstraint.PackDomain == "" {
		for _, componentName := range componentNames {
			component := components[componentName]
			if component.TopologyConstraint == nil {
				errs = append(errs, fmt.Errorf("spec.components[%s].topologyConstraint is required "+
					"because spec.topologyConstraint.packDomain is not set; either set a spec-level "+
					"packDomain or provide a topologyConstraint for every component", componentName))
			}
		}
	}

	if v.shouldValidateGroveClusterTopology(errs, specConstraint.ClusterTopologyName != "") {
		if err := v.validateTopologyDomainsAgainstGroveClusterTopology(ctx, components); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

func (v *DynamoGraphDeploymentBetaValidator) shouldValidateGroveClusterTopology(errs []error, hasClusterTopologyRef bool) bool {
	return len(errs) == 0 &&
		hasClusterTopologyRef &&
		v.mgr != nil &&
		v.deployment.Generation <= 1 &&
		v.isGrovePathway()
}

func (v *DynamoGraphDeploymentBetaValidator) readGroveClusterTopology(ctx context.Context, name string) (*clusterTopologyInfo, error) {
	ct := &grovev1alpha1.ClusterTopology{}
	if err := v.mgr.GetClient().Get(ctx, types.NamespacedName{Name: name}, ct); err != nil {
		return nil, err
	}

	info := &clusterTopologyInfo{
		domainIndex: make(map[string]int, len(ct.Spec.Levels)),
		domains:     make([]string, 0, len(ct.Spec.Levels)),
	}
	for i, level := range ct.Spec.Levels {
		domain := string(level.Domain)
		info.domainIndex[domain] = i
		info.domains = append(info.domains, domain)
	}
	sort.Strings(info.domains)
	return info, nil
}

func (v *DynamoGraphDeploymentBetaValidator) validateTopologyDomainsAgainstGroveClusterTopology(
	ctx context.Context,
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	profileName := v.deployment.Spec.TopologyConstraint.ClusterTopologyName
	if profileName == "" {
		return nil
	}

	info, err := v.readGroveClusterTopology(ctx, profileName)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			return fmt.Errorf("topology-aware scheduling requires a ClusterTopology resource %q but it was not found; "+
				"ensure the cluster topology is configured per the framework documentation", profileName)
		}
		return fmt.Errorf("failed to read ClusterTopology %q for topology validation: %w", profileName, err)
	}
	if info == nil {
		return nil
	}

	type domainCheck struct {
		fieldPath string
		domain    nvidiacomv1beta1.TopologyDomain
	}
	var checks []domainCheck
	if v.deployment.Spec.TopologyConstraint.PackDomain != "" {
		checks = append(checks, domainCheck{
			fieldPath: "spec.topologyConstraint.packDomain",
			domain:    v.deployment.Spec.TopologyConstraint.PackDomain,
		})
	}
	for _, componentName := range sortedBetaComponentNames(components) {
		component := components[componentName]
		if component.TopologyConstraint != nil && component.TopologyConstraint.PackDomain != "" {
			checks = append(checks, domainCheck{
				fieldPath: fmt.Sprintf("spec.components[%s].topologyConstraint.packDomain", componentName),
				domain:    component.TopologyConstraint.PackDomain,
			})
		}
	}

	var errs []error
	for _, c := range checks {
		if _, ok := info.domainIndex[string(c.domain)]; !ok {
			errs = append(errs, fmt.Errorf("%s: domain %q does not exist in ClusterTopology %q; "+
				"available domains: %v", c.fieldPath, c.domain, profileName, info.domains))
		}
	}

	specDomain := v.deployment.Spec.TopologyConstraint.PackDomain
	if specDomain != "" {
		specIdx, specOk := info.domainIndex[string(specDomain)]
		if specOk {
			for _, componentName := range sortedBetaComponentNames(components) {
				component := components[componentName]
				if component.TopologyConstraint == nil || component.TopologyConstraint.PackDomain == "" {
					continue
				}
				componentDomain := component.TopologyConstraint.PackDomain
				componentIdx, componentOk := info.domainIndex[string(componentDomain)]
				if componentOk && componentIdx < specIdx {
					errs = append(errs, fmt.Errorf("spec.components[%s]: topologyConstraint.packDomain %q is broader "+
						"than spec-level %q; component constraints must be equal to or narrower than the "+
						"deployment-level constraint", componentName, componentDomain, specDomain))
				}
			}
		}
	}

	return errors.Join(errs...)
}

func (v *DynamoGraphDeploymentBetaValidator) validateTopologyConstraintImmutability(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
	oldComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	newComponents map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	var errs []error

	if !betaSpecTopologyConstraintsEqual(old.Spec.TopologyConstraint, v.deployment.Spec.TopologyConstraint) {
		errs = append(errs, fmt.Errorf("spec.topologyConstraint is immutable and cannot be added, removed, or changed after creation; "+
			"delete and recreate the DynamoGraphDeployment to change topology constraints"))
	}

	for _, componentName := range sortedBetaComponentNames(newComponents) {
		newComponent := newComponents[componentName]
		oldComponent, exists := oldComponents[componentName]
		if !exists {
			continue
		}
		if !betaTopologyConstraintsEqual(oldComponent.TopologyConstraint, newComponent.TopologyConstraint) {
			errs = append(errs, fmt.Errorf("spec.components[%s].topologyConstraint is immutable and cannot be added, removed, or changed after creation; "+
				"delete and recreate the DynamoGraphDeployment to change topology constraints", componentName))
		}
	}

	return errors.Join(errs...)
}

func (v *DynamoGraphDeploymentBetaValidator) validateKvTransferPolicyImmutability(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if betaKvTransferPoliciesEqual(betaKvTransferPolicyFor(old), betaKvTransferPolicyFor(v.deployment)) {
		return nil
	}
	return fmt.Errorf("spec.experimental.kvTransferPolicy is immutable and cannot be added, removed, or changed after creation; " +
		"delete and recreate the DynamoGraphDeployment to change the KV transfer policy")
}

func (v *DynamoGraphDeploymentBetaValidator) validateNoRestartDuringRollingUpdate(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if old.Status.RollingUpdate == nil {
		return nil
	}
	phase := old.Status.RollingUpdate.Phase
	if phase != nvidiacomv1beta1.RollingUpdatePhasePending && phase != nvidiacomv1beta1.RollingUpdatePhaseInProgress {
		return nil
	}

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

func (v *DynamoGraphDeploymentBetaValidator) validateKvTransferPolicy(ctx context.Context) error {
	if v.deployment.Spec.Experimental == nil {
		return nil
	}
	kvt := v.deployment.Spec.Experimental.KvTransferPolicy
	if kvt == nil {
		return nil
	}

	var errs []error
	const fieldPath = "spec.experimental.kvTransferPolicy"

	hasLabelKey := kvt.LabelKey != ""
	hasClusterTopologyName := kvt.ClusterTopologyName != ""
	if hasLabelKey == hasClusterTopologyName {
		errs = append(errs, fmt.Errorf("%s: exactly one of labelKey or clusterTopologyName is required", fieldPath))
	}
	if hasLabelKey {
		if labelKeyErrs := k8svalidation.IsQualifiedName(kvt.LabelKey); len(labelKeyErrs) > 0 {
			errs = append(errs, fmt.Errorf("%s.labelKey %q is not a valid Kubernetes label key: %s",
				fieldPath, kvt.LabelKey, strings.Join(labelKeyErrs, "; ")))
		}
	}
	if hasClusterTopologyName {
		if nameErrs := k8svalidation.IsDNS1123Subdomain(kvt.ClusterTopologyName); len(nameErrs) > 0 {
			errs = append(errs, fmt.Errorf("%s.clusterTopologyName %q is not a valid Kubernetes resource name: %s",
				fieldPath, kvt.ClusterTopologyName, strings.Join(nameErrs, "; ")))
		}
		if !v.isGrovePathway() {
			errs = append(errs, v.grovePathwayRequiredError("spec.experimental.kvTransferPolicy.clusterTopologyName"))
		}
	}
	if !hasLabelKey && !hasClusterTopologyName {
		return errors.Join(errs...)
	}

	if kvt.Domain == "" {
		errs = append(errs, fmt.Errorf("%s.domain is required", fieldPath))
	} else if !isValidBetaTopologyDomainFormat(kvt.Domain) {
		errs = append(errs, fmt.Errorf("%s.domain %q is not a valid topology domain; "+
			"must match ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", fieldPath, kvt.Domain))
	}

	enforcement := betaEffectiveKvTransferEnforcement(kvt)
	if enforcement != nvidiacomv1beta1.KvTransferEnforcementRequired &&
		enforcement != nvidiacomv1beta1.KvTransferEnforcementPreferred {
		errs = append(errs, fmt.Errorf("%s.enforcement %q is invalid; "+
			"must be \"required\" or \"preferred\"", fieldPath, kvt.Enforcement))
	}
	if enforcement == nvidiacomv1beta1.KvTransferEnforcementPreferred && kvt.PreferredWeight == nil {
		errs = append(errs, fmt.Errorf("%s.preferredWeight is required when enforcement is \"preferred\"", fieldPath))
	}
	if kvt.PreferredWeight != nil {
		if *kvt.PreferredWeight < 0 || *kvt.PreferredWeight > 1 {
			errs = append(errs, fmt.Errorf("%s.preferredWeight %g is invalid; "+
				"must be >= 0 and <= 1", fieldPath, *kvt.PreferredWeight))
		}
		if enforcement == nvidiacomv1beta1.KvTransferEnforcementRequired {
			errs = append(errs, fmt.Errorf("%s.preferredWeight must not be set when enforcement is \"required\"", fieldPath))
		}
	}

	if v.shouldValidateGroveClusterTopology(errs, hasClusterTopologyName) {
		if err := v.validateKvTransferPolicyAgainstGroveClusterTopology(ctx, kvt); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

func (v *DynamoGraphDeploymentBetaValidator) validateKvTransferPolicyAgainstGroveClusterTopology(
	ctx context.Context,
	kvt *nvidiacomv1beta1.KvTransferPolicy,
) error {
	info, err := v.readGroveClusterTopology(ctx, kvt.ClusterTopologyName)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			return fmt.Errorf("spec.experimental.kvTransferPolicy.clusterTopologyName %q references a ClusterTopology resource that was not found",
				kvt.ClusterTopologyName)
		}
		return fmt.Errorf("failed to read ClusterTopology %q for kvTransferPolicy validation: %w", kvt.ClusterTopologyName, err)
	}
	if info == nil {
		return nil
	}
	if _, ok := info.domainIndex[string(kvt.Domain)]; ok {
		return nil
	}
	return fmt.Errorf("spec.experimental.kvTransferPolicy.domain %q does not exist in ClusterTopology %q; available domains: %v",
		kvt.Domain, kvt.ClusterTopologyName, info.domains)
}

func (v *DynamoGraphDeploymentBetaValidator) validateFailoverRequiresDiscoveryMode() error {
	hasIntraPodFailover := false
	for i := range v.deployment.Spec.Components {
		component := &v.deployment.Spec.Components[i]
		failover := betaFailover(component)
		if failover == nil {
			continue
		}
		if betaGMSMode(failover.Mode) == nvidiacomv1beta1.GMSModeIntraPod {
			hasIntraPodFailover = true
			break
		}
	}
	if !hasIntraPodFailover {
		return nil
	}

	annotations := v.deployment.GetAnnotations()
	if annotations == nil || annotations[consts.KubeAnnotationDynamoKubeDiscoveryMode] != "container" {
		return fmt.Errorf(
			"failover requires per-container K8s discovery; set annotation %q to %q on the DynamoGraphDeployment",
			consts.KubeAnnotationDynamoKubeDiscoveryMode, "container")
	}
	return nil
}

// BetaSharedSpecValidator validates v1beta1 component fields embedded in a DGD.
type BetaSharedSpecValidator struct {
	spec         *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
	fieldPath    string
	grovePathway bool
	mgr          ctrl.Manager
}

func NewBetaSharedSpecValidator(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fieldPath string,
	grovePathway bool,
	mgr ctrl.Manager,
) *BetaSharedSpecValidator {
	return &BetaSharedSpecValidator{
		spec:         spec,
		fieldPath:    fieldPath,
		grovePathway: grovePathway,
		mgr:          mgr,
	}
}

func (v *BetaSharedSpecValidator) Validate(ctx context.Context) (admission.Warnings, error) {
	if v.spec.Replicas != nil && *v.spec.Replicas < 0 {
		return nil, fmt.Errorf("%s.replicas must be non-negative", v.fieldPath)
	}
	if err := v.validateMinAvailable(); err != nil {
		return nil, err
	}
	if err := v.validatePodTemplate(); err != nil {
		return nil, err
	}
	if err := v.validateCompilationCache(); err != nil {
		return nil, err
	}
	if err := v.validateSharedMemorySize(); err != nil {
		return nil, err
	}
	if err := v.validateCheckpointConfig(); err != nil {
		return nil, err
	}
	if err := v.validateGMSClientContainerNames(); err != nil {
		return nil, err
	}
	if err := v.validatePodTemplateAnnotations(); err != nil {
		return nil, err
	}
	if err := v.validateEPPConfig(ctx); err != nil {
		return nil, err
	}
	if err := v.validateGPUMemoryService(); err != nil {
		return nil, err
	}
	if err := v.validateFailover(); err != nil {
		return nil, err
	}
	if err := v.validateSnapshotWithGPUMemoryService(); err != nil {
		return nil, err
	}
	return nil, nil
}

func (v *BetaSharedSpecValidator) validateMinAvailable() error {
	if v.spec.MinAvailable == nil {
		return nil
	}
	if !v.grovePathway {
		return fmt.Errorf("%s.minAvailable is currently supported only for Grove-backed DynamoGraphDeployment components", v.fieldPath)
	}
	if *v.spec.MinAvailable <= 0 {
		return fmt.Errorf("%s.minAvailable must be greater than 0", v.fieldPath)
	}
	replicas := int32(1)
	if v.spec.Replicas != nil {
		replicas = *v.spec.Replicas
	}
	if replicas > 0 && replicas < *v.spec.MinAvailable {
		return fmt.Errorf("%s.replicas must be 0 or greater than or equal to minAvailable", v.fieldPath)
	}
	return nil
}

func (v *BetaSharedSpecValidator) validatePodTemplate() error {
	if v.spec.PodTemplate == nil {
		if v.spec.FrontendSidecar != nil {
			return fmt.Errorf("%s.frontendSidecar requires podTemplate.spec.containers", v.fieldPath)
		}
		return nil
	}

	containers := v.spec.PodTemplate.Spec.Containers
	if v.spec.FrontendSidecar != nil {
		target := *v.spec.FrontendSidecar
		if target == "" {
			return fmt.Errorf("%s.frontendSidecar must not be empty", v.fieldPath)
		}
		found := false
		for i := range containers {
			if containers[i].Name == target {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("%s.frontendSidecar %q does not match any podTemplate.spec.containers name",
				v.fieldPath, target)
		}
	}

	for i := range containers {
		container := containers[i]
		if container.Name != consts.MainContainerName && container.Image == "" {
			return fmt.Errorf("%s.podTemplate.spec.containers[%d].image is required for sidecar container %q",
				v.fieldPath, i, container.Name)
		}
	}
	for i := range v.spec.PodTemplate.Spec.InitContainers {
		container := v.spec.PodTemplate.Spec.InitContainers[i]
		if container.Image == "" {
			return fmt.Errorf("%s.podTemplate.spec.initContainers[%d].image is required for init container %q",
				v.fieldPath, i, container.Name)
		}
	}
	return nil
}

func (v *BetaSharedSpecValidator) validateCompilationCache() error {
	if v.spec.CompilationCache == nil {
		return nil
	}
	if v.spec.CompilationCache.PVCName == "" {
		return fmt.Errorf("%s.compilationCache.pvcName is required", v.fieldPath)
	}
	return nil
}

func (v *BetaSharedSpecValidator) validateSharedMemorySize() error {
	if v.spec.SharedMemorySize == nil {
		return nil
	}
	if v.spec.SharedMemorySize.Sign() < 0 {
		return fmt.Errorf("%s.sharedMemorySize must be non-negative", v.fieldPath)
	}
	return nil
}

func (v *BetaSharedSpecValidator) validateCheckpointConfig() error {
	checkpoint := betaCheckpoint(v.spec)
	if checkpoint == nil {
		return nil
	}
	if checkpoint.Job != nil && checkpoint.CheckpointRef != nil && *checkpoint.CheckpointRef != "" {
		return fmt.Errorf("%s.experimental.checkpoint.job cannot be set when checkpointRef is specified", v.fieldPath)
	}
	if checkpoint.TargetContainerName != "" {
		if errs := k8svalidation.IsDNS1123Label(checkpoint.TargetContainerName); len(errs) > 0 {
			return fmt.Errorf(
				"%s.experimental.checkpoint.targetContainerName %q is not a valid Kubernetes container name: %s",
				v.fieldPath,
				checkpoint.TargetContainerName,
				strings.Join(errs, "; "),
			)
		}
	}
	if checkpoint.Job == nil {
		return nil
	}
	if len(checkpoint.Job.GMSClientContainers) > 0 {
		gms := betaGPUMemoryService(v.spec)
		if gms == nil {
			return fmt.Errorf("%s.experimental.checkpoint.job.gmsClientContainers requires gpuMemoryService to be set", v.fieldPath)
		}
		if betaGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod {
			return fmt.Errorf("%s.experimental.checkpoint.job.gmsClientContainers is only supported with gpuMemoryService.mode=IntraPod", v.fieldPath)
		}
	}
	for i, name := range checkpoint.Job.GMSClientContainers {
		if errs := k8svalidation.IsDNS1123Label(name); len(errs) > 0 {
			return fmt.Errorf(
				"%s.experimental.checkpoint.job.gmsClientContainers[%d] %q is not a valid Kubernetes container name: %s",
				v.fieldPath,
				i,
				name,
				strings.Join(errs, "; "),
			)
		}
	}
	return nil
}

func (v *BetaSharedSpecValidator) validateGMSClientContainerNames() error {
	gms := betaGPUMemoryService(v.spec)
	if gms == nil {
		return nil
	}
	for i, name := range gms.ExtraClientContainers {
		if errs := k8svalidation.IsDNS1123Label(name); len(errs) > 0 {
			return fmt.Errorf(
				"%s.experimental.gpuMemoryService.extraClientContainers[%d] %q is not a valid Kubernetes container name: %s",
				v.fieldPath,
				i,
				name,
				strings.Join(errs, "; "),
			)
		}
	}
	if len(gms.ExtraClientContainers) > 0 && betaGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod {
		return fmt.Errorf("%s.experimental.gpuMemoryService.extraClientContainers is only supported with mode=IntraPod", v.fieldPath)
	}
	if len(gms.ExtraClientPods) > 0 {
		return fmt.Errorf("%s.experimental.gpuMemoryService.extraClientPods is reserved for inter-pod GMS and is not implemented yet", v.fieldPath)
	}
	return nil
}

func (v *BetaSharedSpecValidator) validatePodTemplateAnnotations() error {
	annotations := dynamo.GetPodTemplateAnnotations(v.spec)
	if annotations == nil {
		return nil
	}
	if value, exists := annotations[consts.KubeAnnotationVLLMDistributedExecutorBackend]; exists {
		switch strings.ToLower(value) {
		case vllmDistributedExecutorBackendMP, vllmDistributedExecutorBackendRay:
		default:
			return fmt.Errorf("%s.podTemplate.metadata.annotations[%s] has invalid value %q: must be \"mp\" or \"ray\"",
				v.fieldPath, consts.KubeAnnotationVLLMDistributedExecutorBackend, value)
		}
	}
	return nil
}

func (v *BetaSharedSpecValidator) validateEPPConfig(ctx context.Context) error {
	if v.spec.ComponentType != nvidiacomv1beta1.ComponentTypeEPP {
		return nil
	}
	if v.mgr != nil {
		if err := v.checkInferencePoolAPIAvailability(ctx); err != nil {
			return fmt.Errorf("%s: cannot deploy EPP component: %w", v.fieldPath, err)
		}
	}
	if v.spec.IsMultinode() {
		return fmt.Errorf("%s: EPP component cannot be multinode (multinode field must be nil or nodeCount must be 1)", v.fieldPath)
	}
	if v.spec.Replicas != nil && *v.spec.Replicas != 1 {
		return fmt.Errorf("%s: EPP component must have exactly 1 replica (found %d replicas)", v.fieldPath, *v.spec.Replicas)
	}
	if v.spec.EPPConfig == nil {
		return fmt.Errorf("%s.eppConfig is required for EPP components", v.fieldPath)
	}
	if v.spec.EPPConfig.ConfigMapRef == nil && v.spec.EPPConfig.Config == nil {
		return fmt.Errorf("%s.eppConfig: either configMapRef or config must be specified (no default configuration provided)", v.fieldPath)
	}
	if v.spec.EPPConfig.ConfigMapRef != nil && v.spec.EPPConfig.Config != nil {
		return fmt.Errorf("%s.eppConfig: configMapRef and config are mutually exclusive, only one can be specified", v.fieldPath)
	}
	if v.spec.EPPConfig.ConfigMapRef != nil && v.spec.EPPConfig.ConfigMapRef.Name == "" {
		return fmt.Errorf("%s.eppConfig.configMapRef.name is required", v.fieldPath)
	}
	return nil
}

func (v *BetaSharedSpecValidator) checkInferencePoolAPIAvailability(ctx context.Context) error {
	if v.mgr == nil {
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

func (v *BetaSharedSpecValidator) validateGPUMemoryService() error {
	gms := betaGPUMemoryService(v.spec)
	if gms == nil {
		return nil
	}

	isWorker := v.spec.ComponentType == nvidiacomv1beta1.ComponentTypeWorker ||
		v.spec.ComponentType == nvidiacomv1beta1.ComponentTypePrefill ||
		v.spec.ComponentType == nvidiacomv1beta1.ComponentTypeDecode
	if !isWorker {
		return fmt.Errorf(
			"%s.experimental.gpuMemoryService: GPU memory service is only supported for worker components (type must be worker, prefill, or decode)",
			v.fieldPath)
	}

	gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(dynamo.GetMainContainerResources(v.spec))
	if err != nil || gpuCount < 1 {
		return fmt.Errorf(
			"%s.experimental.gpuMemoryService: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
			v.fieldPath)
	}

	return nil
}

func (v *BetaSharedSpecValidator) validateFailover() error {
	failover := betaFailover(v.spec)
	if failover == nil {
		return nil
	}

	var errs []error
	gms := betaGPUMemoryService(v.spec)
	failoverMode := betaGMSMode(failover.Mode)

	switch failoverMode {
	case nvidiacomv1beta1.GMSModeIntraPod:
		if gms == nil {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: intraPod failover requires gpuMemoryService to be set",
				v.fieldPath))
		} else if betaGMSMode(gms.Mode) != nvidiacomv1beta1.GMSModeIntraPod {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: failover.mode %q must match gpuMemoryService.mode %q",
				v.fieldPath, failoverMode, gms.Mode))
		}
		if failover.NumShadows != 0 && failover.NumShadows != 1 {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover.numShadows=%d is invalid for mode=%q: intraPod uses a fixed 1 primary + 1 shadow sidecar; "+
					"use failover.mode=%q to configure numShadows",
				v.fieldPath, failover.NumShadows, nvidiacomv1beta1.GMSModeIntraPod, nvidiacomv1beta1.GMSModeInterPod))
		}

	case nvidiacomv1beta1.GMSModeInterPod:
		if gms == nil {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: interPod failover requires gpuMemoryService.mode=%q",
				v.fieldPath, nvidiacomv1beta1.GMSModeInterPod))
		} else if betaGMSMode(gms.Mode) != nvidiacomv1beta1.GMSModeInterPod {
			detected := string(gms.Mode)
			if detected == "" {
				detected = unsetValue
			}
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: interPod failover requires gpuMemoryService.mode=%q (got %q)",
				v.fieldPath, nvidiacomv1beta1.GMSModeInterPod, detected))
		}
		if failover.NumShadows < 1 {
			errs = append(errs, fmt.Errorf("%s.experimental.failover.numShadows must be >= 1", v.fieldPath))
		}

		gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(dynamo.GetMainContainerResources(v.spec))
		if err != nil {
			errs = append(errs, fmt.Errorf("%s.podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu: %w", v.fieldPath, err))
		} else if gpuCount < 1 {
			errs = append(errs, fmt.Errorf("%s: GMS failover requires at least 1 GPU in podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu", v.fieldPath))
		}

		switch v.spec.ComponentType {
		case nvidiacomv1beta1.ComponentTypeEPP, nvidiacomv1beta1.ComponentTypeFrontend, nvidiacomv1beta1.ComponentTypePlanner:
			errs = append(errs, fmt.Errorf("%s: GMS failover is not supported for type %q", v.fieldPath, v.spec.ComponentType))
		}

	default:
		errs = append(errs, fmt.Errorf("%s.experimental.failover.mode %q is invalid", v.fieldPath, failover.Mode))
	}

	return errors.Join(errs...)
}

func (v *BetaSharedSpecValidator) validateSnapshotWithGPUMemoryService() error {
	checkpoint := betaCheckpoint(v.spec)
	gms := betaGPUMemoryService(v.spec)
	if checkpoint == nil || !checkpoint.Enabled || gms == nil {
		return nil
	}
	if os.Getenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar) == "1" {
		return nil
	}
	return fmt.Errorf(
		"%s.experimental.checkpoint: GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate",
		v.fieldPath)
}

func betaComponentsByName(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, error) {
	if dgd == nil {
		return nil, fmt.Errorf("DynamoGraphDeployment is nil")
	}
	components := make(map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, len(dgd.Spec.Components))
	lowerNames := make(map[string]string, len(dgd.Spec.Components))
	var errs []error
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		name := component.ComponentName
		if name == "" {
			errs = append(errs, fmt.Errorf("spec.components[%d].name is required", i))
			continue
		}
		lowerName := strings.ToLower(name)
		if existingName, exists := lowerNames[lowerName]; exists {
			errs = append(errs, fmt.Errorf("spec.components[%d].name %q duplicates component %q case-insensitively", i, name, existingName))
			continue
		}
		lowerNames[lowerName] = name
		components[name] = component
	}
	return components, errors.Join(errs...)
}

func sortedBetaComponentNames(components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) []string {
	names := make([]string, 0, len(components))
	for name := range components {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func betaComponentNameSet(components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) map[string]struct{} {
	names := make(map[string]struct{}, len(components))
	for name := range components {
		names[name] = struct{}{}
	}
	return names
}

func betaSpecTopologyConstraintsEqual(a, b *nvidiacomv1beta1.SpecTopologyConstraint) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.ClusterTopologyName == b.ClusterTopologyName && a.PackDomain == b.PackDomain
}

func betaTopologyConstraintsEqual(a, b *nvidiacomv1beta1.TopologyConstraint) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.PackDomain == b.PackDomain
}

func betaKvTransferPolicyFor(dgd *nvidiacomv1beta1.DynamoGraphDeployment) *nvidiacomv1beta1.KvTransferPolicy {
	if dgd == nil || dgd.Spec.Experimental == nil {
		return nil
	}
	return dgd.Spec.Experimental.KvTransferPolicy
}

func betaKvTransferPoliciesEqual(a, b *nvidiacomv1beta1.KvTransferPolicy) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.ClusterTopologyName == b.ClusterTopologyName &&
		a.LabelKey == b.LabelKey &&
		a.Domain == b.Domain &&
		betaEffectiveKvTransferEnforcement(a) == betaEffectiveKvTransferEnforcement(b) &&
		betaKvTransferPreferredWeightsEqual(a.PreferredWeight, b.PreferredWeight)
}

func betaEffectiveKvTransferEnforcement(kvt *nvidiacomv1beta1.KvTransferPolicy) nvidiacomv1beta1.KvTransferEnforcement {
	if kvt == nil || kvt.Enforcement == "" {
		return nvidiacomv1beta1.KvTransferEnforcementRequired
	}
	return kvt.Enforcement
}

func betaKvTransferPreferredWeightsEqual(a, b *float32) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

func betaGPUMemoryService(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) *nvidiacomv1beta1.GPUMemoryServiceSpec {
	if component == nil || component.Experimental == nil {
		return nil
	}
	return component.Experimental.GPUMemoryService
}

func betaFailover(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) *nvidiacomv1beta1.FailoverSpec {
	if component == nil || component.Experimental == nil {
		return nil
	}
	return component.Experimental.Failover
}

func betaCheckpoint(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) *nvidiacomv1beta1.ComponentCheckpointConfig {
	if component == nil || component.Experimental == nil {
		return nil
	}
	return component.Experimental.Checkpoint
}

func betaGMSMode(mode nvidiacomv1beta1.GPUMemoryServiceMode) nvidiacomv1beta1.GPUMemoryServiceMode {
	if mode == "" {
		return nvidiacomv1beta1.GMSModeIntraPod
	}
	return mode
}

func isValidBetaTopologyDomainFormat(d nvidiacomv1beta1.TopologyDomain) bool {
	return betaTopologyDomainRegex.MatchString(string(d))
}
