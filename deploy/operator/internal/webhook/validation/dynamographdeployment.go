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
	"regexp"
	"sort"
	"strings"

	semver "github.com/Masterminds/semver/v3"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
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

// DynamoGraphDeploymentValidator validates v1beta1 DynamoGraphDeployment resources.
type DynamoGraphDeploymentValidator struct {
	mgr          ctrl.Manager
	groveEnabled bool
}

// NewDynamoGraphDeploymentValidator creates a validator for v1beta1 DynamoGraphDeployment.
func NewDynamoGraphDeploymentValidator(
	mgr ctrl.Manager,
	groveEnabled bool,
) *DynamoGraphDeploymentValidator {
	return &DynamoGraphDeploymentValidator{
		mgr:          mgr,
		groveEnabled: groveEnabled,
	}
}

type dynamoGraphDeploymentValidation struct {
	deployment   *nvidiacomv1beta1.DynamoGraphDeployment
	mgr          ctrl.Manager
	groveEnabled bool
}

// Validate performs stateless validation on the v1beta1 DynamoGraphDeployment.
func (v *DynamoGraphDeploymentValidator) Validate(
	ctx context.Context,
	deployment *nvidiacomv1beta1.DynamoGraphDeployment,
) (admission.Warnings, error) {
	return (&dynamoGraphDeploymentValidation{
		deployment:   deployment,
		mgr:          v.mgr,
		groveEnabled: v.groveEnabled,
	}).validate(ctx)
}

// ValidateUpdate performs stateful validation comparing old and new v1beta1 DGD objects.
// userInfo is used for identity-based validation (replica protection).
// If userInfo is nil, replica changes for DGDSA-enabled components are rejected (fail closed).
// operatorPrincipal is the full Kubernetes SA username of the operator for authorization.
func (v *DynamoGraphDeploymentValidator) ValidateUpdate(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
	new *nvidiacomv1beta1.DynamoGraphDeployment,
	userInfo *authenticationv1.UserInfo,
	operatorPrincipal string,
) (admission.Warnings, error) {
	return (&dynamoGraphDeploymentValidation{
		deployment:   new,
		mgr:          v.mgr,
		groveEnabled: v.groveEnabled,
	}).validateUpdate(old, userInfo, operatorPrincipal)
}

func (v *dynamoGraphDeploymentValidation) validate(ctx context.Context) (admission.Warnings, error) {
	var errs []error
	components, err := betaComponentsByName(v.deployment)
	if err != nil {
		errs = append(errs, err)
	}
	if len(v.deployment.Spec.Components) == 0 {
		errs = append(errs, fmt.Errorf("spec.components must have at least one component"))
	}

	if err := v.validateAnnotations(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateRestart(components); err != nil {
		errs = append(errs, err)
	}
	if err := v.validatePriorityClassName(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateTopologyConstraints(ctx, components); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateKvTransferPolicy(ctx); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateFailoverRequiresDiscoveryMode(); err != nil {
		errs = append(errs, err)
	}
	var allWarnings admission.Warnings
	alphaWarnings, err := v.validateAlphaCompatibility()
	if err != nil {
		errs = append(errs, err)
	}
	allWarnings = append(allWarnings, alphaWarnings...)

	for i := range v.deployment.Spec.Components {
		component := &v.deployment.Spec.Components[i]
		warnings, err := v.validateComponent(ctx, component)
		if err != nil {
			errs = append(errs, err)
		}
		allWarnings = append(allWarnings, warnings...)
	}

	return allWarnings, errors.Join(errs...)
}

// validateUpdate performs stateful validation comparing old and new v1beta1 DGD objects.
func (v *dynamoGraphDeploymentValidation) validateUpdate(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
	userInfo *authenticationv1.UserInfo,
	operatorPrincipal string,
) (admission.Warnings, error) {
	var warnings admission.Warnings
	var errs []error

	if old == nil {
		errs = append(errs, fmt.Errorf("old DynamoGraphDeployment is nil"))
	}
	if len(errs) > 0 {
		return warnings, errors.Join(errs...)
	}

	oldComponents, err := betaComponentsByName(old)
	if err != nil {
		errs = append(errs, err)
	}
	newComponents, err := betaComponentsByName(v.deployment)
	if err != nil {
		errs = append(errs, err)
	}

	if err := v.validateImmutableFields(old, oldComponents, newComponents, &warnings); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateComponentTopology(oldComponents, newComponents); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateReplicasChanges(oldComponents, newComponents, userInfo, operatorPrincipal); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateNoRestartDuringRollingUpdate(old); err != nil {
		errs = append(errs, err)
	}

	return warnings, errors.Join(errs...)
}

func (v *dynamoGraphDeploymentValidation) validateImmutableFields(
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

		// Validate inter-pod GMS layout and failover immutability.
		//
		// Flipping the inter-pod GMS layout or toggling failover within an
		// inter-pod layout both change the PodClique topology (weight-server PCLQ,
		// per-rank engine PCLQs, shadow PCLQs, DRA ResourceClaimTemplates), which
		// Grove cannot transform in place. Force the user to delete and recreate.
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

func (v *dynamoGraphDeploymentValidation) validateComponentTopology(
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

func (v *dynamoGraphDeploymentValidation) validateReplicasChanges(
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

func (v *dynamoGraphDeploymentValidation) validateComponent(
	ctx context.Context,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (admission.Warnings, error) {
	componentName := component.ComponentName
	fieldPath := fmt.Sprintf("spec.components[%s]", componentName)

	// The inter-pod GMS layout (with or without failover) requires the Grove
	// pathway: the weight-server pod, per-rank PCLQs, and DRA ResourceClaim
	// templates are all wired at the PodCliqueScalingGroup level, which only
	// the Grove renderer produces.
	if component.IsInterPodGMSEnabled() && !v.isGrovePathway() {
		return nil, v.grovePathwayRequiredError(fmt.Sprintf(
			"%s: experimental.gpuMemoryService.mode=%q",
			fieldPath, nvidiacomv1beta1.GMSModeInterPod))
	}

	// The inter-pod GMS layout is currently implemented only for vLLM (the
	// engine relies on vLLM-specific runtime hooks like --load-format gms; the
	// failover variant additionally enables vLLM shadow mode). Fail fast at
	// admission rather than producing a broken deployment when another or no
	// backend is configured; an empty BackendFramework means the operator cannot
	// confirm the engine speaks vLLM, which is a hard prerequisite for inter-pod
	// GMS (both standalone and with failover).
	if component.IsInterPodGMSEnabled() &&
		v.deployment.Spec.BackendFramework != string(dynamo.BackendFrameworkVLLM) {
		detected := v.deployment.Spec.BackendFramework
		if detected == "" {
			detected = unsetValue
		}
		return nil, fmt.Errorf(
			"%s: the inter-pod GMS layout (experimental.gpuMemoryService.mode=%q) is currently supported only for vLLM (detected: %s); "+
				"set spec.backendFramework=%q",
			fieldPath, nvidiacomv1beta1.GMSModeInterPod, detected, dynamo.BackendFrameworkVLLM)
	}

	if v.isGrovePathway() {
		if err := v.validateComponentNameLength(componentName, component); err != nil {
			return nil, err
		}
	}

	sharedValidator := NewSharedSpecValidatorV1Beta1(component, fieldPath, v.isGrovePathway(), v.mgr)
	return sharedValidator.Validate(ctx)
}

// validateComponentNameLength ensures Grove-rendered resource names stay within
// the pod-name budget after the PCS, PCSG, and PodClique names are combined.
//
// Grove builds pod names from these generated pieces:
//   - PCS name: derived from the DynamoGraphDeployment name
//
// For multi-node and inter-pod GMS components:
//   - PCSG name: lowercase(componentName)
//   - PodClique names: rendered role names for the component
//
// For single-node components:
//   - PodClique name: lowercase(componentName)
//
// The combined length of these names must not exceed maxCombinedResourceNameLength.
func (v *dynamoGraphDeploymentValidation) validateComponentNameLength(
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

func (v *dynamoGraphDeploymentValidation) isGrovePathway() bool {
	if !v.groveEnabled {
		return false
	}
	return v.deployment.Annotations == nil ||
		strings.ToLower(v.deployment.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse
}

func (v *dynamoGraphDeploymentValidation) grovePathwayRequiredError(subject string) error {
	if !v.groveEnabled {
		return fmt.Errorf("%s requires the Grove pathway, but Grove is disabled in the operator configuration", subject)
	}
	return fmt.Errorf("%s requires the Grove pathway; remove or unset the %q annotation (currently %q)",
		subject, consts.KubeAnnotationEnableGrove, v.deployment.Annotations[consts.KubeAnnotationEnableGrove])
}

func (v *dynamoGraphDeploymentValidation) validateRestart(
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

func (v *dynamoGraphDeploymentValidation) validateRestartStrategyOrder(
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

func (v *dynamoGraphDeploymentValidation) validatePriorityClassName() error {
	if v.deployment.Spec.PriorityClassName == "" || v.isGrovePathway() {
		return nil
	}
	return v.grovePathwayRequiredError("spec.priorityClassName")
}

func (v *dynamoGraphDeploymentValidation) validateAnnotations() error {
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
	if err := validateVLLMDistributedExecutorBackendAnnotation("", annotations); err != nil {
		errs = append(errs, err)
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

func (v *dynamoGraphDeploymentValidation) validateTopologyConstraints(
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

func (v *dynamoGraphDeploymentValidation) shouldValidateGroveClusterTopology(errs []error, hasClusterTopologyRef bool) bool {
	return len(errs) == 0 &&
		hasClusterTopologyRef &&
		v.mgr != nil &&
		v.deployment.Generation <= 1 &&
		v.isGrovePathway()
}

func (v *dynamoGraphDeploymentValidation) readGroveClusterTopology(ctx context.Context, name string) (*clusterTopologyInfo, error) {
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

func (v *dynamoGraphDeploymentValidation) validateTopologyDomainsAgainstGroveClusterTopology(
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

func (v *dynamoGraphDeploymentValidation) validateTopologyConstraintImmutability(
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

func (v *dynamoGraphDeploymentValidation) validateKvTransferPolicyImmutability(
	old *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if betaKvTransferPoliciesEqual(betaKvTransferPolicyFor(old), betaKvTransferPolicyFor(v.deployment)) {
		return nil
	}
	return fmt.Errorf("spec.experimental.kvTransferPolicy is immutable and cannot be added, removed, or changed after creation; " +
		"delete and recreate the DynamoGraphDeployment to change the KV transfer policy")
}

func (v *dynamoGraphDeploymentValidation) validateNoRestartDuringRollingUpdate(
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

func (v *dynamoGraphDeploymentValidation) validateKvTransferPolicy(ctx context.Context) error {
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

func (v *dynamoGraphDeploymentValidation) validateKvTransferPolicyAgainstGroveClusterTopology(
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

func (v *dynamoGraphDeploymentValidation) validateFailoverRequiresDiscoveryMode() error {
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

func (v *dynamoGraphDeploymentValidation) validateAlphaCompatibility() (admission.Warnings, error) {
	// Reconstruct the v1alpha1 view so alpha-only fields preserved by
	// conversion still get the webhook validation they had before the DGD
	// webhook moved to v1beta1.
	alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(v.deployment); err != nil {
		return nil, fmt.Errorf("cannot validate preserved v1alpha1 DynamoGraphDeployment fields: failed to reconstruct compatibility view: %w", err)
	}
	if !hasAlphaCompatibilityFields(alpha) {
		return nil, nil
	}

	return validateAlphaCompatibility(alpha)
}

func hasAlphaCompatibilityFields(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) bool {
	if len(dgd.Spec.PVCs) > 0 {
		return true
	}
	for _, service := range dgd.Spec.Services {
		if service == nil {
			return true
		}
		hasDeprecatedAutoscaling := false
		//nolint:staticcheck // SA1019: Intentionally checking deprecated field to preserve v1alpha1 warnings.
		if service.Autoscaling != nil {
			hasDeprecatedAutoscaling = true
		}
		if service.Ingress != nil ||
			len(service.Annotations) > 0 ||
			service.DynamoNamespace != nil ||
			hasDeprecatedAutoscaling ||
			len(service.VolumeMounts) > 0 ||
			service.SharedMemory != nil ||
			service.FrontendSidecar != nil ||
			(service.GPUMemoryService != nil && !service.GPUMemoryService.Enabled) {
			return true
		}
	}
	return false
}

func validateAlphaCompatibility(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) (admission.Warnings, error) {
	var warnings admission.Warnings
	var errs []error

	if err := validateAlphaCompatibilityPVCs(dgd.Spec.PVCs); err != nil {
		errs = append(errs, err)
	}
	for serviceName, service := range dgd.Spec.Services {
		fieldPath := fmt.Sprintf("spec.services[%s]", serviceName)
		if service == nil {
			errs = append(errs, fmt.Errorf("%s must not be null", fieldPath))
			continue
		}
		warnings = append(warnings, alphaCompatibilityWarningsForService(dgd, fieldPath, service)...)
		if err := validateAlphaCompatibilityService(fieldPath, service); err != nil {
			errs = append(errs, err)
		}
	}

	return warnings, errors.Join(errs...)
}

func alphaCompatibilityWarningsForService(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	fieldPath string,
	service *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
) admission.Warnings {
	var warnings admission.Warnings
	if service.DynamoNamespace != nil && *service.DynamoNamespace != "" {
		warnings = append(warnings, fmt.Sprintf(
			"%s.dynamoNamespace is deprecated and ignored. Value '%s' will be replaced with '%s'. "+
				"Remove this field from your configuration",
			fieldPath, *service.DynamoNamespace, dgd.GetDynamoNamespaceForService(service)))
	}

	//nolint:staticcheck // SA1019: Intentionally checking deprecated field to warn users.
	if service.Autoscaling != nil {
		warnings = append(warnings, fmt.Sprintf(
			"%s.autoscaling is deprecated and ignored. Use DynamoGraphDeploymentScalingAdapter "+
				"with HPA, KEDA, or Planner for autoscaling instead. See docs/kubernetes/autoscaling.md",
			fieldPath))
	}
	return warnings
}

func validateAlphaCompatibilityPVCs(pvcs []nvidiacomv1alpha1.PVC) error {
	var errs []error
	for i := range pvcs {
		if err := validateAlphaCompatibilityPVC(i, &pvcs[i]); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func validateAlphaCompatibilityPVC(index int, pvc *nvidiacomv1alpha1.PVC) error {
	var err error
	if pvc.Name == nil || *pvc.Name == "" {
		err = errors.Join(err, fmt.Errorf("spec.pvcs[%d].name is required", index))
	}
	if pvc.Create != nil && *pvc.Create {
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

func validateAlphaCompatibilityService(
	fieldPath string,
	service *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
) error {
	var errs []error
	if service.Ingress != nil && service.Ingress.Enabled && service.Ingress.Host == "" {
		errs = append(errs, fmt.Errorf("%s.ingress.host is required when ingress is enabled", fieldPath))
	}
	if err := validateAlphaCompatibilityServiceAnnotations(fieldPath, service.Annotations); err != nil {
		errs = append(errs, err)
	}
	if err := validateAlphaCompatibilityVolumeMounts(fieldPath, service.VolumeMounts); err != nil {
		errs = append(errs, err)
	}
	if service.SharedMemory != nil && !service.SharedMemory.Disabled && service.SharedMemory.Size.IsZero() {
		errs = append(errs, fmt.Errorf("%s.sharedMemory.size is required when disabled is false", fieldPath))
	}
	if err := validateAlphaCompatibilityFrontendSidecar(fieldPath, service); err != nil {
		errs = append(errs, err)
	}
	if err := validateAlphaCompatibilityGMSClientContainerNames(fieldPath, service.GPUMemoryService); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateAlphaCompatibilityServiceAnnotations(fieldPath string, annotations map[string]string) error {
	return validateVLLMDistributedExecutorBackendAnnotation(fieldPath+".annotations", annotations)
}

func validateAlphaCompatibilityFrontendSidecar(
	fieldPath string,
	service *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
) error {
	if service.FrontendSidecar == nil ||
		service.ExtraPodSpec == nil ||
		service.ExtraPodSpec.PodSpec == nil {
		return nil
	}
	for _, container := range service.ExtraPodSpec.PodSpec.Containers {
		if container.Name == consts.FrontendSidecarContainerName {
			return fmt.Errorf(
				"%s: cannot inject frontend sidecar: a container named %q already exists in extraPodSpec.containers",
				fieldPath, consts.FrontendSidecarContainerName)
		}
	}
	return nil
}

func validateAlphaCompatibilityVolumeMounts(
	fieldPath string,
	volumeMounts []nvidiacomv1alpha1.VolumeMount,
) error {
	var errs []error
	for i := range volumeMounts {
		volumeMount := &volumeMounts[i]
		if !volumeMount.UseAsCompilationCache && volumeMount.MountPoint == "" {
			errs = append(errs, fmt.Errorf("%s.volumeMounts[%d].mountPoint is required when useAsCompilationCache is false",
				fieldPath, i))
		}
	}
	return errors.Join(errs...)
}

func validateAlphaCompatibilityGMSClientContainerNames(
	fieldPath string,
	gms *nvidiacomv1alpha1.GPUMemoryServiceSpec,
) error {
	if gms == nil {
		return nil
	}
	var errs []error
	for i, name := range gms.ExtraClientContainers {
		if validationErrs := k8svalidation.IsDNS1123Label(name); len(validationErrs) > 0 {
			errs = append(errs, fmt.Errorf(
				"%s.gpuMemoryService.extraClientContainers[%d] %q is not a valid Kubernetes container name: %s",
				fieldPath,
				i,
				name,
				strings.Join(validationErrs, "; "),
			))
		}
	}
	return errors.Join(errs...)
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
	if a == nil || b == nil {
		return a == b
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
