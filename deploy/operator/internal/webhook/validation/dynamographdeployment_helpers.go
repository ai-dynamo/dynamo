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
	"maps"
	"slices"
	"sort"
	"strconv"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
)

type clusterTopologyInfo struct {
	name        string
	domainIndex map[string]int
	domains     []string
}

// lpxComponentGateErrors freezes existing LPX graph components when the integration is disabled.
// ctx, newSpec, and fldPath are non-nil; oldSpec is nil for a new component.
func lpxComponentGateErrors(
	ctx context.Context,
	newSpec, oldSpec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
	source runtimeVersionValidationSource,
) field.ErrorList {
	if !newSpec.IsLPX() || features.MustGateFrom(ctx).Enabled(features.LPX) ||
		(oldSpec != nil && apiequality.Semantic.DeepEqual(newSpec, oldSpec)) {
		return nil
	}

	// Conversion preserves the source service name, so gate diagnostics can target the submitted API.
	typePath := fldPath.Child("type")
	if source == runtimeVersionSourceV1Alpha1 {
		typePath = field.NewPath("spec", "services").Key(newSpec.ComponentName).Child("componentType")
	}
	return field.ErrorList{field.Forbidden(typePath, "LPX components require lpx.enabled=true")}
}

// invalidDynamoGraphDeploymentError converts allErrs for dgd into an API error.
// dgd must not be nil.
func invalidDynamoGraphDeploymentError(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	allErrs field.ErrorList,
) error {
	if len(allErrs) == 0 {
		return nil
	}
	return k8serrors.NewInvalid(nvidiacomv1beta1.DynamoGraphDeploymentGVK.GroupKind(), dgd.Name, allErrs)
}

// alphaDynamoGraphDeploymentForValidation reconstructs the compatibility view.
// dgd must not be nil.
func alphaDynamoGraphDeploymentForValidation(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (*nvidiacomv1alpha1.DynamoGraphDeployment, error) {
	alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(dgd); err != nil {
		return nil, fmt.Errorf("failed to reconstruct compatibility view: %w", err)
	}
	return alpha, nil
}

func hasV1Alpha1CompatibilityFields(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) bool {
	if len(dgd.Spec.PVCs) > 0 {
		return true
	}
	for _, service := range dgd.Spec.Services {
		if service == nil {
			return true
		}
		hasDeprecatedAutoscaling := false
		//nolint:staticcheck // SA1019: Intentionally checking deprecated fields preserved by conversion.
		if service.Autoscaling != nil {
			hasDeprecatedAutoscaling = true
		}
		if service.Ingress != nil ||
			len(service.Annotations) > 0 ||
			service.DynamoNamespace != nil ||
			hasDeprecatedAutoscaling ||
			len(service.VolumeMounts) > 0 ||
			service.SharedMemory != nil ||
			service.EPPConfig != nil ||
			service.FrontendSidecar != nil ||
			service.Failover != nil ||
			(service.GPUMemoryService != nil && !service.GPUMemoryService.Enabled) {
			return true
		}
	}
	return false
}

func sortedV1Alpha1ServiceNames(
	services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
) []string {
	names := make([]string, 0, len(services))
	for name := range services {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// readGroveClusterTopology reads a topology by name. ctx and mgr must not be nil.
func readGroveClusterTopology(ctx context.Context, mgr ctrl.Manager, name string) (*clusterTopologyInfo, error) {
	clusterTopology := &grovev1alpha1.ClusterTopologyBinding{}
	if err := mgr.GetClient().Get(ctx, types.NamespacedName{Name: name}, clusterTopology); err != nil {
		return nil, err
	}

	info := &clusterTopologyInfo{
		name:        name,
		domainIndex: make(map[string]int, len(clusterTopology.Spec.Levels)),
		domains:     make([]string, 0, len(clusterTopology.Spec.Levels)),
	}
	for i, level := range clusterTopology.Spec.Levels {
		domain := string(level.Domain)
		info.domainIndex[domain] = i
		info.domains = append(info.domains, domain)
	}
	sort.Strings(info.domains)
	return info, nil
}

// supportedWorkloadProviders lists the workload programs the controller
// implements. Both the create-side and update-side metadata rules read this so
// the set is stated once.
func supportedWorkloadProviders() []string {
	return []string{
		consts.WorkloadProviderComponent,
		consts.WorkloadProviderGrove,
		consts.WorkloadProviderDisaggregatedSet,
	}
}

// isSupportedWorkloadProvider reports whether value names a workload program
// the controller implements.
func isSupportedWorkloadProvider(value string) bool {
	return slices.Contains(supportedWorkloadProviders(), value)
}

func grovePathwayForDynamoGraphDeployment(
	groveEnabled bool,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, string) {
	// A durable selection takes precedence over current capabilities and the
	// original routing annotation. Availability is reported by the selected
	// workload program, while admission retains that program's API semantics.
	if provider, exists := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; exists {
		switch provider {
		case consts.WorkloadProviderGrove:
			return true, ""
		case consts.WorkloadProviderComponent:
			return false, fmt.Sprintf(
				"requires the Grove pathway, but workload provider %q is selected",
				provider,
			)
		case consts.WorkloadProviderDisaggregatedSet:
			return false, fmt.Sprintf(
				"requires the Grove pathway, but workload provider %q is selected",
				provider,
			)
		default:
			return false, fmt.Sprintf(
				"requires the Grove pathway, but annotation %q has unsupported value %q",
				consts.KubeAnnotationWorkloadProvider,
				provider,
			)
		}
	}

	if !groveEnabled {
		return false, "requires the Grove pathway, but Grove is disabled in the operator configuration"
	}
	annotationValue := strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableGrove])
	if annotationValue == consts.KubeLabelValueFalse {
		return false, fmt.Sprintf(
			"requires the Grove pathway; remove or unset annotation %q (currently %q)",
			consts.KubeAnnotationEnableGrove,
			dgd.Annotations[consts.KubeAnnotationEnableGrove],
		)
	}
	return true, ""
}

func hasIntraPodFailover(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) bool {
	for i := range spec.Components {
		failover := failoverFor(&spec.Components[i])
		if failover != nil && effectiveGMSMode(failover.Mode) == nvidiacomv1beta1.GMSModeIntraPod {
			return true
		}
	}
	return false
}

func componentsByName(
	components []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	byName := make(map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, len(components))
	for i := range components {
		byName[components[i].ComponentName] = &components[i]
	}
	return byName
}

func dgdPowerLimit(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (string, bool) {
	if component.PodTemplate == nil {
		return "", false
	}
	value, exists := component.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]
	return value, exists
}

// validateDGDPowerLimitValue checks that the power-limit annotation value is a
// positive decimal integer. The annotation is made immutable on creation, so an
// invalid value that slips through requires a DGD delete-and-recreate to correct.
func validateDGDPowerLimitValue(value string, fldPath *field.Path) *field.Error {
	watts, err := strconv.Atoi(value)
	if err != nil {
		return field.Invalid(fldPath, value, "must be a decimal integer")
	}
	if watts <= 0 {
		return field.Invalid(fldPath, value, "must be greater than zero")
	}
	return nil
}

func dgdDRAPath(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) *field.Path {
	if gpuMemoryServiceFor(component) != nil {
		return fldPath.Child("experimental", "gpuMemoryService")
	}
	if component.PodTemplate == nil {
		return nil
	}

	// The claim's DeviceClass is external to the DGD, so fail closed for every consumed claim.
	containersPath := fldPath.Child("podTemplate", "spec", "containers")
	for i := range component.PodTemplate.Spec.Containers {
		if len(component.PodTemplate.Spec.Containers[i].Resources.Claims) != 0 {
			return containersPath.Index(i).Child("resources", "claims")
		}
	}
	initContainersPath := fldPath.Child("podTemplate", "spec", "initContainers")
	for i := range component.PodTemplate.Spec.InitContainers {
		if len(component.PodTemplate.Spec.InitContainers[i].Resources.Claims) != 0 {
			return initContainersPath.Index(i).Child("resources", "claims")
		}
	}
	return nil
}

// effectiveNumberOfGPUs captures the effective scalar GPU count and its source field.
type effectiveNumberOfGPUs struct {
	value   string
	present bool
	path    *field.Path
}

func (input effectiveNumberOfGPUs) equal(other effectiveNumberOfGPUs) bool {
	return input.present == other.present && (!input.present || input.value == other.value)
}

func (input effectiveNumberOfGPUs) invalidValue() any {
	if !input.present {
		return nil
	}
	return input.value
}

// effectiveNumberOfGPUsV1Beta1 returns the main container's scalar GPU count, preferring limits to requests.
func effectiveNumberOfGPUsV1Beta1(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) effectiveNumberOfGPUs {
	containersPath := fldPath.Child("podTemplate", "spec", "containers")
	if component.PodTemplate == nil {
		return effectiveNumberOfGPUs{path: containersPath}
	}

	// Phase-1 power admission rejects DRA-backed components, matching the Planner's scalar-only
	// GPU count. Read the limit before the request, as the Planner does.
	resourceName := corev1.ResourceName(consts.KubeResourceGPUNvidia)
	for i := range component.PodTemplate.Spec.Containers {
		container := &component.PodTemplate.Spec.Containers[i]
		if container.Name != consts.MainContainerName {
			continue
		}

		resourcesPath := containersPath.Index(i).Child("resources")
		if quantity, exists := container.Resources.Limits[resourceName]; exists {
			return effectiveNumberOfGPUs{
				value:   quantity.String(),
				present: true,
				path:    resourcesPath.Child("limits").Key(consts.KubeResourceGPUNvidia),
			}
		}
		if quantity, exists := container.Resources.Requests[resourceName]; exists {
			return effectiveNumberOfGPUs{
				value:   quantity.String(),
				present: true,
				path:    resourcesPath.Child("requests").Key(consts.KubeResourceGPUNvidia),
			}
		}
		return effectiveNumberOfGPUs{path: resourcesPath}
	}
	return effectiveNumberOfGPUs{path: containersPath}
}

// effectivePodGPUCountV1Beta1 returns the scalar GPU scheduling footprint of the component Pod template.
func effectivePodGPUCountV1Beta1(
	ctx context.Context,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) (effectiveNumberOfGPUs, error) {
	podSpecPath := fldPath.Child("podTemplate", "spec")
	if component.PodTemplate == nil {
		return effectiveNumberOfGPUs{path: podSpecPath}, nil
	}

	count, err := dra.ResolvePodGPUCount(ctx, nil, "", &component.PodTemplate.Spec)
	if err != nil {
		return effectiveNumberOfGPUs{}, err
	}
	return effectiveNumberOfGPUs{
		value:   strconv.Itoa(count),
		present: true,
		path:    podSpecPath,
	}, nil
}

func sortedComponentNames(
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) []string {
	names := make([]string, 0, len(components))
	for name := range components {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func componentNameSet(
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) map[string]struct{} {
	names := make(map[string]struct{}, len(components))
	for name := range components {
		names[name] = struct{}{}
	}
	return names
}

func kvTransferPolicyFor(
	experimental *nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec,
) *nvidiacomv1beta1.KvTransferPolicy {
	if experimental == nil {
		return nil
	}
	return experimental.KvTransferPolicy
}

func kvTransferPoliciesEqual(a, b *nvidiacomv1beta1.KvTransferPolicy) bool {
	if b == nil {
		return false
	}
	return a.ClusterTopologyName == b.ClusterTopologyName &&
		a.LabelKey == b.LabelKey &&
		a.Domain == b.Domain &&
		effectiveKvTransferEnforcement(a) == effectiveKvTransferEnforcement(b) &&
		k8sptr.Equal(a.PreferredWeight, b.PreferredWeight)
}

func effectiveKvTransferEnforcement(policy *nvidiacomv1beta1.KvTransferPolicy) nvidiacomv1beta1.KvTransferEnforcement {
	if policy.Enforcement == "" {
		return nvidiacomv1beta1.KvTransferEnforcementRequired
	}
	return policy.Enforcement
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

// difference returns elements in set a that are not in set b (a - b).
func difference(a, b map[string]struct{}) []string {
	var result []string
	for name := range a {
		if _, exists := b[name]; !exists {
			result = append(result, name)
		}
	}
	return result
}

// generatedDynamoGraphDeploymentNameViolations reports generated names Kubernetes would reject:
// the DGD label value, <dgd>-<component>, and outside Grove the worker DCD name. Service names
// are checked only for components that get a Service. dgd must not be nil.
func generatedDynamoGraphDeploymentNameViolations(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	grovePathway bool,
	discoveryBackend configv1alpha1.DiscoveryBackend,
	lwsEnabled bool,
	source runtimeVersionValidationSource,
) []generatedNameViolation {
	var violations []generatedNameViolation
	if msgs := k8svalidation.IsValidLabelValue(dgd.Name); len(msgs) > 0 {
		violations = append(violations, generatedNameViolation{
			key: generatedNameViolationKey{generatedName: dgd.Name, rule: consts.KubeLabelDynamoGraphDeploymentName},
			err: field.Invalid(
				field.NewPath("metadata", "name"),
				dgd.Name,
				fmt.Sprintf(
					"invalid as the %s label value (%s); choose a shorter DynamoGraphDeployment name",
					consts.KubeLabelDynamoGraphDeploymentName,
					strings.Join(msgs, "; "),
				),
			),
		})
	}

	disaggregatedSet := dgd.Annotations[consts.KubeAnnotationWorkloadProvider] == consts.WorkloadProviderDisaggregatedSet ||
		(!grovePathway && dgd.Annotations[consts.KubeAnnotationWorkloadProvider] == "" && strings.EqualFold(dgd.Annotations[consts.KubeAnnotationEnableDisaggregatedSet], "true"))
	violations = append(violations, generatedProviderHeadlessServiceNameViolations(dgd, grovePathway, disaggregatedSet)...)

	// A stable placeholder compares name budgets without treating every new worker hash as a new rule.
	workerHashPlaceholder := strings.Repeat("0", dynamo.WorkerHashLength)
	componentsPath := field.NewPath("spec", "components")
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ManagedByExternalController() && !component.IsLPX() {
			continue
		}
		namePath := componentsPath.Index(i).Child("name")
		if source == runtimeVersionSourceV1Alpha1 {
			namePath = field.NewPath("spec", "services").Key(component.ComponentName)
		}

		// Converted alpha keys also become child spec.name and component labels verbatim.
		if msgs := k8svalidation.IsDNS1123Label(strings.ToLower(component.ComponentName)); len(msgs) > 0 {
			violations = append(violations, generatedNameViolation{
				key: generatedNameViolationKey{component: component.ComponentName, generatedName: component.ComponentName, rule: "component name"},
				err: field.Invalid(namePath, component.ComponentName, "component name must be a non-empty alphanumeric name with internal hyphens: "+strings.Join(msgs, "; ")),
			})
		}

		baseName := dynamo.GetDCDResourceName(dgd, component.ComponentName, "")
		name := baseName
		action := "choose a shorter or letter-leading DynamoGraphDeployment name, or a shorter component name"
		if !grovePathway {
			name = dynamo.GetDCDResourceName(dgd, component.ComponentName, workerHashPlaceholder)
			if name != baseName {
				action = fmt.Sprintf("worker DynamoComponentDeployment names end with an %d-character worker hash (shown as zeros); choose a shorter DynamoGraphDeployment or component name", dynamo.WorkerHashLength)
			}
		}
		dsWorker := disaggregatedSet && component.IsMultinode() && dynamo.IsWorkerComponent(string(component.ComponentType))
		checks := generatedNameChecks{
			dcd:      !grovePathway && !dsWorker && !component.IsLPX(),
			service:  !component.IsLPX() && dynamoGraphDeploymentComponentHasService(dgd, component, grovePathway, discoveryBackend),
			selector: grovePathway || !(dsWorker || (lwsEnabled && component.IsMultinode())),
		}
		for _, violation := range generatedNameViolations(namePath, name, checks, action) {
			violation.key.component = component.ComponentName
			violations = append(violations, violation)
		}

		// The LWS and elastic-EP leader Services have names distinct from discovery Services.
		if !grovePathway && !dsWorker && lwsEnabled && component.IsMultinode() {
			for _, violation := range generatedHeadlessServiceNameViolations(namePath, dynamo.LeaderWorkerSetNameForWorkload(name), "LeaderWorkerSet headless Service name") {
				violation.key.component = component.ComponentName
				violations = append(violations, violation)
			}
		}
		if grovePathway && !component.IsLPX() && dynamo.IsSinglePodElasticEPLeader(component) {
			serviceName := dynamo.ElasticEPLeaderServiceName(baseName)
			if msgs := k8svalidation.IsDNS1035Label(serviceName); len(msgs) > 0 {
				violations = append(violations, generatedNameViolation{
					key: generatedNameViolationKey{component: component.ComponentName, generatedName: serviceName, rule: "Service name"},
					err: field.Invalid(namePath, serviceName, fmt.Sprintf("generated name is invalid (Service name: %s); %s", strings.Join(msgs, "; "), action)),
				})
			}
		}
	}
	return violations
}

// generatedProviderHeadlessServiceNameViolations checks names created independently of discovery.
// dgd must not be nil.
func generatedProviderHeadlessServiceNameViolations(dgd *nvidiacomv1beta1.DynamoGraphDeployment, grovePathway, disaggregatedSet bool) []generatedNameViolation {
	var violations []generatedNameViolation
	if grovePathway {
		pcsName := dynamo.PCSNameForDGD(dgd, (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController)
		violations = append(violations, generatedHeadlessServiceNameViolations(field.NewPath("metadata", "name"), pcsName+"-0", "Grove PodCliqueSet headless Service name")...)
	}
	if disaggregatedSet {
		violations = append(violations, generatedHeadlessServiceNameViolations(field.NewPath("metadata", "name"), dynamo.DisaggregatedSetName(dgd), "DisaggregatedSet child Service prefix")...)
	}
	return violations
}

// dynamoGraphDeploymentComponentHasService reports whether the controller renders a Service named
// from component: a frontend always gets one, and other components get one only under
// Kubernetes discovery. The elastic-EP leader Service is checked separately.
// dgd and component must not be nil.
func dynamoGraphDeploymentComponentHasService(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	grovePathway bool,
	discoveryBackend configv1alpha1.DiscoveryBackend,
) bool {
	if component.ComponentType == nvidiacomv1beta1.ComponentTypeFrontend {
		return true
	}
	// Grove reads the DGD annotation; child DCDs inherit it unless the component overrides it.
	annotations := map[string]string{}
	if backend, ok := dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend]; ok && (grovePathway || component.PodTemplate != nil) {
		annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = backend
	}
	if !grovePathway {
		maps.Copy(annotations, dynamo.GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component))
	}
	return commoncontroller.IsK8sDiscoveryEnabled(discoveryBackend, annotations)
}

// generatedHeadlessServiceNameViolations checks the unnormalized names provider controllers use.
// fldPath must not be nil.
func generatedHeadlessServiceNameViolations(fldPath *field.Path, name, rule string) []generatedNameViolation {
	msgs := k8svalidation.IsDNS1035Label(name)
	if len(msgs) == 0 {
		return nil
	}
	return []generatedNameViolation{{
		key: generatedNameViolationKey{generatedName: name, rule: rule},
		err: field.Invalid(fldPath, name, fmt.Sprintf("generated name is invalid (%s: %s); choose a shorter name starting with a letter and without dots", rule, strings.Join(msgs, "; "))),
	}}
}
