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
	"strconv"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
)

const (
	// maxCombinedResourceNameLength is kept as a local alias for readability.
	maxCombinedResourceNameLength = consts.MaxCombinedGroveResourceNameLength
)

type clusterTopologyInfo struct {
	name        string
	domainIndex map[string]int
	domains     []string
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

func dgdComponentResourceNameLength(
	dgdName string,
	components []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (int, string) {
	pcsName := dynamo.PCSNameForDGD(dgdName, components)
	componentName := component.ComponentName
	combinedLength := len(pcsName) + len(strings.ToLower(componentName))
	detail := "PCS name + component name"

	if component.UsesPCSG() {
		longestPodCliqueName := dynamo.LongestPodCliqueNameForDGDComponent(componentName, component)
		combinedLength += len(longestPodCliqueName)
		detail = fmt.Sprintf("PCS name + PCSG name + longest PodClique name %q", longestPodCliqueName)
	}
	return combinedLength, detail
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
