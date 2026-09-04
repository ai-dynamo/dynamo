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

// dgdPowerLimitWatts is a parsed power-limit annotation value. parsed is the
// explicit presence boolean for watts: a syntactically invalid annotation is
// reported by validateDGDPowerLimitValue and yields no usable watt value, which
// a bare zero would silently conflate with a legal reading.
type dgdPowerLimitWatts struct {
	value  string
	watts  int64
	parsed bool
}

// validateDGDPowerLimitValue checks that the power-limit annotation value is a
// positive decimal integer, and returns the parsed watts so the GPU-product
// range check reuses this single parse. The annotation is made immutable on
// creation, so an invalid value that slips through requires a DGD
// delete-and-recreate to correct. fldPath must not be nil.
func validateDGDPowerLimitValue(value string, fldPath *field.Path) (dgdPowerLimitWatts, field.ErrorList) {
	watts, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return dgdPowerLimitWatts{value: value}, field.ErrorList{
			field.Invalid(fldPath, value, "must be a decimal integer"),
		}
	}
	if watts <= 0 {
		return dgdPowerLimitWatts{value: value}, field.ErrorList{
			field.Invalid(fldPath, value, "must be greater than zero"),
		}
	}
	return dgdPowerLimitWatts{value: value, watts: watts, parsed: true}, nil
}

// gpuProductNodeSelectorLabel is the GPU Feature Discovery node label naming the
// exact GPU product allocatable on a node. internal/gpu declares the same key as
// gpu.LabelGPUProduct, and this local copy duplicates it deliberately: that
// package carries DCGM and node-discovery dependencies unrelated to admission,
// and it reads the label from a corev1.Node rather than from a Pod node
// selector. Do not "fix" this by importing internal/gpu.
const gpuProductNodeSelectorLabel = "nvidia.com/gpu.product"

// powerRangeW is the inclusive whole-watt settable Total Graphics Power interval
// of one GPU product. Every entry satisfies 0 < Min <= Max.
type powerRangeW struct {
	Min int64
	Max int64
}

// powerRanges is the operator-owned admission catalog of settable TGP ranges,
// keyed by an exact reviewed public nvidia.com/gpu.product GFD label.
//
// Fractional source bounds are rounded inward — ceil(Min) and floor(Max) — so
// admission never widens a physical settable interval. Entries whose key is not
// an exact public GFD label, such as internal bring-up-board names, are excluded.
//
// Membership is not an exhaustive hardware list and not a Dynamo support
// guarantee: it states only which products this release can range-check a power
// annotation against. Adding an entry in a later release is strictly additive —
// it turns rejections into acceptances and never changes an existing object's
// update contract. The configured NVML or DCGM actuator, not this map, remains
// authoritative at runtime; the Power Agent does not consume it.
var powerRanges = map[string]powerRangeW{
	"NVIDIA-A10":                     {Min: 100, Max: 150},
	"NVIDIA-A100-80GB-PCIe":          {Min: 150, Max: 300},
	"NVIDIA-A100-PCIE-40GB":          {Min: 150, Max: 250},
	"NVIDIA-A100-SXM4-40GB":          {Min: 100, Max: 400},
	"NVIDIA-A100-SXM4-80GB":          {Min: 100, Max: 400},
	"NVIDIA-A16":                     {Min: 49, Max: 62},
	"NVIDIA-A2":                      {Min: 35, Max: 60},
	"NVIDIA-A30":                     {Min: 100, Max: 165},
	"NVIDIA-A40":                     {Min: 100, Max: 300},
	"NVIDIA-AX800":                   {Min: 250, Max: 350},
	"NVIDIA-B200":                    {Min: 200, Max: 1000},
	"NVIDIA-GH200-120GB":             {Min: 100, Max: 900},
	"NVIDIA-GH200-480GB":             {Min: 100, Max: 900},
	"NVIDIA-GeForce-RTX-2070-SUPER":  {Min: 125, Max: 260},
	"NVIDIA-GeForce-RTX-3070":        {Min: 100, Max: 240},
	"NVIDIA-GeForce-RTX-3070-Ti":     {Min: 100, Max: 320},
	"NVIDIA-GeForce-RTX-3080":        {Min: 100, Max: 370},
	"NVIDIA-GeForce-RTX-3090":        {Min: 100, Max: 400},
	"NVIDIA-GeForce-RTX-4090":        {Min: 10, Max: 600},
	"NVIDIA-GeForce-RTX-5080":        {Min: 250, Max: 360},
	"NVIDIA-GeForce-RTX-5090":        {Min: 400, Max: 600},
	"NVIDIA-H100":                    {Min: 200, Max: 700},
	"NVIDIA-H100-80GB-HBM3":          {Min: 200, Max: 700},
	"NVIDIA-H100-NVL":                {Min: 200, Max: 400},
	"NVIDIA-H100-PCIe":               {Min: 200, Max: 350},
	"NVIDIA-H20":                     {Min: 200, Max: 500},
	"NVIDIA-H20-3e":                  {Min: 200, Max: 500},
	"NVIDIA-H200":                    {Min: 200, Max: 700},
	"NVIDIA-H200-NVL":                {Min: 200, Max: 600},
	"NVIDIA-H800-NVL":                {Min: 200, Max: 400},
	"NVIDIA-L2":                      {Min: 40, Max: 72},
	"NVIDIA-L20":                     {Min: 150, Max: 350},
	"NVIDIA-L4":                      {Min: 40, Max: 72},
	"NVIDIA-L40":                     {Min: 100, Max: 300},
	"NVIDIA-L40G":                    {Min: 100, Max: 300},
	"NVIDIA-L40S":                    {Min: 100, Max: 350},
	"NVIDIA-RTX-4500-Ada-Generation": {Min: 100, Max: 210},
	"NVIDIA-RTX-5000-Ada-Generation": {Min: 100, Max: 250},
	"NVIDIA-RTX-6000-Ada-Generation": {Min: 100, Max: 300},
	"NVIDIA-RTX-A4000":               {Min: 100, Max: 140},
	"NVIDIA-RTX-A5000":               {Min: 100, Max: 230},
	"NVIDIA-RTX-A6000":               {Min: 100, Max: 300},
	"NVIDIA-RTX-PRO-2000-Blackwell":  {Min: 56, Max: 70},
	"NVIDIA-RTX-PRO-4500-Blackwell-Server-Edition":      {Min: 100, Max: 165},
	"NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition":      {Min: 300, Max: 600},
	"NVIDIA-RTX-PRO-6000-Blackwell-Workstation-Edition": {Min: 150, Max: 600},
	"Quadro-GP100":         {Min: 118, Max: 235},
	"Quadro-RTX-6000":      {Min: 100, Max: 260},
	"Quadro-RTX-8000":      {Min: 100, Max: 260},
	"Tesla-P100-PCIE-16GB": {Min: 125, Max: 250},
	"Tesla-T4":             {Min: 60, Max: 70},
	"Tesla-V100-PCIE-32GB": {Min: 100, Max: 250},
	"Tesla-V100-SXM2-16GB": {Min: 100, Max: 300},
	"Tesla-V100-SXM3-32GB": {Min: 100, Max: 350},
}

// dgdGPUProductSelector returns the exact GPU product a component pins through
// podTemplate.spec.nodeSelector, or "" when the component has no pod template,
// no node selector, or no such key. component must not be nil.
func dgdGPUProductSelector(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) string {
	if component.PodTemplate == nil {
		return ""
	}
	return component.PodTemplate.Spec.NodeSelector[gpuProductNodeSelectorLabel]
}

// powerProductContract is the complete normalized input set for the GPU-product
// power rule. What counts as an input depends on whether the component pins a
// product: an exact product selector bounds the GPU product on its own, so
// nothing else about placement can change the rule's outcome. Without one, every
// placement input can move the component to different hardware, so every
// placement input is a rule input.
//
// Compare two contracts with equality.Semantic.DeepEqual rather than ==;
// corev1.Affinity contains slices and is not comparable. The fields are exported
// on this unexported type because that comparator walks the struct by reflection
// and panics on an unexported field.
type powerProductContract struct {
	PowerLimit string // annotation value; "" when absent
	HasPower   bool   // explicit presence boolean; "" is a distinct, invalid state
	GPUProduct string // nodeSelector["nvidia.com/gpu.product"]; "" when absent or podTemplate is nil
	NodeName   string

	// NodeSelector and Affinity are populated only when GPUProduct is empty.
	// A legacy component has no product bound, so its complete placement is part
	// of the rule input set.
	//
	// NodeSelector deliberately retains the nvidia.com/gpu.product key itself.
	// Pruning it as redundant would make a missing product key and an explicitly
	// empty one compare equal, so a missing -> "" transition would be ratcheted
	// instead of rejected. Retaining the key keeps them distinguishable by map
	// length, which Semantic.DeepEqual does distinguish even though it equates a
	// nil map with an empty one — and equating those two is correct here, since
	// both express exactly the same placement.
	NodeSelector map[string]string
	Affinity     *corev1.Affinity
}

// dgdPowerProductContract derives the complete normalized power-product rule
// inputs for component. component must not be nil.
//
// The conditional shape below is entirely internal to this function. Callers
// compare two returned contracts and nothing else: they must not branch on
// GPUProduct == "", reconstruct the condition, or reach past the contract into
// the component to decide what to compare. A caller that needs to know why two
// contracts differ is a signal that the logic belongs here.
func dgdPowerProductContract(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) powerProductContract {
	contract := powerProductContract{}
	contract.PowerLimit, contract.HasPower = dgdPowerLimit(component)
	if component.PodTemplate == nil {
		return contract
	}
	podSpec := &component.PodTemplate.Spec
	contract.GPUProduct = podSpec.NodeSelector[gpuProductNodeSelectorLabel]
	contract.NodeName = podSpec.NodeName
	if contract.GPUProduct == "" {
		contract.NodeSelector = podSpec.NodeSelector
		contract.Affinity = podSpec.Affinity
	}
	return contract
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
