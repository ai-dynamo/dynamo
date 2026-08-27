/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dra

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// ClaimName is the pod-level DRA ResourceClaim name for shared GPU access.
	ClaimName = "intrapod-shared-gpu"

	// DefaultDeviceClassName is the default DRA DeviceClass name used when a
	// component does not specify an explicit gpuType. It matches the
	// DeviceClass that ships with the NVIDIA DRA Driver and is the single
	// source of truth for this string across the operator.
	DefaultDeviceClassName = "gpu.nvidia.com"
)

// deviceDriverEqualityPattern extracts positive device.driver equality constraints
// from CEL conjunctions. classifyDeviceClass deliberately ignores disjunctions,
// where one matching branch does not classify every device selected by the class.
var deviceDriverEqualityPattern = regexp.MustCompile(`(?:^|&&)\s*\(*\s*device\.driver\s*==\s*["']([^"']+)["']`)

// ApplyClaim replaces the first container's scalar GPU resources with a shared
// DRA ResourceClaim. Every container that references this claim name will share
// the same physical GPUs.
func ApplyClaim(podSpec *corev1.PodSpec, claimTemplateName string) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for DRA claim")
	}

	// Replace GPU resources with the shared DRA claim. The resource name can be
	// nvidia.com/gpu or a MIG shape such as nvidia.com/mig-3g.20gb.
	container := &podSpec.Containers[0]
	RemoveGPUResources(container.Resources.Limits)
	RemoveGPUResources(container.Resources.Requests)
	hasClaim := false
	for i := range container.Resources.Claims {
		if container.Resources.Claims[i].Name == ClaimName {
			hasClaim = true
			break
		}
	}
	if !hasClaim {
		container.Resources.Claims = append(container.Resources.Claims, corev1.ResourceClaim{Name: ClaimName})
	}

	// GPU nodes are typically tainted with nvidia.com/gpu=NoSchedule. DRA
	// bypasses the device-plugin toleration injection, so add it explicitly.
	hasToleration := false
	for i := range podSpec.Tolerations {
		toleration := podSpec.Tolerations[i]
		if toleration.Key == commonconsts.KubeResourceGPUNvidia && toleration.Effect == corev1.TaintEffectNoSchedule {
			hasToleration = true
			break
		}
	}
	if !hasToleration {
		podSpec.Tolerations = append(podSpec.Tolerations, corev1.Toleration{
			Key:      commonconsts.KubeResourceGPUNvidia,
			Operator: corev1.TolerationOpExists,
			Effect:   corev1.TaintEffectNoSchedule,
		})
	}

	claim := corev1.PodResourceClaim{
		Name:                      ClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	}
	for i := range podSpec.ResourceClaims {
		if podSpec.ResourceClaims[i].Name == ClaimName {
			podSpec.ResourceClaims[i] = claim
			return nil
		}
	}
	podSpec.ResourceClaims = append(podSpec.ResourceClaims, claim)
	return nil
}

// ResourceClaimTemplateName returns the deterministic name for the
// ResourceClaimTemplate associated with a component.
func ResourceClaimTemplateName(parentName, componentName string) string {
	return fmt.Sprintf("%s-%s-gpu", parentName, strings.ToLower(componentName))
}

func ExtractGPUCountFromResourceRequirements(resources corev1.ResourceRequirements) (int, error) {
	if name, q, ok := gpuResourceQuantity(resources.Limits); ok {
		return gpuCountFromQuantity(name, q)
	}
	if name, q, ok := gpuResourceQuantity(resources.Requests); ok {
		return gpuCountFromQuantity(name, q)
	}
	return 0, nil
}

// ResolveGPUCount returns the number of GPUs requested by a container, including
// devices requested through DRA ResourceClaims. Scalar resources take precedence
// because operator-managed DRA paths intentionally retain them until rendering.
func ResolveGPUCount(
	ctx context.Context,
	cl client.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	resources corev1.ResourceRequirements,
) (int, error) {
	scalarCount, err := ExtractGPUCountFromResourceRequirements(resources)
	if err != nil || scalarCount > 0 || len(resources.Claims) == 0 {
		return scalarCount, err
	}
	if cl == nil {
		return 0, fmt.Errorf("cannot resolve GPU ResourceClaims without a Kubernetes client")
	}
	if podSpec == nil {
		return 0, fmt.Errorf("cannot resolve GPU ResourceClaims without a pod spec")
	}

	podClaims := make(map[string]corev1.PodResourceClaim, len(podSpec.ResourceClaims))
	for _, claim := range podSpec.ResourceClaims {
		podClaims[claim.Name] = claim
	}

	total := 0
	seen := make(map[string]struct{}, len(resources.Claims))
	deviceClasses := newDeviceClassResolver(cl)
	for _, containerClaim := range resources.Claims {
		key := containerClaim.Name + "\x00" + containerClaim.Request
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}

		podClaim, ok := podClaims[containerClaim.Name]
		if !ok {
			return 0, fmt.Errorf("container ResourceClaim %q has no matching pod resourceClaim", containerClaim.Name)
		}
		claimSpec, err := getClaimSpec(ctx, cl, namespace, podClaim)
		if err != nil {
			return 0, err
		}
		count, err := gpuCountFromClaimSpec(ctx, deviceClasses, claimSpec, containerClaim.Request)
		if err != nil {
			return 0, fmt.Errorf("cannot determine GPU count for ResourceClaim %q: %w", containerClaim.Name, err)
		}
		total += count
	}
	return total, nil
}

// PodSpecMultiplicity describes how many copies of a rendered Pod spec belong
// to one component replica.
type PodSpecMultiplicity struct {
	PodSpec *corev1.PodSpec
	Count   int32
}

type claimSelection struct {
	claims     []corev1.ResourceClaim
	requests   map[string]struct{}
	selectsAll bool
}

func (s *claimSelection) add(claim corev1.ResourceClaim) {
	if claim.Request == "" {
		s.claims = []corev1.ResourceClaim{claim}
		s.selectsAll = true
		return
	}
	if s.selectsAll {
		return
	}
	if s.requests == nil {
		s.requests = make(map[string]struct{})
	}
	if _, ok := s.requests[claim.Request]; ok {
		return
	}
	s.requests[claim.Request] = struct{}{}
	s.claims = append(s.claims, claim)
}

// ResolvePodGPUCount returns the unique GPUs allocated to all regular
// containers in one Pod. Scalar GPU resources and DRA allocations are
// additive; repeated references to the same DRA request are counted once.
func ResolvePodGPUCount(
	ctx context.Context,
	cl client.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
) (int, error) {
	return ResolvePodSetGPUCount(ctx, cl, namespace, []PodSpecMultiplicity{{
		PodSpec: podSpec,
		Count:   1,
	}})
}

// ResolvePodSetGPUCount returns the unique GPU allocation across all rendered
// Pods in one component replica. ResourceClaimTemplates are charged per Pod;
// repeated references to the same concrete ResourceClaim are charged once
// across the whole set.
func ResolvePodSetGPUCount(
	ctx context.Context,
	cl client.Reader,
	namespace string,
	pods []PodSpecMultiplicity,
) (int, error) {
	total := 0
	directClaimOrder := make([]string, 0)
	directClaims := make(map[string]*claimSelection)
	for _, pod := range pods {
		if pod.PodSpec == nil {
			return 0, fmt.Errorf("cannot resolve Pod GPU count without a pod spec")
		}
		if pod.Count <= 0 {
			return 0, fmt.Errorf("rendered Pod multiplicity must be positive, got %d", pod.Count)
		}

		podClaims := make(map[string]corev1.PodResourceClaim, len(pod.PodSpec.ResourceClaims))
		for _, podClaim := range pod.PodSpec.ResourceClaims {
			podClaims[podClaim.Name] = podClaim
		}
		claimOrder := make([]string, 0)
		claimSelections := make(map[string]*claimSelection)
		for i := range pod.PodSpec.Containers {
			container := &pod.PodSpec.Containers[i]
			scalarCount, err := ExtractGPUCountFromResourceRequirements(container.Resources)
			if err != nil {
				return 0, fmt.Errorf("container %q: %w", container.Name, err)
			}
			total += scalarCount * int(pod.Count)

			for _, claim := range container.Resources.Claims {
				selection, ok := claimSelections[claim.Name]
				if !ok {
					selection = &claimSelection{}
					claimSelections[claim.Name] = selection
					claimOrder = append(claimOrder, claim.Name)
				}
				selection.add(claim)
			}
		}

		for _, claimName := range claimOrder {
			selection := claimSelections[claimName]
			podClaim, ok := podClaims[claimName]
			if !ok {
				return 0, fmt.Errorf("container ResourceClaim %q has no matching pod resourceClaim", claimName)
			}
			if podClaim.ResourceClaimTemplateName == nil &&
				podClaim.ResourceClaimName != nil && *podClaim.ResourceClaimName != "" {
				directName := *podClaim.ResourceClaimName
				directSelection, ok := directClaims[directName]
				if !ok {
					directSelection = &claimSelection{}
					directClaims[directName] = directSelection
					directClaimOrder = append(directClaimOrder, directName)
				}
				for _, claim := range selection.claims {
					claim.Name = directName
					directSelection.add(claim)
				}
				continue
			}

			count, err := ResolveGPUCount(ctx, cl, namespace, pod.PodSpec, corev1.ResourceRequirements{
				Claims: selection.claims,
			})
			if err != nil {
				return 0, err
			}
			total += count * int(pod.Count)
		}
	}

	for _, directName := range directClaimOrder {
		selection := directClaims[directName]
		count, err := ResolveGPUCount(
			ctx,
			cl,
			namespace,
			&corev1.PodSpec{ResourceClaims: []corev1.PodResourceClaim{{
				Name:              directName,
				ResourceClaimName: ptr.To(directName),
			}}},
			corev1.ResourceRequirements{Claims: selection.claims},
		)
		if err != nil {
			return 0, err
		}
		total += count
	}
	return total, nil
}

func getClaimSpec(
	ctx context.Context,
	cl client.Reader,
	namespace string,
	podClaim corev1.PodResourceClaim,
) (*resourcev1.ResourceClaimSpec, error) {
	switch {
	case podClaim.ResourceClaimTemplateName != nil && *podClaim.ResourceClaimTemplateName != "":
		name := *podClaim.ResourceClaimTemplateName
		template := &resourcev1.ResourceClaimTemplate{}
		if err := cl.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, template); err != nil {
			return nil, fmt.Errorf("failed to get ResourceClaimTemplate %s/%s: %w", namespace, name, err)
		}
		return &template.Spec.Spec, nil
	case podClaim.ResourceClaimName != nil && *podClaim.ResourceClaimName != "":
		name := *podClaim.ResourceClaimName
		claim := &resourcev1.ResourceClaim{}
		if err := cl.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, claim); err != nil {
			return nil, fmt.Errorf("failed to get ResourceClaim %s/%s: %w", namespace, name, err)
		}
		return &claim.Spec, nil
	default:
		return nil, fmt.Errorf("pod resourceClaim %q does not reference a ResourceClaim or ResourceClaimTemplate", podClaim.Name)
	}
}

func gpuCountFromClaimSpec(
	ctx context.Context,
	deviceClasses *deviceClassResolver,
	spec *resourcev1.ResourceClaimSpec,
	requestName string,
) (int, error) {
	total := 0
	foundRequest := requestName == ""
	for _, request := range spec.Devices.Requests {
		if requestName != "" && request.Name != requestName {
			continue
		}
		foundRequest = true

		count, isGPU, err := gpuCountFromDeviceRequest(ctx, deviceClasses, request)
		if err != nil {
			return 0, err
		}
		if isGPU {
			total += count
		}
	}
	if !foundRequest {
		return 0, fmt.Errorf("device request %q was not found", requestName)
	}
	return total, nil
}

func gpuCountFromDeviceRequest(
	ctx context.Context,
	deviceClasses *deviceClassResolver,
	request resourcev1.DeviceRequest,
) (int, bool, error) {
	if request.Exactly != nil {
		isGPU, err := deviceClasses.isGPU(ctx, request.Exactly.DeviceClassName, request.Exactly.Selectors)
		if err != nil {
			return 0, false, err
		}
		if !isGPU {
			return 0, false, nil
		}
		count, err := exactDeviceCount(request.Exactly.AllocationMode, request.Exactly.Count)
		return count, true, err
	}
	if len(request.FirstAvailable) == 0 {
		return 0, false, fmt.Errorf("device request %q must set exactly or firstAvailable", request.Name)
	}

	var count int
	var hasGPU, hasNonGPU bool
	for _, subRequest := range request.FirstAvailable {
		isGPU, err := deviceClasses.isGPU(ctx, subRequest.DeviceClassName, subRequest.Selectors)
		if err != nil {
			return 0, false, err
		}
		if !isGPU {
			hasNonGPU = true
			continue
		}
		candidateCount, err := exactDeviceCount(subRequest.AllocationMode, subRequest.Count)
		if err != nil {
			return 0, false, err
		}
		if hasGPU && candidateCount != count {
			return 0, false, fmt.Errorf("GPU device request %q has firstAvailable alternatives with different counts", request.Name)
		}
		count = candidateCount
		hasGPU = true
	}
	if hasGPU && hasNonGPU {
		return 0, false, fmt.Errorf("device request %q mixes GPU and non-GPU firstAvailable alternatives", request.Name)
	}
	return count, hasGPU, nil
}

func exactDeviceCount(mode resourcev1.DeviceAllocationMode, count int64) (int, error) {
	switch mode {
	case "", resourcev1.DeviceAllocationModeExactCount:
		if count == 0 {
			return 1, nil
		}
		if count < 0 {
			return 0, fmt.Errorf("GPU device count must be positive")
		}
		return int(count), nil
	case resourcev1.DeviceAllocationModeAll:
		return 0, fmt.Errorf("GPU allocationMode %q has no deterministic per-node device count", mode)
	default:
		return 0, fmt.Errorf("unsupported GPU allocationMode %q", mode)
	}
}

func looksLikeGPUIdentifier(name string) bool {
	normalized := strings.ToLower(name)
	return normalized == "gpu" ||
		strings.HasPrefix(normalized, "gpu.") ||
		strings.HasPrefix(normalized, "gpu/") ||
		strings.HasSuffix(normalized, "/gpu") ||
		strings.Contains(normalized, "/gpu.") ||
		strings.Contains(normalized, "/gpu-")
}

type deviceClassResolver struct {
	reader client.Reader
	cache  map[string]*resourcev1.DeviceClass
}

func newDeviceClassResolver(reader client.Reader) *deviceClassResolver {
	return &deviceClassResolver{
		reader: reader,
		cache:  map[string]*resourcev1.DeviceClass{},
	}
}

func (r *deviceClassResolver) isGPU(ctx context.Context, name string, requestSelectors []resourcev1.DeviceSelector) (bool, error) {
	deviceClass, ok := r.cache[name]
	if !ok {
		deviceClass = &resourcev1.DeviceClass{}
		if err := r.reader.Get(ctx, types.NamespacedName{Name: name}, deviceClass); err != nil {
			return false, fmt.Errorf("failed to get DeviceClass %q: %w", name, err)
		}
		r.cache[name] = deviceClass
	}

	isGPU, known, err := classifyDeviceRequest(deviceClass, requestSelectors)
	if err != nil {
		return false, err
	}
	if !known {
		return false, fmt.Errorf(
			"cannot determine whether DeviceClass %q provides GPUs: set spec.extendedResourceName or constrain device.driver in a DeviceClass or request CEL selector",
			name,
		)
	}
	return isGPU, nil
}

func classifyDeviceRequest(deviceClass *resourcev1.DeviceClass, requestSelectors []resourcev1.DeviceSelector) (bool, bool, error) {
	// Kubernetes requires both DeviceClass and request selectors to match.
	selectors := make([]resourcev1.DeviceSelector, 0, len(deviceClass.Spec.Selectors)+len(requestSelectors))
	selectors = append(selectors, deviceClass.Spec.Selectors...)
	selectors = append(selectors, requestSelectors...)
	var isGPU, hasDriver bool
	for _, selector := range selectors {
		if selector.CEL == nil || strings.Contains(selector.CEL.Expression, "||") {
			continue
		}
		matches := deviceDriverEqualityPattern.FindAllStringSubmatch(selector.CEL.Expression, -1)
		for _, match := range matches {
			candidateIsGPU := looksLikeGPUIdentifier(match[1])
			if hasDriver && candidateIsGPU != isGPU {
				return false, false, fmt.Errorf("DeviceClass %q and its request selectors constrain device.driver to both GPU and non-GPU drivers", deviceClass.Name)
			}
			isGPU = candidateIsGPU
			hasDriver = true
		}
	}
	if hasDriver {
		return isGPU, true, nil
	}
	if deviceClass.Spec.ExtendedResourceName != nil {
		return isGPUResourceName(corev1.ResourceName(*deviceClass.Spec.ExtendedResourceName)), true, nil
	}
	// Keep canonical GPU class names working only when the class has no selectors.
	// A present but unsupported class or request selector must remain unclassified
	// instead of being overridden by a suggestive class name.
	if len(selectors) == 0 && looksLikeGPUIdentifier(deviceClass.Name) {
		return true, true, nil
	}
	return false, false, nil
}

// RemoveGPUResources deletes all scalar GPU resource entries from a resource list.
func RemoveGPUResources(resources corev1.ResourceList) {
	for _, name := range gpuResourceNames(resources) {
		delete(resources, name)
	}
}

func gpuResourceQuantity(resources corev1.ResourceList) (corev1.ResourceName, resource.Quantity, bool) {
	names := gpuResourceNames(resources)
	if len(names) == 0 {
		return "", resource.Quantity{}, false
	}
	return names[0], resources[names[0]], true
}

func gpuCountFromQuantity(name corev1.ResourceName, q resource.Quantity) (int, error) {
	value := q.Value()
	if q.CmpInt64(value) != 0 {
		return 0, fmt.Errorf("GPU resource %q quantity %q must be a whole number", name, q.String())
	}
	return int(value), nil
}

func isGPUResourceName(name corev1.ResourceName) bool {
	normalized := strings.ToLower(string(name))
	return normalized == commonconsts.KubeResourceGPUNvidia ||
		normalized == "gpu" ||
		strings.HasSuffix(normalized, "/gpu") ||
		strings.Contains(normalized, "/mig-") ||
		strings.HasPrefix(normalized, "gpu.")
}

func gpuResourceNames(resources corev1.ResourceList) []corev1.ResourceName {
	matches := make([]string, 0)
	for name := range resources {
		if isGPUResourceName(name) {
			matches = append(matches, string(name))
		}
	}
	sort.Strings(matches)
	result := make([]corev1.ResourceName, 0, len(matches))
	for _, name := range matches {
		result = append(result, corev1.ResourceName(name))
	}
	return result
}

// GenerateResourceClaimTemplate builds the ResourceClaimTemplate that provides
// shared GPU access to all containers in a pod via DRA.
//
// When gpuCount <= 0 it returns the template skeleton with toDelete=true so
// that SyncResource cleans up any previously created template. Pass cl=nil to
// skip the DeviceClass existence check.
func GenerateResourceClaimTemplate(
	ctx context.Context,
	cl client.Client,
	claimTemplateName, namespace string,
	gpuCount int,
	deviceClassName string,
) (*resourcev1.ResourceClaimTemplate, bool, error) {
	template := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      claimTemplateName,
			Namespace: namespace,
		},
	}

	if gpuCount <= 0 {
		return template, true, nil
	}

	if deviceClassName == "" {
		deviceClassName = DefaultDeviceClassName
	}

	if cl != nil {
		dc := &resourcev1.DeviceClass{}
		if err := cl.Get(ctx, types.NamespacedName{Name: deviceClassName}, dc); err != nil {
			if apierrors.IsNotFound(err) {
				return nil, false, fmt.Errorf(
					"DeviceClass %q not found: ensure the GPU DRA driver is installed and the device class is registered",
					deviceClassName)
			}
			return nil, false, fmt.Errorf("failed to verify DeviceClass %q: %w", deviceClassName, err)
		}
	}

	template.Spec = resourcev1.ResourceClaimTemplateSpec{
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{
					{
						Name: "gpus",
						Exactly: &resourcev1.ExactDeviceRequest{
							DeviceClassName: deviceClassName,
							AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
							Count:           int64(gpuCount),
						},
					},
				},
			},
		},
	}

	return template, false, nil
}
