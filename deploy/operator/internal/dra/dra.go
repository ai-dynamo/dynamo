/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dra

import (
	"context"
	"fmt"
	"sort"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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
