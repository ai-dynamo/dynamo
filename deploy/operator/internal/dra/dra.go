/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dra

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// ClaimName is the pod-level DRA ResourceClaim name for shared GPU access.
	ClaimName = "intrapod-shared-gpu"

	defaultDeviceClassName = "gpu.nvidia.com"
)

// ApplyClaim replaces the first container's nvidia.com/gpu resources with a
// shared DRA ResourceClaim. Every container that references this claim name
// will share the same physical GPUs. The function is idempotent — calling it
// on a pod that already has the claim is a no-op.
func ApplyClaim(podSpec *corev1.PodSpec, claimTemplateName string) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for DRA claim")
	}

	// Skip if the pod-level claim already exists (idempotent).
	for i := range podSpec.ResourceClaims {
		if podSpec.ResourceClaims[i].Name == ClaimName {
			return nil
		}
	}

	// Replace nvidia.com/gpu with the shared DRA claim.
	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	delete(podSpec.Containers[0].Resources.Limits, gpuResource)
	delete(podSpec.Containers[0].Resources.Requests, gpuResource)
	podSpec.Containers[0].Resources.Claims = append(podSpec.Containers[0].Resources.Claims, corev1.ResourceClaim{
		Name: ClaimName,
	})

	// GPU nodes are typically tainted with nvidia.com/gpu=NoSchedule. DRA
	// bypasses the device-plugin toleration injection, so add it explicitly.
	podSpec.Tolerations = append(podSpec.Tolerations, corev1.Toleration{
		Key:      commonconsts.KubeResourceGPUNvidia,
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	})

	podSpec.ResourceClaims = append(podSpec.ResourceClaims, corev1.PodResourceClaim{
		Name:                      ClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	})

	return nil
}

// ResourceClaimTemplateName returns the deterministic name for the
// ResourceClaimTemplate associated with a component.
func ResourceClaimTemplateName(parentName, serviceName string) string {
	return fmt.Sprintf("%s-%s-gpu", parentName, strings.ToLower(serviceName))
}

// ExtractGPUParams extracts the GPU count and device class name from API types
// shared by DGD components and DynamoCheckpoint specs. Returns gpuCount=0 when
// GMS is not enabled, which tells GenerateResourceClaimTemplate to delete.
func ExtractGPUParams(gmsSpec *v1alpha1.GPUMemoryServiceSpec, resources *v1alpha1.Resources) (gpuCount int, deviceClassName string) {
	if gmsSpec == nil || !gmsSpec.Enabled {
		return 0, ""
	}
	deviceClassName = gmsSpec.DeviceClassName
	if resources != nil {
		gpuStr := ""
		if resources.Limits != nil {
			gpuStr = resources.Limits.GPU
		}
		if gpuStr == "" && resources.Requests != nil {
			gpuStr = resources.Requests.GPU
		}
		gpuCount, _ = strconv.Atoi(gpuStr)
	}
	return gpuCount, deviceClassName
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
		deviceClassName = defaultDeviceClassName
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
