/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	gmsruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	defaultDeviceClassName = "gpu.nvidia.com"
)

func isGMSEnabled(component *v1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	return component.GPUMemoryService != nil && component.GPUMemoryService.Enabled
}

// getGPUCount extracts the GPU count from the component resource spec.
func getGPUCount(component *v1alpha1.DynamoComponentDeploymentSharedSpec) (int, error) {
	if component.Resources == nil {
		return 0, fmt.Errorf("resources must be specified when GPU memory service is enabled")
	}

	gpuStr := ""
	if component.Resources.Limits != nil && component.Resources.Limits.GPU != "" {
		gpuStr = component.Resources.Limits.GPU
	} else if component.Resources.Requests != nil && component.Resources.Requests.GPU != "" {
		gpuStr = component.Resources.Requests.GPU
	}

	if gpuStr == "" {
		return 0, fmt.Errorf("GPU count must be specified when GPU memory service is enabled")
	}

	count, err := strconv.Atoi(gpuStr)
	if err != nil {
		return 0, fmt.Errorf("invalid GPU count %q: %w", gpuStr, err)
	}
	return count, nil
}

// getDeviceClassName returns the DRA DeviceClass name for the component.
func getDeviceClassName(component *v1alpha1.DynamoComponentDeploymentSharedSpec) string {
	if component.Resources != nil && component.Resources.Limits != nil && component.Resources.Limits.GPUType != "" {
		return component.Resources.Limits.GPUType
	}
	return defaultDeviceClassName
}

// resolveMainContainer finds the container named "main" in the pod spec.
// Falls back to Containers[0] if there is exactly one container.
func resolveMainContainer(podSpec *corev1.PodSpec) (*corev1.Container, error) {
	if len(podSpec.Containers) == 0 {
		return nil, fmt.Errorf("pod spec must have at least one container for GPU memory service")
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			return &podSpec.Containers[i], nil
		}
	}
	if len(podSpec.Containers) == 1 {
		return &podSpec.Containers[0], nil
	}
	return nil, fmt.Errorf("pod spec has %d containers but none named %q", len(podSpec.Containers), commonconsts.MainContainerName)
}

// applyGPUMemoryService transforms a pod spec to include GMS server sidecars
// with DRA shared GPU access. The main container's GPU resources are replaced
// with a DRA ResourceClaim.
func applyGPUMemoryService(
	podSpec *corev1.PodSpec,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
	parentName string,
	serviceName string,
) error {
	gpuCount, err := getGPUCount(component)
	if err != nil {
		return err
	}
	_ = gpuCount // GPU count is used for DRA claim template; sidecar discovers devices via pynvml

	mainContainer, err := resolveMainContainer(podSpec)
	if err != nil {
		return err
	}

	// Replace GPU resources with DRA claim on main container
	removeGPUResources(mainContainer)
	mainContainer.Resources.Claims = append(mainContainer.Resources.Claims, corev1.ResourceClaim{
		Name: gmsruntime.DRAClaimName,
	})

	// Add GMS server sidecar, shared volume, and socket env vars.
	// The sidecar gets DRA claims copied from main automatically.
	gmsruntime.EnsureServerSidecar(podSpec, mainContainer)

	// DRA replaces normal GPU scheduling, so the default GPU toleration that
	// kubelet/device-plugin would add is lost. Re-add it explicitly.
	podSpec.Tolerations = append(podSpec.Tolerations, corev1.Toleration{
		Key:      commonconsts.KubeResourceGPUNvidia,
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	})

	// Add pod-level DRA resource claim referencing the ResourceClaimTemplate
	claimTemplateName := GMSResourceClaimTemplateName(parentName, serviceName)
	podSpec.ResourceClaims = append(podSpec.ResourceClaims, corev1.PodResourceClaim{
		Name:                      gmsruntime.DRAClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	})

	return nil
}

// removeGPUResources strips nvidia.com/gpu from container resource limits and requests.
// GPU allocation is handled by DRA when GMS is enabled.
func removeGPUResources(container *corev1.Container) {
	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	delete(container.Resources.Limits, gpuResource)
	delete(container.Resources.Requests, gpuResource)
}

// GMSResourceClaimTemplateName returns the deterministic name for the
// ResourceClaimTemplate associated with a GMS-enabled component.
func GMSResourceClaimTemplateName(parentName, serviceName string) string {
	return fmt.Sprintf("%s-%s-gpu", parentName, strings.ToLower(serviceName))
}

// GenerateGMSResourceClaimTemplate builds the ResourceClaimTemplate that
// provides shared GPU access to all containers in a GMS-enabled pod via DRA.
//
// When GMS is not enabled for the component, it returns the template skeleton
// with toDelete=true so that SyncResource cleans up any previously created template.
//
// The cl parameter is used to verify the DeviceClass exists before creating
// the template. Pass nil to skip the DeviceClass check.
func GenerateGMSResourceClaimTemplate(
	ctx context.Context,
	cl client.Client,
	parentName, namespace, serviceName string,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
) (*resourcev1.ResourceClaimTemplate, bool, error) {
	name := GMSResourceClaimTemplateName(parentName, serviceName)

	template := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}

	if !isGMSEnabled(component) {
		return template, true, nil
	}

	gpuCount, err := getGPUCount(component)
	if err != nil {
		return nil, false, fmt.Errorf("failed to get GPU count for ResourceClaimTemplate: %w", err)
	}

	deviceClassName := getDeviceClassName(component)

	// Verify the DeviceClass exists before creating the template
	if cl != nil {
		dc := &resourcev1.DeviceClass{}
		if err := cl.Get(ctx, types.NamespacedName{Name: deviceClassName}, dc); err != nil {
			return nil, false, fmt.Errorf(
				"DeviceClass %q not found: ensure the GPU DRA driver is installed and the device class is registered: %w",
				deviceClassName, err)
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
