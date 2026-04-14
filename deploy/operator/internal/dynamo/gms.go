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
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

const (
	gmsSharedVolumeName      = "gms-shared"
	gmsSharedMountPath       = "/shared"
	gmsDRAClaimName          = "shared-gpu"
	defaultDeviceClassName   = "gpu.nvidia.com"
	gmsProcessesPerGPU       = 2
	gmsStartupProbeTimeout   = 2 * time.Minute
	gmsStartupProbePeriodSec = 2
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
// It reads from GPUMemoryServiceSpec.DeviceClassName, falling back to the default.
func getDeviceClassName(component *v1alpha1.DynamoComponentDeploymentSharedSpec) string {
	if component.GPUMemoryService != nil && component.GPUMemoryService.DeviceClassName != "" {
		return component.GPUMemoryService.DeviceClassName
	}
	return defaultDeviceClassName
}

// applyGPUMemoryService transforms a pod spec to include a GMS sidecar with
// DRA shared GPU access. The main container's GPU resources are replaced with
// a DRA ResourceClaim, and a GMS init container is added.
//
// claimTemplateName is the name of the ResourceClaimTemplate that will provide
// shared GPU access; callers should compute it via GMSResourceClaimTemplateName.
func applyGPUMemoryService(
	podSpec *corev1.PodSpec,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
	claimTemplateName string,
) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for GPU memory service")
	}

	gpuCount, err := getGPUCount(component)
	if err != nil {
		return err
	}

	mainContainer := &podSpec.Containers[0]

	// Replace GPU resources with DRA claim on main container
	removeGPUResources(mainContainer)
	mainContainer.Resources.Claims = append(mainContainer.Resources.Claims, corev1.ResourceClaim{
		Name: gmsDRAClaimName,
	})

	// Add shared volume mount and TMPDIR to main container
	mainContainer.VolumeMounts = append(mainContainer.VolumeMounts, corev1.VolumeMount{
		Name:      gmsSharedVolumeName,
		MountPath: gmsSharedMountPath,
	})
	mainContainer.Env = append(mainContainer.Env, corev1.EnvVar{
		Name: "TMPDIR", Value: gmsSharedMountPath,
	})

	// Add GMS sidecar
	gmsSidecar := buildGMSSidecar(mainContainer.Image, gpuCount)
	podSpec.InitContainers = append(podSpec.InitContainers, gmsSidecar)

	// Add shared volume
	podSpec.Volumes = append(podSpec.Volumes, gmsSharedVolume())

	// GPU nodes are typically tainted with nvidia.com/gpu=NoSchedule. With
	// traditional scheduling the device-plugin injects the matching toleration,
	// but DRA bypasses that path. Re-add the toleration explicitly so the pod
	// can schedule on GPU nodes.
	podSpec.Tolerations = append(podSpec.Tolerations, corev1.Toleration{
		Key:      commonconsts.KubeResourceGPUNvidia,
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	})

	// Add pod-level DRA resource claim referencing the ResourceClaimTemplate
	podSpec.ResourceClaims = append(podSpec.ResourceClaims, corev1.PodResourceClaim{
		Name:                      gmsDRAClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	})

	return nil
}

// removeGPUResources strips nvidia.com/gpu from container resource limits and requests.
// GPU allocation is handled by DRA when GMS is enabled.
func removeGPUResources(container *corev1.Container) {
	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	if container.Resources.Limits != nil {
		delete(container.Resources.Limits, gpuResource)
	}
	if container.Resources.Requests != nil {
		delete(container.Resources.Requests, gpuResource)
	}
}

// buildGMSSidecar creates the GMS weight server as a sidecar init container
// (restartPolicy: Always). kubelet starts it before regular containers and
// keeps it running for the pod's lifetime.
//
// Each GPU gets two GMS subprocesses (weights + kv_cache) via a bash wrapper
// that forwards signals and exits if any child dies. TMPDIR is set so
// UUID-based sockets land in the shared volume.
func buildGMSSidecar(image string, gpuCount int) corev1.Container {
	return corev1.Container{
		Name:          "gms-weights",
		Image:         image,
		Command:       []string{"bash", "-c"},
		Args:          []string{gmsWrapperScript(gpuCount)},
		RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
		Env: []corev1.EnvVar{
			{Name: "TMPDIR", Value: gmsSharedMountPath},
		},
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      gmsSharedVolumeName,
				MountPath: gmsSharedMountPath,
			},
		},
		StartupProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: gmsReadyCheckCommand(gpuCount),
				},
			},
			PeriodSeconds:    int32(gmsStartupProbePeriodSec),
			FailureThreshold: int32(gmsStartupProbeTimeout/time.Second) / int32(gmsStartupProbePeriodSec),
		},
		Resources: corev1.ResourceRequirements{
			Claims: []corev1.ResourceClaim{
				{Name: gmsDRAClaimName},
			},
		},
	}
}

// gmsWrapperScript generates a bash script that launches two GMS subprocesses
// per GPU device (one for weights, one for kv_cache), waits for any to exit,
// then tears down the process group.
func gmsWrapperScript(gpuCount int) string {
	devList := make([]string, gpuCount)
	for i := range gpuCount {
		devList[i] = strconv.Itoa(i)
	}
	return fmt.Sprintf(
		`trap 'kill 0 2>/dev/null || true' EXIT
for dev in %s; do
  python3 -m gpu_memory_service --device "$dev" --tag weights &
  echo "Started GMS device=$dev tag=weights pid=$!"
  python3 -m gpu_memory_service --device "$dev" --tag kv_cache &
  echo "Started GMS device=$dev tag=kv_cache pid=$!"
done
wait -n
echo "A GMS subprocess exited, shutting down"`, strings.Join(devList, " "))
}

// gmsReadyCheckCommand returns the exec probe command that verifies the
// expected number of GMS UDS sockets exist on the shared volume.
// With 2-tag GMS (weights + kv_cache), there are 2 sockets per GPU.
func gmsReadyCheckCommand(gpuCount int) []string {
	return []string{
		"sh", "-c",
		fmt.Sprintf("test $(ls %s/gms_*.sock 2>/dev/null | wc -l) -ge %d", gmsSharedMountPath, gpuCount*gmsProcessesPerGPU),
	}
}

func gmsSharedVolume() corev1.Volume {
	return corev1.Volume{
		Name: gmsSharedVolumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}
}

// GMSResourceClaimTemplateName returns the deterministic name for the
// ResourceClaimTemplate associated with a GMS-enabled component.
func GMSResourceClaimTemplateName(parentName, serviceName string) string {
	return fmt.Sprintf("%s-%s-gpu", parentName, strings.ToLower(serviceName))
}

// GenerateGMSResourceClaimTemplate builds the ResourceClaimTemplate that
// provides shared GPU access to all containers in a GMS-enabled pod via DRA.
//
// claimTemplateName is the deterministic name for the template; callers should
// compute it via GMSResourceClaimTemplateName.
//
// When GMS is not enabled for the component, it returns the template skeleton
// with toDelete=true so that SyncResource cleans up any previously created template.
//
// The cl parameter is used to verify the DeviceClass exists before creating
// the template. Pass nil to skip the DeviceClass check.
func GenerateGMSResourceClaimTemplate(
	ctx context.Context,
	cl client.Client,
	claimTemplateName, namespace string,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
) (*resourcev1.ResourceClaimTemplate, bool, error) {
	template := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      claimTemplateName,
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
