/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

const (
	checkpointInterPodCompatibilityMessage = "Snapshot with gpuMemoryService.mode=InterPod is unsupported"
	checkpointFailoverCompatibilityMessage = "Snapshot with active/passive failover is temporarily unsupported"
)

// ValidateCheckpointCompatibility returns unsupported checkpoint combinations
// in stable policy order.
func ValidateCheckpointCompatibility(experimental *nvidiacomv1beta1.ExperimentalSpec) []error {
	if experimental == nil ||
		experimental.Checkpoint == nil || !experimental.Checkpoint.Enabled {
		return nil
	}

	var violations []error
	if experimental.GPUMemoryService != nil &&
		experimental.GPUMemoryService.Mode == nvidiacomv1beta1.GMSModeInterPod {
		violations = append(violations, errors.New(checkpointInterPodCompatibilityMessage))
	}
	if experimental.Failover != nil {
		violations = append(violations, errors.New(checkpointFailoverCompatibilityMessage))
	}

	return violations
}

// snapshotRestoreEnvironmentNames must match KUBERNETES_REQUIRED_ENV_NAMES,
// KUBERNETES_OPTIONAL_ENV_NAMES, and RESTORE_RUNTIME_ENV_NAMES in
// components/src/dynamo/common/snapshot/constants.py. Those values name the
// destination Pod and runtime, so they are intentionally not part of a portable
// snapshot's compatibility identity.
var snapshotRestoreEnvironmentNames = map[string]struct{}{
	"CONTAINER_NAME":                        {},
	"DYN_COMPONENT":                         {},
	"DYN_DISCOVERY_BACKEND":                 {},
	"DYN_EVENT_PLANE":                       {},
	"DYN_GMS_USE_V1":                        {},
	"DYN_HEALTH_CHECK_ENABLED":              {},
	"DYN_KUBE_DISCOVERY_MODE":               {},
	"DYN_NAMESPACE":                         {},
	"DYN_NAMESPACE_WORKER_SUFFIX":           {},
	"DYN_PARENT_DGD_K8S_NAME":               {},
	"DYN_PARENT_DGD_K8S_NAMESPACE":          {},
	"DYN_REQUEST_PLANE":                     {},
	"DYN_SYSTEM_HEALTH_PATH":                {},
	"DYN_SYSTEM_HOST":                       {},
	"DYN_SYSTEM_LIVE_PATH":                  {},
	"DYN_SYSTEM_PORT":                       {},
	"DYN_SYSTEM_STARTING_HEALTH_STATUS":     {},
	"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": {},
	"DYN_VLLM_GMS_SHADOW_MODE":              {},
	"ENGINE_ID":                             {},
	"ETCD_ENDPOINTS":                        {},
	"FAILOVER_LOCK_PATH":                    {},
	"MODEL_EXPRESS_URL":                     {},
	"NATS_SERVER":                           {},
	"POD_NAME":                              {},
	"POD_NAMESPACE":                         {},
	"POD_UID":                               {},
	"PROMETHEUS_ENDPOINT":                   {},
}

type snapshotCompatibilityContract struct {
	Version               string                     `json:"version"`
	BackendFramework      string                     `json:"backendFramework"`
	GMSMode               string                     `json:"gmsMode"`
	GMSDeviceClassName    string                     `json:"gmsDeviceClassName,omitempty"`
	TargetContainer       corev1.Container           `json:"targetContainer"`
	InitContainers        []corev1.Container         `json:"initContainers,omitempty"`
	Volumes               []corev1.Volume            `json:"volumes,omitempty"`
	HostNetwork           bool                       `json:"hostNetwork,omitempty"`
	HostPID               bool                       `json:"hostPID,omitempty"`
	HostIPC               bool                       `json:"hostIPC,omitempty"`
	ShareProcessNamespace *bool                      `json:"shareProcessNamespace,omitempty"`
	SecurityContext       *corev1.PodSecurityContext `json:"securityContext,omitempty"`
	RuntimeClassName      *string                    `json:"runtimeClassName,omitempty"`
	// NodeName records only an explicit pod-template pin. The scheduler-assigned
	// node is not present in a PodTemplateSpec and therefore is never hashed.
	NodeName       string                    `json:"nodeName,omitempty"`
	NodeSelector   map[string]string         `json:"nodeSelector,omitempty"`
	NodeAffinity   *corev1.NodeAffinity      `json:"nodeAffinity,omitempty"`
	SchedulerName  string                    `json:"schedulerName,omitempty"`
	ResourceClaims []corev1.PodResourceClaim `json:"resourceClaims,omitempty"`
}

// ComputeSnapshotCompatibilityHash returns the portable v2 compatibility
// identity for one captured process. It deliberately excludes rollout and
// graph-incarnation identity, along with restore-time environment that Dynamo
// refreshes in the destination Pod.
func ComputeSnapshotCompatibilityHash(
	podTemplate *corev1.PodTemplateSpec,
	targetContainerName string,
	backendFramework string,
	gmsMode string,
	gmsDeviceClassName string,
) (string, error) {
	if podTemplate == nil {
		return "", fmt.Errorf("snapshot compatibility pod template is required")
	}
	if targetContainerName == "" {
		return "", fmt.Errorf("snapshot compatibility target container is required")
	}

	var target *corev1.Container
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name == targetContainerName {
			target = podTemplate.Spec.Containers[i].DeepCopy()
			break
		}
	}
	if target == nil {
		return "", fmt.Errorf("snapshot compatibility target container %q not found", targetContainerName)
	}

	contract := snapshotCompatibilityContract{
		Version:               consts.SnapshotCompatibilityVersion,
		BackendFramework:      backendFramework,
		GMSMode:               gmsMode,
		GMSDeviceClassName:    gmsDeviceClassName,
		TargetContainer:       canonicalSnapshotContainer(*target, false),
		HostNetwork:           podTemplate.Spec.HostNetwork,
		HostPID:               podTemplate.Spec.HostPID,
		HostIPC:               podTemplate.Spec.HostIPC,
		ShareProcessNamespace: podTemplate.Spec.ShareProcessNamespace,
		SecurityContext:       podTemplate.Spec.SecurityContext,
		RuntimeClassName:      podTemplate.Spec.RuntimeClassName,
		NodeName:              podTemplate.Spec.NodeName,
		NodeSelector:          podTemplate.Spec.NodeSelector,
		SchedulerName:         podTemplate.Spec.SchedulerName,
		ResourceClaims:        podTemplate.Spec.ResourceClaims,
	}
	if podTemplate.Spec.Affinity != nil {
		contract.NodeAffinity = podTemplate.Spec.Affinity.NodeAffinity
	}
	contract.ResourceClaims = append([]corev1.PodResourceClaim(nil), contract.ResourceClaims...)
	sort.Slice(contract.ResourceClaims, func(i, j int) bool {
		return contract.ResourceClaims[i].Name < contract.ResourceClaims[j].Name
	})
	for _, container := range podTemplate.Spec.InitContainers {
		contract.InitContainers = append(contract.InitContainers, canonicalSnapshotContainer(container, true))
	}
	contract.Volumes = snapshotCompatibilityVolumes(&podTemplate.Spec, target)
	sort.Slice(contract.Volumes, func(i, j int) bool { return contract.Volumes[i].Name < contract.Volumes[j].Name })

	data, err := json.Marshal(contract)
	if err != nil {
		return "", fmt.Errorf("marshal snapshot compatibility contract: %w", err)
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

func snapshotCompatibilityVolumes(podSpec *corev1.PodSpec, target *corev1.Container) []corev1.Volume {
	referenced := make(map[string]struct{})
	addContainerVolumes := func(container *corev1.Container) {
		for _, mount := range container.VolumeMounts {
			referenced[mount.Name] = struct{}{}
		}
		for _, device := range container.VolumeDevices {
			referenced[device.Name] = struct{}{}
		}
	}
	addContainerVolumes(target)
	for i := range podSpec.InitContainers {
		addContainerVolumes(&podSpec.InitContainers[i])
	}

	volumes := make([]corev1.Volume, 0, len(referenced))
	for _, volume := range podSpec.Volumes {
		if _, used := referenced[volume.Name]; used {
			volumes = append(volumes, *volume.DeepCopy())
		}
	}
	return volumes
}

func canonicalSnapshotContainer(container corev1.Container, keepName bool) corev1.Container {
	container = *container.DeepCopy()
	if !keepName {
		container.Name = ""
	}

	env := make([]corev1.EnvVar, 0, len(container.Env))
	for _, variable := range container.Env {
		if _, restored := snapshotRestoreEnvironmentNames[variable.Name]; !restored {
			env = append(env, variable)
		}
	}
	container.Env = env
	sort.SliceStable(container.VolumeMounts, func(i, j int) bool {
		if container.VolumeMounts[i].MountPath != container.VolumeMounts[j].MountPath {
			return container.VolumeMounts[i].MountPath < container.VolumeMounts[j].MountPath
		}
		return container.VolumeMounts[i].Name < container.VolumeMounts[j].Name
	})
	sort.SliceStable(container.VolumeDevices, func(i, j int) bool {
		if container.VolumeDevices[i].DevicePath != container.VolumeDevices[j].DevicePath {
			return container.VolumeDevices[i].DevicePath < container.VolumeDevices[j].DevicePath
		}
		return container.VolumeDevices[i].Name < container.VolumeDevices[j].Name
	})
	sort.SliceStable(container.Resources.Claims, func(i, j int) bool {
		if container.Resources.Claims[i].Name != container.Resources.Claims[j].Name {
			return container.Resources.Claims[i].Name < container.Resources.Claims[j].Name
		}
		return container.Resources.Claims[i].Request < container.Resources.Claims[j].Request
	})

	// Health and termination policy affect Kubernetes lifecycle, not whether a
	// captured process image can resume in this container.
	container.Lifecycle = nil
	container.LivenessProbe = nil
	container.ReadinessProbe = nil
	container.StartupProbe = nil
	container.Ports = nil
	container.TerminationMessagePath = ""
	container.TerminationMessagePolicy = ""
	container.ResizePolicy = nil
	container.Stdin = false
	container.StdinOnce = false
	container.TTY = false
	return container
}
