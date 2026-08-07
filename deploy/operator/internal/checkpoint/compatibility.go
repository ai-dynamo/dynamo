/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"errors"
	"fmt"
	"reflect"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// IsDGDControlled reports whether a DCD has an exact controller owner
// reference to a live-identity DynamoGraphDeployment.
func IsDGDControlled(dcd *nvidiacomv1beta1.DynamoComponentDeployment) bool {
	if dcd == nil {
		return false
	}
	owner := metav1.GetControllerOf(dcd)
	return owner != nil &&
		owner.APIVersion == nvidiacomv1beta1.GroupVersion.String() &&
		owner.Kind == "DynamoGraphDeployment" &&
		owner.Name != "" &&
		owner.UID != ""
}

const (
	checkpointInterPodCompatibilityMessage = "Snapshot with gpuMemoryService.mode=InterPod is unsupported"
	checkpointFailoverCompatibilityMessage = "Snapshot with active/passive failover requires an operator-managed automatic vLLM Worker checkpoint"
)

// HasCheckpointEnabledFailover reports whether checkpoint failover policy
// requires a resolved backend.
func HasCheckpointEnabledFailover(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) bool {
	return component != nil &&
		component.Experimental != nil &&
		component.Experimental.Checkpoint != nil &&
		component.Experimental.Checkpoint.Enabled &&
		component.Experimental.Failover != nil
}

// CompatibilityContext identifies which controller owns a checkpoint+failover
// component and therefore whether checkpointRef must be absent or present.
type CompatibilityContext int

const (
	compatibilityContextInvalid CompatibilityContext = iota
	CompatibilityContextDGDSource
	CompatibilityContextGeneratedDCD
	CompatibilityContextStandaloneDCD
)

// ValidateCheckpointCompatibility returns unsupported checkpoint combinations
// in stable policy order.
func ValidateCheckpointCompatibility(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework string,
	context CompatibilityContext,
) []error {
	if component == nil || component.Experimental == nil {
		return nil
	}
	experimental := component.Experimental
	config := experimental.Checkpoint
	if config == nil || !config.Enabled {
		return nil
	}

	var violations []error
	if experimental.GPUMemoryService != nil &&
		experimental.GPUMemoryService.Mode == nvidiacomv1beta1.GMSModeInterPod {
		violations = append(violations, errors.New(checkpointInterPodCompatibilityMessage))
	}
	if experimental.Failover == nil {
		return violations
	}

	var profile []error
	hasRef := config.CheckpointRef != nil && *config.CheckpointRef != ""
	switch context {
	case CompatibilityContextDGDSource:
		if hasRef {
			profile = append(profile, errors.New("checkpointRef must be omitted for the DGD-managed automatic checkpoint"))
		}
	case CompatibilityContextGeneratedDCD:
		if !hasRef {
			profile = append(profile, errors.New("checkpointRef must name the DGD-managed automatic checkpoint"))
		}
	case CompatibilityContextStandaloneDCD:
		profile = append(profile, errors.New("checkpoint failover is only supported for an operator-generated DCD"))
	default:
		profile = append(profile, errors.New("checkpoint failover compatibility context is invalid"))
	}
	if config.Mode != "" && config.Mode != nvidiacomv1beta1.CheckpointModeAuto {
		profile = append(profile, errors.New("checkpoint mode must be automatic"))
	}
	if config.DeletionPolicy != "" &&
		config.DeletionPolicy != nvidiacomv1beta1.CheckpointDeletionPolicyDelete {
		profile = append(profile, errors.New("deletionPolicy must be Delete"))
	}
	if config.TargetContainerName != "" &&
		config.TargetContainerName != nvidiacomv1beta1.MainContainerName {
		profile = append(profile, errors.New("targetContainerName must be main"))
	}
	if backendFramework != "vllm" {
		profile = append(profile, errors.New("backendFramework must be vllm"))
	}
	if component.ComponentType != nvidiacomv1beta1.ComponentTypeWorker {
		profile = append(profile, errors.New("component type must be Worker"))
	}
	if experimental.GPUMemoryService == nil ||
		(experimental.GPUMemoryService.Mode != "" &&
			experimental.GPUMemoryService.Mode != nvidiacomv1beta1.GMSModeIntraPod) {
		profile = append(profile, errors.New("gpuMemoryService.mode must be IntraPod"))
	}
	if experimental.Failover.Mode != "" &&
		experimental.Failover.Mode != nvidiacomv1beta1.GMSModeIntraPod {
		profile = append(profile, errors.New("failover.mode must be IntraPod"))
	}
	numShadows := experimental.Failover.NumShadows
	if numShadows == 0 {
		numShadows = 1
	}
	if numShadows < 1 || numShadows > 2 {
		profile = append(profile, errors.New("failover.numShadows must be 1 or 2"))
	}
	if component.GetNumberOfNodes() != 1 {
		profile = append(profile, errors.New("multinode/model-parallel worker topology is unsupported"))
	}
	if mainGPUCount(component) != 1 {
		profile = append(profile, errors.New("main container must request exactly one GPU"))
	}
	profile = append(profile, validateAutomaticFailoverCheckpointJob(config.Job)...)
	if len(profile) != 0 {
		violations = append(violations, fmt.Errorf(
			"%s: %w",
			checkpointFailoverCompatibilityMessage,
			errors.Join(profile...),
		))
	}
	return violations
}

func mainGPUCount(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) int64 {
	if component.PodTemplate == nil {
		return 0
	}
	for i := range component.PodTemplate.Spec.Containers {
		container := &component.PodTemplate.Spec.Containers[i]
		if container.Name != nvidiacomv1beta1.MainContainerName {
			continue
		}
		gpu := container.Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]
		return gpu.Value()
	}
	return 0
}

func validateAutomaticFailoverCheckpointJob(
	job *nvidiacomv1beta1.ComponentCheckpointJobConfig,
) []error {
	if job == nil {
		return nil
	}
	if job.PodTemplate == nil {
		if len(job.GMSClientContainers) == 0 {
			return nil
		}
		return []error{errors.New("checkpoint.job GMS helpers require podTemplate")}
	}

	spec := job.PodTemplate.Spec
	containers := spec.Containers
	spec.Containers = nil
	if !reflect.DeepEqual(spec, corev1.PodSpec{}) {
		return []error{errors.New("checkpoint.job may only customize metadata and helper containers")}
	}

	clients := make(map[string]struct{}, len(job.GMSClientContainers))
	for _, name := range job.GMSClientContainers {
		clients[name] = struct{}{}
	}
	var violations []error
	for _, container := range containers {
		if container.Name == nvidiacomv1beta1.MainContainerName {
			violations = append(violations, errors.New("checkpoint.job must not override main"))
			continue
		}
		if _, ok := clients[container.Name]; !ok {
			violations = append(violations, fmt.Errorf(
				"checkpoint.job helper %q must be declared in gmsClientContainers",
				container.Name,
			))
			continue
		}
		delete(clients, container.Name)
	}
	for name := range clients {
		violations = append(violations, fmt.Errorf(
			"checkpoint.job GMS client helper %q is missing from podTemplate",
			name,
		))
	}
	return violations
}
