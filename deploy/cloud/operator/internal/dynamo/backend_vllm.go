package dynamo

import (
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

type VLLMBackend struct{}

func (b *VLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) {
	isMultinode := numberOfNodes > 1

	if isMultinode {
		// Apply multinode-specific argument modifications
		updateVLLMMultinodeArgs(container, role, multinodeDeploymentType, serviceName)

		// Remove probes for multinode worker and leader
		if role == RoleWorker || role == RoleLeader {
			container.LivenessProbe = nil
			container.ReadinessProbe = nil
			container.StartupProbe = nil
		}
	}
}

func (b *VLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) {
	// do nothing
}

// updateVLLMMultinodeArgs applies Ray-specific modifications for multinode deployments
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) {
	switch role {
	case RoleLeader:
		if len(container.Args) > 0 {
			// Prepend ray start --head command to existing args
			container.Args = []string{fmt.Sprintf("ray start --head --port=6379 && %s", strings.Join(container.Args, " "))}
		}
	case RoleWorker:
		// Worker nodes only run Ray, completely replace args
		if multinodeDeploymentType == commonconsts.MultinodeDeploymentTypeGrove {
			leaderHostname := generateGroveLeaderHostname(serviceName)
			container.Args = []string{fmt.Sprintf("ray start --address=%s:6379 --block", leaderHostname)}
		} else {
			container.Args = []string{"ray start --address=${LWS_LEADER_ADDRESS}:6379 --block"}
		}
	}
}
