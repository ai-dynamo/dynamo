package dynamo

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/render"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	SglangPort = "29500"
)

type SGLangBackend struct{}

func isPythonCommand(command string) bool {
	return render.IsPythonCommand(command)
}

func (b *SGLangBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer, _ ContainerGPUCount) error {
	if component.CompilationCache != nil {
		logger := log.Log.WithName("sglang-backend")
		logger.Info("Compilation cache configured for SGLang but not yet fully supported",
			"backend", "sglang",
			"status", "partial-support",
			"cache-dir", component.CompilationCache.MountPath,
			"env-vars-set", false,
			"next-steps", "upstream SGLang changes needed")
	}

	// For single node, nothing to do
	if numberOfNodes <= 1 {
		return nil
	}

	if err := (render.EngineMutations{
		{
			Applies: workerRole,
			Mutation: render.RemoveProbesMutation{
				ContainerName: commonconsts.MainContainerName,
				Liveness:      true,
				Readiness:     true,
				Startup:       true,
			},
		},
	}).Apply(component, role, container); err != nil {
		return err
	}

	flags, needsShell := b.getMultinodeFlags(numberOfNodes, role, serviceName, multinodeDeployer)
	return (render.EngineMutations{
		{
			Mutation: render.AddFlagsMutation{
				ContainerName: commonconsts.MainContainerName,
				Flags:         flags,
				NeedsShell:    needsShell,
				Framework:     "sglang",
			},
		},
	}).Apply(component, role, container)
}

func (b *SGLangBackend) UpdatePodSpec(*corev1.PodSpec, int32, Role, *v1beta1.DynamoComponentDeploymentSharedSpec, string, MultinodeDeployer) error {
	return nil
}

// getMultinodeFlags returns the multinode flags and whether shell interpretation is needed
func (b *SGLangBackend) getMultinodeFlags(numberOfNodes int32, role Role, serviceName string, multinodeDeployer MultinodeDeployer) (string, bool) {
	leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)

	var nodeRank string
	var needsShell bool

	if role == RoleLeader {
		nodeRank = "0"
		needsShell = false
	} else {
		nodeRank, needsShell = multinodeDeployer.GetNodeRank()
	}
	distInitAddr := fmt.Sprintf("%s:%s", leaderHostname, SglangPort)

	flags := fmt.Sprintf("--dist-init-addr %s --nnodes %d --node-rank %s", distInitAddr, numberOfNodes, nodeRank)
	return flags, needsShell
}
