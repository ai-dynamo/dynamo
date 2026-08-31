package dynamo

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
	sglangmutation "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation/sglang"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	SglangPort = "29500"
)

type SGLangBackend struct{}

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

	mutations := mutation.Concat(
		sglangmutation.MultinodePodWiring(),
		sglangAutomaticMutations(component, numberOfNodes, role, serviceName, multinodeDeployer),
	)
	return mutations.Apply(component, role, container)
}

func sglangAutomaticMutations(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	numberOfNodes int32,
	role Role,
	serviceName string,
	multinodeDeployer MultinodeDeployer,
) mutation.EngineMutations {
	if IsManualFlagsInjection(component) {
		return nil
	}

	var workerRank string
	var workerRankNeedsShell bool
	if role == RoleWorker {
		workerRank, workerRankNeedsShell = multinodeDeployer.GetNodeRank()
	}
	return sglangmutation.AutomaticMultinode(sglangmutation.MultinodeValues{
		NumberOfNodes:        numberOfNodes,
		DistributedInitAddr:  fmt.Sprintf("%s:%s", multinodeDeployer.GetLeaderHostname(serviceName), SglangPort),
		WorkerRank:           workerRank,
		WorkerRankNeedsShell: workerRankNeedsShell,
	})
}

func (b *SGLangBackend) UpdatePodSpec(*corev1.PodSpec, int32, Role, *v1beta1.DynamoComponentDeploymentSharedSpec, string, MultinodeDeployer) error {
	return nil
}
