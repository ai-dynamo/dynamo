package dynamo

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
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

	var workerRank string
	var workerRankNeedsShell bool
	if role == RoleWorker {
		workerRank, workerRankNeedsShell = multinodeDeployer.GetNodeRank()
	}
	mutations := sglangmutation.Multinode(sglangmutation.MultinodeValues{
		NumberOfNodes:        numberOfNodes,
		DistributedInitAddr:  fmt.Sprintf("%s:%s", multinodeDeployer.GetLeaderHostname(serviceName), SglangPort),
		WorkerRank:           workerRank,
		WorkerRankNeedsShell: workerRankNeedsShell,
	})
	return mutations.Apply(component, role, container)
}

func (b *SGLangBackend) UpdatePodSpec(*corev1.PodSpec, int32, Role, *v1beta1.DynamoComponentDeploymentSharedSpec, string, MultinodeDeployer) error {
	return nil
}
