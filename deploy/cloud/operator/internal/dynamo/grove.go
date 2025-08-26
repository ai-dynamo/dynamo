package dynamo

import (
	"context"
	"fmt"
	"strings"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
)

type GroveMultinodeDeployer struct {
	MultinodeDeployer
}

func (d *GroveMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	return fmt.Sprintf("${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-%s-%s-0.${GROVE_HEADLESS_SERVICE}", serviceName, commonconsts.GroveRoleSuffixLeader)
}

func (d *GroveMultinodeDeployer) GetNodeRank() string {
	return "$((GROVE_PCLQ_POD_INDEX + 1))"
}

func (d *GroveMultinodeDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, 0, numberOfNodes)
	leaderHostname := d.GetLeaderHostname(serviceName)
	hostnames = append(hostnames, leaderHostname)
	// Add worker hostnames
	for i := int32(0); i < numberOfNodes-1; i++ {
		workerHostname := fmt.Sprintf("${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-%s-%s-%d.${GROVE_HEADLESS_SERVICE}",
			serviceName, commonconsts.GroveRoleSuffixWorker, i)
		hostnames = append(hostnames, workerHostname)
	}
	return hostnames
}

// EvaluateAllComponentsReady determines if all Grove components are ready
// - PodCliques: spec.replicas == status.readyReplicas
// - PodCliqueScalingGroups: spec.replicas == status.availableReplicas
func EvaluateAllComponentsReady(ctx context.Context, client client.Client, dgd *nvidiacomv1alpha1.DynamoGraphDeployment) (bool, string) {
	logger := log.FromContext(ctx)
	var notReadyComponents []string

	for serviceName, component := range dgd.Spec.Services {
		numberOfNodes := component.GetNumberOfNodes()
		isMultinode := numberOfNodes > 1
		resourceName := fmt.Sprintf("%s-0-%s", dgd.Name, strings.ToLower(serviceName))

		if isMultinode {
			// Check PodCliqueScalingGroup: spec.replicas == status.availableReplicas
			if ok, reason := checkPCSGReady(ctx, client, resourceName, dgd.Namespace, logger); !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("pcsg/%s: %s", resourceName, reason))
			}
		} else {
			// Check PodClique: spec.replicas == status.readyReplicas
			if ok, reason := checkPodCliqueReady(ctx, client, resourceName, dgd.Namespace, logger); !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("podclique/%s: %s", resourceName, reason))
			}
		}
	}

	if len(notReadyComponents) > 0 {
		return false, strings.Join(notReadyComponents, "; ")
	}

	return true, ""
}

// checkPodCliqueReady checks if a PodClique has spec.replicas == status.readyReplicas
func checkPodCliqueReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string) {
	podClique := &grovev1alpha1.PodClique{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, podClique)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodClique not found", "resourceName", resourceName)
			return false, "resource not found"
		}
		logger.V(1).Info("Failed to get PodClique", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err)
	}

	desiredReplicas := podClique.Spec.Replicas
	readyReplicas := podClique.Status.ReadyReplicas

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready
		return true, ""
	}

	if desiredReplicas != readyReplicas {
		logger.V(1).Info("PodClique not ready", "resourceName", resourceName, "desired", desiredReplicas, "ready", readyReplicas)
		return false, fmt.Sprintf("desired=%d, ready=%d", desiredReplicas, readyReplicas)
	}

	return true, ""
}

// checkPCSGReady checks if a PodCliqueScalingGroup has spec.replicas == status.availableReplicas
func checkPCSGReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string) {
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, pcsg)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodCliqueScalingGroup not found", "resourceName", resourceName)
			return false, "resource not found"
		}
		logger.V(1).Info("Failed to get PodCliqueScalingGroup", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err)
	}

	desiredReplicas := pcsg.Spec.Replicas
	availableReplicas := pcsg.Status.AvailableReplicas

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready
		return true, ""
	}

	if desiredReplicas != availableReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not ready", "resourceName", resourceName, "desired", desiredReplicas, "available", availableReplicas)
		return false, fmt.Sprintf("desired=%d, available=%d", desiredReplicas, availableReplicas)
	}

	return true, ""
}
