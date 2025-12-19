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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	ctrl "sigs.k8s.io/controller-runtime"
)

type GroveMultinodeDeployer struct {
	MultinodeDeployer
}

func (d *GroveMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	return fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-0.$(GROVE_HEADLESS_SERVICE)", strings.ToLower(serviceName), commonconsts.GroveRoleSuffixLeader)
}

func (d *GroveMultinodeDeployer) GetNodeRank() (string, bool) {
	// This requires shell expansion for arithmetic expression
	return "$((GROVE_PCLQ_POD_INDEX + 1))", true
}

func (d *GroveMultinodeDeployer) NeedsDNSWait() bool {
	// Grove doesn't need DNS wait - it handles startup coordination differently
	return false
}

func (d *GroveMultinodeDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, 0, numberOfNodes)
	leaderHostname := d.GetLeaderHostname(serviceName)
	hostnames = append(hostnames, leaderHostname)
	// Add worker hostnames
	for i := int32(0); i < numberOfNodes-1; i++ {
		workerHostname := fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-%d.$(GROVE_HEADLESS_SERVICE)",
			strings.ToLower(serviceName), commonconsts.GroveRoleSuffixWorker, i)
		hostnames = append(hostnames, workerHostname)
	}
	return hostnames
}

// GetComponentReadinessAndServiceReplicaStatuses determines if all Grove components are ready
// and returns the service replica statuses for each component.
// - PodCliques: spec.replicas == status.readyReplicas
// - PodCliqueScalingGroups: spec.replicas == status.availableReplicas
func GetComponentReadinessAndServiceReplicaStatuses(ctx context.Context, client client.Client, dgd *nvidiacomv1alpha1.DynamoGraphDeployment) (bool, string, map[string]v1alpha1.ServiceReplicaStatus) {
	logger := log.FromContext(ctx)
	var notReadyComponents []string

	serviceStatuses := make(map[string]v1alpha1.ServiceReplicaStatus, len(dgd.Spec.Services))

	for serviceName, component := range dgd.Spec.Services {
		isMultinode := component.GetNumberOfNodes() > 1
		resourceName := fmt.Sprintf("%s-0-%s", dgd.Name, strings.ToLower(serviceName))

		if isMultinode {
			// Check PodCliqueScalingGroup: spec.replicas == status.availableReplicas
			ok, reason, serviceStatus := checkPCSGReady(ctx, client, resourceName, dgd.Namespace, logger)
			serviceStatuses[serviceName] = serviceStatus
			if !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("pcsg/%s: %s", resourceName, reason))
			}
		} else {
			// Check PodClique: spec.replicas == status.readyReplicas
			ok, reason, serviceStatus := checkPodCliqueReady(ctx, client, resourceName, dgd.Namespace, logger)
			serviceStatuses[serviceName] = serviceStatus
			if !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("podclique/%s: %s", resourceName, reason))
			}
		}
	}

	if len(notReadyComponents) > 0 {
		return false, strings.Join(notReadyComponents, "; "), serviceStatuses
	}

	return true, "", serviceStatuses
}

// checkPodCliqueReady checks if a PodClique has spec.replicas == status.readyReplicas
func checkPodCliqueReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1alpha1.ServiceReplicaStatus) {
	podClique := &grovev1alpha1.PodClique{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, podClique)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodClique not found", "resourceName", resourceName)
			return false, "resource not found", v1alpha1.ServiceReplicaStatus{}
		}
		logger.V(1).Info("Failed to get PodClique", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err), v1alpha1.ServiceReplicaStatus{}
	}

	desiredReplicas := podClique.Spec.Replicas
	readyReplicas := podClique.Status.ReadyReplicas

	serviceStatus := v1alpha1.ServiceReplicaStatus{
		ComponentKind:   v1alpha1.ComponentKindPodClique,
		ComponentName:   resourceName,
		Replicas:        podClique.Status.Replicas,
		UpdatedReplicas: podClique.Status.UpdatedReplicas,
		ReadyReplicas:   &readyReplicas,
	}

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready
		return true, "", serviceStatus
	}

	if desiredReplicas != readyReplicas {
		logger.V(1).Info("PodClique not ready", "resourceName", resourceName, "desired", desiredReplicas, "ready", readyReplicas)
		return false, fmt.Sprintf("desired=%d, ready=%d", desiredReplicas, readyReplicas), serviceStatus
	}

	return true, "", serviceStatus
}

// checkPCSGReady checks if a PodCliqueScalingGroup has spec.replicas == status.availableReplicas
func checkPCSGReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1alpha1.ServiceReplicaStatus) {
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, pcsg)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodCliqueScalingGroup not found", "resourceName", resourceName)
			return false, "resource not found", v1alpha1.ServiceReplicaStatus{}
		}
		logger.V(1).Info("Failed to get PodCliqueScalingGroup", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err), v1alpha1.ServiceReplicaStatus{}
	}

	desiredReplicas := pcsg.Spec.Replicas
	availableReplicas := pcsg.Status.AvailableReplicas

	serviceStatus := v1alpha1.ServiceReplicaStatus{
		ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
		ComponentName:     resourceName,
		Replicas:          pcsg.Status.Replicas,
		UpdatedReplicas:   pcsg.Status.UpdatedReplicas,
		AvailableReplicas: &availableReplicas,
	}

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready
		return true, "", serviceStatus
	}

	if desiredReplicas != availableReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not ready", "resourceName", resourceName, "desired", desiredReplicas, "available", availableReplicas)
		return false, fmt.Sprintf("desired=%d, available=%d", desiredReplicas, availableReplicas), serviceStatus
	}

	return true, "", serviceStatus
}

// resolveKaiSchedulerQueueName extracts the queue name from annotations or returns default
// This is the shared logic between DetermineKaiSchedulerQueue and ResolveKaiSchedulerQueue
func resolveKaiSchedulerQueueName(annotations map[string]string) string {
	queueName := commonconsts.DefaultKaiSchedulerQueue
	if annotations != nil {
		if annotationQueue, exists := annotations[commonconsts.KubeAnnotationKaiSchedulerQueue]; exists && strings.TrimSpace(annotationQueue) != "" {
			queueName = strings.TrimSpace(annotationQueue)
		}
	}
	return queueName
}

// ensureQueueExists validates that a Queue resource with the given name exists in the cluster
// Returns an error if the queue doesn't exist or if validation fails
func ensureQueueExists(ctx context.Context, dynamicClient dynamic.Interface, queueName string) error {
	logger := log.FromContext(ctx)

	// Try to get the queue resource using the predefined GVR
	_, err := dynamicClient.Resource(commonconsts.QueueGVR).Get(ctx, queueName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			logger.Error(err, "Queue not found", "queueName", queueName)
			return fmt.Errorf("queue '%s' not found in cluster. Ensure the queue exists before using kai-scheduler", queueName)
		}
		logger.Error(err, "Failed to validate queue", "queueName", queueName)
		return fmt.Errorf("failed to validate queue '%s': %w", queueName, err)
	}

	logger.Info("Queue validation successful", "queueName", queueName)
	return nil
}

// DetermineKaiSchedulerQueue determines the queue name for kai-scheduler from deployment annotations or returns default
// Also validates that the queue exists in the cluster
func DetermineKaiSchedulerQueue(ctx context.Context, annotations map[string]string) (string, error) {
	// Get the queue name from annotation or use default
	queueName := resolveKaiSchedulerQueueName(annotations)

	// Create a dynamic client for CRD validation (Queue CRD might not be in the standard client scheme)
	cfg, err := ctrl.GetConfig()
	if err != nil {
		return "", fmt.Errorf("failed to get kubernetes config for queue validation: %w", err)
	}

	dynamicClient, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to create dynamic client for queue validation: %w", err)
	}

	// Validate that the queue exists
	if err := ensureQueueExists(ctx, dynamicClient, queueName); err != nil {
		return "", fmt.Errorf("kai-scheduler queue validation failed: %w", err)
	}

	return queueName, nil
}

// ResolveKaiSchedulerQueue determines the queue name for kai-scheduler from deployment annotations or returns default
// Does NOT validate - use DetermineKaiSchedulerQueue for validation
func ResolveKaiSchedulerQueue(annotations map[string]string) string {
	return resolveKaiSchedulerQueueName(annotations)
}

// injectKaiSchedulerIfEnabled injects kai-scheduler settings into a clique if kai-scheduler is enabled and grove is enabled
func injectKaiSchedulerIfEnabled(
	clique *grovev1alpha1.PodCliqueTemplateSpec,
	controllerConfig controller_common.Config,
	validatedQueueName string,
) {
	// Only proceed if grove is enabled, kai-scheduler is enabled, and no manual schedulerName is set
	if !controllerConfig.Grove.Enabled || !controllerConfig.KaiScheduler.Enabled {
		return
	}

	// Check if user has manually set schedulerName - if so, respect their choice
	if clique.Spec.PodSpec.SchedulerName != "" && clique.Spec.PodSpec.SchedulerName != commonconsts.KaiSchedulerName {
		return
	}

	// Use the pre-validated queue name
	queueName := validatedQueueName

	// Inject schedulerName
	clique.Spec.PodSpec.SchedulerName = commonconsts.KaiSchedulerName

	// Inject queue label
	if clique.Labels == nil {
		clique.Labels = make(map[string]string)
	}
	clique.Labels[commonconsts.KubeLabelKaiSchedulerQueue] = queueName
}
