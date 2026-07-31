package dynamo

import (
	"context"
	"fmt"
	"strings"

	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	ctrl "sigs.k8s.io/controller-runtime"
)

// legacyGroveConditionReasonInsufficientScheduledPCSGReplicas can remain on
// persisted PCSG status after upgrading from Grove versions that emitted it.
const legacyGroveConditionReasonInsufficientScheduledPCSGReplicas = "InsufficientScheduledPodCliqueScalingGroupReplicas"

type GroveMultinodeDeployer struct {
	MultinodeDeployer
	// IsInterPodGMS is true when this deployer produces pod specs for an
	// engine PCLQ that uses the inter-pod GMS *layout* (one engine pod per
	// rank, per shadow, with a dedicated GMS weight server pod). It is a
	// layout/topology flag — not a failover policy flag — and governs how
	// hostnames, node ranks, and per-pod wiring are computed. Today this
	// layout is only produced when inter-pod GMS failover is enabled, but
	// the deployer itself should not encode that assumption.
	IsInterPodGMS bool
	Rank          int32 // explicit node rank (used when IsInterPodGMS is true)
}

func (d *GroveMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	if d.IsInterPodGMS {
		// GMS: each PCLQ has multiple replicas; pods at the same index across
		// ranks form a communication group, so use the dynamic pod index.
		return fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-$(GROVE_PCLQ_POD_INDEX).$(GROVE_HEADLESS_SERVICE)",
			strings.ToLower(serviceName), commonconsts.GroveRoleSuffixLeader)
	}
	return fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-0.$(GROVE_HEADLESS_SERVICE)",
		strings.ToLower(serviceName), commonconsts.GroveRoleSuffixLeader)
}

func (d *GroveMultinodeDeployer) GetNodeRank() (string, bool) {
	if d.IsInterPodGMS {
		return fmt.Sprintf("%d", d.Rank), false
	}
	return "$((GROVE_PCLQ_POD_INDEX + 1))", true
}

func (d *GroveMultinodeDeployer) NeedsDNSWait() bool {
	return false
}

func (d *GroveMultinodeDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, 0, numberOfNodes)
	hostnames = append(hostnames, d.GetLeaderHostname(serviceName))

	if d.IsInterPodGMS {
		for rank := int32(1); rank < numberOfNodes; rank++ {
			hostname := fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-%d-$(GROVE_PCLQ_POD_INDEX).$(GROVE_HEADLESS_SERVICE)",
				strings.ToLower(serviceName), commonconsts.GroveRoleSuffixWorker, rank)
			hostnames = append(hostnames, hostname)
		}
	} else {
		for i := int32(0); i < numberOfNodes-1; i++ {
			hostname := fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-%d.$(GROVE_HEADLESS_SERVICE)",
				strings.ToLower(serviceName), commonconsts.GroveRoleSuffixWorker, i)
			hostnames = append(hostnames, hostname)
		}
	}
	return hostnames
}

// GetComponentReadinessAndServiceReplicaStatuses determines if all Grove components are ready
// and returns the replica statuses for each component.
// - PodCliques: spec.replicas == status.readyReplicas
// - PodCliqueScalingGroups: spec.replicas == status.availableReplicas
func GetComponentReadinessAndServiceReplicaStatuses(ctx context.Context, client client.Client, dgd *v1beta1.DynamoGraphDeployment) (bool, string, map[string]v1beta1.ComponentReplicaStatus, error) {
	allReady, _, message, componentStatuses, err := evaluateGroveComponents(ctx, client, dgd)
	return allReady, message, componentStatuses, err
}

// ClassifyGroveReadiness returns the DGD-level Ready condition reason for a
// Grove-backed DGD one of the v1beta1.DGDReadyReason* constants.
// It performs the same Grove status reads as
// GetComponentReadinessAndServiceReplicaStatuses; callers that need both the
// reason and the readiness/message/component-status detail in the same
// reconcile should prefer calling evaluateGroveComponents directly to avoid
// reading Grove status twice. A non-nil error indicates a transient read
// failure that should be retried, not a classification.
func ClassifyGroveReadiness(ctx context.Context, client client.Client, dgd *v1beta1.DynamoGraphDeployment) (string, error) {
	_, reason, _, _, err := evaluateGroveComponents(ctx, client, dgd)
	return reason, err
}

// evaluateGroveComponents is the single per-component evaluation loop shared
// by GetComponentReadinessAndServiceReplicaStatuses and ClassifyGroveReadiness.
//
// Each Check*Ready call returns the DGD-level Ready reason its component would
// imply (a v1beta1.DGDReadyReason* value) when that component is not ready. The
// reasons are aggregated in place: if every not-ready component implies the
// same reason, that reason is used; if they disagree, the result is
// MixedNotReadyReasons.
func evaluateGroveComponents(ctx context.Context, client client.Client, dgd *v1beta1.DynamoGraphDeployment) (allReady bool, classificationReason string, message string, componentStatuses map[string]v1beta1.ComponentReplicaStatus, err error) {
	logger := log.FromContext(ctx)
	var notReadyComponents []string
	aggregatedReason := ""

	componentStatuses = make(map[string]v1beta1.ComponentReplicaStatus, len(dgd.Spec.Components))

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentName := component.ComponentName
		usesPCSG := component.GetNumberOfNodes() > 1 || component.IsInterPodGMSEnabled()
		resourceName := fmt.Sprintf("%s-0-%s", PCSNameForDGD(dgd.Name, dgd.Spec.Components), strings.ToLower(componentName))

		var ok bool
		var reason string
		var componentStatus v1beta1.ComponentReplicaStatus
		var componentReason string
		var checkErr error

		if usesPCSG {
			ok, reason, componentStatus, componentReason, checkErr = CheckPCSGReady(ctx, client, resourceName, dgd.Namespace, logger)
		} else {
			ok, reason, componentStatus, componentReason, checkErr = CheckPodCliqueReady(ctx, client, resourceName, dgd.Namespace, logger)
		}
		// A non-NotFound read error is a transient failure to determine
		// readiness. Propagate it (rather than folding it into a not-ready
		// result) so the reconcile retries with backoff and does not advance
		// ObservedGeneration on a blip. NotFound is handled inside Check* as a
		// legitimate not-ready state and never surfaces here.
		if checkErr != nil {
			return false, "", "", nil, fmt.Errorf("component %q: %w", componentName, checkErr)
		}
		componentStatus.RuntimeNamespace = dgd.GetDynamoNamespaceForComponent(component)
		componentStatuses[componentName] = componentStatus
		if !ok {
			notReadyComponents = append(notReadyComponents, fmt.Sprintf("%s: %s", componentName, reason))
			switch aggregatedReason {
			case "":
				aggregatedReason = componentReason
			case componentReason:
				// same reason as seen so far; keep it
			default:
				aggregatedReason = v1beta1.DGDReadyReasonMixedNotReadyReasons
			}
		}
	}

	if len(notReadyComponents) > 0 {
		return false, aggregatedReason, strings.Join(notReadyComponents, "; "), componentStatuses, nil
	}

	return true, v1beta1.DGDReadyReasonAllResourcesReady, "", componentStatuses, nil
}

// CheckPodCliqueReady determines if a Grove PodClique is fully ready and available.
// It checks various status fields to ensure all replicas are available and the PodClique
// configuration has been fully applied. This is the PodClique equivalent of IsDeploymentReady
// for standard Kubernetes Deployments.
//
// The returned reason string is the DGD-level Ready reason this component
// implies when it is not ready (a v1beta1.DGDReadyReason* value):
// InsufficientCapacity for a scheduling/capacity blocker, Updating while the
// rollout is unfinished, PodsNotReady when scheduled but not enough replicas
// are ready, or SomeResourcesNotReady when the cause cannot be determined. It
// is empty when the component is ready.
func CheckPodCliqueReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1beta1.ComponentReplicaStatus, string, error) {
	podClique := &grovev1alpha1.PodClique{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, podClique)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodClique not found", "resourceName", resourceName)
			// The backing PodClique is not created yet. Return a valid status
			// entry (with the known kind and expected name) rather than an empty
			// ComponentReplicaStatus{}
			return false, "resource not found", v1beta1.ComponentReplicaStatus{
				ComponentKind:  v1beta1.ComponentKindPodClique,
				ComponentNames: []string{resourceName},
			}, v1beta1.DGDReadyReasonSomeResourcesNotReady, nil
		}
		// A non-NotFound error is a transient failure to determine readiness,
		// not a legitimate not-ready state. Return it so the reconcile retries
		// with backoff and does not advance ObservedGeneration on a blip.
		logger.V(1).Info("Failed to get PodClique", "error", err, "resourceName", resourceName)
		return false, "", v1beta1.ComponentReplicaStatus{}, "", fmt.Errorf("failed to get PodClique %s/%s: %w", namespace, resourceName, err)
	}

	desiredReplicas := podClique.Spec.Replicas
	readyReplicas := podClique.Status.ReadyReplicas
	updatedReplicas := podClique.Status.UpdatedReplicas
	replicas := podClique.Status.Replicas
	scheduledReplicas := podClique.Status.ScheduledReplicas
	scheduleGatedReplicas := podClique.Status.ScheduleGatedReplicas
	observedGeneration := podClique.Status.ObservedGeneration
	generation := podClique.Generation

	logger.V(1).Info("CheckPodCliqueFullyUpdated",
		"resourceName", resourceName,
		"generation", podClique.Generation,
		"observedGeneration", podClique.Status.ObservedGeneration,
		"desiredReplicas", desiredReplicas,
		"readyReplicas", readyReplicas,
		"updatedReplicas", updatedReplicas,
		"replicas", replicas,
		"scheduledReplicas", scheduledReplicas,
		"scheduleGatedReplicas", scheduleGatedReplicas,
	)

	serviceStatus := v1beta1.ComponentReplicaStatus{
		ComponentKind:   v1beta1.ComponentKindPodClique,
		ComponentNames:  []string{resourceName},
		Replicas:        podClique.Status.Replicas,
		UpdatedReplicas: podClique.Status.UpdatedReplicas,
		ReadyReplicas:   &readyReplicas,
	}

	if observedGeneration == nil {
		logger.V(1).Info("PodClique observedGeneration is nil", "resourceName", resourceName)
		return false, "observedGeneration is nil", serviceStatus, v1beta1.DGDReadyReasonSomeResourcesNotReady, nil
	}

	if observedGeneration != nil && *observedGeneration < generation {
		logger.V(1).Info("PodClique spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), serviceStatus, v1beta1.DGDReadyReasonSomeResourcesNotReady, nil
	}

	serviceStatus.ScheduledReplicas = &scheduledReplicas

	if desiredReplicas == 0 {
		return true, "", serviceStatus, "", nil
	}

	// Fully ready: replicas exist, are updated, and are ready. Checked first so
	// a healthy component is never mis-diagnosed as InsufficientCapacity when
	// Grove does not populate scheduledReplicas on a ready PodClique.
	if replicas == desiredReplicas && updatedReplicas == desiredReplicas && readyReplicas == desiredReplicas {
		return true, "", serviceStatus, "", nil
	}

	// Not ready: classify capacity signals, in order of reliability:
	//   1. scheduleGatedReplicas > 0            (explicit gated count)
	//   2. PodCliqueScheduled condition = False  (explicit scheduling signal)
	//   3. 0 < scheduledReplicas < desired       (genuine partial scheduling)
	if scheduleGatedReplicas > 0 {
		logger.V(1).Info("PodClique has schedule-gated replicas", "resourceName", resourceName, "scheduleGated", scheduleGatedReplicas)
		return false, fmt.Sprintf("schedule-gated replicas: %d", scheduleGatedReplicas), serviceStatus, v1beta1.DGDReadyReasonInsufficientCapacity, nil
	}
	if cond := meta.FindStatusCondition(podClique.Status.Conditions, groveconstants.ConditionTypePodCliqueScheduled); cond != nil &&
		cond.Status == metav1.ConditionFalse &&
		cond.Reason == groveconstants.ConditionReasonInsufficientScheduledPods {
		logger.V(1).Info("PodClique scheduling condition reports insufficient capacity", "resourceName", resourceName, "reason", cond.Reason, "message", cond.Message)
		return false, fmt.Sprintf("scheduling condition %s: %s", cond.Reason, cond.Message), serviceStatus, v1beta1.DGDReadyReasonInsufficientCapacity, nil
	}
	if scheduledReplicas > 0 && scheduledReplicas < desiredReplicas {
		logger.V(1).Info("PodClique partially scheduled", "resourceName", resourceName, "desired", desiredReplicas, "scheduled", scheduledReplicas)
		return false, fmt.Sprintf("insufficient scheduled replicas: scheduled=%d/%d", scheduledReplicas, desiredReplicas), serviceStatus, v1beta1.DGDReadyReasonInsufficientCapacity, nil
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodClique not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), serviceStatus, v1beta1.DGDReadyReasonUpdating, nil
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodClique performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), serviceStatus, v1beta1.DGDReadyReasonUpdating, nil
	}

	// Scheduled and rolled out, but not enough ready replicas.
	logger.V(1).Info("PodClique not ready", "resourceName", resourceName, "desired", desiredReplicas, "ready", readyReplicas)
	return false, fmt.Sprintf("scheduled but ready=%d/%d", readyReplicas, desiredReplicas), serviceStatus, v1beta1.DGDReadyReasonPodsNotReady, nil
}

// CheckPCSGReady determines if a Grove PodCliqueScalingGroup is fully ready and available.
// It checks various status fields to ensure all replicas are available and the PCSG
// configuration has been fully applied. This is the PodCliqueScalingGroup equivalent of IsDeploymentReady
// for standard Kubernetes Deployments.
func CheckPCSGReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1beta1.ComponentReplicaStatus, string, error) {
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, pcsg)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodCliqueScalingGroup not found", "resourceName", resourceName)
			// The backing PodCliqueScalingGroup is not created yet. Return a valid
			// status entry (with the known kind and expected name) rather than an
			// empty ComponentReplicaStatus{}
			return false, "resource not found", v1beta1.ComponentReplicaStatus{
				ComponentKind:  v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames: []string{resourceName},
			}, v1beta1.DGDReadyReasonSomeResourcesNotReady, nil
		}
		// A non-NotFound error is a transient failure to determine readiness,
		// not a legitimate not-ready state. Return it so the reconcile retries
		// with backoff and does not advance ObservedGeneration on a blip.
		logger.V(1).Info("Failed to get PodCliqueScalingGroup", "error", err, "resourceName", resourceName)
		return false, "", v1beta1.ComponentReplicaStatus{}, "", fmt.Errorf("failed to get PodCliqueScalingGroup %s/%s: %w", namespace, resourceName, err)
	}

	desiredReplicas := pcsg.Spec.Replicas
	availableReplicas := pcsg.Status.AvailableReplicas
	updatedReplicas := pcsg.Status.UpdatedReplicas
	replicas := pcsg.Status.Replicas
	scheduledReplicas := pcsg.Status.ScheduledReplicas
	observedGeneration := pcsg.Status.ObservedGeneration
	generation := pcsg.Generation

	logger.V(1).Info("CheckPCSGFullyUpdated",
		"resourceName", resourceName,
		"generation", pcsg.Generation,
		"observedGeneration", pcsg.Status.ObservedGeneration,
		"desiredReplicas", desiredReplicas,
		"availableReplicas", availableReplicas,
		"updatedReplicas", updatedReplicas,
		"replicas", replicas,
		"scheduledReplicas", scheduledReplicas,
	)

	serviceStatus := v1beta1.ComponentReplicaStatus{
		ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
		ComponentNames:    []string{resourceName},
		Replicas:          pcsg.Status.Replicas,
		UpdatedReplicas:   pcsg.Status.UpdatedReplicas,
		AvailableReplicas: &availableReplicas,
	}

	if observedGeneration == nil {
		logger.V(1).Info("PodCliqueScalingGroup observedGeneration is nil", "resourceName", resourceName)
		return false, "observedGeneration is nil", serviceStatus, v1beta1.DGDReadyReasonSomeResourcesNotReady, nil
	}

	if observedGeneration != nil && *observedGeneration < generation {
		logger.V(1).Info("PodCliqueScalingGroup spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), serviceStatus, v1beta1.DGDReadyReasonSomeResourcesNotReady, nil
	}

	serviceStatus.ScheduledReplicas = &scheduledReplicas

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready
		return true, "", serviceStatus, "", nil
	}

	// Fully ready: replicas exist, are updated, and are available. Checked
	// first so a healthy PCSG is never mis-diagnosed as InsufficientCapacity
	// when Grove does not populate scheduledReplicas on a ready group.
	if replicas == desiredReplicas && updatedReplicas == desiredReplicas && availableReplicas == desiredReplicas {
		return true, "", serviceStatus, "", nil
	}

	// Not ready: the explicit MinAvailableBreached scheduling condition,
	// and a genuine partial scheduled count (0 < scheduled < desired).
	//
	// Grove alpha.8 polarity note: on the MinAvailableBreached condition, the
	// *scheduling* shortfall reason (InsufficientScheduledPodCliqueScalingGroupReplicas)
	// is emitted with Status=False, while Status=True is paired with the
	// *availability* reason (InsufficientAvailablePodCliqueScalingGroupReplicas).
	if cond := meta.FindStatusCondition(pcsg.Status.Conditions, groveconstants.ConditionTypeMinAvailableBreached); cond != nil &&
		cond.Status == metav1.ConditionFalse &&
		cond.Reason == legacyGroveConditionReasonInsufficientScheduledPCSGReplicas {
		logger.V(1).Info("PodCliqueScalingGroup MinAvailableBreached reports insufficient capacity", "resourceName", resourceName, "reason", cond.Reason, "message", cond.Message)
		return false, fmt.Sprintf("min-available breached (%s): %s", cond.Reason, cond.Message), serviceStatus, v1beta1.DGDReadyReasonInsufficientCapacity, nil
	}
	if scheduledReplicas > 0 && scheduledReplicas < desiredReplicas {
		logger.V(1).Info("PodCliqueScalingGroup partially scheduled", "resourceName", resourceName, "desired", desiredReplicas, "scheduled", scheduledReplicas)
		return false, fmt.Sprintf("insufficient scheduled replicas: scheduled=%d/%d", scheduledReplicas, desiredReplicas), serviceStatus, v1beta1.DGDReadyReasonInsufficientCapacity, nil
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), serviceStatus, v1beta1.DGDReadyReasonUpdating, nil
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodCliqueScalingGroup performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), serviceStatus, v1beta1.DGDReadyReasonUpdating, nil
	}

	// Scheduled and rolled out, but not enough available replicas.
	logger.V(1).Info("PodCliqueScalingGroup not ready", "resourceName", resourceName, "desired", desiredReplicas, "available", availableReplicas)
	return false, fmt.Sprintf("scheduled but available=%d/%d", availableReplicas, desiredReplicas), serviceStatus, v1beta1.DGDReadyReasonPodsNotReady, nil
}

// specToGroveTopologyConstraint converts a deployment-level topology constraint
// to the current Grove API shape.
func specToGroveTopologyConstraint(tc *v1beta1.SpecTopologyConstraint) *grovev1alpha1.TopologyConstraint {
	if tc == nil {
		return nil
	}
	return groveTopologyConstraint(tc.ClusterTopologyName, tc.PackDomain)
}

// toGroveTopologyConstraint converts a component-level topology constraint to
// the current Grove API shape. Components inherit topologyName from a
// constrained PodCliqueSet. When the deployment has no packing constraint,
// there is no parent Grove constraint to inherit from, so each constrained
// component carries the deployment's topologyName explicitly.
func toGroveTopologyConstraint(tc *v1beta1.TopologyConstraint, deploymentTC *v1beta1.SpecTopologyConstraint) *grovev1alpha1.TopologyConstraint {
	if tc == nil || tc.PackDomain == "" {
		return nil
	}
	topologyName := ""
	if deploymentTC != nil && deploymentTC.PackDomain == "" {
		topologyName = deploymentTC.ClusterTopologyName
	}
	return groveTopologyConstraint(topologyName, tc.PackDomain)
}

func groveTopologyConstraint(topologyName string, packDomain v1beta1.TopologyDomain) *grovev1alpha1.TopologyConstraint {
	if packDomain == "" {
		return nil
	}
	return &grovev1alpha1.TopologyConstraint{
		TopologyName: topologyName,
		Pack: &grovev1alpha1.TopologyPackConstraint{
			RequiredDomain: grovev1alpha1.TopologyDomain(packDomain),
		},
	}
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

func resolveVolcanoQueueName(annotations map[string]string) string {
	if annotations == nil {
		return ""
	}
	return strings.TrimSpace(annotations[commonconsts.KubeAnnotationVolcanoQueue])
}

// injectKaiSchedulerIfEnabled injects kai-scheduler settings into a clique if kai-scheduler is enabled and grove is enabled
func injectKaiSchedulerIfEnabled(
	clique *grovev1alpha1.PodCliqueTemplateSpec,
	runtimeConfig *controller_common.RuntimeConfig,
	validatedQueueName string,
) {
	// Only proceed if grove is enabled, kai-scheduler is enabled, and no manual schedulerName is set
	if !runtimeConfig.Gate.Enabled(features.Grove) || !runtimeConfig.Gate.Enabled(features.KaiScheduler) {
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

// injectVolcanoSchedulerIfEnabled injects Volcano scheduler settings into a clique if Volcano scheduler integration is enabled.
func injectVolcanoSchedulerIfEnabled(
	clique *grovev1alpha1.PodCliqueTemplateSpec,
	runtimeConfig *controller_common.RuntimeConfig,
) {
	if !runtimeConfig.Gate.Enabled(features.Grove) || !runtimeConfig.Gate.Enabled(features.VolcanoScheduler) {
		return
	}

	// Check if user has manually set schedulerName - if so, respect their choice
	if clique.Spec.PodSpec.SchedulerName != "" && clique.Spec.PodSpec.SchedulerName != commonconsts.VolcanoSchedulerName {
		return
	}

	clique.Spec.PodSpec.SchedulerName = commonconsts.VolcanoSchedulerName
}

// injectVolcanoQueueAnnotation maps the Dynamo Volcano queue annotation onto
// the generated PodCliqueSet annotation consumed by Grove's Volcano backend.
func injectVolcanoQueueAnnotation(
	gangSet *grovev1alpha1.PodCliqueSet,
	annotations map[string]string,
	runtimeConfig *controller_common.RuntimeConfig,
) {
	if !runtimeConfig.Gate.Enabled(features.Grove) || !runtimeConfig.Gate.Enabled(features.VolcanoScheduler) {
		return
	}

	queueName := resolveVolcanoQueueName(annotations)
	if queueName == "" {
		return
	}

	if gangSet.Annotations == nil {
		gangSet.Annotations = make(map[string]string)
	}
	gangSet.Annotations[commonconsts.GroveAnnotationVolcanoQueue] = queueName
}
