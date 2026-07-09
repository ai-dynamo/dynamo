package dynamo

import (
	"context"
	"fmt"
	"strings"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	ctrl "sigs.k8s.io/controller-runtime"
)

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
	logger := log.FromContext(ctx)
	var notReadyComponents []string

	componentStatuses := make(map[string]v1beta1.ComponentReplicaStatus, len(dgd.Spec.Components))

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentName := component.ComponentName
		usesPCSG := component.GetNumberOfNodes() > 1 || component.IsInterPodGMSEnabled()
		resourceName := fmt.Sprintf("%s-0-%s", PCSNameForDGD(dgd.Name, dgd.Spec.Components), strings.ToLower(componentName))

		if usesPCSG {
			ok, reason, componentStatus, err := CheckPCSGReady(ctx, client, resourceName, dgd.Namespace, logger)
			if err != nil {
				return false, "", nil, fmt.Errorf("failed to check component %q (pcsg/%s): %w", componentName, resourceName, err)
			}
			componentStatuses[componentName] = componentStatus
			if !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("pcsg/%s: %s", resourceName, reason))
			}
		} else {
			ok, reason, componentStatus, err := CheckPodCliqueReady(ctx, client, resourceName, dgd.Namespace, logger)
			if err != nil {
				return false, "", nil, fmt.Errorf("failed to check component %q (podclique/%s): %w", componentName, resourceName, err)
			}
			componentStatuses[componentName] = componentStatus
			if !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("podclique/%s: %s", resourceName, reason))
			}
		}
	}

	if len(notReadyComponents) > 0 {
		return false, strings.Join(notReadyComponents, "; "), componentStatuses, nil
	}

	return true, "", componentStatuses, nil
}

// CheckPodCliqueReady determines if a Grove PodClique is fully ready and available.
// It checks various status fields to ensure all replicas are available and the PodClique
// configuration has been fully applied. This is the PodClique equivalent of IsDeploymentReady
// for standard Kubernetes Deployments.
func CheckPodCliqueReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1beta1.ComponentReplicaStatus, error) {
	serviceStatus := v1beta1.ComponentReplicaStatus{
		ComponentKind:  v1beta1.ComponentKindPodClique,
		ComponentNames: []string{resourceName},
	}

	podClique := &grovev1alpha1.PodClique{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, podClique)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodClique not found", "resourceName", resourceName)
			return false, "resource not found", serviceStatus, nil
		}
		return false, "", v1beta1.ComponentReplicaStatus{}, fmt.Errorf("failed to get PodClique %s/%s: %w", namespace, resourceName, err)
	}

	desiredReplicas := podClique.Spec.Replicas
	readyReplicas := podClique.Status.ReadyReplicas
	updatedReplicas := podClique.Status.UpdatedReplicas
	replicas := podClique.Status.Replicas
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
	)

	serviceStatus.Replicas = podClique.Status.Replicas
	serviceStatus.UpdatedReplicas = podClique.Status.UpdatedReplicas
	serviceStatus.ReadyReplicas = &readyReplicas

	if observedGeneration == nil {
		logger.V(1).Info("PodClique observedGeneration is nil", "resourceName", resourceName)
		return false, "observedGeneration is nil", serviceStatus, nil
	}

	if observedGeneration != nil && *observedGeneration < generation {
		logger.V(1).Info("PodClique spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), serviceStatus, nil
	}

	if desiredReplicas == 0 {
		return true, "", serviceStatus, nil
	}

	if desiredReplicas != readyReplicas {
		logger.V(1).Info("PodClique not ready", "resourceName", resourceName, "desired", desiredReplicas, "ready", readyReplicas)
		return false, fmt.Sprintf("desired=%d, ready=%d", desiredReplicas, readyReplicas), serviceStatus, nil
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodClique not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), serviceStatus, nil
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodClique performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), serviceStatus, nil
	}

	return true, "", serviceStatus, nil
}

// CheckPCSGReady determines if a Grove PodCliqueScalingGroup is fully ready and available.
// It checks various status fields to ensure all replicas are available and the PodClique
// configuration has been fully applied. This is the PodCliqueScalingGroup equivalent of IsDeploymentReady
// for standard Kubernetes Deployments.
func CheckPCSGReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1beta1.ComponentReplicaStatus, error) {
	serviceStatus := v1beta1.ComponentReplicaStatus{
		ComponentKind:  v1beta1.ComponentKindPodCliqueScalingGroup,
		ComponentNames: []string{resourceName},
	}

	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, pcsg)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodCliqueScalingGroup not found", "resourceName", resourceName)
			return false, "resource not found", serviceStatus, nil
		}
		return false, "", v1beta1.ComponentReplicaStatus{}, fmt.Errorf("failed to get PodCliqueScalingGroup %s/%s: %w", namespace, resourceName, err)
	}

	desiredReplicas := pcsg.Spec.Replicas
	availableReplicas := pcsg.Status.AvailableReplicas
	updatedReplicas := pcsg.Status.UpdatedReplicas
	replicas := pcsg.Status.Replicas
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
	)

	serviceStatus.Replicas = pcsg.Status.Replicas
	serviceStatus.UpdatedReplicas = pcsg.Status.UpdatedReplicas
	serviceStatus.AvailableReplicas = &availableReplicas

	if observedGeneration == nil {
		logger.V(1).Info("PodCliqueScalingGroup observedGeneration is nil", "resourceName", resourceName)
		return false, "observedGeneration is nil", serviceStatus, nil
	}

	if observedGeneration != nil && *observedGeneration < generation {
		logger.V(1).Info("PodCliqueScalingGroup spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), serviceStatus, nil
	}

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready
		return true, "", serviceStatus, nil
	}

	if desiredReplicas != availableReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not ready", "resourceName", resourceName, "desired", desiredReplicas, "available", availableReplicas)
		return false, fmt.Sprintf("desired=%d, available=%d", desiredReplicas, availableReplicas), serviceStatus, nil
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), serviceStatus, nil
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodCliqueScalingGroup performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), serviceStatus, nil
	}

	return true, "", serviceStatus, nil
}

// AggregatePlacementScore reads PodGangStatutes from the given PodCliqueSet
// and derives the DGD-level placement score. This implements the "Grove
// mirror" placement-score source alternative; the source of truth is still
// under discussion (see the placement-score DEP Open Questions).
//
// When multiple PodGangs report a score the minimum is used: the DGD-level
// value is a worst-placement signal, matching the DEP's rationale that one
// poorly placed serving unit should not be hidden by a weighted average.
//
// Absence is non-failing:
//   - nil PodCliqueSet: Unsupported (no Grove object to read from).
//   - PodGangStatutes empty: Unknown (Grove hasn't populated the status yet,
//     which is indeterminate rather than a permanent absence).
//
// NOTE: The current Grove alpha.8 PodGangStatus has no PlacementScore field.
// Until Grove exposes it, this function reports Unsupported when gangs are
// present. Replace the TODO below with the actual field read (and any
// partial-report handling) once available.
func AggregatePlacementScore(pcs *grovev1alpha1.PodCliqueSet) (score *float64, state v1beta1.PlacementScoreState) {
	if pcs == nil {
		return nil, v1beta1.PlacementScoreStateUnsupported
	}

	gangs := pcs.Status.PodGangStatutes
	if len(gangs) == 0 {
		return nil, v1beta1.PlacementScoreStateUnknown
	}

	// TODO: read gang.PlacementScore once Grove PodGangStatus exposes the
	// field. Until then, report Unsupported so absence is explicit and
	// non-failing.
	return nil, v1beta1.PlacementScoreStateUnsupported
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
	if !runtimeConfig.GroveEnabled || !runtimeConfig.KaiSchedulerEnabled {
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
	if !runtimeConfig.GroveEnabled || !runtimeConfig.VolcanoSchedulerEnabled {
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
	if !runtimeConfig.GroveEnabled || !runtimeConfig.VolcanoSchedulerEnabled {
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
