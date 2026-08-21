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

// GroveComponentResourceName returns the Grove child resource name for a DGD
// component. Grove currently creates one PodClique or PodCliqueScalingGroup
// instance per component at PodCliqueSet replica index zero.
func GroveComponentResourceName(dgd *v1beta1.DynamoGraphDeployment, componentName string) string {
	return fmt.Sprintf(
		"%s-0-%s",
		PCSNameForDGD(dgd.Name, dgd.Spec.Components),
		strings.ToLower(componentName),
	)
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

// GroveReadiness is one coherent observation of every Grove component backing
// a DGD. Callers should reuse it for readiness, status, and reason projection
// rather than rereading the same resources.
type GroveReadiness struct {
	Ready             bool
	Classification    string
	Message           string
	ComponentStatuses map[string]v1beta1.ComponentReplicaStatus
}

// EvaluateGroveReadiness resolves one Grove readiness snapshot from the
// supplied PCS observation. A nil PCS represents an observed missing PCS.
func EvaluateGroveReadiness(
	ctx context.Context,
	reader client.Reader,
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) (GroveReadiness, error) {
	allReady, classification, message, componentStatuses, err := evaluateGroveComponents(ctx, reader, dgd, pcs)
	if err != nil {
		return GroveReadiness{}, err
	}
	return GroveReadiness{
		Ready:             allReady,
		Classification:    classification,
		Message:           message,
		ComponentStatuses: componentStatuses,
	}, nil
}

// evaluateGroveComponents is the single per-component evaluation loop behind
// all public Grove readiness views.
//
// Each Check*Ready call returns the DGD-level Ready reason its component would
// imply (a v1beta1.DGDReadyReason* value) when that component is not ready. The
// reasons are aggregated in place: if every not-ready component implies the
// same reason, that reason is used; if they disagree, the result is
// MixedNotReadyReasons.
func evaluateGroveComponents(ctx context.Context, reader client.Reader, dgd *v1beta1.DynamoGraphDeployment, pcs *grovev1alpha1.PodCliqueSet) (allReady bool, classificationReason string, message string, componentStatuses map[string]v1beta1.ComponentReplicaStatus, err error) {
	logger := log.FromContext(ctx)
	var notReadyComponents []string
	aggregatedReason := ""
	componentReadinesses := make(map[string]groveComponentReadiness, len(dgd.Spec.Components))

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentName := component.ComponentName
		resourceName := GroveComponentResourceName(dgd, componentName)

		var componentReadiness groveComponentReadiness
		var checkErr error
		if component.UsesPCSG() {
			componentReadiness, checkErr = observePCSGReadiness(ctx, reader, resourceName, dgd.Namespace, logger)
		} else {
			componentReadiness, checkErr = observePodCliqueReadiness(ctx, reader, resourceName, dgd.Namespace, logger)
		}
		// A non-NotFound read error is a transient failure to determine
		// readiness. Propagate it (rather than folding it into a not-ready
		// result) so the reconcile retries with backoff and does not advance
		// ObservedGeneration on a blip. NotFound is handled inside observe* as a
		// legitimate not-ready state and never surfaces here.
		if checkErr != nil {
			return false, "", "", nil, fmt.Errorf("component %q: %w", componentName, checkErr)
		}
		componentReadinesses[componentName] = componentReadiness
		if !componentReadiness.ready {
			notReadyComponents = append(notReadyComponents, fmt.Sprintf("%s: %s", componentName, componentReadiness.reason))
			switch aggregatedReason {
			case "":
				aggregatedReason = componentReadiness.classification
			case componentReadiness.classification:
				// same reason as seen so far; keep it
			default:
				aggregatedReason = v1beta1.DGDReadyReasonMixedNotReadyReasons
			}
		}
	}

	namespacePlan, err := newGroveRuntimeNamespacePlan(dgd, pcs, componentReadinesses)
	if err != nil {
		return false, "", "", nil, err
	}

	componentStatuses = make(map[string]v1beta1.ComponentReplicaStatus, len(dgd.Spec.Components))
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentReadiness := componentReadinesses[component.ComponentName]
		componentStatus := componentReadiness.status
		componentStatus.RuntimeNamespace = namespacePlan.runtimeNamespace(dgd, component)
		componentStatuses[component.ComponentName] = componentStatus
	}

	if len(notReadyComponents) > 0 {
		return false, aggregatedReason, strings.Join(notReadyComponents, "; "), componentStatuses, nil
	}

	return true, v1beta1.DGDReadyReasonAllResourcesReady, "", componentStatuses, nil
}

// getAcceptedPCSRevisionHash returns the current PCS revision after Grove has
// observed exactly the latest PCS generation. It returns nil when pcs is nil,
// stale, or has not published a current revision.
func getAcceptedPCSRevisionHash(pcs *grovev1alpha1.PodCliqueSet) *string {
	if pcs == nil ||
		pcs.Status.ObservedGeneration == nil ||
		*pcs.Status.ObservedGeneration != pcs.Generation ||
		pcs.Status.CurrentGenerationHash == nil {
		return nil
	}
	return pcs.Status.CurrentGenerationHash
}

// groveComponentReadiness is the complete result of a single child read. The
// exported Check* helpers expose its readiness fields; EvaluateGroveReadiness
// also uses the observed revision to select the runtime namespace without a
// second read or an observer side channel.
type groveComponentReadiness struct {
	ready          bool
	reason         string
	classification string
	status         v1beta1.ComponentReplicaStatus
	revision       groveComponentRevisionState
}

func (r groveComponentReadiness) withResult(ready bool, reason, classification string) groveComponentReadiness {
	r.ready = ready
	r.reason = reason
	r.classification = classification
	return r
}

// groveComponentRevisionState is the child status needed to decide whether a
// worker revision has cut over.
type groveComponentRevisionState struct {
	generationObserved     bool
	currentPCSRevisionHash *string
	replicas               int32
	updatedReplicas        int32
	desiredReplicas        int32
	updateInProgress       bool
	updateEnded            bool
}

// hasCompletedAcceptedPCSRevision reports whether this child completed the
// accepted PCS revision. A nil acceptedPCSRevisionHash means no revision has
// been accepted yet.
// A child completes either during its initial realization with no update
// progress, or after a tracked update has ended.
func (s groveComponentRevisionState) hasCompletedAcceptedPCSRevision(acceptedPCSRevisionHash *string) bool {
	return acceptedPCSRevisionHash != nil &&
		s.generationObserved &&
		s.currentPCSRevisionHash != nil &&
		*s.currentPCSRevisionHash == *acceptedPCSRevisionHash &&
		s.replicas == s.desiredReplicas &&
		s.updatedReplicas == s.desiredReplicas &&
		(!s.updateInProgress || s.updateEnded)
}

// groveRuntimeNamespacePlan is derived from one accepted PCS observation and
// one read of every child. Worker namespaces change together, matching the DCD
// path's cohort-wide cutover.
type groveRuntimeNamespacePlan struct {
	acceptedPCSRevisionHash *string
	workerHash              string
	workersUseHashSuffix    bool
	workersCompleted        bool
}

func newGroveRuntimeNamespacePlan(
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	componentReadinesses map[string]groveComponentReadiness,
) (groveRuntimeNamespacePlan, error) {
	acceptedPCSRevisionHash := getAcceptedPCSRevisionHash(pcs)
	workerHash, workersUseHashSuffix, err := acceptedGroveWorkerHash(dgd, pcs, acceptedPCSRevisionHash)
	if err != nil {
		return groveRuntimeNamespacePlan{}, err
	}

	return groveRuntimeNamespacePlan{
		acceptedPCSRevisionHash: acceptedPCSRevisionHash,
		workerHash:              workerHash,
		workersUseHashSuffix:    workersUseHashSuffix,
		workersCompleted:        groveWorkersCompletedAcceptedPCSRevision(dgd, componentReadinesses, acceptedPCSRevisionHash),
	}, nil
}

func (p groveRuntimeNamespacePlan) runtimeNamespace(
	dgd *v1beta1.DynamoGraphDeployment,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
) string {
	baseNamespace := dgd.GetDynamoNamespaceForComponent(component)
	if !IsWorkerComponent(string(component.ComponentType)) {
		return baseNamespace
	}

	previousNamespace := dgd.Status.Components[component.ComponentName].RuntimeNamespace
	if p.acceptedPCSRevisionHash == nil || !p.workersCompleted {
		return previousNamespace
	}
	if !p.workersUseHashSuffix {
		return baseNamespace
	}
	return ComponentRuntimeNamespace(baseNamespace, string(component.ComponentType), p.workerHash)
}

// groveWorkersCompletedAcceptedPCSRevision reports whether every worker child
// completed the accepted PCS revision. dgd must contain at least one worker.
func groveWorkersCompletedAcceptedPCSRevision(
	dgd *v1beta1.DynamoGraphDeployment,
	componentReadinesses map[string]groveComponentReadiness,
	acceptedPCSRevisionHash *string,
) bool {
	workerCount := 0
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		workerCount++
		if !componentReadinesses[component.ComponentName].revision.hasCompletedAcceptedPCSRevision(acceptedPCSRevisionHash) {
			return false
		}
	}
	return workerCount > 0
}

// acceptedGroveWorkerHash returns the worker hash rendered into the accepted
// PCS revision. The second result reports whether that accepted revision is suffixed.
func acceptedGroveWorkerHash(
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	acceptedPCSRevisionHash *string,
) (string, bool, error) {
	if acceptedPCSRevisionHash == nil {
		return "", false, nil
	}

	workerHash := ""
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		clique := grovePodCliqueSetCliqueForComponent(pcs, component.ComponentName)
		if clique == nil {
			return "", false, fmt.Errorf("accepted Grove PodCliqueSet revision %q has no worker clique for component %q", *acceptedPCSRevisionHash, component.ComponentName)
		}
		cliqueHash := clique.Labels[commonconsts.KubeLabelDynamoWorkerHash]
		if cliqueHash == "" {
			if workerHash != "" {
				return "", false, fmt.Errorf("accepted Grove PodCliqueSet revision %q mixes suffixed and legacy worker cliques", *acceptedPCSRevisionHash)
			}
			continue
		}
		if workerHash != "" && workerHash != cliqueHash {
			return "", false, fmt.Errorf("accepted Grove PodCliqueSet revision %q has inconsistent worker hashes", *acceptedPCSRevisionHash)
		}
		workerHash = cliqueHash
	}

	if workerHash == "" {
		return "", false, nil
	}

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		clique := grovePodCliqueSetCliqueForComponent(pcs, component.ComponentName)
		if clique.Labels[commonconsts.KubeLabelDynamoWorkerHash] != workerHash {
			return "", false, fmt.Errorf("accepted Grove PodCliqueSet revision %q mixes suffixed and legacy worker cliques", *acceptedPCSRevisionHash)
		}
	}

	return workerHash, true, nil
}

// grovePodCliqueSetCliqueForComponent returns the rendered clique for a DGD component.
func grovePodCliqueSetCliqueForComponent(pcs *grovev1alpha1.PodCliqueSet, componentName string) *grovev1alpha1.PodCliqueTemplateSpec {
	if pcs == nil {
		return nil
	}
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique != nil && clique.Labels[commonconsts.KubeLabelDynamoComponent] == componentName {
			return clique
		}
	}
	return nil
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
func CheckPodCliqueReady(ctx context.Context, reader client.Reader, resourceName, namespace string, logger logr.Logger) (bool, string, v1beta1.ComponentReplicaStatus, string, error) {
	componentReadiness, err := observePodCliqueReadiness(ctx, reader, resourceName, namespace, logger)
	return componentReadiness.ready, componentReadiness.reason, componentReadiness.status, componentReadiness.classification, err
}

func observePodCliqueReadiness(ctx context.Context, reader client.Reader, resourceName, namespace string, logger logr.Logger) (groveComponentReadiness, error) {
	podClique := &grovev1alpha1.PodClique{}
	err := reader.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, podClique)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodClique not found", "resourceName", resourceName)
			// The backing PodClique is not created yet. Return a valid status
			// entry (with the known kind and expected name) rather than an empty
			// ComponentReplicaStatus{}.
			serviceStatus := v1beta1.ComponentReplicaStatus{
				ComponentKind:  v1beta1.ComponentKindPodClique,
				ComponentNames: []string{resourceName},
			}
			return groveComponentReadiness{status: serviceStatus}.withResult(false, "resource not found", v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
		}
		// A non-NotFound error is a transient failure to determine readiness,
		// not a legitimate not-ready state. Return it so the reconcile retries
		// with backoff and does not advance ObservedGeneration on a blip.
		logger.V(1).Info("Failed to get PodClique", "error", err, "resourceName", resourceName)
		return groveComponentReadiness{}, fmt.Errorf("failed to get PodClique %s/%s: %w", namespace, resourceName, err)
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

	componentReadiness := groveComponentReadiness{
		status: v1beta1.ComponentReplicaStatus{
			ComponentKind:   v1beta1.ComponentKindPodClique,
			ComponentNames:  []string{resourceName},
			Replicas:        podClique.Status.Replicas,
			UpdatedReplicas: podClique.Status.UpdatedReplicas,
			ReadyReplicas:   &readyReplicas,
		},
		revision: groveComponentRevisionState{
			generationObserved:     observedGeneration != nil && *observedGeneration >= generation,
			currentPCSRevisionHash: podClique.Status.CurrentPodCliqueSetGenerationHash,
			replicas:               replicas,
			updatedReplicas:        updatedReplicas,
			desiredReplicas:        desiredReplicas,
			updateInProgress:       podClique.Status.UpdateProgress != nil,
			updateEnded: podClique.Status.UpdateProgress != nil &&
				podClique.Status.UpdateProgress.UpdateEndedAt != nil,
		},
	}

	if observedGeneration == nil {
		logger.V(1).Info("PodClique observedGeneration is nil", "resourceName", resourceName)
		return componentReadiness.withResult(false, "observedGeneration is nil", v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
	}

	if *observedGeneration < generation {
		logger.V(1).Info("PodClique spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return componentReadiness.withResult(false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
	}

	componentReadiness.status.ScheduledReplicas = &scheduledReplicas

	if desiredReplicas == 0 {
		return componentReadiness.withResult(true, "", ""), nil
	}

	// Fully ready: replicas exist, are updated, and are ready. Checked first so
	// a healthy component is never mis-diagnosed as InsufficientCapacity when
	// Grove does not populate scheduledReplicas on a ready PodClique.
	if replicas == desiredReplicas && updatedReplicas == desiredReplicas && readyReplicas == desiredReplicas {
		return componentReadiness.withResult(true, "", ""), nil
	}

	// Not ready: classify capacity signals, in order of reliability:
	//   1. scheduleGatedReplicas > 0            (explicit gated count)
	//   2. PodCliqueScheduled condition = False  (explicit scheduling signal)
	//   3. 0 < scheduledReplicas < desired       (genuine partial scheduling)
	if scheduleGatedReplicas > 0 {
		logger.V(1).Info("PodClique has schedule-gated replicas", "resourceName", resourceName, "scheduleGated", scheduleGatedReplicas)
		return componentReadiness.withResult(false, fmt.Sprintf("schedule-gated replicas: %d", scheduleGatedReplicas), v1beta1.DGDReadyReasonInsufficientCapacity), nil
	}
	if cond := meta.FindStatusCondition(podClique.Status.Conditions, groveconstants.ConditionTypePodCliqueScheduled); cond != nil &&
		cond.Status == metav1.ConditionFalse &&
		cond.Reason == groveconstants.ConditionReasonInsufficientScheduledPods {
		logger.V(1).Info("PodClique scheduling condition reports insufficient capacity", "resourceName", resourceName, "reason", cond.Reason, "message", cond.Message)
		return componentReadiness.withResult(false, fmt.Sprintf("scheduling condition %s: %s", cond.Reason, cond.Message), v1beta1.DGDReadyReasonInsufficientCapacity), nil
	}
	if scheduledReplicas > 0 && scheduledReplicas < desiredReplicas {
		logger.V(1).Info("PodClique partially scheduled", "resourceName", resourceName, "desired", desiredReplicas, "scheduled", scheduledReplicas)
		return componentReadiness.withResult(false, fmt.Sprintf("insufficient scheduled replicas: scheduled=%d/%d", scheduledReplicas, desiredReplicas), v1beta1.DGDReadyReasonInsufficientCapacity), nil
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodClique not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return componentReadiness.withResult(false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), v1beta1.DGDReadyReasonUpdating), nil
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodClique performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return componentReadiness.withResult(false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), v1beta1.DGDReadyReasonUpdating), nil
	}

	// Scheduled and rolled out, but not enough ready replicas.
	logger.V(1).Info("PodClique not ready", "resourceName", resourceName, "desired", desiredReplicas, "ready", readyReplicas)
	return componentReadiness.withResult(false, fmt.Sprintf("scheduled but ready=%d/%d", readyReplicas, desiredReplicas), v1beta1.DGDReadyReasonPodsNotReady), nil
}

// CheckPCSGReady determines if a Grove PodCliqueScalingGroup is fully ready and available.
// It checks various status fields to ensure all replicas are available and the PodCliqueScalingGroup
// configuration has been fully applied. This is the PodCliqueScalingGroup equivalent of IsDeploymentReady
// for standard Kubernetes Deployments.
func CheckPCSGReady(ctx context.Context, reader client.Reader, resourceName, namespace string, logger logr.Logger) (bool, string, v1beta1.ComponentReplicaStatus, string, error) {
	componentReadiness, err := observePCSGReadiness(ctx, reader, resourceName, namespace, logger)
	return componentReadiness.ready, componentReadiness.reason, componentReadiness.status, componentReadiness.classification, err
}

func observePCSGReadiness(ctx context.Context, reader client.Reader, resourceName, namespace string, logger logr.Logger) (groveComponentReadiness, error) {
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	err := reader.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, pcsg)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodCliqueScalingGroup not found", "resourceName", resourceName)
			// The backing PodCliqueScalingGroup is not created yet. Return a valid
			// status entry (with the known kind and expected name) rather than an
			// empty ComponentReplicaStatus{}.
			serviceStatus := v1beta1.ComponentReplicaStatus{
				ComponentKind:  v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames: []string{resourceName},
			}
			return groveComponentReadiness{status: serviceStatus}.withResult(false, "resource not found", v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
		}
		// A non-NotFound error is a transient failure to determine readiness,
		// not a legitimate not-ready state. Return it so the reconcile retries
		// with backoff and does not advance ObservedGeneration on a blip.
		logger.V(1).Info("Failed to get PodCliqueScalingGroup", "error", err, "resourceName", resourceName)
		return groveComponentReadiness{}, fmt.Errorf("failed to get PodCliqueScalingGroup %s/%s: %w", namespace, resourceName, err)
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

	componentReadiness := groveComponentReadiness{
		status: v1beta1.ComponentReplicaStatus{
			ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
			ComponentNames:    []string{resourceName},
			Replicas:          pcsg.Status.Replicas,
			UpdatedReplicas:   pcsg.Status.UpdatedReplicas,
			AvailableReplicas: &availableReplicas,
		},
		revision: groveComponentRevisionState{
			generationObserved:     observedGeneration != nil && *observedGeneration >= generation,
			currentPCSRevisionHash: pcsg.Status.CurrentPodCliqueSetGenerationHash,
			replicas:               replicas,
			updatedReplicas:        updatedReplicas,
			desiredReplicas:        desiredReplicas,
			updateInProgress:       pcsg.Status.UpdateProgress != nil,
			updateEnded: pcsg.Status.UpdateProgress != nil &&
				pcsg.Status.UpdateProgress.UpdateEndedAt != nil,
		},
	}

	if observedGeneration == nil {
		logger.V(1).Info("PodCliqueScalingGroup observedGeneration is nil", "resourceName", resourceName)
		return componentReadiness.withResult(false, "observedGeneration is nil", v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
	}

	if *observedGeneration < generation {
		logger.V(1).Info("PodCliqueScalingGroup spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return componentReadiness.withResult(false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), v1beta1.DGDReadyReasonSomeResourcesNotReady), nil
	}

	componentReadiness.status.ScheduledReplicas = &scheduledReplicas

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready.
		return componentReadiness.withResult(true, "", ""), nil
	}

	// Fully ready: replicas exist, are updated, and are available. Checked
	// first so a healthy PCSG is never mis-diagnosed as InsufficientCapacity
	// when Grove does not populate scheduledReplicas on a ready group.
	if replicas == desiredReplicas && updatedReplicas == desiredReplicas && availableReplicas == desiredReplicas {
		return componentReadiness.withResult(true, "", ""), nil
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
		return componentReadiness.withResult(false, fmt.Sprintf("min-available breached (%s): %s", cond.Reason, cond.Message), v1beta1.DGDReadyReasonInsufficientCapacity), nil
	}
	if scheduledReplicas > 0 && scheduledReplicas < desiredReplicas {
		logger.V(1).Info("PodCliqueScalingGroup partially scheduled", "resourceName", resourceName, "desired", desiredReplicas, "scheduled", scheduledReplicas)
		return componentReadiness.withResult(false, fmt.Sprintf("insufficient scheduled replicas: scheduled=%d/%d", scheduledReplicas, desiredReplicas), v1beta1.DGDReadyReasonInsufficientCapacity), nil
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return componentReadiness.withResult(false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), v1beta1.DGDReadyReasonUpdating), nil
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodCliqueScalingGroup performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return componentReadiness.withResult(false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), v1beta1.DGDReadyReasonUpdating), nil
	}

	// Scheduled and rolled out, but not enough available replicas.
	logger.V(1).Info("PodCliqueScalingGroup not ready", "resourceName", resourceName, "desired", desiredReplicas, "available", availableReplicas)
	return componentReadiness.withResult(false, fmt.Sprintf("scheduled but available=%d/%d", availableReplicas, desiredReplicas), v1beta1.DGDReadyReasonPodsNotReady), nil
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
