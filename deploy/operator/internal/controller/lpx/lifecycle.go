/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

const (
	lpxDeploymentUIDAnnotation = dynamo.LPXDeploymentUIDAnnotation
	lpxOwnerUIDLabel           = lpxDeploymentUIDAnnotation
	lpxModelAnnotation         = "scheduling.lpu.nvidia.com/dynamo-model"
	lpxPodCliqueSetKind        = "PodCliqueSet"
	lpxRetirementRequeueAfter  = 5 * time.Second
	maxDGDConditionMessageSize = 32768
)

const (
	lpxRetiringReason               string = "LPXRetiring"
	lpxSchedulerStatusInvalidReason string = "LPXSchedulerStatusInvalid"
)

// lpxClassification is deliberately private and closed: reconciliation can
// report only one legal protocol state, never a freely assembled collection
// of correlated booleans.
type lpxClassification interface {
	lpxClassification()
}

type lpxRejected struct {
	reason string
}

type lpxMaterializing struct {
	workloadDigest string
	requests       []lpxModelMaterializing
	workload       *lpx.SelectedWorkload
	plan           *lpx.MaterializationPlan
}

type reconcileOutcome struct {
	State           nvidiacomv1beta1.DGDState
	Reason          string
	Message         string
	retirementScope lpxRetirementScope
}

type lpxRetirementScope uint8

const (
	lpxWorkloadRetirement lpxRetirementScope = iota
	lpxRequestRetirement
)

// lpxModelMaterializing is the request-local state for one model projection.
// The enclosing lpxMaterializing owns the shared Grove identity.
type lpxModelMaterializing struct {
	requestName     string
	modelProjection *lpx.ModelProjection
	replicaIndex    int32
	agentCliqueName string
}

type lpxGroveIdentity struct {
	pcs  *grovev1alpha1.PodCliqueSet
	plan *lpx.MaterializationPlan
}

type lpxClosed struct {
	incomplete string
}

type lpxOpen struct{}

type lpxBound struct{}

type lpxSchedulerObserved lpxv1alpha1.LPUPipelineRequestStatus

type lpxRetiring struct {
	retirementReason string
	scope            lpxRetirementScope
}

func (*lpxRejected) lpxClassification()          {}
func (*lpxClosed) lpxClassification()            {}
func (*lpxOpen) lpxClassification()              {}
func (*lpxBound) lpxClassification()             {}
func (*lpxSchedulerObserved) lpxClassification() {}
func (*lpxRetiring) lpxClassification()          {}

func lpxResult(classification lpxClassification) reconcileOutcome {
	result := reconcileOutcome{State: nvidiacomv1beta1.DGDStatePending}
	switch state := classification.(type) {
	case *lpxRejected:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = "LPXRejected"
		result.Message = state.reason
	case *lpxClosed:
		result.Reason = "LPXPublicationPending"
		result.Message = state.incomplete
	case *lpxOpen:
		result.Reason = "LPXPublished"
		result.Message = "LPX request is published"
	case *lpxBound:
		result.Reason = "AwaitingLPXRuntimeActivation"
		result.Message = "LPX scheduling is bound; runtime activation and serving readiness remain external"
	case *lpxSchedulerObserved:
		result = lpxSchedulerResult(state)
	case *lpxRetiring:
		result.Reason = lpxRetiringReason
		result.Message = state.retirementReason
		result.retirementScope = state.scope
	case *lpxDeadlineExceeded:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = lpxSchedulingDeadlineExceededReason
		result.Message = "An LPX request exceeded its scheduling deadline"
	default:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = "LPXInternalClassificationError"
		result.Message = "Unknown LPX reconciliation classification"
	}
	return result
}

// lpxSchedulerResult classifies one receipt and formats its diagnostics once.
func lpxSchedulerResult(state *lpxSchedulerObserved) reconcileOutcome {
	result := reconcileOutcome{State: nvidiacomv1beta1.DGDStatePending}
	var summary string
	switch state.Phase {
	case lpxv1alpha1.RequestPhasePending:
		result.Reason = "LPXSchedulerPending"
		summary = "LPX scheduler is waiting to plan the current request"
	case lpxv1alpha1.RequestPhaseNoFit:
		result.Reason = "LPXNoFit"
		summary = "LPX scheduler found no placement for the current request"
	case lpxv1alpha1.RequestPhaseUnsupported:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = "LPXUnsupported"
		summary = "LPX scheduler cannot support the current request"
	case lpxv1alpha1.RequestPhasePlanned:
		result.Reason = "LPXPlanned"
		summary = "LPX scheduler committed a placement plan for the current request"
	case lpxv1alpha1.RequestPhaseReserving:
		result.Reason = "LPXReserving"
		summary = "LPX scheduler is reserving the planned LPU allocation"
	case lpxv1alpha1.RequestPhaseBinding:
		result.Reason = "LPXBinding"
		summary = "LPX scheduler is binding the planned workload Pods"
	case lpxv1alpha1.RequestPhaseDegraded:
		result.Reason = "LPXDegraded"
		summary = "LPX scheduler is repairing or releasing a degraded placement"
	case lpxv1alpha1.RequestPhaseReleasing:
		result.Reason = "LPXReleasing"
		summary = "LPX scheduler is releasing the current plan"
	case lpxv1alpha1.RequestPhaseReleased:
		result.Reason = "LPXReleased"
		summary = "LPX scheduler released the current plan"
	case lpxv1alpha1.RequestPhaseBound:
		return lpxResult(&lpxBound{})
	default:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = lpxSchedulerStatusInvalidReason
		summary = fmt.Sprintf("LPX scheduler reported unknown phase %q", state.Phase)
	}
	result.Message = lpxMessageWithDiagnostics(summary, state.Diagnostics)
	return result
}

func lpxMessageWithDiagnostics(summary string, diagnostics []lpxv1alpha1.StatusDiagnostic) string {
	if len(diagnostics) == 0 {
		return summary
	}
	var message strings.Builder
	message.WriteString(summary)
	message.WriteString(". LPX diagnostics: ")
	for index, diagnostic := range diagnostics {
		if index > 0 {
			message.WriteString("; ")
		}
		message.WriteString(diagnostic.Code)
		message.WriteString(" [")
		message.WriteString(diagnostic.Subject)
		message.WriteString("]: ")
		message.WriteString(diagnostic.Detail)

		// Later diagnostics cannot change the retained prefix.
		if message.Len() > maxDGDConditionMessageSize {
			break
		}
	}
	return truncateLPXMessage(message.String())
}

func truncateLPXMessage(message string) string {
	if len(message) <= maxDGDConditionMessageSize {
		return message
	}
	const suffix = "…"
	last := maxDGDConditionMessageSize - len(suffix)
	for last > 0 && !utf8.ValidString(message[:last]) {
		last--
	}
	return message[:last] + suffix
}

func overlayLPXResult(
	base reconcileOutcome,
	classification lpxClassification,
) reconcileOutcome {
	if _, ok := classification.(*lpxBound); ok {
		return base
	}
	return lpxResult(classification)
}

// reconcileLPXKnownIntentFence deletes a publication before any new graph can
// be rendered when selection, DGD generation, model, or DGD incarnation changes.
// pcs is nil when its stable name was not found.
func (r *graphReconciler) reconcileLPXKnownIntentFence(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	requests []lpxv1alpha1.LPUPipelineRequest,
) (lpxClassification, []lpxv1alpha1.LPUPipelineRequest, error) {
	var err error
	if requests == nil {
		requests, err = r.listOwnedLPXRequests(ctx, deployment, pcs)
	}
	if err != nil {
		return nil, nil, err
	}
	selected := source.HasLPXComponent()

	for index := range requests {
		request := &requests[index]
		keep := selected &&
			request.Annotations[lpxDeploymentUIDAnnotation] == string(deployment.UID) &&
			request.Annotations[lpx.ExecutionBackendAnnotation] == ""
		if keep {
			continue
		}
		reason := "LPX scheduler was deselected"
		if selected {
			reason = "The DGD incarnation, generation, model, or selected input changed"
		}
		retiring, retireErr := r.retireLPXEngineRequest(ctx, deployment, request, reason)
		return retiring, nil, retireErr
	}
	return nil, requests, nil
}

// prepareLPXMaterializing resolves the selected workload against the observed
// Grove capacity. pcs is nil when the stable PodCliqueSet name was not found.
func (r *graphReconciler) prepareLPXMaterializing(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) (*lpxMaterializing, lpxClassification, error) {
	workload, err := lpx.ResolveSelectedWorkload(ctx, source, r.modelRegistry)
	if err != nil {
		if errors.Is(err, lpx.ErrBuildSnapshotAcquisition) {
			return nil, nil, err
		}
		return nil, &lpxRejected{reason: err.Error()}, nil
	}
	projections := workload.ModelProjections()
	plan, err := workload.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(deployment))
	if err != nil {
		return nil, &lpxRejected{reason: err.Error()}, nil
	}
	// Omitted replicas leave the native scaling group in charge of engine count.
	if lpx.ServingComponent(source).Replicas == nil {
		group := &grovev1alpha1.PodCliqueScalingGroup{}
		if err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: plan.LPXScalingGroup}, group); err != nil {
			if !apierrors.IsNotFound(err) {
				return nil, nil, err
			}
			plan.Replicas = ptr.Deref(deployment.Status.RetainedReplicas, plan.Replicas)
		} else {
			// Reject an unrelated PCS before consulting any group it happens to own.
			if pcs != nil && !metav1.IsControlledBy(pcs, deployment) {
				return nil, nil, fmt.Errorf("refusing to use replicas from a foreign LPX PodCliqueSet")
			}
			// A group outliving its PCS is normal during background Grove cleanup.
			if pcs == nil || !pcs.DeletionTimestamp.IsZero() || !metav1.IsControlledBy(group, pcs) {
				return nil, &lpxRetiring{retirementReason: "Waiting for the previous LPX scaling group to finish cleanup"}, nil
			}
			plan.Replicas = group.Spec.Replicas
		}
		if err := plan.ValidateReplicaCount(); err != nil {
			return nil, &lpxRejected{reason: err.Error()}, nil
		}
	}
	workloadDigest := workload.Digest().String()
	requests := make([]lpxModelMaterializing, 0, len(projections)*int(plan.Replicas))
	for replicaIndex := int32(0); replicaIndex < plan.Replicas; replicaIndex++ {
		for projectionIndex, projection := range projections {
			requestDigest := digestAttemptKey(
				deployment.Namespace,
				deployment.Name,
				deployment.UID,
				projection.Model(),
				replicaIndex,
			)
			requests = append(requests, lpxModelMaterializing{
				requestName:     lpxRequestName(deployment.Name, requestDigest),
				modelProjection: projection,
				replicaIndex:    replicaIndex,
				agentCliqueName: plan.ForReplica(replicaIndex).Agents[projectionIndex].CliqueName,
			})
		}
	}
	return &lpxMaterializing{
		workloadDigest: workloadDigest,
		requests:       requests,
		workload:       workload,
		plan:           plan,
	}, nil, nil
}

// retireInvalidLPXWorkload also retires staging that never acquired an LPR.
// Healthy publication and spec-write fences intentionally do not use this path.
// pcs is nil when its stable name was not found.
func (r *graphReconciler) retireInvalidLPXWorkload(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	reason string,
) (*lpxRetiring, error) {
	selector := client.MatchingLabels{lpxOwnerUIDLabel: string(deployment.UID)}
	owned := make([]client.Object, 0)
	podCliqueSets := &grovev1alpha1.PodCliqueSetList{}
	if err := r.apiReader.List(ctx, podCliqueSets, client.InNamespace(deployment.Namespace), selector); err != nil {
		return nil, err
	}
	for index := range podCliqueSets.Items {
		if pcs := &podCliqueSets.Items[index]; metav1.IsControlledBy(pcs, deployment) {
			owned = append(owned, pcs)
		}
	}

	services := &corev1.ServiceList{}
	if err := r.apiReader.List(ctx, services, client.InNamespace(deployment.Namespace), selector); err != nil {
		return nil, err
	}
	for index := range services.Items {
		if service := &services.Items[index]; metav1.IsControlledBy(service, deployment) {
			owned = append(owned, service)
		}
	}

	configMaps := &corev1.ConfigMapList{}
	if err := r.apiReader.List(ctx, configMaps, client.InNamespace(deployment.Namespace), selector); err != nil {
		return nil, err
	}
	for index := range configMaps.Items {
		if configMap := &configMaps.Items[index]; metav1.IsControlledBy(configMap, deployment) {
			owned = append(owned, configMap)
		}
	}

	requests, err := r.listOwnedLPXRequests(ctx, deployment, pcs)
	if err != nil {
		return nil, err
	}
	if len(owned) == 0 && len(requests) == 0 {
		return nil, nil
	}
	if len(owned) > 0 {
		// A newer child seen during listing must not lose its staging to this stale failure.
		if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
			return nil, err
		}

		propagation := metav1.DeletePropagationBackground
		for _, resource := range owned {
			if !resource.GetDeletionTimestamp().IsZero() {
				continue
			}
			uid, resourceVersion := resource.GetUID(), resource.GetResourceVersion()
			if err := r.Delete(ctx, resource, &client.DeleteOptions{
				Preconditions:     &metav1.Preconditions{UID: &uid, ResourceVersion: &resourceVersion},
				PropagationPolicy: &propagation,
			}); err != nil && !apierrors.IsNotFound(err) {
				return nil, err
			}
		}
	}

	return &lpxRetiring{retirementReason: reason}, nil
}

func digestAttemptKey(
	namespace string,
	name string,
	uid types.UID,
	model string,
	replicaIndex int32,
) string {
	h := sha256.New()
	writeLPXHashField(h, "namespace", namespace)
	writeLPXHashField(h, "name", name)
	writeLPXHashField(h, "uid", string(uid))
	writeLPXHashField(h, "model", model)
	if replicaIndex > 0 {
		writeLPXHashField(h, "group-replica", strconv.FormatInt(int64(replicaIndex), 10))
	}
	return fmt.Sprintf("sha256:%x", h.Sum(nil))
}

func writeLPXHashField(h hash.Hash, tag, value string) {
	var size [8]byte
	for _, field := range []string{tag, value} {
		binary.BigEndian.PutUint64(size[:], uint64(len(field)))
		_, _ = h.Write(size[:])
		_, _ = h.Write([]byte(field))
	}
}

// hasCurrentLPXPublicationAnnotations requires nonnil desired and its DGD; a nil object interface does not match.
func (desired *lpxMaterializing) hasCurrentLPXPublicationAnnotations(
	object metav1.Object,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
) bool {
	if object == nil || !object.GetDeletionTimestamp().IsZero() {
		return false
	}
	annotations := object.GetAnnotations()
	executionBackend := annotations[lpx.ExecutionBackendAnnotation]
	return annotations[lpxDeploymentUIDAnnotation] == string(deployment.UID) &&
		executionBackend == ""
}

func lpxRequestName(dgdName, digest string) string {
	prefix := strings.ReplaceAll(dgdName, ".", "-")
	if len(prefix) > 23 {
		prefix = strings.TrimRight(prefix[:23], "-")
	}
	hexDigest := strings.TrimPrefix(digest, "sha256:")
	return fmt.Sprintf("lpx-%s-%s", prefix, hexDigest[:32])
}

// reconcileLPXPublicationFence requires a nonnil desired whose requests remain immutable during the call.
// pcs is nil when the desired PodCliqueSet was not found; it cannot be recreated until garbage collection
// removes every request from its previous incarnation.
func (r *graphReconciler) reconcileLPXPublicationFence(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	pcs *grovev1alpha1.PodCliqueSet,
) (map[string]*lpxv1alpha1.LPUPipelineRequest, lpxClassification, error) {
	// Retain each desired position for digest lookup and collision precedence.
	desiredByName := make(map[string]int, len(desired.requests))
	for index := range desired.requests {
		desiredByName[desired.requests[index].requestName] = index
	}
	currents := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(desired.requests))
	foreignCollision := len(desired.requests)
	stale := make([]*lpxv1alpha1.LPUPipelineRequest, 0)
	requests, err := r.listLPXRequestCandidates(ctx, deployment)
	if err != nil {
		return nil, nil, err
	}
	for index := range requests.Items {
		request := &requests.Items[index]
		desiredIndex, desiredName := desiredByName[request.Name]
		ownedByChild := ownsLPXRequest(deployment, pcs, request)
		if !ownedByChild {
			// Preserve desired-order error precedence without retaining collision state.
			if desiredName {
				foreignCollision = min(foreignCollision, desiredIndex)
			}
			continue
		}
		if desiredName {
			currents[request.Name] = request
			continue
		}
		stale = append(stale, request)
	}
	// A deterministic-name collision must never be adopted or deleted.
	if foreignCollision < len(desired.requests) {
		return nil, nil, fmt.Errorf("LPX request name %q is occupied by a foreign owner", desired.requests[foreignCollision].requestName)
	}
	if pcs == nil && len(currents)+len(stale) > 0 {
		return currents, &lpxRetiring{retirementReason: "Waiting for garbage collection to remove the previous LPX requests"}, nil
	}
	if pcs != nil && pcs.Annotations[lpx.WorkloadDigestAnnotation] != desired.workloadDigest {
		if err := r.deleteLPXPodCliqueSet(ctx, deployment, pcs.Name, pcs.UID); err != nil {
			return nil, nil, err
		}
		return currents, &lpxRetiring{retirementReason: "The selected LPX workload shape changed"}, nil
	}
	if pcs != nil && !pcs.DeletionTimestamp.IsZero() {
		return currents, &lpxRetiring{retirementReason: "Waiting for the current PodCliqueSet to finish cleanup"}, nil
	}
	if len(stale) > 0 {
		if err := r.deleteStaleLPXRequests(ctx, deployment, stale); err != nil {
			return nil, nil, err
		}
		return currents, &lpxRetiring{
			retirementReason: "One or more engines are no longer requested",
			scope:            lpxRequestRetirement,
		}, nil
	}

	// Grove must not recreate pods while a retiring request still protects their names.
	for _, request := range desired.requests {
		if live := currents[request.requestName]; live != nil && !live.DeletionTimestamp.IsZero() {
			return currents, &lpxRetiring{
				retirementReason: "Waiting for the current LPX request to finish cleanup",
				scope:            lpxRequestRetirement,
			}, nil
		}
	}

	// A recorded deadline failure fences this generation even if the scheduler later reports a terminal phase.
	if lpxDeadlineFailureCurrent(deployment) {
		return currents, &lpxDeadlineExceeded{}, nil
	}
	return currents, nil, nil
}

// listLPXRequestCandidates uses the propagated deployment identity to avoid a
// namespace-wide scan. Ownership is still checked before a returned request is
// observed or changed.
func (r *graphReconciler) listLPXRequestCandidates(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
) (*lpxv1alpha1.LPUPipelineRequestList, error) {
	list := &lpxv1alpha1.LPUPipelineRequestList{}
	err := r.apiReader.List(ctx, list,
		client.InNamespace(deployment.Namespace),
		client.MatchingLabels{lpxOwnerUIDLabel: string(deployment.UID)},
	)
	return list, err
}

// observeCurrentLPXPodCliqueSet returns nil when the deployment's stable PCS
// name is absent from the informer cache.
func (r *graphReconciler) observeCurrentLPXPodCliqueSet(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
) (*grovev1alpha1.PodCliqueSet, error) {
	pcs := &grovev1alpha1.PodCliqueSet{}
	err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: dynamo.PCSNameForLPX(deployment)}, pcs)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	return pcs, err
}

// ownsLPXRequest follows the request's exact controller chain using the PCS
// already observed by the caller. pcs is nil after that PCS disappears.
func ownsLPXRequest(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	request *lpxv1alpha1.LPUPipelineRequest,
) bool {
	owner := metav1.GetControllerOf(request)
	if owner == nil || owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() || owner.Kind != lpxPodCliqueSetKind {
		return false
	}
	if pcs != nil {
		return metav1.IsControlledBy(pcs, deployment) && metav1.IsControlledBy(request, pcs)
	}

	// Keep observing requests from the missing PCS, not another owner with a copied label.
	return owner.Name == dynamo.PCSNameForLPX(deployment) && request.Labels[lpxOwnerUIDLabel] == string(deployment.UID)
}

// listOwnedLPXRequests filters the deployment-scoped list against the caller's
// PCS observation. pcs is nil after that owner disappears.
func (r *graphReconciler) listOwnedLPXRequests(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) ([]lpxv1alpha1.LPUPipelineRequest, error) {
	requests, err := r.listLPXRequestCandidates(ctx, deployment)
	if err != nil {
		return nil, err
	}
	owned := make([]lpxv1alpha1.LPUPipelineRequest, 0)
	for index := range requests.Items {
		request := &requests.Items[index]
		if ownsLPXRequest(deployment, pcs, request) {
			owned = append(owned, *request)
		}
	}
	sort.Slice(owned, func(i, j int) bool { return owned[i].Name < owned[j].Name })
	return owned, nil
}

// deleteLPXPodCliqueSet starts background cascading deletion of an exact PCS.
// A same-name replacement proves that the old UID is already absent and is
// never adopted or deleted.
func (r *graphReconciler) deleteLPXPodCliqueSet(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcsName string,
	pcsUID types.UID,
) error {
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.apiReader.Get(ctx, types.NamespacedName{Namespace: deployment.Namespace, Name: pcsName}, pcs); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if pcs.UID != pcsUID {
		return nil
	}
	if !metav1.IsControlledBy(pcs, deployment) {
		return fmt.Errorf("refusing to delete PodCliqueSet %q without the exact LPXGraphDeployment controller owner", pcs.Name)
	}
	if !pcs.DeletionTimestamp.IsZero() {
		return nil
	}
	if deployment.DeletionTimestamp.IsZero() {
		if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
			return err
		}
	}
	propagation := metav1.DeletePropagationBackground
	if err := r.Delete(ctx, pcs, &client.DeleteOptions{
		Preconditions:     &metav1.Preconditions{UID: &pcs.UID, ResourceVersion: &pcs.ResourceVersion},
		PropagationPolicy: &propagation,
	}); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// deleteStaleLPXRequests removes scaled-in scheduler intent without disturbing
// the shared PCS. Grove and the LPX scheduler converge their pods independently.
func (r *graphReconciler) deleteStaleLPXRequests(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	requests []*lpxv1alpha1.LPUPipelineRequest,
) error {
	if deployment.DeletionTimestamp.IsZero() {
		if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
			return err
		}
	}
	for _, request := range requests {
		if !request.DeletionTimestamp.IsZero() {
			continue
		}
		if request.UID == "" {
			return fmt.Errorf("refusing to delete LPX request %q without an immutable UID", request.Name)
		}
		uid, resourceVersion := request.UID, request.ResourceVersion
		if err := r.Delete(ctx, request, &client.DeleteOptions{Preconditions: &metav1.Preconditions{
			UID: &uid, ResourceVersion: &resourceVersion,
		}}); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete stale LPX request %q: %w", request.Name, err)
		}
	}
	return nil
}

// retireLPXEngineRequest removes the PCS whose immutable request no longer
// matches the selected workload.
func (r *graphReconciler) retireLPXEngineRequest(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	request *lpxv1alpha1.LPUPipelineRequest,
	reason string,
) (lpxClassification, error) {
	owner := metav1.GetControllerOf(request)
	if owner == nil || owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() ||
		owner.Kind != lpxPodCliqueSetKind || owner.UID == "" {
		return nil, fmt.Errorf("LPX request %q does not have an exact PodCliqueSet controller owner", request.Name)
	}
	if err := r.deleteLPXPodCliqueSet(ctx, deployment, owner.Name, owner.UID); err != nil {
		return nil, err
	}
	return &lpxRetiring{retirementReason: reason}, nil
}

func (r *graphReconciler) reconcileSelectedLPXFromCurrentRequests(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	currents map[string]*lpxv1alpha1.LPUPipelineRequest,
) (lpxClassification, error) {
	if currents == nil {
		currents = make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(desired.requests))
	}

	pcs, classification, err := r.reconcileSelectedLPXPodCliqueSet(
		ctx,
		deployment,
		desired,
		currents,
	)
	if classification != nil || err != nil {
		return classification, err
	}

	for index := range desired.requests {
		request := &desired.requests[index]
		if currents[request.requestName] != nil {
			continue
		}
		groveIdentity := &lpxGroveIdentity{pcs: pcs, plan: desired.plan.ForReplica(request.replicaIndex)}
		live, createErr := r.createPublishedLPXRequest(ctx, deployment, request, groveIdentity)
		if createErr != nil {
			return nil, createErr
		}
		currents[request.requestName] = live
	}

	return classifyLPXCurrentRequests(desired.requests, currents), nil
}

// classifyLPXCurrentRequests accepts absent requests.
func classifyLPXCurrentRequests(
	requests []lpxModelMaterializing,
	currents map[string]*lpxv1alpha1.LPUPipelineRequest,
) lpxClassification {
	// Stream terminal, side-effect-free classifications in request order.
	var firstNonBound lpxClassification
	for _, request := range requests {
		if live := currents[request.requestName]; live != nil {
			classification := classifyPublishedLPX(live)
			if state, ok := classification.(*lpxSchedulerObserved); ok && lpxSchedulerResult(state).State == nvidiacomv1beta1.DGDStateFailed {
				return classification
			}
			if _, bound := classification.(*lpxBound); !bound && firstNonBound == nil {
				firstNonBound = classification
			}
		}
	}

	if firstNonBound != nil {
		return firstNonBound
	}
	return &lpxBound{}
}

func (r *graphReconciler) reconcileSelectedLPXPodCliqueSet(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	currents map[string]*lpxv1alpha1.LPUPipelineRequest,
) (*grovev1alpha1.PodCliqueSet, lpxClassification, error) {
	pcs, incomplete, err := r.observeLPXPodCliqueSet(ctx, deployment, desired, len(currents) == 0)
	if err != nil {
		return nil, nil, err
	}

	if pcs == nil || incomplete != "" {
		if len(currents) == 0 {
			return nil, &lpxClosed{incomplete: incomplete}, nil
		}
		for _, request := range desired.requests {
			if live := currents[request.requestName]; live != nil {
				retiring, err := r.retireLPXEngineRequest(ctx, deployment, live, fmt.Sprintf("The published PCS identity became unavailable: %q", incomplete))
				return nil, retiring, err
			}
		}
		return nil, &lpxClosed{incomplete: incomplete}, nil
	}

	// The PCS owns each request; LPX resolves the replica-local clique references itself.
	for index := range desired.requests {
		request := &desired.requests[index]
		identity := &lpxGroveIdentity{pcs: pcs, plan: desired.plan.ForReplica(request.replicaIndex)}
		if live := currents[request.requestName]; live != nil {
			if err := validateCurrentLPXRequest(deployment, request, identity, live); err != nil {
				retiring, retireErr := r.retireLPXEngineRequest(ctx, deployment, live, err.Error())
				return nil, retiring, retireErr
			}
		}
	}
	return pcs, nil, nil
}

// classifyPublishedLPX classifies a non-nil published request from its current scheduler receipt.
func classifyPublishedLPX(current *lpxv1alpha1.LPUPipelineRequest) lpxClassification {
	if current == nil || current.Status == nil || current.Status.ObservedGeneration == nil ||
		*current.Status.ObservedGeneration != current.Generation {
		return &lpxOpen{}
	}
	if current.Status.Phase == lpxv1alpha1.RequestPhaseBound {
		return &lpxBound{}
	}
	observed := lpxSchedulerObserved(*current.Status.DeepCopy())
	return &observed
}

func (r *graphReconciler) createPublishedLPXRequest(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	materializing *lpxModelMaterializing,
	grove *lpxGroveIdentity,
) (*lpxv1alpha1.LPUPipelineRequest, error) {
	if err := r.validateLPXPublicationSource(ctx, deployment); err != nil {
		return nil, err
	}
	spec := materializing.modelProjection.RequestSpec(grove.plan, materializing.agentCliqueName)
	request := &lpxv1alpha1.LPUPipelineRequest{
		TypeMeta: metav1.TypeMeta{
			APIVersion: lpxv1alpha1.GroupVersion.String(),
			Kind:       "LpuPipelineRequest",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      materializing.requestName,
			Namespace: deployment.Namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: metav1.GetControllerOf(deployment).Name,
				lpxOwnerUIDLabel: string(deployment.UID),
			},
			Annotations: lpxRequestAnnotations(deployment, materializing),
		},
		Spec: spec,
	}
	if err := controllerutil.SetControllerReference(grove.pcs, request, r.Client.Scheme()); err != nil {
		return nil, err
	}
	if err := r.Create(ctx, request); err != nil {
		return nil, fmt.Errorf(
			"create LPX request %q for model %q: %w",
			request.Name,
			materializing.modelProjection.Model(),
			err,
		)
	}
	return request, nil
}

func lpxRequestAnnotations(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	materializing *lpxModelMaterializing,
) map[string]string {
	// Callers validate the exact source controller owner before constructing requests.
	sourceOwner := metav1.GetControllerOf(deployment)
	return map[string]string{
		lpx.DGDUIDAnnotation:                         string(sourceOwner.UID),
		lpx.DeploymentNameAnnotation:                 deployment.Name,
		lpxDeploymentUIDAnnotation:                   string(deployment.UID),
		lpxModelAnnotation:                           materializing.modelProjection.Model(),
		lpx.WorkloadDigestAnnotation:                 materializing.modelProjection.Digest().String(),
		lpxv1alpha1.CompilerSnapshotDigestAnnotation: materializing.modelProjection.CompilerSnapshotDigest(),
	}
}

func validateCurrentLPXRequest(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxModelMaterializing,
	grove *lpxGroveIdentity,
	request *lpxv1alpha1.LPUPipelineRequest,
) error {
	if !metav1.IsControlledBy(request, grove.pcs) {
		return fmt.Errorf("LPX request %q does not have the exact PCS owner", request.Name)
	}
	if request.UID == "" {
		return fmt.Errorf("LPX request %q has no immutable UID", request.Name)
	}
	wantAnnotations := lpxRequestAnnotations(deployment, desired)
	for key, value := range wantAnnotations {
		if request.Annotations[key] != value {
			return fmt.Errorf("LPX request %q immutable annotation %q no longer matches the current deployment", request.Name, key)
		}
	}
	wantSpec := desired.modelProjection.RequestSpec(grove.plan, desired.agentCliqueName)
	if !apiequality.Semantic.DeepEqual(request.Spec, wantSpec) {
		return fmt.Errorf("LPX request %q immutable intent no longer matches the current publication", request.Name)
	}
	return nil
}

func (r *graphReconciler) observeLPXPodCliqueSet(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	noCurrentRequests bool,
) (*grovev1alpha1.PodCliqueSet, string, error) {
	pcs := &grovev1alpha1.PodCliqueSet{}
	key := types.NamespacedName{Namespace: deployment.Namespace, Name: desired.plan.PodCliqueSetName}
	if err := r.apiReader.Get(ctx, key, pcs); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, "Waiting for the selected Grove PodCliqueSet", nil
		}
		return nil, "", err
	}
	if noCurrentRequests && !metav1.IsControlledBy(pcs, deployment) {
		return nil, "", fmt.Errorf("refusing to inspect PodCliqueSet %q without the exact LPXGraphDeployment controller owner", pcs.Name)
	}
	if pcs.UID == "" || !pcs.DeletionTimestamp.IsZero() || !metav1.IsControlledBy(pcs, deployment) {
		return nil, "The selected Grove PodCliqueSet lacks the exact LPXGraphDeployment owner identity", nil
	}

	incomplete := ""
	if !desired.hasCurrentLPXPublicationAnnotations(pcs, deployment) {
		incomplete = "The selected Grove PodCliqueSet is stale"
	}
	if pcs.Spec.Replicas != 1 {
		if incomplete == "" {
			incomplete = fmt.Sprintf("The selected Grove PodCliqueSet has %d replicas, want 1", pcs.Spec.Replicas)
		}
	}
	return pcs, incomplete, nil
}
