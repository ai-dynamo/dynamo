/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	groveschedulerv1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

const (
	lpxAttemptDigestAnnotation = "scheduling.lpu.nvidia.com/dynamo-attempt-digest"
	lpxDeploymentUIDAnnotation = dynamo.LPXDeploymentUIDAnnotation
	lpxModelAnnotation         = "scheduling.lpu.nvidia.com/dynamo-model"
	lpxPCSUIDAnnotation        = "scheduling.lpu.nvidia.com/podcliqueset-uid"
	lpxPodGangUIDAnnotation    = "scheduling.lpu.nvidia.com/podgang-uid"
	lpxRetirementRequeueAfter  = 5 * time.Second
	lpxLifecycleListPageSize   = 64
	maxDGDConditionMessageSize = 32768
)

const (
	lpxRetiringReason               string = "LPXRetiring"
	lpxSchedulerStatusInvalidReason string = "LPXSchedulerStatusInvalid"
)

var errLPXEngineReplacing = errors.New("Grove is replacing the engine's PodCliques")

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
	workloadDigest  string
	requests        []lpxModelMaterializing
	workload        *lpx.SelectedWorkload
	plan            *lpx.MaterializationPlan
	deadlineSeconds *int64
}

type reconcileOutcome struct {
	State   nvidiacomv1beta1.DGDState
	Reason  string
	Message string
}

// lpxModelMaterializing is the request-local state for one model projection.
// The enclosing lpxMaterializing owns the shared Grove identity.
type lpxModelMaterializing struct {
	requestName     string
	attemptDigest   string
	modelProjection *lpx.ModelProjection
	replicaIndex    int32
}

type lpxGroveSnapshot struct {
	podCliqueSet   *grovev1alpha1.PodCliqueSet
	scalingGroup   *grovev1alpha1.PodCliqueScalingGroup
	generationHash string
	incomplete     string
	podGangs       []groveschedulerv1alpha1.PodGang
}

type lpxGroveIdentity struct {
	pcsUID       types.UID
	podGangName  string
	podGangUID   types.UID
	cyborgClique *lpxv1alpha1.CyborgPodCliqueReference
	agentUIDs    []types.UID
	complete     bool
}

type lpxClosed struct {
	incomplete string
}

type lpxOpen struct{}

type lpxBound struct{}

type lpxSchedulerObserved lpxv1alpha1.LPUPipelineRequestStatus

type lpxRetiring struct {
	retirementReason string
}

// Error carries retirement through Grove's error-shaped write boundary.
func (*lpxRetiring) Error() string { return "LPX publication retired before Grove spec write" }

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
		result.Message = lpxMessageWithDiagnostics(result.Message, state.Diagnostics)
	case *lpxRetiring:
		result.Reason = lpxRetiringReason
		result.Message = state.retirementReason
	case *lpxDeadlineTransition:
		result = state.result
	default:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = "LPXInternalClassificationError"
		result.Message = "Unknown LPX reconciliation classification"
	}
	return result
}

// lpxSchedulerResult classifies the receipt without formatting diagnostics before selection.
func lpxSchedulerResult(state *lpxSchedulerObserved) reconcileOutcome {
	result := reconcileOutcome{State: nvidiacomv1beta1.DGDStatePending}
	switch state.Phase {
	case lpxv1alpha1.RequestPhasePending:
		result.Reason = "LPXSchedulerPending"
		result.Message = "LPX scheduler is waiting to plan the current request"
	case lpxv1alpha1.RequestPhaseNoFit:
		result.Reason = "LPXNoFit"
		result.Message = "LPX scheduler found no placement for the current request"
	case lpxv1alpha1.RequestPhaseUnsupported:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = "LPXUnsupported"
		result.Message = "LPX scheduler cannot support the current request"
	case lpxv1alpha1.RequestPhasePlanned:
		result.Reason = "LPXPlanned"
		result.Message = "LPX scheduler committed a placement plan for the current request"
	case lpxv1alpha1.RequestPhaseReserving:
		result.Reason = "LPXReserving"
		result.Message = "LPX scheduler is reserving the planned LPU allocation"
	case lpxv1alpha1.RequestPhaseBinding:
		result.Reason = "LPXBinding"
		result.Message = "LPX scheduler is binding the planned workload Pods"
	case lpxv1alpha1.RequestPhaseDegraded:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = "LPXDegraded"
		result.Message = "LPX scheduler detected that the committed plan is no longer valid"
	case lpxv1alpha1.RequestPhaseReleasing, lpxv1alpha1.RequestPhaseReleased:
		result = lpxSchedulerReleaseResult(state)
	case lpxv1alpha1.RequestPhaseBound:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = lpxSchedulerStatusInvalidReason
		result.Message = "LPX scheduler reported Bound without the exact current-generation plan proof required by Dynamo"
	default:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = lpxSchedulerStatusInvalidReason
		result.Message = fmt.Sprintf("LPX scheduler reported unknown phase %q", state.Phase)
	}
	return result
}

func lpxSchedulerReleaseResult(state *lpxSchedulerObserved) reconcileOutcome {
	result := reconcileOutcome{State: nvidiacomv1beta1.DGDStatePending}
	phase := "releasing"
	result.Reason = "LPXReleasing"
	result.Message = "LPX scheduler is releasing the current plan"
	if state.Phase == lpxv1alpha1.RequestPhaseReleased {
		phase = "released"
		result.Reason = "LPXReleased"
		result.Message = "LPX scheduler released the current plan and is finalizing its journal"
	}
	if state.Committed == nil || state.Committed.Execution.Release == nil {
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = lpxSchedulerStatusInvalidReason
		result.Message = fmt.Sprintf("LPX scheduler reported %s without the required release journal", phase)
		return result
	}
	releaseReason := state.Committed.Execution.Release.Reason
	switch releaseReason {
	case lpxv1alpha1.ReleaseReasonDependencyChanged, lpxv1alpha1.ReleaseReasonBindingFailed:
		result.State = nvidiacomv1beta1.DGDStateFailed
	case lpxv1alpha1.ReleaseReasonRequestChanged, lpxv1alpha1.ReleaseReasonRequestDeleted:
	default:
		result.State = nvidiacomv1beta1.DGDStateFailed
		result.Reason = lpxSchedulerStatusInvalidReason
		result.Message = fmt.Sprintf("LPX scheduler reported %s with unknown release reason %q", phase, releaseReason)
		return result
	}
	result.Message = fmt.Sprintf("%s after %s", result.Message, releaseReason)
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
func (r *graphReconciler) reconcileLPXKnownIntentFence(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	requests []lpxv1alpha1.LPUPipelineRequest,
) (lpxClassification, []lpxv1alpha1.LPUPipelineRequest, error) {
	var err error
	if requests == nil {
		requests, err = r.listOwnedLPXRequests(ctx, deployment)
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
		retiring, retireErr := r.retireLPXRequest(ctx, deployment, "", request, reason)
		return retiring, nil, retireErr
	}
	return nil, requests, nil
}

func (r *graphReconciler) prepareLPXMaterializing(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
) (*lpxMaterializing, *lpxRejected, error) {
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
		} else {
			pcs := &grovev1alpha1.PodCliqueSet{}
			if err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: plan.PodCliqueSetName}, pcs); err != nil {
				return nil, nil, err
			}
			if !metav1.IsControlledBy(pcs, deployment) || !metav1.IsControlledBy(group, pcs) {
				return nil, nil, fmt.Errorf("refusing to use replicas from a foreign LPX scaling group")
			}
			plan.Replicas = group.Spec.Replicas
		}
	}
	if err := plan.ValidateReplicaCount(); err != nil {
		return nil, &lpxRejected{reason: err.Error()}, nil
	}
	workloadDigest := workload.Digest().String()
	requests := make([]lpxModelMaterializing, 0, len(projections)*int(plan.Replicas))
	for replicaIndex := int32(0); replicaIndex < plan.Replicas; replicaIndex++ {
		for _, projection := range projections {
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
			})
		}
	}
	return &lpxMaterializing{
		workloadDigest:  workloadDigest,
		requests:        requests,
		workload:        workload,
		plan:            plan,
		deadlineSeconds: lpxAttemptDeadlineSeconds(source),
	}, nil, nil
}

// retireInvalidLPXWorkload also retires staging that never acquired an LPR.
// Healthy publication and spec-write fences intentionally do not use this path.
func (r *graphReconciler) retireInvalidLPXWorkload(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	reason string,
) (*lpxRetiring, error) {
	// Preserve request-first retirement before removing unpublished staging.
	requests, err := r.listOwnedLPXRequests(ctx, deployment)
	if err != nil {
		return nil, err
	}
	if len(requests) > 0 {
		return r.retireLPXRequest(ctx, deployment, "", &requests[0], reason)
	}

	// Source edits can rename resources; discover only this exact child's old objects.
	owned := make([]client.Object, 0)
	for _, list := range []client.ObjectList{&corev1.ServiceList{}, &grovev1alpha1.PodCliqueSetList{}, &corev1.ConfigMapList{}} {
		if err := visitLifecycleObjectPages(ctx, r.apiReader, list, func(object k8sruntime.Object) error {
			resource := object.(client.Object)
			if metav1.IsControlledBy(resource, deployment) {
				owned = append(owned, resource)
			}
			return nil
		}, client.InNamespace(deployment.Namespace)); err != nil {
			return nil, err
		}
	}
	if len(owned) == 0 {
		return nil, nil
	}

	// A newer child seen during listing must not lose its staging to this stale failure.
	if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
		return nil, err
	}
	for _, resource := range owned {
		if !resource.GetDeletionTimestamp().IsZero() {
			continue
		}
		uid, resourceVersion := resource.GetUID(), resource.GetResourceVersion()
		if err := r.Delete(ctx, resource, &client.DeleteOptions{Preconditions: &metav1.Preconditions{
			UID: &uid, ResourceVersion: &resourceVersion,
		}}); err != nil && !apierrors.IsNotFound(err) {
			return nil, err
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

// hasCurrentLPXAttemptAnnotations requires nonnil desired and its DGD; a nil object interface does not match.
func (desired *lpxMaterializing) hasCurrentLPXAttemptAnnotations(
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

// reconcileLPXAttemptFence requires a nonnil desired whose requests remain immutable during the call.
func (r *graphReconciler) reconcileLPXAttemptFence(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
) (map[string]*lpxv1alpha1.LPUPipelineRequest, lpxClassification, error) {
	// Retain each desired position for digest lookup and collision precedence.
	desiredByName := make(map[string]int, len(desired.requests))
	for index := range desired.requests {
		desiredByName[desired.requests[index].requestName] = index
	}
	currents := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(desired.requests))
	foreignCollision := len(desired.requests)
	var stale *lpxv1alpha1.LPUPipelineRequest
	if err := visitLifecycleObjectPages(
		ctx,
		r.apiReader,
		&lpxv1alpha1.LPUPipelineRequestList{},
		func(object k8sruntime.Object) error {
			request := object.(*lpxv1alpha1.LPUPipelineRequest)
			desiredIndex, desiredName := desiredByName[request.Name]
			if !metav1.IsControlledBy(request, deployment) {
				// Preserve desired-order error precedence without retaining collision state.
				if desiredName {
					foreignCollision = min(foreignCollision, desiredIndex)
				}
				return nil
			}
			if desiredName {
				currents[request.Name] = request
				return nil
			}
			if stale == nil || request.Name < stale.Name {
				stale = request
			}
			return nil
		},
		client.InNamespace(deployment.Namespace),
	); err != nil {
		return nil, nil, err
	}
	if stale != nil {
		// Persist cancellation or a replacement scheduling batch before retiring removed engines.
		if attempt := currentLPXAttemptStatus(deployment); attempt != nil && attempt.ExceededAt == nil {
			want := desiredLPXAttemptStatus(desired, deployment.Generation, attempt.PodCliqueSetUID)
			if transition := r.reconcileLPXAttemptPreparation(deployment, desired.deadlineSeconds, want, currents); transition != nil {
				return currents, transition, nil
			}
		}
		retiring, retireErr := r.retireLPXEngineRequest(ctx, deployment, stale, "The engine is no longer requested", true)
		return currents, retiring, retireErr
	}
	// Keep exact cleanup authority until the replaced request is authoritatively absent.
	if attempt := currentLPXAttemptStatus(deployment); attempt != nil {
		for index, row := range attempt.Requests {
			if row.UID == "" || row.AttemptDigest != "" {
				continue
			}
			if request := currents[row.Name]; request != nil && request.UID == row.UID {
				retiring, retireErr := r.retireLPXEngineRequest(ctx, deployment, request, "The engine's previous request is retiring", true)
				return currents, retiring, retireErr
			}
			next := attempt.DeepCopy()
			next.Requests[index].UID = ""
			return currents, newLPXDeadlineTransition(next, nvidiacomv1beta1.DGDStatePending,
				lpxAttemptPreparedReason, "The previous engine request has finished cleanup", time.Nanosecond), nil
		}
	}
	// A deterministic-name collision must never be adopted or deleted.
	if foreignCollision < len(desired.requests) {
		return nil, nil, fmt.Errorf("LPX request name %q is occupied by a foreign owner", desired.requests[foreignCollision].requestName)
	}
	return currents, nil, nil
}

func visitLifecycleObjectPages(
	ctx context.Context,
	reader client.Reader,
	list client.ObjectList,
	visit func(k8sruntime.Object) error,
	opts ...client.ListOption,
) error {
	continueToken := ""
	pageOptions := make([]client.ListOption, 0, len(opts)+2)
	for {
		page := list.DeepCopyObject().(client.ObjectList)
		pageOptions = pageOptions[:0]
		pageOptions = append(pageOptions, opts...)
		pageOptions = append(pageOptions, client.Limit(lpxLifecycleListPageSize))
		if continueToken != "" {
			pageOptions = append(pageOptions, client.Continue(continueToken))
		}
		if err := reader.List(ctx, page, pageOptions...); err != nil {
			return err
		}
		if err := meta.EachListItem(page, visit); err != nil {
			return err
		}
		nextToken := page.GetContinue()
		if nextToken == "" {
			break
		}
		if nextToken == continueToken {
			return fmt.Errorf("lifecycle list %T repeated continuation token %q", list, nextToken)
		}
		continueToken = nextToken
	}
	return nil
}

func (r *graphReconciler) listOwnedLPXRequests(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
) ([]lpxv1alpha1.LPUPipelineRequest, error) {
	requests, available, err := r.listOwnedLPXRequestsIfAvailable(ctx, deployment)
	if err == nil && !available {
		return nil, fmt.Errorf("LPX request API is unavailable after startup discovery")
	}
	return requests, err
}

func (r *graphReconciler) listOwnedLPXRequestsIfAvailable(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
) ([]lpxv1alpha1.LPUPipelineRequest, bool, error) {
	list := &lpxv1alpha1.LPUPipelineRequestList{}
	owned := make([]lpxv1alpha1.LPUPipelineRequest, 0)
	visit := func(object k8sruntime.Object) error {
		request := object.(*lpxv1alpha1.LPUPipelineRequest)
		if metav1.IsControlledBy(request, deployment) {
			owned = append(owned, *request)
		}
		return nil
	}
	err := visitLifecycleObjectPages(
		ctx,
		r.apiReader,
		list,
		visit,
		client.InNamespace(deployment.Namespace),
	)
	if err != nil {
		if meta.IsNoMatchError(err) || k8sruntime.IsNotRegisteredError(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	sort.Slice(owned, func(i, j int) bool { return owned[i].Name < owned[j].Name })
	return owned, true, nil
}

// retireFirstOwnedLPXRequest uses the sorted authoritative first request as the retirement witness.
func (r *graphReconciler) retireFirstOwnedLPXRequest(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcsName string,
	reason string,
) (*lpxRetiring, error) {
	requests, err := r.listOwnedLPXRequests(ctx, deployment)
	if err != nil || len(requests) == 0 {
		return nil, err
	}
	return r.retireLPXRequest(ctx, deployment, pcsName, &requests[0], reason)
}

func (r *graphReconciler) retireLPXRequest(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcsName string,
	request *lpxv1alpha1.LPUPipelineRequest,
	reason string,
) (*lpxRetiring, error) {
	if request == nil || !metav1.IsControlledBy(request, deployment) {
		return nil, fmt.Errorf("refusing to retire an LPX request without the exact LPXGraphDeployment controller owner")
	}
	if err := r.releaseLPXAttemptRecordingFinalizer(ctx, request); err != nil {
		return nil, err
	}

	// Read the complete publication set directly from the API before changing shared Grove state.
	type requestDeletePrecondition struct {
		name            string
		uid             types.UID
		resourceVersion string
	}
	owned := make([]requestDeletePrecondition, 0)
	if err := visitLifecycleObjectPages(
		ctx,
		r.apiReader,
		&lpxv1alpha1.LPUPipelineRequestList{},
		func(object k8sruntime.Object) error {
			live := object.(*lpxv1alpha1.LPUPipelineRequest)
			if metav1.IsControlledBy(live, deployment) && live.DeletionTimestamp.IsZero() {
				owned = append(owned, requestDeletePrecondition{
					name:            live.Name,
					uid:             live.UID,
					resourceVersion: live.ResourceVersion,
				})
			}
			return nil
		},
		client.InNamespace(deployment.Namespace),
	); err != nil {
		return nil, fmt.Errorf("list the exact LPXGraphDeployment-owned LPX requests for attempt-wide retirement: %w", err)
	}

	// Delete every live publication with exact-object preconditions before scaling the shared PCS.
	for _, current := range owned {
		live := &lpxv1alpha1.LPUPipelineRequest{ObjectMeta: metav1.ObjectMeta{
			Namespace:       deployment.Namespace,
			Name:            current.name,
			UID:             current.uid,
			ResourceVersion: current.resourceVersion,
		}}
		if err := r.Delete(ctx, live, &client.DeleteOptions{
			Preconditions: &metav1.Preconditions{UID: &current.uid, ResourceVersion: &current.resourceVersion},
		}); err != nil && !apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("delete exact LPXGraphDeployment-owned LPX request %q: %w", current.name, err)
		}
	}

	// Hold the shared Grove attempt at zero only after every publication is retiring.
	if err := r.scaleDownLPXPodCliqueSet(ctx, deployment, pcsName, ""); err != nil {
		return nil, fmt.Errorf("scale down the LPX PodCliqueSet after retiring all attempt requests: %w", err)
	}
	return &lpxRetiring{
		retirementReason: reason,
	}, nil
}

// scaleDownLPXPodCliqueSet holds the complete Grove attempt at zero while LPX
// releases its deleted requests. LPX requires every exact old workload Pod to be
// absent and treats a same-name replacement UID as a cleanup blocker.
// Grove supports mutable PodCliqueSet scale-to-zero and deletes every child
// replica without changing the immutable clique composition.
func (r *graphReconciler) scaleDownLPXPodCliqueSet(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcsName string,
	pcsUID types.UID,
) error {
	scale := func(pcs *grovev1alpha1.PodCliqueSet) error {
		if pcsUID != "" && pcs.UID != pcsUID {
			return nil
		}
		if !metav1.IsControlledBy(pcs, deployment) {
			return fmt.Errorf("refusing to scale down PodCliqueSet %q without the exact LPXGraphDeployment controller owner", pcs.Name)
		}
		if !pcs.DeletionTimestamp.IsZero() || pcs.Spec.Replicas == 0 {
			return nil
		}
		pcs.Spec.Replicas = 0
		return r.Update(ctx, pcs)
	}
	if pcsName != "" {
		pcs := &grovev1alpha1.PodCliqueSet{}
		if err := r.apiReader.Get(ctx, types.NamespacedName{Namespace: deployment.Namespace, Name: pcsName}, pcs); err != nil {
			return client.IgnoreNotFound(err)
		}
		return scale(pcs)
	}
	return visitLifecycleObjectPages(ctx, r.apiReader, &grovev1alpha1.PodCliqueSetList{}, func(object k8sruntime.Object) error {
		pcs := object.(*grovev1alpha1.PodCliqueSet)
		if !metav1.IsControlledBy(pcs, deployment) {
			return nil
		}
		return scale(pcs)
	}, client.InNamespace(deployment.Namespace))
}

func (r *graphReconciler) reconcileSelectedLPX(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
) (lpxClassification, error) {
	// Retiring a removed engine must not block publication for surviving engines.
	currents, retiring, err := r.reconcileLPXAttemptFence(ctx, deployment, desired)
	if err != nil {
		return nil, err
	}
	if _, recording := retiring.(*lpxDeadlineTransition); recording {
		return retiring, nil
	}
	classification, err := r.reconcileSelectedLPXFromCurrentRequests(ctx, deployment, desired, currents, true)
	if retiring != nil && err == nil {
		if _, recording := classification.(*lpxDeadlineTransition); !recording {
			return retiring, nil
		}
	}
	return classification, err
}

//nolint:gocyclo // Current-request retirement must remain visibly ordered before Grove observation.
func (r *graphReconciler) reconcileSelectedLPXFromCurrentRequests(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	currents map[string]*lpxv1alpha1.LPUPipelineRequest,
	allowPublication bool,
) (lpxClassification, error) {
	if currents == nil {
		currents = make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(desired.requests))
	}

	groveIdentities, pending, classification, err := r.reconcileSelectedLPXGroveIdentityPrefix(
		ctx,
		deployment,
		desired,
		currents,
	)
	if classification != nil || err != nil {
		return classification, err
	}

	// Validate staged identities without crossing the scheduler-publication boundary.
	if !allowPublication {
		for index := range desired.requests {
			request := &desired.requests[index]
			identity := groveIdentities[request.replicaIndex]
			if identity != nil && identity.complete && currents[request.requestName] == nil {
				return &lpxClosed{
					incomplete: "The Grove publication identities are staged; LPX publication waits for Grove synchronization",
				}, nil
			}
		}
		if pending != nil {
			return pending, nil
		}
	}

	// Prepare the batch from any ready engine, independently of replica zero's rollout.
	var attemptProjection *nvidiacomv1beta1.LPXAttemptStatus
	trackAttempt := desired.deadlineSeconds != nil || currentLPXAttemptStatus(deployment) != nil
	for _, baseGroveIdentity := range groveIdentities {
		if baseGroveIdentity == nil || !baseGroveIdentity.complete {
			continue
		}
		if trackAttempt {
			attemptProjection = desiredLPXAttemptStatus(desired, deployment.Generation, baseGroveIdentity.pcsUID)
		}
		transition := r.reconcileLPXAttemptPreparation(deployment, desired.deadlineSeconds, attemptProjection, currents)
		if transition != nil {
			return transition, nil
		}
		break
	}
	for index := range desired.requests {
		request := &desired.requests[index]
		groveIdentity := groveIdentities[request.replicaIndex]
		if currents[request.requestName] != nil || groveIdentity == nil || !groveIdentity.complete {
			continue
		}
		if attemptProjection != nil {
			// Classify known authority loss before attempting a replacement publication.
			attempt := currentLPXAttemptStatus(deployment)
			if _, _, missing := exactLPXAttemptState(attempt, currents); missing {
				return newLPXDeadlineTransition(attempt, nvidiacomv1beta1.DGDStateFailed,
					lpxAttemptAuthorityLostReason, "An exact LPX request disappeared; retiring its scheduling batch without replacement", 0), nil
			}
			if err := r.revalidateLPXAttemptPublication(ctx, deployment, desired.plan.PodCliqueSetName); err != nil {
				return nil, err
			}
		}
		live, createErr := r.createPublishedLPXRequest(ctx, deployment, request, groveIdentity, trackAttempt)
		if createErr != nil {
			return nil, createErr
		}
		currents[request.requestName] = live
		transition := r.reconcileLPXAttemptPreparation(deployment, desired.deadlineSeconds, attemptProjection, currents)
		if transition != nil {
			return transition, nil
		}
	}

	// Stream terminal, side-effect-free classifications in request order.
	var firstNonBound lpxClassification
	for _, request := range desired.requests {
		if live := currents[request.requestName]; live != nil {
			classification := classifyPublishedLPX(live)
			if state, ok := classification.(*lpxSchedulerObserved); ok && lpxSchedulerResult(state).State == nvidiacomv1beta1.DGDStateFailed {
				return classification, nil
			}
			if _, bound := classification.(*lpxBound); !bound && firstNonBound == nil {
				firstNonBound = classification
			}
		}
	}

	// Keep pending behind every published non-bound request.
	if firstNonBound != nil {
		return firstNonBound, nil
	}
	if pending != nil {
		return pending, nil
	}
	return &lpxBound{}, nil
}

func (r *graphReconciler) reconcileSelectedLPXGroveIdentityPrefix(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	currents map[string]*lpxv1alpha1.LPUPipelineRequest,
) ([]*lpxGroveIdentity, *lpxClosed, lpxClassification, error) {
	snapshot, incomplete, err := r.observeLPXGroveSnapshot(ctx, deployment, desired, len(currents) == 0)
	if err != nil {
		return nil, nil, nil, err
	}
	if snapshot == nil {
		return nil, nil, &lpxClosed{incomplete: incomplete}, nil
	}

	groveIdentities := make([]*lpxGroveIdentity, desired.plan.Replicas)
	replicaPlans := make([]*lpx.MaterializationPlan, desired.plan.Replicas)
	allAgentGroups := make(map[string]struct{}, int(desired.plan.Replicas)*len(desired.plan.Agents))
	for replicaIndex := int32(0); replicaIndex < desired.plan.Replicas; replicaIndex++ {
		plan := desired.plan.ForReplica(replicaIndex)
		replicaPlans[replicaIndex] = plan
		for _, agent := range plan.Agents {
			allAgentGroups[agent.CliqueName] = struct{}{}
		}
	}
	var pending *lpxClosed
	retire := make(map[int32]bool)
	intentional := make(map[int32]bool)
	for replicaIndex, plan := range replicaPlans {
		identity, incomplete, err := r.observeLPXGroveIdentity(ctx, deployment, desired, snapshot, plan, allAgentGroups)
		if err != nil {
			var drift *lpxRetiring
			if errors.Is(err, errLPXEngineReplacing) {
				intentional[int32(replicaIndex)] = true
				incomplete = err.Error()
			} else if errors.As(err, &drift) {
				incomplete = drift.retirementReason
			} else {
				return nil, nil, nil, err
			}
			retire[int32(replicaIndex)] = true
		}
		if incomplete != "" {
			if pending == nil {
				pending = &lpxClosed{incomplete: incomplete}
			}
		}
		if identity != nil {
			identity.complete = incomplete == ""
			groveIdentities[replicaIndex] = identity
		}
	}

	// Observe retirement per engine; an incomplete sibling never fences this engine.
	for index := range desired.requests {
		request := &desired.requests[index]
		live := currents[request.requestName]
		identity := groveIdentities[request.replicaIndex]
		if identity != nil && identity.complete {
			spec, err := json.Marshal(struct {
				Spec      lpxv1alpha1.LPUPipelineRequestSpec
				AgentUIDs []types.UID
			}{request.modelProjection.RequestSpec(deployment.Namespace, identity.podGangName, identity.cyborgClique), identity.agentUIDs})
			if err != nil {
				return nil, nil, nil, err
			}
			request.attemptDigest = fmt.Sprintf("sha256:%x", sha256.Sum256(spec))
		} else if live != nil {
			request.attemptDigest = live.Annotations[lpxAttemptDigestAnnotation]
		}
		if live == nil {
			continue
		}
		if identity != nil && identity.complete && validateCurrentLPXRequest(deployment, request, identity, live) != nil {
			intentional[request.replicaIndex] = true
		}
		if intentional[request.replicaIndex] || !live.DeletionTimestamp.IsZero() || lpx.ValidateRequestSize(live) != nil {
			retire[request.replicaIndex] = true
		}
	}

	// All model requests of a replacing engine must finish before any is republished.
	for index := range desired.requests {
		request := &desired.requests[index]
		if !retire[request.replicaIndex] {
			continue
		}
		request.attemptDigest = ""
		groveIdentities[request.replicaIndex] = nil
		pending = &lpxClosed{incomplete: "Waiting for the replaced engine's scheduler requests to finish cleanup"}
		if live := currents[request.requestName]; live != nil {
			classification, err := r.retireLPXEngineRequest(ctx, deployment, live, pending.incomplete, intentional[request.replicaIndex])
			if transition, recording := classification.(*lpxDeadlineTransition); recording {
				if _, completed, _ := exactLPXAttemptState(currentLPXAttemptStatus(deployment), currents); completed {
					disarmedAt := metav1.Now()
					transition.attempt.DisarmedAt = &disarmedAt
				}
				return nil, nil, classification, err
			}
			if err != nil {
				return nil, nil, classification, err
			}
			delete(currents, request.requestName)
		}
	}
	return groveIdentities, pending, nil, nil
}

// classifyPublishedLPX classifies a non-nil published request from its current scheduler receipt.
func classifyPublishedLPX(current *lpxv1alpha1.LPUPipelineRequest) lpxClassification {
	// Expose only scheduler status observed for the current request generation.
	status := current.Status
	if current.Generation <= 0 || status == nil || status.ObservedGeneration == nil || *status.ObservedGeneration != current.Generation {
		return &lpxOpen{}
	}

	// Bound requires one complete committed receipt for this exact request and plan.
	committed := status.Committed
	if status.Phase == lpxv1alpha1.RequestPhaseBound && committed != nil &&
		committed.Execution.AcceptedGeneration != nil && *committed.Execution.AcceptedGeneration == current.Generation &&
		committed.Plan.PlannedFromGeneration == current.Generation &&
		committed.Plan.Revision > 0 && committed.Plan.Revision == status.LastPlanRevision &&
		committed.Plan.PlanDigest != "" && committed.Plan.Placement.ExecutionBackend == current.Spec.ExecutionBackend &&
		current.Spec.ExecutionBackend == lpxv1alpha1.ExecutionBackendNodeLocal &&
		committed.Plan.Placement.NodeLocal != nil && committed.Plan.Placement.RemoteDra == nil &&
		committed.Execution.NodeLocal != nil && committed.Execution.RemoteDra == nil {
		return &lpxBound{}
	}
	return (*lpxSchedulerObserved)(status)
}

func (r *graphReconciler) fenceLPXPublicationBeforeGroveSpecWrite(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcsName string,
) error {
	if err := r.validateLPXPublicationSource(ctx, deployment); err != nil {
		return err
	}
	retiring, err := r.retireFirstOwnedLPXRequest(ctx, deployment, pcsName, "LPX publication was retired before synchronizing a Grove PodCliqueSet spec change")
	if err == nil && retiring != nil {
		err = retiring
	}
	return err
}

func (r *graphReconciler) createPublishedLPXRequest(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	materializing *lpxModelMaterializing,
	grove *lpxGroveIdentity,
	recordAttempt bool,
) (*lpxv1alpha1.LPUPipelineRequest, error) {
	if err := r.validateLPXPublicationSource(ctx, deployment); err != nil {
		return nil, err
	}
	spec := materializing.modelProjection.RequestSpec(deployment.Namespace, grove.podGangName, grove.cyborgClique)
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
			},
			Annotations: lpxRequestAnnotations(deployment, materializing, grove),
		},
		Spec: spec,
	}
	if recordAttempt {
		controllerutil.AddFinalizer(request, lpxAttemptRecordingFinalizer)
	}
	if err := controllerutil.SetControllerReference(deployment, request, r.Client.Scheme()); err != nil {
		return nil, err
	}
	if err := lpx.ValidateRequestSize(request); err != nil {
		return nil, fmt.Errorf("constructed LPX request exceeds the scheduler budget: %w", err)
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
	grove *lpxGroveIdentity,
) map[string]string {
	// Callers validate the exact source controller owner before constructing requests.
	sourceOwner := metav1.GetControllerOf(deployment)
	return map[string]string{
		lpxAttemptDigestAnnotation:   materializing.attemptDigest,
		lpx.DGDUIDAnnotation:         string(sourceOwner.UID),
		lpxDeploymentUIDAnnotation:   string(deployment.UID),
		lpxModelAnnotation:           materializing.modelProjection.Model(),
		lpxPCSUIDAnnotation:          string(grove.pcsUID),
		lpxPodGangUIDAnnotation:      string(grove.podGangUID),
		lpx.WorkloadDigestAnnotation: materializing.modelProjection.Digest().String(),
	}
}

func validateCurrentLPXRequest(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxModelMaterializing,
	grove *lpxGroveIdentity,
	request *lpxv1alpha1.LPUPipelineRequest,
) error {
	if request.UID == "" {
		return fmt.Errorf("LPX request %q has no immutable UID", request.Name)
	}
	wantAnnotations := lpxRequestAnnotations(deployment, desired, grove)
	for key, value := range wantAnnotations {
		if request.Annotations[key] != value {
			return fmt.Errorf("LPX request %q immutable annotation %q no longer matches the current attempt", request.Name, key)
		}
	}
	wantSpec := desired.modelProjection.RequestSpec(deployment.Namespace, grove.podGangName, grove.cyborgClique)
	if !apiequality.Semantic.DeepEqual(request.Spec, wantSpec) {
		return fmt.Errorf("LPX request %q immutable intent no longer matches the current attempt", request.Name)
	}
	return nil
}

func (r *graphReconciler) observeLPXGroveSnapshot(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	noCurrentRequests bool,
) (*lpxGroveSnapshot, string, error) {
	pcs, incomplete, err := r.observeLPXPodCliqueSet(ctx, deployment, desired, noCurrentRequests)
	if err != nil || pcs == nil {
		return nil, incomplete, err
	}
	generationHash, wait := observedLPXPCSGenerationHash(pcs)
	if wait != "" {
		if incomplete == "" {
			incomplete = wait
		}
	}
	podGangs, err := r.listLPXPublicationPodGangs(ctx, deployment, pcs)
	if err != nil {
		return nil, "", err
	}

	// Every resolved runtime has one serving clique in this scaling group.
	scalingGroupName := desired.plan.LPXScalingGroup
	scalingGroup := &grovev1alpha1.PodCliqueScalingGroup{}
	key := types.NamespacedName{Namespace: deployment.Namespace, Name: scalingGroupName}
	if err := r.apiReader.Get(ctx, key, scalingGroup); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, fmt.Sprintf("Waiting for current LPU PodCliqueScalingGroup %q", scalingGroupName), nil
		}
		return nil, "", err
	}
	if scalingGroup.UID == "" || !scalingGroup.DeletionTimestamp.IsZero() ||
		!metav1.IsControlledBy(scalingGroup, pcs) {
		return nil, fmt.Sprintf(
			"LPU PodCliqueScalingGroup %q lacks the current PodCliqueSet owner identity",
			scalingGroupName,
		), nil
	}
	return &lpxGroveSnapshot{
		podCliqueSet:   pcs,
		scalingGroup:   scalingGroup,
		generationHash: generationHash,
		incomplete:     incomplete,
		podGangs:       podGangs,
	}, "", nil
}

func (r *graphReconciler) observeLPXGroveIdentity(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	desired *lpxMaterializing,
	snapshot *lpxGroveSnapshot,
	plan *lpx.MaterializationPlan,
	allAgentGroups map[string]struct{},
) (*lpxGroveIdentity, string, error) {
	incomplete := snapshot.incomplete
	// Observe native retirement before gang availability or successor revision readiness.
	agentUIDs := make([]types.UID, 0, len(plan.Agents))
	projections := desired.workload.ModelProjections()
	for index, agent := range plan.Agents {
		clique := &grovev1alpha1.PodClique{}
		if err := r.apiReader.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: agent.CliqueName}, clique); err != nil {
			if apierrors.IsNotFound(err) {
				incomplete = fmt.Sprintf("Waiting for Agent PodClique %q", agent.CliqueName)
				continue
			}
			return nil, "", err
		}
		if !hasCurrentLPXScalingGroupOwner(clique, snapshot.scalingGroup) || clique.UID == "" {
			incomplete = fmt.Sprintf("Waiting for owned Agent PodClique %q", agent.CliqueName)
			continue
		}
		if !clique.DeletionTimestamp.IsZero() {
			return nil, "", errLPXEngineReplacing
		}
		agentUIDs = append(agentUIDs, clique.UID)
		if !desired.hasCurrentLPXAttemptAnnotations(clique, deployment) ||
			clique.Annotations[lpx.WorkloadDigestAnnotation] != projections[index].Digest().String() ||
			clique.Status.CurrentPodCliqueSetGenerationHash == nil || *clique.Status.CurrentPodCliqueSetGenerationHash != snapshot.generationHash {
			incomplete = fmt.Sprintf("Waiting for current Agent PodClique %q", agent.CliqueName)
		}
	}
	lpxPodGang, ordinaryPodGang, wait := selectLPXPublicationPodGangs(
		deployment.Namespace,
		snapshot.podGangs,
		plan,
		allAgentGroups,
	)
	if wait != "" {
		return nil, wait, nil
	}

	identity := &lpxGroveIdentity{
		pcsUID:      snapshot.podCliqueSet.UID,
		podGangName: lpxPodGang.Name,
		podGangUID:  lpxPodGang.UID,
		agentUIDs:   agentUIDs,
	}
	if !desired.hasCurrentLPXAttemptAnnotations(lpxPodGang, deployment) ||
		!desired.hasCurrentLPXAttemptAnnotations(ordinaryPodGang, deployment) {
		if incomplete == "" {
			incomplete = "Waiting for Grove to publish current-generation PodGangs"
		}
	}

	// A conductor needs complete identity; Cyborg can retain its reference while hashes converge.
	cyborg, wait, err := r.observeLPXServingClique(
		ctx,
		deployment,
		snapshot,
		ordinaryPodGang.Name,
		desired,
		plan,
	)
	if err != nil || (wait != "" && cyborg == nil) {
		return nil, wait, err
	}
	identity.cyborgClique = cyborg
	if wait != "" {
		if incomplete == "" {
			incomplete = wait
		}
	}
	return identity, incomplete, nil
}

func (r *graphReconciler) listLPXPublicationPodGangs(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) ([]groveschedulerv1alpha1.PodGang, error) {
	podGangs := make([]groveschedulerv1alpha1.PodGang, 0, 2)
	if err := visitLifecycleObjectPages(
		ctx,
		r.apiReader,
		&groveschedulerv1alpha1.PodGangList{},
		func(object k8sruntime.Object) error {
			podGang := object.(*groveschedulerv1alpha1.PodGang)
			schedulerName := podGang.Labels[grovecommon.LabelSchedulerName]
			if (schedulerName != lpx.SchedulerName && schedulerName != corev1.DefaultSchedulerName) ||
				podGang.UID == "" || !podGang.DeletionTimestamp.IsZero() ||
				!metav1.IsControlledBy(podGang, pcs) {
				return nil
			}
			podGangs = append(podGangs, *podGang)
			return nil
		},
		client.InNamespace(deployment.Namespace),
		client.MatchingLabels{
			grovecommon.LabelPartOfKey: pcs.Name,
		},
	); err != nil {
		return nil, err
	}
	return podGangs, nil
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
	if noCurrentRequests && pcs.Spec.Replicas == 0 {
		return nil, "The retired Grove attempt is held at zero; successor publication waits for post-retirement Grove synchronization", nil
	}
	if pcs.UID == "" || !pcs.DeletionTimestamp.IsZero() || !metav1.IsControlledBy(pcs, deployment) {
		return nil, "The selected Grove PodCliqueSet lacks the exact LPXGraphDeployment owner identity", nil
	}

	incomplete := ""
	if !desired.hasCurrentLPXAttemptAnnotations(pcs, deployment) {
		incomplete = "The selected Grove PodCliqueSet is stale"
	}
	if pcs.Spec.Replicas != 1 {
		if incomplete == "" {
			incomplete = fmt.Sprintf("The selected Grove PodCliqueSet has %d replicas, want 1", pcs.Spec.Replicas)
		}
	}
	return pcs, incomplete, nil
}

func observedLPXPCSGenerationHash(pcs *grovev1alpha1.PodCliqueSet) (string, string) {
	if pcs.Status.CurrentGenerationHash == nil || *pcs.Status.CurrentGenerationHash == "" {
		return "", fmt.Sprintf(
			"Waiting for Grove to publish PodCliqueSet %q generation identity",
			pcs.Name,
		)
	}
	return *pcs.Status.CurrentGenerationHash, ""
}

func hasCurrentLPXScalingGroupOwner(
	clique *grovev1alpha1.PodClique,
	scalingGroup *grovev1alpha1.PodCliqueScalingGroup,
) bool {
	owner := metav1.GetControllerOf(clique)
	return owner != nil && owner.APIVersion == grovev1alpha1.SchemeGroupVersion.String() &&
		owner.Kind == "PodCliqueScalingGroup" && owner.Name == scalingGroup.Name &&
		owner.UID == scalingGroup.UID
}

func (r *graphReconciler) observeLPXServingClique(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	snapshot *lpxGroveSnapshot,
	ordinaryPodGangName string,
	desired *lpxMaterializing,
	plan *lpx.MaterializationPlan,
) (*lpxv1alpha1.CyborgPodCliqueReference, string, error) {
	// Every supported plan has exactly one ordinary-scheduler serving clique.
	isCyborg := plan.CyborgClique != ""
	name, role := plan.ConductorClique, lpxv1alpha1.PodRoleConductor
	roleName, waitingRoleName := "Conductor", "conductor"
	if isCyborg {
		name, role = plan.CyborgClique, lpxv1alpha1.PodRoleCyborgWorker
		roleName, waitingRoleName = "Cyborg", "Cyborg"
	}

	// Read the single serving witness and validate its shared scheduler identity.
	clique := &grovev1alpha1.PodClique{}
	key := types.NamespacedName{Namespace: deployment.Namespace, Name: name}
	if err := r.apiReader.Get(ctx, key, clique); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, fmt.Sprintf("Waiting for %s PodClique %q", waitingRoleName, name), nil
		}
		return nil, "", err
	}
	if clique.UID == "" ||
		clique.Labels[grovecommon.LabelPartOfKey] != snapshot.podCliqueSet.Name ||
		clique.Labels[grovecommon.LabelPodCliqueScalingGroup] != plan.LPXScalingGroup ||
		clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] != strconv.FormatInt(int64(plan.ReplicaIndex), 10) ||
		clique.Labels[grovecommon.LabelPodGang] != ordinaryPodGangName ||
		clique.Spec.PodSpec.SchedulerName != corev1.DefaultSchedulerName ||
		clique.Annotations[lpxv1alpha1.PodRoleAnnotation] != role ||
		!desired.hasCurrentLPXAttemptAnnotations(clique, deployment) ||
		(!isCyborg && !hasCurrentLPXScalingGroupOwner(clique, snapshot.scalingGroup)) {
		return nil, fmt.Sprintf(
			"%s PodClique %q does not match the current ordinary-scheduler identity",
			roleName, name,
		), nil
	}
	if !isCyborg {
		if clique.Status.CurrentPodCliqueSetGenerationHash == nil || *clique.Status.CurrentPodCliqueSetGenerationHash != snapshot.generationHash {
			return nil, fmt.Sprintf("Waiting for current conductor PodClique %q", clique.Name), nil
		}
		return nil, "", nil
	}

	// Cyborg publication additionally binds device claims and exact Grove revision hashes.
	if !hasCurrentLPXScalingGroupOwner(clique, snapshot.scalingGroup) {
		return nil, fmt.Sprintf(
			"Cyborg PodClique %q lacks the current LPU scaling-group owner identity",
			clique.Name,
		), nil
	}
	if !hasLPXCyborgDeviceIntent(clique.Spec.PodSpec) || !hasExpectedLPXCyborgClaimReferences(snapshot.podCliqueSet, clique, plan.CyborgTemplate) {
		reason := fmt.Sprintf("Cyborg PodClique %q GPU intent or DRA Claim references changed during Grove materialization", clique.Name)
		// Until Grove has accepted the PCS edit, differing references may still be its old engine.
		pcs := snapshot.podCliqueSet
		if pcs.Status.ObservedGeneration != nil && *pcs.Status.ObservedGeneration == pcs.Generation &&
			clique.Status.CurrentPodCliqueSetGenerationHash != nil && *clique.Status.CurrentPodCliqueSetGenerationHash == snapshot.generationHash {
			return nil, "", &lpxRetiring{retirementReason: reason}
		}
		return nil, reason, nil
	}
	reference := &lpxv1alpha1.CyborgPodCliqueReference{Name: clique.Name, UID: string(clique.UID)}

	// Require Grove's selected PCS revision and pod-template identities. Deliberately
	// do not require Cyborg ObservedGeneration: LPX must remain able to publish the
	// provision request before Grove advances that compatibility signal.
	if clique.Status.CurrentPodCliqueSetGenerationHash == nil ||
		*clique.Status.CurrentPodCliqueSetGenerationHash != snapshot.generationHash ||
		clique.Status.CurrentPodTemplateHash == nil ||
		*clique.Status.CurrentPodTemplateHash == "" {
		return reference, fmt.Sprintf(
			"Waiting for Grove to materialize the current PodCliqueSet generation in Cyborg PodClique %q",
			clique.Name,
		), nil
	}
	return reference, "", nil
}

func selectLPXPublicationPodGangs(
	namespace string,
	podGangs []groveschedulerv1alpha1.PodGang,
	plan *lpx.MaterializationPlan,
	allAgentGroups map[string]struct{},
) (*groveschedulerv1alpha1.PodGang, *groveschedulerv1alpha1.PodGang, string) {
	var lpxCandidate, ordinaryCandidate *groveschedulerv1alpha1.PodGang
	var lpxCandidateCount, ordinaryCandidateCount int
	ordinaryGroup := plan.ConductorClique
	if ordinaryGroup == "" {
		ordinaryGroup = plan.CyborgClique
	}
	// Count every matching gang before reporting either scheduler's ambiguity.
	for index := range podGangs {
		podGang := &podGangs[index]
		switch podGang.Labels[grovecommon.LabelSchedulerName] {
		case lpx.SchedulerName:
			if hasExactLPXAgentRows(namespace, podGang, plan) {
				lpxCandidate = podGang
				lpxCandidateCount++
			}
		case corev1.DefaultSchedulerName:
			containsAgent, containsOrdinaryGroup := false, ordinaryGroup == ""
			for _, group := range podGang.Spec.PodGroups {
				if _, agent := allAgentGroups[group.Name]; agent {
					containsAgent = true
				}
				if group.Name == ordinaryGroup {
					containsOrdinaryGroup = true
				}
			}
			if !containsAgent && containsOrdinaryGroup {
				ordinaryCandidate = podGang
				ordinaryCandidateCount++
			}
		}
	}
	if lpxCandidateCount != 1 {
		return nil, nil, fmt.Sprintf(
			"Waiting for one replica-%d LPX PodGang; found %d candidates",
			plan.ReplicaIndex,
			lpxCandidateCount,
		)
	}
	if ordinaryCandidateCount != 1 {
		return nil, nil, fmt.Sprintf(
			"Waiting for one replica-%d ordinary-scheduler PodGang; found %d candidates",
			plan.ReplicaIndex,
			ordinaryCandidateCount,
		)
	}
	return lpxCandidate, ordinaryCandidate, ""
}

func hasExactLPXAgentRows(
	namespace string,
	podGang *groveschedulerv1alpha1.PodGang,
	plan *lpx.MaterializationPlan,
) bool {
	if len(podGang.Spec.PodGroups) != len(plan.Agents) {
		return false
	}
	groups := make(map[string]*groveschedulerv1alpha1.PodGroup, len(podGang.Spec.PodGroups))
	for index := range podGang.Spec.PodGroups {
		group := &podGang.Spec.PodGroups[index]
		if _, duplicate := groups[group.Name]; duplicate {
			return false
		}
		groups[group.Name] = group
	}
	for _, agent := range plan.Agents {
		group, found := groups[agent.CliqueName]
		if !found || group.MinReplicas != int32(agent.Replicas) || len(group.PodReferences) != agent.Replicas {
			return false
		}
		seen := make(map[types.NamespacedName]struct{}, agent.Replicas)
		for _, reference := range group.PodReferences {
			key := types.NamespacedName{Namespace: reference.Namespace, Name: reference.Name}
			if reference.Namespace != namespace || strings.TrimSpace(reference.Name) == "" {
				return false
			}
			if _, duplicate := seen[key]; duplicate {
				return false
			}
			seen[key] = struct{}{}
		}
	}
	return true
}

func hasExpectedLPXCyborgClaimReferences(
	pcs *grovev1alpha1.PodCliqueSet,
	cyborg *grovev1alpha1.PodClique,
	templateName string,
) bool {
	var expected *corev1.PodSpec
	for _, template := range pcs.Spec.Template.Cliques {
		if template == nil || template.Name != templateName {
			continue
		}
		if expected != nil {
			return false
		}
		expected = &template.Spec.PodSpec
	}
	if expected == nil {
		return false
	}
	observed := &cyborg.Spec.PodSpec
	return apiequality.Semantic.DeepEqual(podLevelClaimReferences(expected), podLevelClaimReferences(observed)) &&
		apiequality.Semantic.DeepEqual(expected.ResourceClaims, observed.ResourceClaims) &&
		containerClaimReferencesEqual(expected.Containers, observed.Containers) &&
		containerClaimReferencesEqual(expected.InitContainers, observed.InitContainers)
}

func podLevelClaimReferences(spec *corev1.PodSpec) []corev1.ResourceClaim {
	if spec.Resources == nil {
		return nil
	}
	return spec.Resources.Claims
}

func containerClaimReferencesEqual(expected, observed []corev1.Container) bool {
	if len(expected) != len(observed) {
		return false
	}
	for index := range expected {
		if expected[index].Name != observed[index].Name ||
			!apiequality.Semantic.DeepEqual(expected[index].Resources.Claims, observed[index].Resources.Claims) {
			return false
		}
	}
	return true
}

func hasLPXCyborgDeviceIntent(podSpec corev1.PodSpec) bool {
	if len(podSpec.ResourceClaims) != 0 {
		return true
	}
	for _, container := range podSpec.Containers {
		if container.Name != consts.MainContainerName && container.Name != "cyborg" {
			continue
		}
		gpuCount, err := lpx.EffectiveCyborgGPUCount(container.Resources)
		if err != nil {
			return false
		}
		if gpuCount > 0 {
			return true
		}
	}
	return false
}

// finalizeLPXRequests retires every exact owned publication for the shared Grove
// attempt. A wait error keeps the DGD finalizer until LPX's own cleanup
// finalizers have completed.
func (r *graphReconciler) finalizeLPXRequests(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
) error {
	requests, _, err := r.listOwnedLPXRequestsIfAvailable(ctx, deployment)

	// Stop when listing fails or no publication remains.
	if err != nil || len(requests) == 0 {
		return err
	}

	// Start or continue retirement from the authoritative first request.
	if _, err := r.retireLPXRequest(ctx, deployment, "", &requests[0], "LPXGraphDeployment is being deleted"); err != nil {
		return err
	}
	return fmt.Errorf("waiting for LPX request %q to finish fail-closed retirement", requests[0].Name)
}
