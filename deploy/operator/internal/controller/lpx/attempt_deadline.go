/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"sort"
	"time"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	lpxAttemptPreparedReason               string = "LPXAttemptPrepared"
	lpxSchedulingAttemptActiveReason       string = "LPXSchedulingAttemptActive"
	lpxSchedulingDispositionObservedReason string = "LPXSchedulingDispositionObserved"
	lpxSchedulingDeadlineExceededReason    string = "LPXSchedulingDeadlineExceeded"
	lpxAttemptAuthorityLostReason          string = "LPXAttemptAuthorityLost"
	lpxAttemptRecordingFinalizer                  = "scheduling.lpu.nvidia.com/dynamo-attempt-recording"
)

type lpxDeadlineTransition struct {
	result       reconcileOutcome
	attempt      *nvidiacomv1beta1.LPXAttemptStatus
	requeueAfter time.Duration
}

func (*lpxDeadlineTransition) lpxClassification() {}

func newLPXDeadlineTransition(attempt *nvidiacomv1beta1.LPXAttemptStatus, state nvidiacomv1beta1.DGDState, reason string, message string, requeueAfter time.Duration) *lpxDeadlineTransition {
	return &lpxDeadlineTransition{
		result:       reconcileOutcome{State: state, Reason: reason, Message: message},
		attempt:      attempt,
		requeueAfter: requeueAfter,
	}
}

// completeLPXDeadline preserves successful deadline wakes and converts active-deadline errors into bounded retries.
func completeLPXDeadline(
	ctx context.Context,
	deadlineAt time.Time,
	result ctrl.Result,
	reconcileErr error,
) (ctrl.Result, error) {
	// Preserve ordinary results when no scheduling deadline is active.
	if deadlineAt.IsZero() {
		return result, reconcileErr
	}

	// Record failures and bound completion by the remaining scheduling deadline.
	if reconcileErr != nil {
		log.FromContext(ctx).Error(reconcileErr, "retrying reconciliation under the LPX scheduling deadline")
	}
	delay := time.Until(deadlineAt)
	if reconcileErr != nil {
		if delay <= 0 || delay > lpxRetirementRequeueAfter {
			delay = lpxRetirementRequeueAfter
		}
		return ctrl.Result{RequeueAfter: delay}, nil
	}
	if delay <= 0 {
		delay = time.Nanosecond
	}
	if result.RequeueAfter <= 0 || delay < result.RequeueAfter {
		result.RequeueAfter = delay
	}
	return result, nil
}

func (r *graphReconciler) reconcileLPXAttemptDeadline(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
) (lpxClassification, time.Time, []lpxv1alpha1.LPUPipelineRequest, error) {
	// Source edits do not reset a scheduling batch's absolute deadline.
	attempt := currentLPXAttemptStatus(deployment)
	deadlineSeconds := lpxAttemptDeadlineSeconds(source)
	if attempt == nil {
		return nil, time.Time{}, nil, nil
	}
	deadlineAt := time.Time{}
	if attempt.DeadlineAt != nil {
		deadlineAt = attempt.DeadlineAt.Time
	}
	requests, err := r.listOwnedLPXRequests(ctx, deployment)
	if err != nil {
		return nil, deadlineAt, nil, err
	}
	observed := lpxRequestsByName(requests)
	if updateLPXAttemptFromLive(deadlineSeconds, attempt, observed) {
		if attempt.DeadlineAt != nil {
			deadlineAt = attempt.DeadlineAt.Time
		}
		return newLPXDeadlineTransition(
			attempt, nvidiacomv1beta1.DGDStatePending, lpxSchedulingAttemptActiveReason,
			"Persisted the exact LPX request identity and immutable scheduling deadline", time.Nanosecond,
		), deadlineAt, nil, nil
	}
	if attempt.ExceededAt != nil {
		state, err := r.continueTerminalLPXAttempt(
			ctx, deployment, attempt, observed, lpxSchedulingDeadlineExceededReason,
			"LPX scheduling deadline exceeded; no retry is authorized",
		)
		return state, time.Time{}, nil, err
	}
	active, disposition, missing := exactLPXAttemptState(attempt, observed)
	if attempt.DisarmedAt != nil && !missing && !active {
		return nil, time.Time{}, requests, nil
	}
	now := time.Now()
	// Completion is authoritative even when the status write lags a Bound observation.
	if disposition {
		observedAt := metav1.NewTime(now)
		attempt.DisarmedAt = &observedAt
		return newLPXDeadlineTransition(attempt, nvidiacomv1beta1.DGDStatePending,
			lpxSchedulingDispositionObservedReason, "Recorded the scheduling batch disposition", time.Nanosecond), time.Time{}, requests, nil
	}
	if attempt.DeadlineAt != nil && (attempt.DisarmedAt == nil || !missing) && !now.Before(attempt.DeadlineAt.Time) {
		decision := metav1.NewTime(now)
		attempt.ExceededAt = &decision
		attempt.DisarmedAt = nil
		attempt.ObservedGeneration = deployment.Generation
		return newLPXDeadlineTransition(
			attempt, nvidiacomv1beta1.DGDStateFailed, lpxSchedulingDeadlineExceededReason,
			"LPX scheduling deadline exceeded; persisting expiry before exact cleanup", time.Nanosecond,
		), attempt.DeadlineAt.Time, nil, nil
	}
	if missing {
		state, err := r.continueTerminalLPXAttempt(
			ctx, deployment, attempt, observed, lpxAttemptAuthorityLostReason,
			"An exact LPX request disappeared; retiring its scheduling batch without replacement",
		)
		return state, time.Time{}, nil, err
	}
	if attempt.DeadlineAt == nil {
		return nil, time.Time{}, requests, nil
	}
	if attempt.DisarmedAt != nil {
		attempt.DisarmedAt = nil
		return newLPXDeadlineTransition(
			attempt, nvidiacomv1beta1.DGDStatePending, lpxSchedulingAttemptActiveReason,
			"LPX scheduling resumed under the original deadline", time.Nanosecond,
		), attempt.DeadlineAt.Time, nil, nil
	}
	return nil, deadlineAt, requests, nil
}

// retireLPXEngineRequest records intentional removal before deleting the exact request.
func (r *graphReconciler) retireLPXEngineRequest(ctx context.Context, deployment *nvidiacomv1alpha1.LPXGraphDeployment, request *lpxv1alpha1.LPUPipelineRequest, reason string, intentional bool) (lpxClassification, error) {
	// Persist batch membership before deletion can be mistaken for lost authority.
	if current := currentLPXAttemptStatus(deployment); current != nil && intentional && current.ExceededAt == nil {
		for i, row := range current.Requests {
			if row.Name == request.Name && row.AttemptDigest != "" && (row.UID == request.UID || row.UID == "" && row.AttemptDigest == request.Annotations[lpxAttemptDigestAnnotation]) {
				next := current.DeepCopy()
				// Retain exact cleanup authority until a replacement digest can be prepared.
				next.Requests[i].UID = request.UID
				next.Requests[i].AttemptDigest = ""
				return newLPXDeadlineTransition(next, nvidiacomv1beta1.DGDStatePending,
					lpxRetiringReason, reason, time.Nanosecond), nil
			}
		}
	}
	if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
		return nil, err
	}
	if request.DeletionTimestamp.IsZero() {
		uid, rv := request.UID, request.ResourceVersion
		if err := r.Delete(ctx, request, client.Preconditions{UID: &uid, ResourceVersion: &rv}); err != nil && !apierrors.IsNotFound(err) {
			return nil, err
		}
	}
	// Invalid live intent needs exact Pod cleanup; native replacement already owns its Pods.
	if !intentional {
		if err := r.retireLPXAttemptPods(ctx, request); err != nil {
			return nil, err
		}
	}
	if err := r.releaseLPXAttemptRecordingFinalizer(ctx, request); err != nil {
		return nil, err
	}
	return &lpxRetiring{retirementReason: reason}, nil
}

// retireLPXAttemptPods releases only Pods committed to this exact scheduling request.
func (r *graphReconciler) retireLPXAttemptPods(ctx context.Context, request *lpxv1alpha1.LPUPipelineRequest) error {
	if request.Status == nil || request.Status.Committed == nil || request.Status.Committed.Execution.NodeLocal == nil {
		return nil
	}
	for _, partition := range request.Status.Committed.Execution.NodeLocal.PartitionSelections {
		for _, row := range partition.SelectedRows {
			refs := make([]lpxv1alpha1.ObjectReference, 0, 2)
			if row.Current != nil {
				refs = append(refs, row.Current.PodRef)
			}
			if row.Target != nil {
				refs = append(refs, row.Target.PodRef)
			}
			for _, ref := range refs {
				if ref.Namespace != request.Namespace || ref.Name == "" || ref.UID == "" {
					return fmt.Errorf("LPX request %q contains an invalid committed Pod identity", request.Name)
				}
				uid := types.UID(ref.UID)
				pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: ref.Namespace, Name: ref.Name}}
				if err := r.Delete(ctx, pod, client.Preconditions{UID: &uid}); err != nil && !apierrors.IsNotFound(err) && !apierrors.IsConflict(err) {
					return err
				}
			}
		}
	}
	return nil
}

func (r *graphReconciler) reconcileLPXAttemptPreparation(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	deadlineSeconds *int64,
	want *nvidiacomv1beta1.LPXAttemptStatus,
	currents map[string]*lpxv1alpha1.LPUPipelineRequest,
) *lpxDeadlineTransition {
	if want == nil {
		return nil
	}
	// New engines join an active batch, but never a completed batch awaiting disarm.
	current := currentLPXAttemptStatus(deployment)
	if current != nil && current.ExceededAt != nil {
		return newLPXDeadlineTransition(current, nvidiacomv1beta1.DGDStateFailed,
			lpxSchedulingDeadlineExceededReason, "LPX scheduling deadline exceeded; waiting for cleanup and a later edit", lpxRetirementRequeueAfter)
	}
	next := want.DeepCopy()
	previous := make(map[string]nvidiacomv1beta1.LPXAttemptRequestStatus)
	completed := false
	if current != nil {
		var missing bool
		_, completed, missing = exactLPXAttemptState(current, currents)
		active := !completed && (current.DisarmedAt == nil || missing)
		if active {
			next = current.DeepCopy()
		}
		for _, row := range current.Requests {
			if active || row.UID != "" && row.AttemptDigest == "" {
				previous[row.Name] = row
			}
		}
	}
	next.Requests = make([]nvidiacomv1beta1.LPXAttemptRequestStatus, 0, len(want.Requests))
	// Desired rows are already sorted; only newly joining engines need disposition checks.
	for _, row := range want.Requests {
		if old, tracked := previous[row.Name]; tracked {
			if old.UID != "" || row.AttemptDigest == "" {
				row = old
			}
		} else if request := currents[row.Name]; request != nil && request.DeletionTimestamp.IsZero() {
			if row.AttemptDigest == "" {
				row.AttemptDigest = request.Annotations[lpxAttemptDigestAnnotation]
			}
			switch state := classifyPublishedLPX(request).(type) {
			case *lpxBound:
				continue
			case *lpxSchedulerObserved:
				if state.Phase == lpxv1alpha1.RequestPhaseNoFit || state.Phase == lpxv1alpha1.RequestPhaseUnsupported {
					continue
				}
			}
		}
		next.Requests = append(next.Requests, row)
	}
	// Cancel the last pending engine durably; replacement placeholders keep their clock.
	if len(next.Requests) == 0 {
		if current == nil || completed {
			return nil
		}
		next = current.DeepCopy()
		next.Requests = []nvidiacomv1beta1.LPXAttemptRequestStatus{}
		if next.DisarmedAt == nil {
			disarmedAt := metav1.Now()
			next.DisarmedAt = &disarmedAt
		}
	}
	updateLPXAttemptFromLive(deadlineSeconds, next, currents)
	if apiequality.Semantic.DeepEqual(current, next) {
		return nil
	}
	return newLPXDeadlineTransition(
		next, nvidiacomv1beta1.DGDStatePending, lpxAttemptPreparedReason,
		"Prepared the scheduling batch before publication", time.Nanosecond,
	)
}

// revalidateLPXAttemptPublication fences live authority after preparation has
// established the current nonnil attempt and an unpublished desired request.
func (r *graphReconciler) revalidateLPXAttemptPublication(ctx context.Context, deployment *nvidiacomv1alpha1.LPXGraphDeployment, pcsName string) error {
	attempt := currentLPXAttemptStatus(deployment)
	if attempt.ExceededAt != nil || attempt.DisarmedAt != nil {
		return fmt.Errorf("LPX publication is forbidden after a durable attempt decision")
	}
	requests, err := r.listOwnedLPXRequests(ctx, deployment)
	if err != nil {
		return err
	}
	if _, _, missing := exactLPXAttemptState(attempt, lpxRequestsByName(requests)); missing {
		return fmt.Errorf("recorded LPX request authority changed before sibling publication")
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.apiReader.Get(ctx, types.NamespacedName{Namespace: deployment.Namespace, Name: pcsName}, pcs); err != nil {
		return err
	}
	if pcs.UID != attempt.PodCliqueSetUID || !pcs.DeletionTimestamp.IsZero() {
		return fmt.Errorf("PodCliqueSet authority changed before LPX publication")
	}
	if attempt.DeadlineAt != nil && !time.Now().Before(attempt.DeadlineAt.Time) {
		return fmt.Errorf("LPX scheduling deadline elapsed before sibling publication")
	}
	return nil
}

func lpxAttemptDeadlineSeconds(source *nvidiacomv1beta1.DynamoGraphDeployment) *int64 {
	if source == nil || source.Spec.Scheduling == nil {
		return nil
	}
	return source.Spec.Scheduling.AttemptDeadlineSeconds
}

// desiredLPXAttemptStatus projects desired for its reconcile-local LPXGraphDeployment generation; desired must be nonnil.
func desiredLPXAttemptStatus(desired *lpxMaterializing, observedGeneration int64, pcsUID types.UID) *nvidiacomv1beta1.LPXAttemptStatus {
	requests := make([]nvidiacomv1beta1.LPXAttemptRequestStatus, len(desired.requests))
	for i := range desired.requests {
		requests[i] = nvidiacomv1beta1.LPXAttemptRequestStatus{
			Name: desired.requests[i].requestName, AttemptDigest: desired.requests[i].attemptDigest,
		}
	}
	sort.Slice(requests, func(i, j int) bool { return requests[i].Name < requests[j].Name })
	return &nvidiacomv1beta1.LPXAttemptStatus{
		ObservedGeneration: observedGeneration, PodCliqueSetUID: pcsUID, Requests: requests,
	}
}

func currentLPXAttemptStatus(deployment *nvidiacomv1alpha1.LPXGraphDeployment) *nvidiacomv1beta1.LPXAttemptStatus {
	if deployment == nil || deployment.Status.Placement == nil {
		return nil
	}
	return deployment.Status.Placement.LPXAttempt
}

// updateLPXAttemptFromLive incorporates server-persisted requests into the non-nil,
// reconcile-owned attempt and reports whether a status transition is needed.
func updateLPXAttemptFromLive(
	deadlineSeconds *int64,
	attempt *nvidiacomv1beta1.LPXAttemptStatus,
	observed map[string]*lpxv1alpha1.LPUPipelineRequest,
) bool {
	changed := false
	var firstPublishedAt time.Time
	for i := range attempt.Requests {
		row := &attempt.Requests[i]
		if row.AttemptDigest == "" {
			continue
		}
		request := observed[row.Name]
		if request == nil {
			continue
		}
		if row.UID == "" {
			if request.Annotations[lpxAttemptDigestAnnotation] != row.AttemptDigest {
				continue
			}
			changed = true
			row.UID = request.UID
		} else if request.UID != row.UID {
			continue
		}
		if attempt.PodCliqueSetUID == "" {
			changed = true
			attempt.PodCliqueSetUID = types.UID(request.Annotations[lpxPCSUIDAnnotation])
		}
		if attempt.DeadlineAt == nil &&
			(firstPublishedAt.IsZero() || request.CreationTimestamp.Time.Before(firstPublishedAt)) {
			firstPublishedAt = request.CreationTimestamp.Time
		}
	}
	if attempt.DeadlineAt == nil && deadlineSeconds != nil && !firstPublishedAt.IsZero() {
		deadline := metav1.NewTime(firstPublishedAt.Add(
			time.Duration(*deadlineSeconds) * time.Second,
		))
		changed = true
		attempt.DeadlineAt = &deadline
	}
	return changed
}

func lpxRequestsByName(requests []lpxv1alpha1.LPUPipelineRequest) map[string]*lpxv1alpha1.LPUPipelineRequest {
	byName := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(requests))
	for i := range requests {
		byName[requests[i].Name] = &requests[i]
	}
	return byName
}

func (r *graphReconciler) releaseLPXAttemptRecordingFinalizer(ctx context.Context, request *lpxv1alpha1.LPUPipelineRequest) error {
	// The scheduler finishes cleanup and removes scheduling.lpu.nvidia.com/plan-protection
	// independently, never waiting for lpxAttemptRecordingFinalizer. Dynamo releases
	// its finalizer last to retain the request identity and terminal status during cleanup.
	// Unknown finalizers block release; their owners must resolve them, not Dynamo.
	if request.DeletionTimestamp.IsZero() || len(request.Finalizers) != 1 || request.Finalizers[0] != lpxAttemptRecordingFinalizer {
		return nil
	}
	controllerutil.RemoveFinalizer(request, lpxAttemptRecordingFinalizer)
	if err := r.Update(ctx, request); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("release LPX attempt recording finalizer on %q: %w", request.Name, err)
	}
	return nil
}

// exactLPXAttemptState reports aggregate request state; attempt must be nonnil.
func exactLPXAttemptState(
	attempt *nvidiacomv1beta1.LPXAttemptStatus,
	observed map[string]*lpxv1alpha1.LPUPipelineRequest,
) (active bool, disposition bool, missing bool) {
	disposition = len(attempt.Requests) > 0
	for i := range attempt.Requests {
		// Distinguish unpublished rows from recorded request authority loss.
		row := &attempt.Requests[i]
		request := observed[row.Name]
		if row.AttemptDigest == "" || row.UID == "" {
			disposition = false
			continue
		}
		if request == nil || request.UID != row.UID || !request.DeletionTimestamp.IsZero() {
			disposition = false
			missing = true
			continue
		}

		// Reuse the canonical publication state after exact row ownership is proven.
		switch state := classifyPublishedLPX(request).(type) {
		case *lpxBound:
			continue
		case *lpxSchedulerObserved:
			switch state.Phase {
			case lpxv1alpha1.RequestPhasePending, lpxv1alpha1.RequestPhasePlanned,
				lpxv1alpha1.RequestPhaseReserving, lpxv1alpha1.RequestPhaseBinding:
				active = true
			case lpxv1alpha1.RequestPhaseNoFit, lpxv1alpha1.RequestPhaseUnsupported:
				continue
			}
		}
		disposition = false
	}
	return active, disposition, missing
}

func (r *graphReconciler) continueTerminalLPXAttempt(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	attempt *nvidiacomv1beta1.LPXAttemptStatus,
	observed map[string]*lpxv1alpha1.LPUPipelineRequest,
	reason string,
	message string,
) (lpxClassification, error) {
	if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
		return nil, err
	}
	// Retire recorded identities in their canonical request order.
	pending := false
	for _, row := range attempt.Requests {
		request := observed[row.Name]
		if row.UID == "" || request == nil || request.UID != row.UID {
			continue
		}
		pending = true
		if !request.DeletionTimestamp.IsZero() {
			if err := r.retireLPXAttemptPods(ctx, request); err != nil {
				return nil, err
			}
			if err := r.releaseLPXAttemptRecordingFinalizer(ctx, request); err != nil {
				return nil, err
			}
			continue
		}
		uid, resourceVersion := request.UID, request.ResourceVersion
		if err := r.Delete(ctx, request, &client.DeleteOptions{Preconditions: &metav1.Preconditions{
			UID: &uid, ResourceVersion: &resourceVersion,
		}}); err != nil && !apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("delete terminal LPX request %q: %w", request.Name, err)
		}
		if err := r.retireLPXAttemptPods(ctx, request); err != nil {
			return nil, err
		}
	}
	// A failed scheduling batch never scales down or deletes the shared PCS.
	if !pending && attempt.ExceededAt != nil && deployment.Generation > attempt.ObservedGeneration {
		return newLPXDeadlineTransition(nil, nvidiacomv1beta1.DGDStatePending,
			lpxAttemptPreparedReason, "The failed batch is gone; a later edit permits scheduling", time.Nanosecond), nil
	}
	requeueAfter := time.Duration(0)
	if pending {
		requeueAfter = lpxRetirementRequeueAfter
	}
	return newLPXDeadlineTransition(
		attempt, nvidiacomv1beta1.DGDStateFailed, reason, message, requeueAfter,
	), nil
}
