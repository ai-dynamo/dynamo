/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"time"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
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
	// Failed reconciliations can retain a current durable attempt before the
	// complete result is observed. Invalidate only the attempt's own generation,
	// including when the new source no longer selects a scheduling deadline.
	attempt := currentLPXAttemptStatus(deployment)
	if attempt != nil && attempt.ObservedGeneration != deployment.Generation {
		deployment.Status.Placement = nil
		attempt = nil
	}
	deadlineSeconds := lpxAttemptDeadlineSeconds(source)
	if !source.HasLPXComponent() || deadlineSeconds == nil {
		return nil, time.Time{}, nil, nil
	}
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
	if updateLPXAttemptFromLive(deployment, deadlineSeconds, attempt, observed) {
		return newLPXDeadlineTransition(
			attempt, nvidiacomv1beta1.DGDStatePending, lpxSchedulingAttemptActiveReason,
			"Persisted the exact LPX request identity and immutable scheduling deadline", time.Nanosecond,
		), attempt.DeadlineAt.Time, nil, nil
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
	if attempt.DeadlineAt != nil && (attempt.DisarmedAt == nil || !missing) && !now.Before(attempt.DeadlineAt.Time) {
		decision := metav1.NewTime(now)
		attempt.ExceededAt = &decision
		attempt.DisarmedAt = nil
		return newLPXDeadlineTransition(
			attempt, nvidiacomv1beta1.DGDStateFailed, lpxSchedulingDeadlineExceededReason,
			"LPX scheduling deadline exceeded; persisting expiry before exact cleanup", time.Nanosecond,
		), attempt.DeadlineAt.Time, nil, nil
	}
	if missing {
		state, err := r.continueTerminalLPXAttempt(
			ctx, deployment, attempt, observed, lpxAttemptAuthorityLostReason,
			"An exact LPX request disappeared; retiring the aggregate without replacement",
		)
		return state, time.Time{}, nil, err
	}
	if attempt.DeadlineAt == nil {
		return nil, time.Time{}, requests, nil
	}
	if disposition {
		observedAt := metav1.NewTime(now)
		attempt.DisarmedAt = &observedAt
		return newLPXDeadlineTransition(
			attempt, nvidiacomv1beta1.DGDStatePending, lpxSchedulingDispositionObservedReason,
			"Recorded the exact aggregate scheduling disposition before deadline expiry", time.Nanosecond,
		), attempt.DeadlineAt.Time, nil, nil
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

func (r *graphReconciler) reconcileLPXAttemptPreparation(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	deadlineSeconds *int64,
	want *nvidiacomv1beta1.LPXAttemptStatus,
	currents map[string]*lpxv1alpha1.LPUPipelineRequest,
) (*lpxDeadlineTransition, error) {
	if want == nil {
		return nil, nil
	}
	current := currentLPXAttemptStatus(deployment)
	if current == nil || current.ObservedGeneration != deployment.Generation {
		updateLPXAttemptFromLive(deployment, deadlineSeconds, want, currents)
		reason, message := lpxAttemptPreparedReason, "Prepared the complete LPX request set before publication"
		if want.DeadlineAt != nil {
			reason, message = lpxSchedulingAttemptActiveReason, "Recovered the exact first publication and its deadline"
		}
		return newLPXDeadlineTransition(
			want, nvidiacomv1beta1.DGDStatePending, reason, message, time.Nanosecond,
		), nil
	}
	if !sameLPXAttemptProjection(current, want) {
		return nil, fmt.Errorf("current LPX attempt status does not match the desired request projection")
	}
	if current.PodCliqueSetUID != want.PodCliqueSetUID {
		return nil, fmt.Errorf("current LPX attempt status does not match the exact Grove identity")
	}
	if _, _, missing := exactLPXAttemptState(current, currents); missing {
		return newLPXDeadlineTransition(
			current, nvidiacomv1beta1.DGDStateFailed, lpxAttemptAuthorityLostReason,
			"An exact LPX request disappeared; publication remains fenced", time.Nanosecond,
		), nil
	}
	if !updateLPXAttemptFromLive(deployment, deadlineSeconds, current, currents) {
		return nil, nil
	}
	return newLPXDeadlineTransition(
		current, nvidiacomv1beta1.DGDStatePending, lpxSchedulingAttemptActiveReason,
		"Persisted the exact LPX request identity and immutable scheduling deadline", time.Nanosecond,
	), nil
}

// revalidateLPXAttemptPublication fences live authority after preparation has
// established the current nonnil attempt and an unpublished desired request.
func (r *graphReconciler) revalidateLPXAttemptPublication(ctx context.Context, deployment *nvidiacomv1alpha1.LPXGraphDeployment) error {
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
	if err := r.apiReader.Get(ctx, types.NamespacedName{Namespace: deployment.Namespace, Name: deployment.Annotations[dynamo.LPXPCSNameAnnotation]}, pcs); err != nil {
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
	if source.Spec.Scheduling == nil {
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

func sameLPXAttemptProjection(a, b *nvidiacomv1beta1.LPXAttemptStatus) bool {
	if a == nil || b == nil || a.ObservedGeneration != b.ObservedGeneration || len(a.Requests) != len(b.Requests) {
		return false
	}
	for i := range a.Requests {
		if a.Requests[i].Name != b.Requests[i].Name ||
			a.Requests[i].AttemptDigest != b.Requests[i].AttemptDigest {
			return false
		}
	}
	return true
}

// updateLPXAttemptFromLive incorporates server-persisted requests into the non-nil,
// reconcile-owned attempt and reports whether a status transition is needed.
func updateLPXAttemptFromLive(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	deadlineSeconds *int64,
	attempt *nvidiacomv1beta1.LPXAttemptStatus,
	observed map[string]*lpxv1alpha1.LPUPipelineRequest,
) bool {
	changed := false
	var firstPublishedAt time.Time
	for i := range attempt.Requests {
		row := &attempt.Requests[i]
		request := observed[row.Name]
		if request == nil {
			continue
		}
		if row.UID == "" {
			if request.Annotations[lpxAttemptDigestAnnotation] != row.AttemptDigest ||
				request.Annotations[lpxDeploymentGenerationAnnotation] != strconv.FormatInt(deployment.Generation, 10) {
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
	if attempt.DeadlineAt == nil && !firstPublishedAt.IsZero() {
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
	disposition = true
	for i := range attempt.Requests {
		// Distinguish unpublished rows from recorded request authority loss.
		row := &attempt.Requests[i]
		request := observed[row.Name]
		if row.UID == "" {
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
	// Retire recorded identities in their canonical request order.
	pending := false
	for _, row := range attempt.Requests {
		request := observed[row.Name]
		if row.UID == "" || request == nil || request.UID != row.UID {
			continue
		}
		pending = true
		if !request.DeletionTimestamp.IsZero() {
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
	}
	if pending {
		if err := r.scaleDownLPXPodCliqueSet(ctx, deployment, attempt.PodCliqueSetUID); err != nil {
			return nil, err
		}
		return newLPXDeadlineTransition(
			attempt, nvidiacomv1beta1.DGDStateFailed, reason, message, lpxRetirementRequeueAfter,
		), nil
	}
	pending, err := r.retireLPXAttemptPodCliqueSet(ctx, deployment, attempt.PodCliqueSetUID)
	if err != nil {
		return nil, err
	}
	requeueAfter := time.Duration(0)
	if pending {
		requeueAfter = lpxRetirementRequeueAfter
	}
	return newLPXDeadlineTransition(
		attempt, nvidiacomv1beta1.DGDStateFailed, reason, message, requeueAfter,
	), nil
}

func (r *graphReconciler) retireLPXAttemptPodCliqueSet(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	uid types.UID,
) (bool, error) {
	pcs := &grovev1alpha1.PodCliqueSet{}
	key := types.NamespacedName{Namespace: deployment.Namespace, Name: deployment.Annotations[dynamo.LPXPCSNameAnnotation]}
	if err := r.apiReader.Get(ctx, key, pcs); err != nil {
		return false, client.IgnoreNotFound(err)
	}
	if uid == "" || pcs.UID != uid {
		return false, nil
	}
	if !metav1.IsControlledBy(pcs, deployment) {
		return false, fmt.Errorf("refusing to retire PodCliqueSet %q without the exact LPXGraphDeployment owner", pcs.Name)
	}
	if pcs.Spec.Replicas != 0 {
		pcs.Spec.Replicas = 0
		return true, r.Update(ctx, pcs)
	}
	if !pcs.DeletionTimestamp.IsZero() {
		return true, nil
	}
	pcsUID, resourceVersion := pcs.UID, pcs.ResourceVersion
	err := r.Delete(ctx, pcs, &client.DeleteOptions{Preconditions: &metav1.Preconditions{
		UID: &pcsUID, ResourceVersion: &resourceVersion,
	}})
	return true, client.IgnoreNotFound(err)
}
