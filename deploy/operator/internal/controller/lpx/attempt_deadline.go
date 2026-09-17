/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"time"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	lpxSchedulingFailedCondition        = "SchedulingFailed"
	lpxSchedulingDeadlineExceededReason = "LPXSchedulingDeadlineExceeded"
)

type lpxDeadlineExceeded struct {
	requeueAfter  time.Duration
	recordFailure bool
}

func (*lpxDeadlineExceeded) lpxClassification() {}

// completeLPXDeadline preserves deadline wakes and converts errors before an
// active deadline into bounded retries.
func completeLPXDeadline(
	ctx context.Context,
	deadlineAt time.Time,
	result ctrl.Result,
	reconcileErr error,
) (ctrl.Result, error) {
	if deadlineAt.IsZero() {
		return result, reconcileErr
	}
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

// reconcileLPXRequestDeadlines derives every scheduling clock from the start
// of the corresponding desired LPR's current scheduler-owned scheduling cycle.
// Recorded expiry survives phase/input changes until the exact request disappears.
// A nil PCS leaves no active clock; already recorded cleanup waits for owner GC.
func (r *graphReconciler) reconcileLPXRequestDeadlines(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	desired []lpxModelMaterializing,
) (lpxClassification, time.Time, error) {
	deadlineSeconds := lpxRequestDeadlineSeconds(source)
	active := deadlineSeconds != nil && pcs != nil && pcs.DeletionTimestamp.IsZero() && metav1.IsControlledBy(pcs, deployment)
	recorded := sets.New(deployment.Status.ExpiredRequestUIDs...)
	if !active && recorded.Len() == 0 {
		return nil, time.Time{}, nil
	}
	requests, err := r.listOwnedLPXRequests(ctx, deployment, pcs)
	if err != nil {
		return nil, time.Time{}, err
	}
	desiredNames := make(map[string]struct{}, len(desired))
	for index := range desired {
		desiredNames[desired[index].requestName] = struct{}{}
	}

	// Expire each desired nonterminal request independently and retain the earliest future wake.
	now := time.Now()
	expired := make([]*lpxv1alpha1.LPUPipelineRequest, 0)
	deployment.Status.ExpiredRequestUIDs = nil
	recordFailure := false
	var nextDeadline time.Time
	for index := range requests {
		request := &requests[index]
		if recorded.Has(request.UID) {
			deployment.Status.ExpiredRequestUIDs = append(deployment.Status.ExpiredRequestUIDs, request.UID)
			if request.DeletionTimestamp.IsZero() {
				expired = append(expired, request)
			}
			continue
		}
		if !active {
			continue
		}
		if _, current := desiredNames[request.Name]; !current {
			continue
		}
		if !request.DeletionTimestamp.IsZero() || lpxRequestHasTerminalSchedulingDisposition(request) {
			continue
		}
		startedAt, known := lpxSchedulingStartedAt(request)
		if !known {
			continue
		}
		deadline := startedAt.Add(time.Duration(*deadlineSeconds) * time.Second)
		if !now.Before(deadline) {
			deployment.Status.ExpiredRequestUIDs = append(deployment.Status.ExpiredRequestUIDs, request.UID)
			recordFailure = true
			continue
		}
		if nextDeadline.IsZero() || deadline.Before(nextDeadline) {
			nextDeadline = deadline
		}
	}

	// Persist exact request identities and retry authorization before cleanup.
	if recordFailure {
		return &lpxDeadlineExceeded{requeueAfter: time.Nanosecond, recordFailure: true}, time.Time{}, nil
	}
	if recorded.Len() == 0 {
		return nil, nextDeadline, nil
	}

	// Lower Grove before deleting the affected engine suffix; convergence is asynchronous.
	// Requeue even after the last recorded UID disappears so an authorized retry can proceed.
	retirementErr := r.retireExpiredLPXRequests(ctx, deployment, source, pcs, requests, expired)
	return &lpxDeadlineExceeded{requeueAfter: lpxRetirementRequeueAfter}, nextDeadline, retirementErr
}

func lpxRequestHasTerminalSchedulingDisposition(request *lpxv1alpha1.LPUPipelineRequest) bool {
	if request.Status == nil || request.Status.ObservedGeneration == nil ||
		*request.Status.ObservedGeneration != request.Generation {
		return false
	}
	switch request.Status.Phase {
	case lpxv1alpha1.RequestPhaseBound, lpxv1alpha1.RequestPhaseNoFit, lpxv1alpha1.RequestPhaseUnsupported:
		return true
	case lpxv1alpha1.RequestPhaseDegraded, lpxv1alpha1.RequestPhaseReleasing, lpxv1alpha1.RequestPhaseReleased:
		committed := request.Status.Committed
		return committed != nil && committed.Execution.AcceptedGeneration != nil &&
			*committed.Execution.AcceptedGeneration == request.Generation
	default:
		return false
	}
}

// lpxSchedulingStartedAt requires a nonnil request and reports whether the
// scheduler published a usable start for its current scheduling cycle.
func lpxSchedulingStartedAt(request *lpxv1alpha1.LPUPipelineRequest) (time.Time, bool) {
	if request.Status == nil || request.Status.SchedulingStartedAt == nil || request.Status.SchedulingStartedAt.IsZero() {
		return time.Time{}, false
	}
	return request.Status.SchedulingStartedAt.Time, true
}

func lpxRequestDeadlineSeconds(source *nvidiacomv1beta1.DynamoGraphDeployment) *int64 {
	if source == nil || source.Spec.Scheduling == nil {
		return nil
	}
	return source.Spec.Scheduling.AttemptDeadlineSeconds
}

// lpxDeadlineFailureCurrent fences publication until expired requests disappear
// and a later input revision explicitly authorizes a retry.
func lpxDeadlineFailureCurrent(deployment *nvidiacomv1alpha1.LPXGraphDeployment) bool {
	failed := meta.FindStatusCondition(deployment.Status.Conditions, lpxSchedulingFailedCondition)
	return len(deployment.Status.ExpiredRequestUIDs) > 0 || (failed != nil &&
		failed.Status == metav1.ConditionTrue && failed.Reason == lpxSchedulingDeadlineExceededReason &&
		failed.ObservedGeneration >= deployment.Generation)
}

// projectLPXSchedulingFailureCondition keeps retry authorization independent
// from the aggregate Failed condition, which may report another controller error.
func projectLPXSchedulingFailureCondition(
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	previous *nvidiacomv1alpha1.LPXGraphDeploymentStatus,
	state reconcileOutcome,
	recordFailure bool,
) {
	prior := meta.FindStatusCondition(previous.Conditions, lpxSchedulingFailedCondition)
	if state.Reason == lpxSchedulingDeadlineExceededReason {
		observedGeneration := deployment.Generation
		if prior != nil && prior.Status == metav1.ConditionTrue && !recordFailure {
			observedGeneration = prior.ObservedGeneration
		}
		if recordFailure {
			meta.RemoveStatusCondition(&deployment.Status.Conditions, lpxSchedulingFailedCondition)
		}
		meta.SetStatusCondition(&deployment.Status.Conditions, metav1.Condition{
			Type: lpxSchedulingFailedCondition, Status: metav1.ConditionTrue,
			ObservedGeneration: observedGeneration, Reason: state.Reason, Message: state.Message,
		})
		return
	}
	if prior == nil {
		return
	}
	if deployment.Generation <= prior.ObservedGeneration || len(deployment.Status.ExpiredRequestUIDs) > 0 {
		return
	}
	meta.SetStatusCondition(&deployment.Status.Conditions, metav1.Condition{
		Type: lpxSchedulingFailedCondition, Status: metav1.ConditionFalse,
		ObservedGeneration: deployment.Generation, Reason: "LPXSchedulingRetryAuthorized",
		Message: "The expired request was removed and current input may be published",
	})
}
