/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"math"
	"time"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	ctrl "sigs.k8s.io/controller-runtime"
)

// Keep transient errors from delaying an active scheduling deadline indefinitely.
const pipelineRequestDeadlineRetryInterval = 5 * time.Second

// pipelineRequestDeadlineSeconds reads the optional scheduling timeout from a non-nil DGD.
func pipelineRequestDeadlineSeconds(dgd *v1beta1.DynamoGraphDeployment) *int64 {
	if dgd.Spec.Scheduling == nil {
		return nil
	}
	return dgd.Spec.Scheduling.AttemptDeadlineSeconds
}

// pipelineRequestDeadlines examines already-selected, owned requests without I/O.
// Missing schedulingStartedAt means the scheduler has not started a cycle yet.
// A surviving LPR can return to Pending; its scheduler-supplied start resets the clock.
// Creation time and managed fields cannot identify that new scheduling cycle.
func pipelineRequestDeadlines(requests map[string]*lpxv1alpha1.LPUPipelineRequest, seconds *int64) (expired []*lpxv1alpha1.LPUPipelineRequest, next time.Time) {
	if seconds == nil {
		return nil, time.Time{}
	}
	now := time.Now()
	for _, request := range requests {
		if !request.DeletionTimestamp.IsZero() || isPipelineRequestDeadlineExempt(request) {
			continue
		}
		started, known := pipelineRequestSchedulingStartedAt(request)
		if !known {
			continue
		}
		deadline := started.Add(time.Duration(*seconds) * time.Second)
		if !now.Before(deadline) {
			expired = append(expired, request)
		} else if next.IsZero() || deadline.Before(next) {
			next = deadline
		}
	}
	return expired, next
}

// pipelineRequestSchedulingStartedAt requires a nonnil request and reports whether the
// scheduler published a usable start for its current scheduling cycle.
func pipelineRequestSchedulingStartedAt(request *lpxv1alpha1.LPUPipelineRequest) (time.Time, bool) {
	if request.Status == nil || request.Status.SchedulingStartedAt == nil || request.Status.SchedulingStartedAt.IsZero() {
		return time.Time{}, false
	}
	return request.Status.SchedulingStartedAt.Time, true
}

// isPipelineRequestDeadlineExempt recognizes current receipts outside the scheduling timer.
// This is not a readiness test: NoFit is still pending, and Bound can later return to Pending.
// request is non-nil; stale receipts cannot exempt its current generation.
func isPipelineRequestDeadlineExempt(request *lpxv1alpha1.LPUPipelineRequest) bool {
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

// requeueForPipelineRequestDeadline preserves deadline wakes and bounds error retries.
// Without a deadline, errors discard unrelated wakeups and use controller-runtime backoff.
// The caller retains the error until status persistence succeeds.
func requeueForPipelineRequestDeadline(
	deadlineAt time.Time,
	result ctrl.Result,
	reconcileErr error,
) ctrl.Result {
	if deadlineAt.IsZero() {
		if reconcileErr != nil {
			return ctrl.Result{}
		}
		return result
	}
	delay := time.Until(deadlineAt)
	if reconcileErr != nil {
		if delay <= 0 || delay > pipelineRequestDeadlineRetryInterval {
			delay = pipelineRequestDeadlineRetryInterval
		}
		return ctrl.Result{RequeueAfter: delay}
	}
	if delay <= 0 {
		delay = time.Nanosecond
	}
	if result.RequeueAfter <= 0 || delay < result.RequeueAfter {
		result.RequeueAfter = delay
	}
	return result
}

// reconcilePipelineRequestDeadline records expiry before cleanup and blocks republication
// for the failed generation. Call only for expired requests or a current SchedulingFailed
// condition. deployment and requests are validated observations. A nil pcsg still
// permits recording failure, but cleanup must wait for an observed scaling group.
func (r *graphReconciler) reconcilePipelineRequestDeadline(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	pcsg *grovev1alpha1.PodCliqueScalingGroup,
	requests []lpxv1alpha1.LPUPipelineRequest,
	expired []*lpxv1alpha1.LPUPipelineRequest,
	manageReplicas bool,
) (ctrl.Result, error) {
	// Persist a failure covering every expired scheduling cycle before deleting requests.
	if len(expired) > 0 {
		if !schedulingFailureCoversPipelineRequests(deployment, expired) {
			setSchedulingFailedCondition(deployment, true)
			// Status-only LPXGD updates are filtered, so explicitly observe the persisted failure.
			return ctrl.Result{RequeueAfter: time.Nanosecond}, nil
		}
		setSchedulingFailedCondition(deployment, false)
		return r.reconcileExpiredPipelineRequests(ctx, pcsg, requests, expired, manageReplicas)
	}

	// A failed generation must not recreate requests removed by its deadline.
	setSchedulingFailedCondition(deployment, false)
	return ctrl.Result{}, nil
}

// reconcileExpiredPipelineRequests removes only complete expired trailing replicas.
// A nil pcsg defers cleanup: the live replica count is unknown, not zero. Non-nil
// pcsg and all requests are already validated against their PCS. No PodCliques
// are deleted for interior failures.
// manageReplicas is false for omitted DGD replicas: external scaling retains capacity ownership.
func (r *graphReconciler) reconcileExpiredPipelineRequests(
	ctx context.Context,
	pcsg *grovev1alpha1.PodCliqueScalingGroup,
	requests []lpxv1alpha1.LPUPipelineRequest,
	expired []*lpxv1alpha1.LPUPipelineRequest,
	manageReplicas bool,
) (ctrl.Result, error) {
	// Observe capacity before deciding which ordinals are safe to remove; the PCSG watch resumes cleanup.
	if pcsg == nil {
		return ctrl.Result{}, nil
	}
	replicas, removed, err := expiredPipelineRequestSuffix(requests, expired, pcsg.Spec.Replicas)
	if err != nil || len(removed) == 0 {
		return ctrl.Result{}, err
	}

	// Persist the lower PCSG count before asynchronous LPR/pod cleanup starts.
	if manageReplicas {
		if err := r.scaleDownPodCliqueScalingGroup(ctx, pcsg, &replicas); err != nil {
			return ctrl.Result{}, err
		}
	}
	return ctrl.Result{}, r.deletePipelineRequests(ctx, removed)
}

// expiredPipelineRequestSuffix returns complete engine replicas only when every
// expired replica belongs to one contiguous trailing suffix. It orders the
// expired requests last so transient sibling cleanup leaves durable expiry
// evidence for the next reconciliation. An interior failure blocks all
// scale-down so a healthy higher ordinal is never removed.
// For four replicas, expiry at {2,3} removes that suffix; {0,3} removes nothing.
func expiredPipelineRequestSuffix(
	requests []lpxv1alpha1.LPUPipelineRequest,
	expired []*lpxv1alpha1.LPUPipelineRequest,
	replicas int32,
) (int32, []*lpxv1alpha1.LPUPipelineRequest, error) {
	if len(expired) == 0 {
		return replicas, nil, nil
	}
	groupName, _, err := pipelineRequestScalingGroupTarget(expired[0])
	if err != nil {
		return 0, nil, err
	}
	failedReplicas := make(map[int64]struct{}, len(expired))
	for _, request := range expired {
		name, replica, err := pipelineRequestScalingGroupTarget(request)
		if err != nil {
			return 0, nil, err
		}
		if name != groupName {
			return 0, nil, fmt.Errorf("expired LPX requests target different scaling groups")
		}
		failedReplicas[replica] = struct{}{}
	}

	// Find the maximal expired suffix, not a cutoff at the minimum expired ordinal.
	targetReplicas := int64(replicas)
	for targetReplicas > 0 {
		if _, failed := failedReplicas[targetReplicas-1]; !failed {
			break
		}
		targetReplicas--
	}

	// Any hole blocks all cleanup, even a separate expired trailing suffix.
	for replica := range failedReplicas {
		if replica < targetReplicas {
			return replicas, nil, nil
		}
	}

	expiredNames := make(map[string]struct{}, len(expired))
	for _, request := range expired {
		expiredNames[request.Name] = struct{}{}
	}
	siblings := make([]*lpxv1alpha1.LPUPipelineRequest, 0, len(requests))
	failed := make([]*lpxv1alpha1.LPUPipelineRequest, 0, len(expired))
	for index := range requests {
		request := &requests[index]
		name, replica, err := pipelineRequestScalingGroupTarget(request)
		if err != nil {
			return 0, nil, err
		}
		if name != groupName {
			return 0, nil, fmt.Errorf("LPX requests target different scaling groups")
		}

		// Include LPRs beyond the live count to finish cleanup after an earlier scale write.
		if replica < targetReplicas {
			continue
		}
		if _, expired := expiredNames[request.Name]; expired {
			failed = append(failed, request)
		} else {
			siblings = append(siblings, request)
		}
	}
	return int32(targetReplicas), append(siblings, failed...), nil
}

func pipelineRequestScalingGroupTarget(request *lpxv1alpha1.LPUPipelineRequest) (string, int64, error) {
	target := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
	if target == nil || target.Name == "" || target.ReplicaIndex < 0 || target.ReplicaIndex >= math.MaxInt32 {
		return "", 0, fmt.Errorf("LPX request %q lacks a valid scaling group target", request.Name)
	}
	return target.Name, target.ReplicaIndex, nil
}
