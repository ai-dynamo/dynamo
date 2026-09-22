/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"maps"
	"math"
	"slices"
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
// Creation time bounds the initial wait for the scheduler. On reused requests,
// the scheduler-supplied start takes precedence for later scheduling cycles.
func pipelineRequestDeadlines(requests map[string]*lpxv1alpha1.LPUPipelineRequest, seconds *int64) (expired []*lpxv1alpha1.LPUPipelineRequest, next time.Time) {
	if seconds == nil {
		return nil, time.Time{}
	}
	now := time.Now()
	for _, request := range requests {
		if !request.DeletionTimestamp.IsZero() || isPipelineRequestDeadlineExempt(request) {
			continue
		}

		// Desired requests awaiting publication have neither timestamp yet.
		started := pipelineRequestSchedulingStartedAt(request)
		if started.IsZero() {
			continue
		}

		// Track each request's expiration and the earliest remaining deadline.
		deadline := started.Add(time.Duration(*seconds) * time.Second)
		if !now.Before(deadline) {
			expired = append(expired, request)
		} else if next.IsZero() || deadline.Before(next) {
			next = deadline
		}
	}
	return expired, next
}

// pipelineRequestSchedulingStartedAt returns the scheduler's nonzero cycle start,
// or creation time while waiting for the scheduler. request must be non-nil.
func pipelineRequestSchedulingStartedAt(request *lpxv1alpha1.LPUPipelineRequest) time.Time {
	if request.Status == nil || request.Status.SchedulingStartedAt == nil || request.Status.SchedulingStartedAt.IsZero() {
		return request.CreationTimestamp.Time
	}
	return request.Status.SchedulingStartedAt.Time
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

// reconcileSchedulingFailure records graph-wide failure before deadline cleanup.
// Each workload removes only its own expired suffix; omitted replicas retain
// external capacity ownership. All configured groups have been observed.
// deployment and requests are validated observations; replica counts are keyed by PCSG name.
func (r *graphReconciler) reconcileSchedulingFailure(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	groups map[string]*grovev1alpha1.PodCliqueScalingGroup,
	explicitReplicas map[string]*int32,
	requests []lpxv1alpha1.LPUPipelineRequest,
	expired []*lpxv1alpha1.LPUPipelineRequest,
) (ctrl.Result, error) {
	// Persist evidence covering every expired cycle before deleting any expired request.
	if len(expired) > 0 && !schedulingFailureCoversPipelineRequests(deployment, expired) {
		setSchedulingFailedCondition(deployment, true)
		return ctrl.Result{RequeueAfter: time.Nanosecond}, nil
	}
	setSchedulingFailedCondition(deployment, false)

	// Partition both collections by their actual target; ordinals belong to one workload.
	expiredByGroup := make(map[string][]*lpxv1alpha1.LPUPipelineRequest)
	for _, request := range expired {
		name := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name
		expiredByGroup[name] = append(expiredByGroup[name], request)
	}
	requestsByGroup := make(map[string][]lpxv1alpha1.LPUPipelineRequest)
	for _, request := range requests {
		name := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name
		requestsByGroup[name] = append(requestsByGroup[name], request)
	}

	// An interior failure in one workload does not prevent another workload's suffix cleanup.
	for _, name := range slices.Sorted(maps.Keys(expiredByGroup)) {
		if err := r.reconcileExpiredPipelineRequests(ctx, groups[name], requestsByGroup[name], expiredByGroup[name], explicitReplicas[name] != nil); err != nil {
			return ctrl.Result{}, err
		}
	}
	return ctrl.Result{}, nil
}

// reconcileExpiredPipelineRequests removes only complete expired trailing replicas.
// pcsg must be non-nil; it and all requests are already validated against their PCS.
// No PodCliques are deleted for interior failures.
// manageReplicas is false for omitted DGD replicas: external scaling retains capacity ownership.
func (r *graphReconciler) reconcileExpiredPipelineRequests(
	ctx context.Context,
	pcsg *grovev1alpha1.PodCliqueScalingGroup,
	requests []lpxv1alpha1.LPUPipelineRequest,
	expired []*lpxv1alpha1.LPUPipelineRequest,
	manageReplicas bool,
) error {
	// Use the observed capacity to decide which ordinals are safe to remove.
	replicas, removed, err := expiredPipelineRequestSuffix(requests, expired, pcsg.Spec.Replicas)
	if err != nil || len(removed) == 0 {
		return err
	}

	// Persist the lower PCSG count before asynchronous LPR/pod cleanup starts.
	if manageReplicas {
		if err := scaleDownPodCliqueScalingGroup(ctx, r, pcsg, replicas); err != nil {
			return err
		}
	}

	return r.deletePipelineRequests(ctx, removed)
}

// expiredPipelineRequestSuffix returns complete workload replicas only when every
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
