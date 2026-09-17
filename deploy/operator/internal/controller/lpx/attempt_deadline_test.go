// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestLPXDeadlineUsesEachSchedulingCycleStart(t *testing.T) {
	t.Log("Publish two requests created together whose scheduling cycles start at different times")
	dgd, source, _ := newLPXTestDGD(t, "single")
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	firstStarted := time.Now().UTC().Truncate(time.Second)
	created := firstStarted.Add(-time.Hour)
	first := deadlineTestRequest(dgd, pcs, "first", created, lpxv1alpha1.RequestPhasePending)
	first.Status.SchedulingStartedAt = ptr.To(metav1.NewTime(firstStarted))
	second := deadlineTestRequest(dgd, pcs, "second", created, lpxv1alpha1.RequestPhasePending)
	second.Status.SchedulingStartedAt = ptr.To(metav1.NewTime(firstStarted.Add(10 * time.Second)))
	r := newLPXTestReconciler(t, nil, dgd, source, pcs, first, second)

	t.Log("Wake at the first scheduling cycle's deadline without using object creation time")
	classification, wake, err := r.reconcileLPXRequestDeadlines(t.Context(), dgd, source, pcs, deadlineTestDesired(first, second))
	require.NoError(t, err)
	require.Nil(t, classification)
	require.True(t, firstStarted.Add(30*time.Second).Equal(wake))
}

func TestLPXDeadlineIgnoresRequestsOutsideCurrentPublication(t *testing.T) {
	t.Log("Observe one current request and one expired request retired by a source edit")
	dgd, source, _ := newLPXTestDGD(t, "single")
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	started := time.Now().UTC().Truncate(time.Second)
	current := deadlineTestRequest(dgd, pcs, "current", started, lpxv1alpha1.RequestPhasePending)
	stale := deadlineTestRequest(dgd, pcs, "stale", started.Add(-time.Minute), lpxv1alpha1.RequestPhasePending)
	r := newLPXTestReconciler(t, nil, dgd, source, pcs, current, stale)

	t.Log("Keep the current wake without converting stale cleanup into deadline failure")
	classification, wake, err := r.reconcileLPXRequestDeadlines(
		t.Context(), dgd, source, pcs, deadlineTestDesired(current),
	)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.True(t, started.Add(30*time.Second).Equal(wake))
}

func TestLPXDeadlineWaitsForSchedulingCycleStart(t *testing.T) {
	t.Log("Observe an old pending request before the scheduler initializes its cycle timestamp")
	dgd, source, _ := newLPXTestDGD(t, "single")
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	request := deadlineTestRequest(dgd, pcs, "pending", time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending)
	request.Status.SchedulingStartedAt = nil
	r := newLPXTestReconciler(t, nil, dgd, source, pcs, request)

	t.Log("Do not infer a deadline from creation time or the current wall clock")
	classification, wake, err := r.reconcileLPXRequestDeadlines(t.Context(), dgd, source, pcs, deadlineTestDesired(request))
	require.NoError(t, err)
	require.Nil(t, classification)
	require.True(t, wake.IsZero())
}

func TestLPXDeadlineRestartsWhenReleasedRequestReturnsToPending(t *testing.T) {
	t.Log("Observe a previously planned request in a new pending scheduling cycle")
	dgd, source, _ := newLPXTestDGD(t, "single")
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	restarted := time.Now().UTC().Truncate(time.Second)
	created := restarted.Add(-time.Hour)
	request := deadlineTestRequest(dgd, pcs, "pending-again", created, lpxv1alpha1.RequestPhasePending)
	request.Status.LastPlanRevision = 1
	request.Status.SchedulingStartedAt = ptr.To(metav1.NewTime(restarted))
	r := newLPXTestReconciler(t, nil, dgd, source, pcs, request)

	t.Log("Deadline the new cycle from the scheduler's reset timestamp")
	classification, wake, err := r.reconcileLPXRequestDeadlines(t.Context(), dgd, source, pcs, deadlineTestDesired(request))
	require.NoError(t, err)
	require.Nil(t, classification)
	require.True(t, restarted.Add(30*time.Second).Equal(wake))
}

func TestLPXTerminalRequestHasNoDeadline(t *testing.T) {
	t.Log("Observe a bound request")
	dgd, source, _ := newLPXTestDGD(t, "single")
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	request := deadlineTestRequest(dgd, pcs, "bound", time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhaseBound)
	r := newLPXTestReconciler(t, nil, dgd, source, pcs, request)

	t.Log("Terminal scheduler disposition has neither expiry nor a wake")
	classification, wake, err := r.reconcileLPXRequestDeadlines(t.Context(), dgd, source, pcs, deadlineTestDesired(request))
	require.NoError(t, err)
	require.Nil(t, classification)
	require.True(t, wake.IsZero())
}

func TestLPXCompletedSchedulingDoesNotRestartAfterDegradation(t *testing.T) {
	t.Log("Observe a request whose initial scheduling completed before runtime degradation")
	dgd, source, _ := newLPXTestDGD(t, "single")
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	request := deadlineTestRequest(dgd, pcs, "degraded", time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhaseBound)
	request.Status.Phase = lpxv1alpha1.RequestPhaseDegraded
	r := newLPXTestReconciler(t, nil, dgd, source, pcs, request)

	classification, wake, err := r.reconcileLPXRequestDeadlines(t.Context(), dgd, source, pcs, deadlineTestDesired(request))
	require.NoError(t, err)
	require.Nil(t, classification)
	require.True(t, wake.IsZero())
}

func TestLPXDeadlineIgnoresRetiringRequests(t *testing.T) {
	for _, test := range []struct {
		name            string
		deleteRequest   bool
		deletePodClique bool
	}{
		{name: "request is terminating", deleteRequest: true},
		{name: "owner PCS is terminating", deletePodClique: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe an expired request that intentional cleanup already owns")
			dgd, source, _ := newLPXTestDGD(t, "single")
			source.Spec.Scheduling = deadlineTestScheduling()
			pcs := deadlineTestPCS(dgd, "pcs-uid")
			request := deadlineTestRequest(dgd, pcs, "expired", time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending)
			deletingAt := metav1.Now()
			if test.deleteRequest {
				request.Finalizers = []string{"scheduler.example/cleanup"}
				request.DeletionTimestamp = &deletingAt
			}
			if test.deletePodClique {
				pcs.Finalizers = []string{"grove.example/cleanup"}
				pcs.DeletionTimestamp = &deletingAt
			}
			r := newLPXTestReconciler(t, nil, dgd, source, pcs, request)

			t.Log("Do not turn an intentional removal into a scheduling failure")
			classification, wake, err := r.reconcileLPXRequestDeadlines(t.Context(), dgd, source, pcs, deadlineTestDesired(request))
			require.NoError(t, err)
			require.Nil(t, classification)
			require.True(t, wake.IsZero())
		})
	}
}

func TestLPXDeadlineListFailurePropagatesWithoutSyntheticWake(t *testing.T) {
	t.Log("Fail the authoritative LPR list before any deadline can be derived")
	dgd, source, _ := newLPXTestDGD(t, "single")
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	r := newLPXTestReconciler(t, nil, dgd, source, pcs)
	listErr := errors.New("request list unavailable")
	r.apiReader = interceptor.NewClient(r.apiReader.(client.WithWatch), interceptor.Funcs{
		List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
			return listErr
		},
	})

	t.Log("Return the real error so controller-runtime applies error backoff")
	classification, wake, err := r.reconcileLPXRequestDeadlines(t.Context(), dgd, source, pcs, nil)
	require.ErrorIs(t, err, listErr)
	require.Nil(t, classification)
	require.True(t, wake.IsZero())
	result, err := completeLPXDeadline(t.Context(), wake, ctrl.Result{}, err)
	require.ErrorIs(t, err, listErr)
	require.Zero(t, result)
}

func TestLPXDeadlineFailureRequiresPostFailureGeneration(t *testing.T) {
	t.Log("Record a deadline failure at the generation current when the failure occurred")
	dgd, _, _ := newLPXTestDGD(t, "single")
	dgd.Generation++
	dgd.Status.Conditions = []metav1.Condition{{
		Type:               lpxSchedulingFailedCondition,
		Status:             metav1.ConditionTrue,
		Reason:             lpxSchedulingDeadlineExceededReason,
		ObservedGeneration: dgd.Generation,
	}}
	require.True(t, lpxDeadlineFailureCurrent(dgd))

	t.Log("Only a generation after the failure authorizes another publication")
	dgd.Generation++
	require.False(t, lpxDeadlineFailureCurrent(dgd))
}

func TestLPXDeadlineFailureOnlyCoversPreviouslyPublishedRequests(t *testing.T) {
	t.Log("Record a deadline failure for the current deployment generation")
	dgd, _, _ := newLPXTestDGD(t, "single")
	failureTime := time.Now().Add(-time.Minute)
	dgd.Status.Conditions = []metav1.Condition{{
		Type: lpxSchedulingFailedCondition, Status: metav1.ConditionTrue,
		Reason: lpxSchedulingDeadlineExceededReason, ObservedGeneration: dgd.Generation,
		LastTransitionTime: metav1.NewTime(failureTime),
	}}

	t.Log("The failure covers cleanup of a scheduling cycle that started before it")
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	oldRequest := deadlineTestRequest(dgd, pcs, "old", failureTime.Add(-time.Minute), lpxv1alpha1.RequestPhasePending)
	require.True(t, lpxDeadlineFailureCovers(dgd, []*lpxv1alpha1.LPUPipelineRequest{oldRequest}))

	t.Log("A failure from an earlier deployment generation cannot cover current cleanup")
	dgd.Generation++
	require.False(t, lpxDeadlineFailureCovers(dgd, []*lpxv1alpha1.LPUPipelineRequest{oldRequest}))
	dgd.Generation--

	t.Log("A later scheduling cycle on the same request requires a new durable failure")
	restartedRequest := deadlineTestRequest(dgd, pcs, "restarted", failureTime.Add(-time.Hour), lpxv1alpha1.RequestPhasePending)
	restartedRequest.Status.LastPlanRevision = 1
	restartedRequest.Status.SchedulingStartedAt = ptr.To(metav1.NewTime(failureTime.Add(time.Second)))
	require.False(t, lpxDeadlineFailureCovers(dgd, []*lpxv1alpha1.LPUPipelineRequest{restartedRequest}))

	t.Log("An unknown cycle start is never covered by an older failure")
	restartedRequest.Status.SchedulingStartedAt = nil
	require.False(t, lpxDeadlineFailureCovers(dgd, []*lpxv1alpha1.LPUPipelineRequest{restartedRequest}))
}

func TestProjectLPXDeadlineFailureAdvancesAStaleFailureFence(t *testing.T) {
	dgd, _, _ := newLPXTestDGD(t, "single")
	previous := dgd.Status.DeepCopy()
	previous.Conditions = []metav1.Condition{{
		Type: lpxSchedulingFailedCondition, Status: metav1.ConditionTrue,
		Reason: lpxSchedulingDeadlineExceededReason, ObservedGeneration: dgd.Generation,
		LastTransitionTime: metav1.NewTime(time.Now().Add(-time.Minute)),
	}}
	dgd.Status = *previous.DeepCopy()
	dgd.Generation++

	t.Log("A newly expired request replaces the stale failure record")
	projectLPXSchedulingFailureCondition(dgd, previous, reconcileOutcome{
		State: nvidiacomv1beta1.DGDStateFailed, Reason: lpxSchedulingDeadlineExceededReason,
		Message: "expired",
	}, true)
	failed := meta.FindStatusCondition(dgd.Status.Conditions, lpxSchedulingFailedCondition)
	require.NotNil(t, failed)
	require.Equal(t, dgd.Generation, failed.ObservedGeneration)
	require.True(t, failed.LastTransitionTime.After(previous.Conditions[0].LastTransitionTime.Time))
}

func TestProjectLPXDeadlineFailureRemainsStickyAtRecordedGeneration(t *testing.T) {
	t.Log("Record a deadline failure for the current generation")
	dgd, _, _ := newLPXTestDGD(t, "single")
	previous := dgd.Status.DeepCopy()
	previous.Conditions = []metav1.Condition{{
		Type: lpxSchedulingFailedCondition, Status: metav1.ConditionTrue,
		Reason: lpxSchedulingDeadlineExceededReason, ObservedGeneration: dgd.Generation,
	}}
	dgd.Status = *previous.DeepCopy()

	t.Log("Do not let a late successful observation authorize the same generation")
	projectLPXSchedulingFailureCondition(dgd, previous, reconcileOutcome{
		State: nvidiacomv1beta1.DGDStateSuccessful, Reason: "Ready", Message: "ready",
	}, false)
	failed := meta.FindStatusCondition(dgd.Status.Conditions, lpxSchedulingFailedCondition)
	require.NotNil(t, failed)
	require.Equal(t, metav1.ConditionTrue, failed.Status)
	require.Equal(t, lpxSchedulingDeadlineExceededReason, failed.Reason)
}

func TestCompleteLPXDeadlineBoundsTransientErrors(t *testing.T) {
	reconcileErr := errors.New("temporary failure")
	result, err := completeLPXDeadline(context.Background(), time.Now().Add(time.Minute), ctrl.Result{}, reconcileErr)
	require.NoError(t, err)
	require.Positive(t, result.RequeueAfter)
	require.LessOrEqual(t, result.RequeueAfter, lpxRetirementRequeueAfter)

	result, err = completeLPXDeadline(context.Background(), time.Time{}, ctrl.Result{}, reconcileErr)
	require.ErrorIs(t, err, reconcileErr)
	require.Zero(t, result)
}

func deadlineTestRequest(dgd *nvidiacomv1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, name string, created time.Time, phase lpxv1alpha1.RequestPhase) *lpxv1alpha1.LPUPipelineRequest {
	generation := int64(1)
	schedulingStartedAt := metav1.NewTime(created.UTC().Truncate(time.Second))
	request := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: dgd.Namespace, UID: types.UID(name + "-uid"), ResourceVersion: "1",
			Generation: generation, CreationTimestamp: metav1.NewTime(created),
			Labels:          map[string]string{lpxOwnerUIDLabel: string(dgd.UID)},
			Annotations:     map[string]string{lpxDeploymentUIDAnnotation: string(dgd.UID)},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
		},
		Spec: lpxv1alpha1.LPUPipelineRequestSpec{ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal},
		Status: &lpxv1alpha1.LPUPipelineRequestStatus{
			Phase: phase, ObservedGeneration: &generation,
			SchedulingStartedAt: &schedulingStartedAt,
		},
	}
	if phase == lpxv1alpha1.RequestPhaseBound {
		request.Status.LastPlanRevision = 1
		request.Status.Committed = &lpxv1alpha1.Committed{
			Execution: lpxv1alpha1.CommittedExecution{AcceptedGeneration: &generation, NodeLocal: &lpxv1alpha1.NodeLocalExecution{}},
			Plan: lpxv1alpha1.CommittedPlan{
				PlannedFromGeneration: generation, Revision: 1, PlanDigest: "sha256:plan",
				Placement: lpxv1alpha1.PlanPlacement{ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal, NodeLocal: &lpxv1alpha1.NodeLocalPlacement{}},
			},
		}
	}
	return request
}

func deadlineTestDesired(requests ...*lpxv1alpha1.LPUPipelineRequest) []lpxModelMaterializing {
	desired := make([]lpxModelMaterializing, 0, len(requests))
	for _, request := range requests {
		desired = append(desired, lpxModelMaterializing{requestName: request.Name})
	}
	return desired
}

func deadlineTestPCS(dgd *nvidiacomv1alpha1.LPXGraphDeployment, uid types.UID) *grovev1alpha1.PodCliqueSet {
	return &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: dynamo.PCSNameForLPX(dgd), Namespace: dgd.Namespace, UID: uid, ResourceVersion: "1",
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, nvidiacomv1alpha1.LPXGraphDeploymentGVK)},
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{Replicas: 1},
	}
}

func deadlineTestScheduling() *nvidiacomv1beta1.SchedulingSpec {
	return &nvidiacomv1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
}
