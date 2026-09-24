// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"testing"
	"time"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestPipelineRequestDeadlines(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	oldStart := ptr.To(metav1.NewTime(now.Add(-time.Hour)))
	newStart := ptr.To(metav1.NewTime(now.Add(time.Minute)))
	laterStart := ptr.To(metav1.NewTime(now.Add(2 * time.Minute)))
	for _, tc := range []struct {
		name             string
		starts           []*metav1.Time
		created          *metav1.Time
		seconds          *int64
		noStatus         bool
		deleting         bool
		lastPlanRevision int64
		wantExpired      []string
		wantWake         time.Time
		previousWake     time.Time
	}{
		{name: "disabled deadline", starts: []*metav1.Time{oldStart}},
		{name: "no requests", seconds: ptr.To(int64(30))},
		{name: "no scheduler receipt expires from creation", starts: []*metav1.Time{nil}, seconds: ptr.To(int64(30)), noStatus: true, wantExpired: []string{"request-0"}},
		{name: "no scheduling start expires from creation", starts: []*metav1.Time{nil}, seconds: ptr.To(int64(30)), wantExpired: []string{"request-0"}},
		{name: "zero scheduling start expires from creation", starts: []*metav1.Time{{}}, seconds: ptr.To(int64(30)), wantExpired: []string{"request-0"}},
		{name: "new request awaits scheduler with a deadline", starts: []*metav1.Time{nil}, created: ptr.To(metav1.NewTime(now)), seconds: ptr.To(int64(30)), noStatus: true, wantWake: now.Add(30 * time.Second)},
		{name: "unpublished request has no deadline", starts: []*metav1.Time{nil}, created: &metav1.Time{}, seconds: ptr.To(int64(30)), noStatus: true},
		{name: "expired cycle", starts: []*metav1.Time{oldStart}, seconds: ptr.To(int64(30)), wantExpired: []string{"request-0"}},
		{name: "deleting request", starts: []*metav1.Time{oldStart}, seconds: ptr.To(int64(30)), deleting: true},
		{name: "independent starts", starts: []*metav1.Time{laterStart, newStart}, seconds: ptr.To(int64(30)), wantWake: newStart.Add(30 * time.Second)},
		{name: "expired and active cycles", starts: []*metav1.Time{oldStart, newStart}, seconds: ptr.To(int64(30)), wantExpired: []string{"request-0"}, wantWake: newStart.Add(30 * time.Second)},
		{name: "surviving request starts again", starts: []*metav1.Time{newStart}, seconds: ptr.To(int64(30)), lastPlanRevision: 1, wantWake: newStart.Add(30 * time.Second)},
		{name: "preserve earlier workload deadline", starts: []*metav1.Time{newStart}, seconds: ptr.To(int64(30)), previousWake: now, wantWake: now},
		{name: "replace later workload deadline", starts: []*metav1.Time{newStart}, seconds: ptr.To(int64(30)), previousWake: laterStart.Time, wantWake: newStart.Add(30 * time.Second)},
		{name: "unlimited workload preserves prior deadline", starts: []*metav1.Time{oldStart}, previousWake: now, wantWake: now},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Use creation time until the scheduler supplies a scheduling-cycle start")
			requests := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(tc.starts))
			for index, start := range tc.starts {
				request := &lpxv1alpha1.LPUPipelineRequest{
					ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("request-%d", index), Generation: 1, CreationTimestamp: *oldStart},
					Status: &lpxv1alpha1.LPUPipelineRequestStatus{
						Phase: lpxv1alpha1.RequestPhasePending, ObservedGeneration: ptr.To(int64(1)),
						SchedulingStartedAt: start, LastPlanRevision: tc.lastPlanRevision,
					},
				}
				request.Annotations = map[string]string{pipelineRequestModelAnnotation: "default"}
				if tc.created != nil {
					request.CreationTimestamp = *tc.created
				}
				if tc.noStatus {
					request.Status = nil
				}
				if tc.deleting {
					request.DeletionTimestamp = ptr.To(metav1.Now())
				}
				requests[request.Name] = request
			}

			t.Log("Return only expired requests and the earliest still-active deadline")
			expired, wake := pipelineRequestDeadlines(requests, map[string]*int64{"default": tc.seconds}, tc.previousWake)
			var names []string
			for _, request := range expired {
				names = append(names, request.Name)
			}
			require.ElementsMatch(t, tc.wantExpired, names)
			require.True(t, tc.wantWake.Equal(wake), "wake = %s, want %s", wake, tc.wantWake)
		})
	}
}

func TestIsPipelineRequestDeadlineExempt(t *testing.T) {
	for _, tc := range []struct {
		name     string
		phase    lpxv1alpha1.RequestPhase
		observed *int64
		accepted *int64
		noStatus bool
		want     bool
	}{
		{name: "no receipt", noStatus: true},
		{name: "unobserved Bound", phase: lpxv1alpha1.RequestPhaseBound},
		{name: "stale Bound", phase: lpxv1alpha1.RequestPhaseBound, observed: ptr.To(int64(1))},
		{name: "current Bound", phase: lpxv1alpha1.RequestPhaseBound, observed: ptr.To(int64(2)), want: true},
		{name: "NoFit", phase: lpxv1alpha1.RequestPhaseNoFit, observed: ptr.To(int64(2)), want: true},
		{name: "Unsupported", phase: lpxv1alpha1.RequestPhaseUnsupported, observed: ptr.To(int64(2)), want: true},
		{name: "Pending", phase: lpxv1alpha1.RequestPhasePending, observed: ptr.To(int64(2)), accepted: ptr.To(int64(2))},
		{name: "Planned", phase: lpxv1alpha1.RequestPhasePlanned, observed: ptr.To(int64(2))},
		{name: "Reserving", phase: lpxv1alpha1.RequestPhaseReserving, observed: ptr.To(int64(2))},
		{name: "Binding", phase: lpxv1alpha1.RequestPhaseBinding, observed: ptr.To(int64(2))},
		{name: "Degraded without execution", phase: lpxv1alpha1.RequestPhaseDegraded, observed: ptr.To(int64(2))},
		{name: "Degraded stale execution", phase: lpxv1alpha1.RequestPhaseDegraded, observed: ptr.To(int64(2)), accepted: ptr.To(int64(1))},
		{name: "Degraded accepted execution", phase: lpxv1alpha1.RequestPhaseDegraded, observed: ptr.To(int64(2)), accepted: ptr.To(int64(2)), want: true},
		{name: "Releasing without execution", phase: lpxv1alpha1.RequestPhaseReleasing, observed: ptr.To(int64(2))},
		{name: "Releasing accepted execution", phase: lpxv1alpha1.RequestPhaseReleasing, observed: ptr.To(int64(2)), accepted: ptr.To(int64(2)), want: true},
		{name: "Released without execution", phase: lpxv1alpha1.RequestPhaseReleased, observed: ptr.To(int64(2))},
		{name: "Released accepted execution", phase: lpxv1alpha1.RequestPhaseReleased, observed: ptr.To(int64(2)), accepted: ptr.To(int64(2)), want: true},
		{name: "unknown phase", phase: "future", observed: ptr.To(int64(2))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Only current scheduler receipts and accepted executions can exempt a cycle")
			request := &lpxv1alpha1.LPUPipelineRequest{
				ObjectMeta: metav1.ObjectMeta{Generation: 2},
				Status:     &lpxv1alpha1.LPUPipelineRequestStatus{Phase: tc.phase, ObservedGeneration: tc.observed},
			}
			if tc.accepted != nil {
				request.Status.Committed = &lpxv1alpha1.Committed{Execution: lpxv1alpha1.CommittedExecution{AcceptedGeneration: tc.accepted}}
			}
			if tc.noStatus {
				request.Status = nil
			}
			require.Equal(t, tc.want, isPipelineRequestDeadlineExempt(request))
		})
	}
}

func TestRequeueForPipelineRequestDeadline(t *testing.T) {
	transient := errors.New("temporary failure")
	for _, tc := range []struct {
		name       string
		until      time.Duration
		noDeadline bool
		requeue    time.Duration
		err        error
		wantDelay  time.Duration
	}{
		{name: "no deadline uses error backoff", noDeadline: true, err: transient},
		{name: "no deadline discards unrelated retry on error", noDeadline: true, err: transient, requeue: time.Minute},
		{name: "no deadline preserves result", noDeadline: true, requeue: time.Minute, wantDelay: time.Minute},
		{name: "active deadline", until: time.Minute, wantDelay: time.Minute},
		{name: "earlier existing retry", until: time.Minute, requeue: time.Second, wantDelay: time.Second},
		{name: "earlier deadline", until: time.Second, requeue: time.Minute, wantDelay: time.Second},
		{name: "elapsed deadline", until: -time.Minute, wantDelay: time.Nanosecond},
		{name: "error bounded by retry interval", until: time.Minute, err: transient, wantDelay: pipelineRequestDeadlineRetryInterval},
		{name: "error bounded by deadline", until: time.Second, err: transient, wantDelay: time.Second},
		{name: "error after deadline", until: -time.Minute, err: transient, wantDelay: pipelineRequestDeadlineRetryInterval},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Preserve an earlier wakeup and keep active-deadline errors on a bounded retry")
			var deadline time.Time
			if !tc.noDeadline {
				deadline = time.Now().Add(tc.until)
			}
			result := requeueForPipelineRequestDeadline(deadline, ctrl.Result{RequeueAfter: tc.requeue}, tc.err)
			require.LessOrEqual(t, result.RequeueAfter, tc.wantDelay)
			if tc.wantDelay <= time.Nanosecond || tc.wantDelay == tc.requeue || tc.wantDelay == pipelineRequestDeadlineRetryInterval {
				require.Equal(t, tc.wantDelay, result.RequeueAfter)
			} else {
				require.InDelta(t, tc.wantDelay, result.RequeueAfter, float64(time.Second))
			}
		})
	}
}

func TestReconcileExpiredPipelineRequests(t *testing.T) {
	for _, tc := range []struct {
		name           string
		replicas       int32
		expired        []int
		manageReplicas bool
		wantReplicas   int32
		removed        []int
	}{
		{name: "healthy lower engine", replicas: 2, expired: []int{1}, manageReplicas: true, wantReplicas: 1, removed: []int{1}},
		{name: "interior hole", replicas: 3, expired: []int{1}, manageReplicas: true, wantReplicas: 3},
		{name: "hole and suffix", replicas: 3, expired: []int{0, 2}, manageReplicas: true, wantReplicas: 3},
		{name: "maximal suffix", replicas: 4, expired: []int{2, 3}, manageReplicas: true, wantReplicas: 2, removed: []int{2, 3}},
		{name: "all expired", replicas: 2, expired: []int{0, 1}, manageReplicas: true, removed: []int{0, 1}},
		{name: "externally managed capacity", replicas: 1, expired: []int{0}, wantReplicas: 1, removed: []int{0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe independent workload requests with only the selected ordinals expired")
			dgd := loadTestDGD(t, lpx.PipelineSingle, "build-v2")
			deployment := newLPXTestDeployment(t, dgd)
			pcs := newTestPodCliqueSet(deployment)
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "engines", Namespace: deployment.Namespace},
				Spec:       grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: tc.replicas},
			}
			requests := make([]*lpxv1alpha1.LPUPipelineRequest, tc.replicas)
			objects := []client.Object{pcs, pcsg}
			var expired []*lpxv1alpha1.LPUPipelineRequest
			for i := range requests {
				request := newTestPipelineRequest(deployment, pcs, fmt.Sprintf("engine-%d", i), time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhaseBound)
				if slices.Contains(tc.expired, i) {
					request.Status.Phase = lpxv1alpha1.RequestPhasePending
					request.Status.Committed = nil
				}
				request.Spec.MaterializationTarget.PodCliqueScalingGroupRef = &lpxv1alpha1.PodCliqueScalingGroupReference{Name: pcsg.Name, ReplicaIndex: int64(i)}
				requests[i] = request
				objects = append(objects, request)
				if slices.Contains(tc.expired, i) {
					expired = append(expired, request)
				}
			}
			r := newLPXTestReconciler(t, nil, deployment, dgd, objects...)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcsg), pcsg))
			for _, request := range requests {
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(request), request))
			}

			t.Log("Apply suffix cleanup without deleting PodCliques or changing healthy requests")
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
					_, ok := object.(*lpxv1alpha1.LPUPipelineRequest)
					require.True(t, ok, "cleanup must delete only pipeline requests")
					live := &grovev1alpha1.PodCliqueScalingGroup{}
					require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(pcsg), live))
					require.Equal(t, tc.wantReplicas, live.Spec.Replicas, "scale must precede deletion")
					return delegated.Delete(ctx, object, opts...)
				},
			})
			err := r.reconcileExpiredPipelineRequests(t.Context(), pcsg, requests, expired, tc.manageReplicas)
			require.NoError(t, err)
			require.Equal(t, tc.replicas, pcsg.Spec.Replicas, "scale writes must preserve the observation")
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcsg), pcsg))
			require.Equal(t, tc.wantReplicas, pcsg.Spec.Replicas)
			for i, request := range requests {
				got := &lpxv1alpha1.LPUPipelineRequest{}
				err := r.Get(t.Context(), client.ObjectKeyFromObject(request), got)
				if slices.Contains(tc.removed, i) {
					require.True(t, apierrors.IsNotFound(err))
				} else {
					require.NoError(t, err)
					require.Equal(t, request, got)
				}
			}
		})
	}
}

func TestExpiredPipelineRequestSuffix(t *testing.T) {
	for _, tc := range []struct {
		name         string
		replicas     int32
		ordinals     []int64
		expired      []int
		wantReplicas int32
		wantRemoved  []string
	}{
		{name: "no expiry", replicas: 2, ordinals: []int64{0, 1}, wantReplicas: 2},
		{name: "siblings before expired model", replicas: 1, ordinals: []int64{0, 0}, expired: []int{0}, wantRemoved: []string{"request-1", "request-0"}},
		{name: "scale already lowered", replicas: 0, ordinals: []int64{0, 0}, expired: []int{0}, wantRemoved: []string{"request-1", "request-0"}},
		{name: "finish requests beyond live count", replicas: 1, ordinals: []int64{0, 1, 2}, expired: []int{1}, wantReplicas: 1, wantRemoved: []string{"request-2", "request-1"}},
		{name: "hole still blocks cleanup beyond live count", replicas: 2, ordinals: []int64{0, 1, 2}, expired: []int{0, 2}, wantReplicas: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Derive the complete trailing request set while preserving expiry evidence")
			requests := make([]*lpxv1alpha1.LPUPipelineRequest, len(tc.ordinals))
			var expired []*lpxv1alpha1.LPUPipelineRequest
			for i, ordinal := range tc.ordinals {
				requests[i] = &lpxv1alpha1.LPUPipelineRequest{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("request-%d", i)}}
				requests[i].Spec.MaterializationTarget.PodCliqueScalingGroupRef = &lpxv1alpha1.PodCliqueScalingGroupReference{Name: "engines", ReplicaIndex: ordinal}
				if slices.Contains(tc.expired, i) {
					expired = append(expired, requests[i])
				}
			}
			replicas, removed, err := expiredPipelineRequestSuffix(requests, expired, tc.replicas)
			require.NoError(t, err)
			require.Equal(t, tc.wantReplicas, replicas)
			var names []string
			for _, request := range removed {
				names = append(names, request.Name)
			}
			require.Equal(t, tc.wantRemoved, names)
		})
	}
}

func newTestPipelineRequest(deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, name string, created time.Time, phase lpxv1alpha1.RequestPhase) *lpxv1alpha1.LPUPipelineRequest {
	generation := int64(1)
	schedulingStartedAt := metav1.NewTime(created.UTC().Truncate(time.Second))
	request := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: deployment.Namespace, UID: types.UID(name + "-uid"), ResourceVersion: "1",
			Generation: generation, CreationTimestamp: metav1.NewTime(created),
			Labels:          map[string]string{deploymentUIDLabel: string(deployment.UID)},
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

func newTestPodCliqueSet(deployment *v1alpha1.LPXGraphDeployment) *grovev1alpha1.PodCliqueSet {
	return &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: dynamo.PCSNameForLPX(deployment), Namespace: deployment.Namespace, UID: "pcs-uid", ResourceVersion: "1",
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(deployment, v1alpha1.LPXGraphDeploymentGVK)},
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{Replicas: 1},
	}
}

func TestExpiredPipelineRequestsRetrySiblingDeletion(t *testing.T) {
	t.Log("Publish multiple model requests for one workload and expire one of them")
	ctx := t.Context()
	child, dgd, registry := newLPXSpecDecodeTestDGD(t)
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	requests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	pcsgs, err := getPodCliqueScalingGroups(ctx, r.Client, pcs)
	pcsg := pcsgs[desired.plan.LPXScalingGroup]
	require.NoError(t, err)
	require.Greater(t, len(requests), 1)
	expired := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].Name)
	var sibling *lpxv1alpha1.LPUPipelineRequest
	for _, request := range requests {
		if request.Name != expired.Name {
			sibling = request
			break
		}
	}
	require.NotNil(t, sibling)

	t.Log("Fail the first sibling deletion and retain the expired request for a retry")
	deleteErr := errors.New("transient sibling deletion failure")
	base := r.Client
	r.Client = interceptor.NewClient(base.(client.WithWatch), interceptor.Funcs{
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			if request, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok && request.Name == sibling.Name {
				return deleteErr
			}
			return delegated.Delete(ctx, object, opts...)
		},
	})
	err = r.reconcileExpiredPipelineRequests(ctx, pcsg, slices.Collect(maps.Values(requests)), []*lpxv1alpha1.LPUPipelineRequest{expired}, true)
	require.ErrorIs(t, err, deleteErr)
	require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{}))

	t.Log("Retry from the retained expiry proof and remove the complete workload request set")
	r.Client = base
	requests, err = r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	expired = getTestPipelineRequest(t, ctx, r.Client, child.Namespace, expired.Name)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	err = r.reconcileExpiredPipelineRequests(ctx, pcsg, slices.Collect(maps.Values(requests)), []*lpxv1alpha1.LPUPipelineRequest{expired}, true)
	require.NoError(t, err)
	for index := range desired.requests {
		requirePipelineRequestNotFound(t, ctx, r.Client, child.Namespace, desired.requests[index].Name)
	}
}
