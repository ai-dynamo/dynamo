/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

func TestLPXAttemptIsPreparedBeforePublicationAndCannotBeRecreated(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	source.Spec.Scheduling = deadlineTestScheduling()
	reconciler := newLPXTestReconciler(t, registry, dgd, source)
	desired, rejected := requirePreparedLPX(t, reconciler, ctx, dgd, source)
	require.Nil(t, rejected)
	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)

	classification, err := reconcileSelectedLPXForTest(ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	require.Equal(t, lpxAttemptPreparedReason, lpxResult(classification).Reason)
	prepared := applyLPXAttemptTransition(t, dgd, classification)
	require.NotEmpty(t, prepared.PodCliqueSetUID)
	require.Empty(t, ownedLPXRequests(t, ctx, reconciler, dgd))

	_, err = reconcileSelectedLPXForTest(ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	requests := ownedLPXRequests(t, ctx, reconciler, dgd)
	require.Len(t, requests, 1)
	require.Empty(t, requests[0].Finalizers)
	dgd.Status.Placement = nil
	classification, err = reconcileSelectedLPXForTest(ctx, reconciler, dgd, desired)
	require.NoError(t, err)
	recovered := applyLPXAttemptTransition(t, dgd, classification)
	require.Equal(t, requests[0].UID, recovered.Requests[0].UID, "publication reconciliation recovers the existing request")
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: recovered}

	require.NoError(t, reconciler.Delete(ctx, &requests[0]))
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(ctx, dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxAttemptAuthorityLostReason, lpxResult(classification).Reason)

	for _, test := range []struct {
		name                  string
		deadline, terminating bool
	}{
		{name: "absent with deadline", deadline: true},
		{name: "terminating with deadline", deadline: true, terminating: true},
		{name: "absent without deadline"},
		{name: "terminating without deadline", terminating: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Keep ready Grove identities after losing an already-recorded request")
			deployment, source := dgd.DeepCopy(), source.DeepCopy()
			attempt := currentLPXAttemptStatus(deployment)
			attempt.DeadlineAt = nil
			if test.deadline {
				attempt.DeadlineAt = ptr.To(metav1.NewTime(time.Now().Add(time.Hour)))
			} else {
				source.Spec.Scheduling = nil
				attempt.DisarmedAt = ptr.To(metav1.Now())
			}
			objects := lpxMaterializedObjects(t, reconciler, deployment, source, desired)
			pcs := findLPXTestPodCliqueSet(t, objects)
			removed := deadlineTestRequest(deployment, pcs, "removed-engine", time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhaseBound)
			objects = append(objects, removed)
			attempt.Requests = append(attempt.Requests, deadlineTestAttempt(deployment, removed, time.Now()).Requests...)
			if test.terminating {
				request := requests[0].DeepCopy()
				request.DeletionTimestamp = ptr.To(metav1.Now())
				request.Finalizers = []string{"scheduler.example/cleanup"}
				objects = append(objects, request)
			}
			r := newLPXTestReconciler(t, registry, deployment, source, objects...)
			t.Log("Terminal authority loss deletes the PCS before its requests")
			var transition *lpxDeadlineTransition
			for range 2 {
				classification, _, _, err := r.reconcileLPXAttemptDeadline(ctx, deployment, source)
				require.NoError(t, err)
				require.Equal(t, lpxAttemptAuthorityLostReason, lpxResult(classification).Reason)
				transition = classification.(*lpxDeadlineTransition)
				deployment.Status.Placement.LPXAttempt = transition.attempt
			}
			pcs = &grovev1alpha1.PodCliqueSet{}
			require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: desired.plan.PodCliqueSetName}, pcs)))
			removed = getLPXRequest(t, ctx, r.Client, deployment.Namespace, "removed-engine")
			require.NoError(t, r.Delete(ctx, removed), "emulate garbage collection in the fake client")
			live := ownedLPXRequests(t, ctx, r, deployment)
			if test.terminating {
				require.Len(t, live, 1)
				require.Equal(t, requests[0].UID, live[0].UID)
				require.False(t, live[0].DeletionTimestamp.IsZero())
				require.Equal(t, lpxRetirementRequeueAfter, transition.requeueAfter)
			} else {
				require.Empty(t, live)
				require.Equal(t, lpxRetirementRequeueAfter, transition.requeueAfter)
			}
		})
	}
}

func TestLPXPreflightReusesActiveDeadlineRequestObservation(t *testing.T) {
	source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
	dgd := newLPXTestDeployment(t, source)
	source.Spec.Scheduling = deadlineTestScheduling()
	deadline := metav1.NewTime(time.Now().Add(time.Minute))
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: &nvidiacomv1beta1.LPXAttemptStatus{
		ObservedGeneration: dgd.Generation,
		DeadlineAt:         &deadline,
		Requests: []nvidiacomv1beta1.LPXAttemptRequestStatus{{
			Name: "unpublished", AttemptDigest: "sha256:attempt",
		}},
	}}
	reconciler := newLPXTestReconciler(t, nil, dgd, source)
	reader, ok := reconciler.apiReader.(client.WithWatch)
	require.True(t, ok)
	requestLists := 0
	reconciler.apiReader = interceptor.NewClient(reader, interceptor.Funcs{
		List: func(
			ctx context.Context,
			delegated client.WithWatch,
			list client.ObjectList,
			opts ...client.ListOption,
		) error {
			if _, ok := list.(*lpxv1alpha1.LPUPipelineRequestList); ok {
				requestLists++
			}
			return delegated.List(ctx, list, opts...)
		},
	})

	t.Log("Reuse the active deadline request observation during safety preflight")
	classification, wake, requests, err := reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Nil(t, classification)
	selected, classification, err := reconciler.reconcileLPXSafetyPreflight(t.Context(), dgd, source, requests)
	require.NoError(t, err)
	require.Nil(t, selected)
	require.Nil(t, classification)
	require.True(t, wake.Equal(deadline.Time))
	require.Equal(t, 1, requestLists)
}

func TestLPXAttemptPartialPublicationLossFencesEverySibling(t *testing.T) {
	ctx := t.Context()
	dgd, source, registry := newLPXSpecDecodeTestDGD(t)
	source.Spec.Scheduling = deadlineTestScheduling()
	reconciler := newLPXTestReconciler(t, registry, dgd, source)
	desired, rejected := requirePreparedLPX(t, reconciler, ctx, dgd, source)
	require.Nil(t, rejected)
	require.Len(t, desired.requests, 3)
	createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, dgd, source, desired)...)

	for range 3 {
		classification, err := reconcileSelectedLPXForTest(ctx, reconciler, dgd, desired)
		require.NoError(t, err)
		applyLPXAttemptTransition(t, dgd, classification)
	}
	requests := ownedLPXRequests(t, ctx, reconciler, dgd)
	require.Len(t, requests, 2, "preparation plus one persisted transition per successful create")

	dgd.Status.Placement.LPXAttempt.DeadlineAt = ptr.To(metav1.NewTime(time.Now().Add(-time.Second)))
	require.ErrorContains(t, reconciler.revalidateLPXAttemptPublication(ctx, dgd, desired.plan.PodCliqueSetName), "deadline")
	dgd.Status.Placement.LPXAttempt.DeadlineAt = ptr.To(metav1.NewTime(time.Now().Add(time.Minute)))
	require.NoError(t, reconciler.Delete(ctx, &requests[0]))
	require.Error(t, reconciler.revalidateLPXAttemptPublication(ctx, dgd, desired.plan.PodCliqueSetName))
	classification, _, _, err := reconciler.reconcileLPXAttemptDeadline(ctx, dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxAttemptAuthorityLostReason, lpxResult(classification).Reason)
	require.Equal(t, nvidiacomv1beta1.DGDStateFailed, lpxResult(classification).State)
	applyLPXAttemptTransition(t, dgd, classification)

	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(ctx, dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxAttemptAuthorityLostReason, lpxResult(classification).Reason)
	_, _ = reconcileSelectedLPXForTest(ctx, reconciler, dgd, desired)
	require.LessOrEqual(t, len(ownedLPXRequests(t, ctx, reconciler, dgd)), 1)
}

func TestLPXAttemptDeadlineRaceUsesOnlyDurableDisposition(t *testing.T) {
	t.Log("Seed an exact Bound request whose deadline has not yet been recovered")
	source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
	dgd := newLPXTestDeployment(t, source)
	source.Spec.Scheduling = deadlineTestScheduling()
	pcs := deadlineTestPCS(dgd, "pcs-uid")
	request := deadlineTestRequest(dgd, pcs, "attempt", time.Now().UTC().Truncate(time.Second), lpxv1alpha1.RequestPhaseBound)
	running := deadlineTestRequest(dgd, pcs, "older-engine", time.Now().UTC().Truncate(time.Second).Add(-time.Hour), lpxv1alpha1.RequestPhaseBound)
	group := &grovev1alpha1.PodCliqueScalingGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name: "engine-group", Namespace: dgd.Namespace,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
		},
		Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 2},
	}
	request.Spec.MaterializationTarget.PodCliqueScalingGroupRef = &lpxv1alpha1.PodCliqueScalingGroupReference{Name: group.Name, ReplicaIndex: 1}
	running.Spec.MaterializationTarget.PodCliqueScalingGroupRef = &lpxv1alpha1.PodCliqueScalingGroupReference{Name: group.Name, ReplicaIndex: 0}
	request.Finalizers = []string{"scheduling.lpu.nvidia.com/lpx-cleanup"}
	attempt := deadlineTestAttempt(dgd, request, time.Now().Add(time.Minute))
	attempt.DeadlineAt, attempt.PodCliqueSetUID = nil, ""
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: attempt}
	reconciler := newLPXTestReconciler(t, nil, dgd, source, request, running, pcs, group)
	running = getLPXRequest(t, t.Context(), reconciler.Client, running.Namespace, running.Name)

	t.Log("Recover the native deadline and record the Bound scheduling disposition")
	classification, _, _, err := reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	recovered := applyLPXAttemptTransition(t, dgd, classification)
	require.True(t, recovered.DeadlineAt.Time.Equal(request.CreationTimestamp.Add(30*time.Second)))
	require.Equal(t, types.UID(request.Annotations[lpxPCSUIDAnnotation]), recovered.PodCliqueSetUID)
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	disarmed := applyLPXAttemptTransition(t, dgd, classification)
	require.NotNil(t, disarmed.DisarmedAt)

	t.Log("A durably recorded disposition survives the elapsed deadline")
	disarmed.DeadlineAt = ptr.To(metav1.NewTime(time.Now().Add(-time.Minute)))
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Nil(t, classification)

	t.Log("Resumed scheduling expires under the original elapsed deadline")
	request.Status.Phase = lpxv1alpha1.RequestPhasePending
	reconciler = newLPXTestReconciler(t, nil, dgd, source, request, running, pcs, group)
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	result := lpxResult(classification)
	require.Equal(t, lpxSchedulingDeadlineExceededReason, result.Reason)
	transition := classification.(*lpxDeadlineTransition)
	require.NotNil(t, transition.attempt.ExceededAt)
	require.Nil(t, transition.attempt.DisarmedAt)

	t.Log("Persist expiry before cleaning up the exact scheduling request")
	applyLPXAttemptTransition(t, dgd, classification)
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, request.Name).DeletionTimestamp.IsZero())

	t.Log("Scale down the failed engine before deleting its request, preserving the older engine")
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	pcsKey := client.ObjectKeyFromObject(pcs)
	require.NoError(t, reconciler.Get(t.Context(), pcsKey, storedPCS))
	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, request.Name).DeletionTimestamp.IsZero())
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, running.Name).DeletionTimestamp.IsZero())

	t.Log("Retire only the failed request after Grove has accepted the lower scale")
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	deleting := getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, request.Name)
	require.False(t, deleting.DeletionTimestamp.IsZero())
	require.Equal(t, running, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, running.Name))
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)

	t.Log("The request disappears when scheduler cleanup releases its own finalizer")
	deleting.Finalizers = nil
	require.NoError(t, reconciler.Update(t.Context(), deleting))
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.True(t, apierrors.IsNotFound(reconciler.Get(t.Context(), client.ObjectKeyFromObject(request), &lpxv1alpha1.LPUPipelineRequest{})))
	require.Zero(t, classification.(*lpxDeadlineTransition).requeueAfter)

	t.Log("A completed old attempt cannot touch same-name replacement identities")
	replacement := request.DeepCopy()
	replacement.UID = "replacement-uid"
	replacement.ResourceVersion = "2"
	replacement.Annotations[lpxPCSUIDAnnotation] = "replacement-pcs-uid"
	pcs = deadlineTestPCS(dgd, "replacement-pcs-uid")
	reconciler = newLPXTestReconciler(t, nil, dgd, source, replacement, pcs)

	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxSchedulingDeadlineExceededReason, lpxResult(classification).Reason)
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, replacement.Name).DeletionTimestamp.IsZero())
	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(pcs), storedPCS))
	require.Equal(t, int32(1), storedPCS.Spec.Replicas)

	t.Log("A stale cached attempt has no exact objects to clean up after replacement")
	key := client.ObjectKeyFromObject(dgd)
	require.NoError(t, reconciler.Get(t.Context(), key, dgd))
	cached := dgd.DeepCopy()
	current := dgd.DeepCopy()
	current.Status.Placement.LPXAttempt = deadlineTestAttempt(current, replacement, time.Now().Add(time.Minute))
	require.NoError(t, reconciler.Status().Update(t.Context(), current))
	require.NotEqual(t, cached.ResourceVersion, current.ResourceVersion)
	running.ResourceVersion = ""
	require.NoError(t, reconciler.Create(t.Context(), running))
	cached.Status.Placement.LPXAttempt.Requests = append(cached.Status.Placement.LPXAttempt.Requests,
		deadlineTestAttempt(cached, running, time.Now()).Requests[0])
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), cached, source)
	require.NoError(t, err)
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, running.Name).DeletionTimestamp.IsZero())
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, replacement.Name).DeletionTimestamp.IsZero())

	t.Log("An exact Bound observation is authoritative even before its disposition is persisted")
	late := deadlineTestAttempt(dgd, request, time.Now().Add(-time.Minute))
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: late}
	request.Status.Phase = lpxv1alpha1.RequestPhaseBound
	reconciler = newLPXTestReconciler(t, nil, dgd, source, request)
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxSchedulingDispositionObservedReason, lpxResult(classification).Reason)
	require.NotNil(t, classification.(*lpxDeadlineTransition).attempt.DisarmedAt)
}

func TestLPXAttemptBatchEditsPreserveUnchangedEngines(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		phase                lpxv1alpha1.RequestPhase
		add, remove, untimed bool
		unrecorded, deleting bool
	}{
		{name: "bound-before-disarm", phase: lpxv1alpha1.RequestPhaseBound, add: true},
		{name: "no-fit", phase: lpxv1alpha1.RequestPhaseNoFit, add: true},
		{name: "unsupported", phase: lpxv1alpha1.RequestPhaseUnsupported, add: true},
		{name: "active-addition", phase: lpxv1alpha1.RequestPhasePending, add: true},
		{name: "timeout-option-removed", phase: lpxv1alpha1.RequestPhasePending, add: true, untimed: true},
		{name: "metadata", phase: lpxv1alpha1.RequestPhasePending},
		{name: "intentional-removal", phase: lpxv1alpha1.RequestPhasePending, remove: true},
		{name: "remove-and-add", phase: lpxv1alpha1.RequestPhasePending, remove: true, add: true},
		{name: "remove-and-replace", phase: lpxv1alpha1.RequestPhasePending, remove: true, add: true, deleting: true},
		{name: "unrecorded-pending", phase: lpxv1alpha1.RequestPhasePending, unrecorded: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Keep the existing engine and its deadline across a new source generation")
			source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
			deployment := newLPXTestDeployment(t, source)
			pcs := deadlineTestPCS(deployment, "pcs-uid")
			now := time.Now().UTC().Truncate(time.Second)
			old := deadlineTestRequest(deployment, pcs, "old", now.Add(-time.Hour), tc.phase)
			deadline := now.Add(time.Minute)
			if tc.phase != lpxv1alpha1.RequestPhasePending {
				deadline = now.Add(-time.Minute)
			}
			current := deadlineTestAttempt(deployment, old, deadline)
			if tc.unrecorded {
				current.Requests[0].UID = ""
			}
			deployment.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: current}
			deployment.Generation++
			want := &nvidiacomv1beta1.LPXAttemptStatus{ObservedGeneration: deployment.Generation, PodCliqueSetUID: current.PodCliqueSetUID}
			currents := map[string]*lpxv1alpha1.LPUPipelineRequest{old.Name: old}
			if !tc.remove {
				want.Requests = append(want.Requests, current.Requests[0])
				want.Requests[0].AttemptDigest = ""
			}
			if tc.add {
				added := deadlineTestRequest(deployment, pcs, "new", now, lpxv1alpha1.RequestPhasePending)
				if tc.deleting {
					added.DeletionTimestamp = ptr.To(metav1.Now())
				}
				currents[added.Name] = added
				want.Requests = append(want.Requests, nvidiacomv1beta1.LPXAttemptRequestStatus{Name: added.Name})
			}

			t.Log("New engines get a fresh batch only after the old scheduling batch completes")
			r := newLPXTestReconciler(t, nil, deployment, source)
			timeout := ptr.To[int64](30)
			if tc.untimed {
				timeout = nil
			}
			if transition := r.reconcileLPXAttemptPreparation(deployment, timeout, want, currents); transition != nil {
				current = applyLPXAttemptTransition(t, deployment, transition)
			}
			if tc.phase != lpxv1alpha1.RequestPhasePending {
				require.Len(t, current.Requests, 1)
				require.Equal(t, "new", current.Requests[0].Name)
				require.Equal(t, now.Add(30*time.Second), current.DeadlineAt.Time)
			} else {
				require.Equal(t, deadline, current.DeadlineAt.Time)
				require.Equal(t, len(want.Requests), len(current.Requests))
			}
			if tc.add || tc.unrecorded {
				require.Nil(t, current.DisarmedAt)
				if !tc.deleting {
					require.NotEmpty(t, current.Requests[0].UID)
				}
			}
			require.True(t, old.DeletionTimestamp.IsZero())
			require.Equal(t, types.UID("old-uid"), old.UID)
		})
	}
}

func TestLPXAttemptReplacementAdoptsNewPodCliqueSetIdentity(t *testing.T) {
	for _, timed := range []bool{false, true} {
		t.Run(fmt.Sprintf("timed=%t", timed), func(t *testing.T) {
			t.Log("Record an active attempt after every old request has finished garbage collection")
			source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
			deployment := newLPXTestDeployment(t, source)
			deployment.Generation = 2
			current := &nvidiacomv1beta1.LPXAttemptStatus{
				ObservedGeneration: 1,
				PodCliqueSetUID:    "previous-pcs-uid",
				Requests: []nvidiacomv1beta1.LPXAttemptRequestStatus{{
					Name: "engine",
				}},
			}
			var deadlineSeconds *int64
			if timed {
				current.DeadlineAt = ptr.To(metav1.NewTime(time.Now().Add(time.Minute)))
				deadlineSeconds = ptr.To[int64](30)
			}
			deployment.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: current}
			replacement := deadlineTestPCS(deployment, "replacement-pcs-uid")
			reconciler := newLPXTestReconciler(t, nil, deployment, source, replacement)
			want := &nvidiacomv1beta1.LPXAttemptStatus{
				ObservedGeneration: deployment.Generation,
				PodCliqueSetUID:    replacement.UID,
				Requests: []nvidiacomv1beta1.LPXAttemptRequestStatus{{
					Name: "engine", AttemptDigest: "sha256:replacement",
				}},
			}

			t.Log("Move the preserved attempt to the replacement PCS before publishing its request")
			transition := reconciler.reconcileLPXAttemptPreparation(
				deployment,
				deadlineSeconds,
				want,
				map[string]*lpxv1alpha1.LPUPipelineRequest{},
			)
			require.NotNil(t, transition)
			next := applyLPXAttemptTransition(t, deployment, transition)
			require.Equal(t, deployment.Generation, next.ObservedGeneration)
			require.Equal(t, replacement.UID, next.PodCliqueSetUID)
			require.Equal(t, current.DeadlineAt, next.DeadlineAt)
			require.Equal(t, want.Requests, next.Requests)
			require.NoError(t, reconciler.revalidateLPXAttemptPublication(t.Context(), deployment, replacement.Name))
		})
	}
}

func TestLPXAttemptScaleInCancellationPrecedesDeadline(t *testing.T) {
	for _, tc := range []struct {
		name  string
		phase lpxv1alpha1.RequestPhase
	}{
		{name: "pending", phase: lpxv1alpha1.RequestPhasePending},
		{name: "completed", phase: lpxv1alpha1.RequestPhaseBound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Publish two engines, with only the second engine recorded in the scheduling batch")
			ctx := t.Context()
			deployment, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			source.Spec.Components[0].Replicas = ptr.To[int32](2)
			r, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, source)
			objects := lpxMaterializedObjects(t, r, deployment, source, desired)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			publishSelectedLPXForTest(t, ctx, r, deployment, desired)
			running := getLPXRequest(t, ctx, r.Client, deployment.Namespace, desired.requests[0].requestName)
			running.Status = newLPXNodeLocalBoundStatus(running.Generation, 1, "sha256:bound")
			pending := getLPXRequest(t, ctx, r.Client, deployment.Namespace, desired.requests[1].requestName)
			pending.Finalizers = []string{"scheduler.example/cleanup"}
			pending.Status = &lpxv1alpha1.LPUPipelineRequestStatus{
				Phase: lpxv1alpha1.RequestPhasePending, ObservedGeneration: ptr.To(pending.Generation),
			}
			if tc.phase == lpxv1alpha1.RequestPhaseBound {
				pending.Status = newLPXNodeLocalBoundStatus(pending.Generation, 1, "sha256:bound")
			}
			deadline := time.Now().UTC().Truncate(time.Second).Add(-time.Minute)
			deployment.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: deadlineTestAttempt(deployment, pending, deadline)}
			source.Spec.Scheduling = deadlineTestScheduling()
			source.Spec.Components[0].Replicas = ptr.To[int32](1)
			objects = append(objects, running, pending)
			r = newLPXTestReconciler(t, registry, deployment, source, objects...)
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
					return delegated.Delete(ctx, object.DeepCopyObject().(client.Object), opts...)
				},
			})
			key := client.ObjectKeyFromObject(deployment)

			t.Log("Persist cancellation before evaluating the expired deadline")
			_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, key, deployment))
			attempt := currentLPXAttemptStatus(deployment)
			require.Equal(t, []nvidiacomv1beta1.LPXAttemptRequestStatus{}, attempt.Requests)
			require.Nil(t, attempt.ExceededAt)
			require.NotNil(t, attempt.DisarmedAt)
			require.True(t, deadline.Equal(attempt.DeadlineAt.Time))

			t.Log("A source read failure on the next reconcile cannot expire the canceled batch")
			readError := errors.New("temporary source observation failure")
			original := r.Client
			r.Client = interceptor.NewClient(original.(client.WithWatch), interceptor.Funcs{
				Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					if _, source := object.(*nvidiacomv1beta1.DynamoGraphDeployment); source {
						return readError
					}
					return delegated.Get(ctx, key, object, opts...)
				},
			})
			_, readErr := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			r.Client = original
			require.NoError(t, r.Get(ctx, key, deployment))
			require.Nil(t, currentLPXAttemptStatus(deployment).ExceededAt)
			require.ErrorIs(t, readErr, readError)

			t.Log("Cancellation remains disarmed while only the removed request retires")
			for range 4 {
				_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
				require.NoError(t, err)
				require.NoError(t, r.Get(ctx, key, deployment))
				attempt = currentLPXAttemptStatus(deployment)
				require.True(t, deadline.Equal(attempt.DeadlineAt.Time))
			}
			require.Nil(t, attempt.ExceededAt)
			require.NotNil(t, attempt.DisarmedAt)
			pcs := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: desired.plan.PodCliqueSetName}, pcs))
			require.Equal(t, types.UID("pcs-uid"), pcs.UID)
			require.Equal(t, running, getLPXRequest(t, ctx, r.Client, deployment.Namespace, running.Name))
			stored := getLPXRequest(t, ctx, r.Client, deployment.Namespace, pending.Name)
			require.False(t, stored.DeletionTimestamp.IsZero())
		})
	}
}
func TestLPXAttemptRetryRequiresLaterEditAndCompletedCleanup(t *testing.T) {
	for _, tc := range []struct {
		name               string
		laterEdit, oldGone bool
	}{{name: "same-generation", oldGone: true}, {name: "cleanup-pending", laterEdit: true}, {name: "later-edit", laterEdit: true, oldGone: true}} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Keep the failed batch fenced until both a later edit and exact-request cleanup")
			source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
			deployment := newLPXTestDeployment(t, source)
			pcs := deadlineTestPCS(deployment, "pcs-uid")
			old := deadlineTestRequest(deployment, pcs, "old", time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhasePending)
			old.Finalizers = []string{"scheduler.example/cleanup"}
			attempt := deadlineTestAttempt(deployment, old, time.Now().Add(-time.Second))
			attempt.ExceededAt = ptr.To(metav1.Now())
			deployment.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: attempt}
			if tc.laterEdit {
				deployment.Generation++
			}
			var objects []client.Object
			if !tc.oldGone {
				objects = append(objects, old)
			}
			r := newLPXTestReconciler(t, nil, deployment, source, objects...)
			var requests []lpxv1alpha1.LPUPipelineRequest
			if !tc.oldGone {
				requests = []lpxv1alpha1.LPUPipelineRequest{*old}
			}
			classification, err := r.continueTerminalLPXAttempt(t.Context(), deployment, attempt, requests, lpxSchedulingDeadlineExceededReason, "expired")
			require.NoError(t, err)
			transition := classification.(*lpxDeadlineTransition)
			if tc.laterEdit && tc.oldGone {
				require.Nil(t, transition.attempt)
			} else {
				require.Equal(t, lpxSchedulingDeadlineExceededReason, transition.result.Reason)
				require.Equal(t, attempt, transition.attempt)
			}
		})
	}
}

func TestLPXAttemptRecoversUIDWithoutTimeoutOrSource(t *testing.T) {
	for _, sourceAvailable := range []bool{false, true} {
		t.Log("Recover an already-published UID even when its timeout option or source is unavailable", sourceAvailable)
		source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
		deployment := newLPXTestDeployment(t, source)
		pcs := deadlineTestPCS(deployment, "pcs-uid")
		request := deadlineTestRequest(deployment, pcs, "unrecorded", time.Now(), lpxv1alpha1.RequestPhasePending)
		attempt := deadlineTestAttempt(deployment, request, time.Now())
		attempt.DeadlineAt, attempt.Requests[0].UID = nil, ""
		deployment.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: attempt}
		r := newLPXTestReconciler(t, nil, deployment, source, request, pcs)
		if !sourceAvailable {
			source = nil
		}
		classification, wake, _, err := r.reconcileLPXAttemptDeadline(t.Context(), deployment, source)
		require.NoError(t, err)
		recovered := applyLPXAttemptTransition(t, deployment, classification)
		require.Equal(t, request.UID, recovered.Requests[0].UID)
		require.Nil(t, recovered.DeadlineAt)
		require.True(t, wake.IsZero())
	}
}

func applyLPXAttemptTransition(t *testing.T, dgd *nvidiacomv1alpha1.LPXGraphDeployment, classification lpxClassification) *nvidiacomv1beta1.LPXAttemptStatus {
	t.Helper()
	transition := classification.(*lpxDeadlineTransition)
	require.NotNil(t, transition.attempt)
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: transition.attempt.DeepCopy()}
	return dgd.Status.Placement.LPXAttempt
}

func deadlineTestAttempt(dgd *nvidiacomv1alpha1.LPXGraphDeployment, request *lpxv1alpha1.LPUPipelineRequest, deadline time.Time) *nvidiacomv1beta1.LPXAttemptStatus {
	return &nvidiacomv1beta1.LPXAttemptStatus{
		ObservedGeneration: dgd.Generation,
		PodCliqueSetUID:    types.UID(request.Annotations[lpxPCSUIDAnnotation]),
		DeadlineAt:         ptr.To(metav1.NewTime(deadline)),
		Requests: []nvidiacomv1beta1.LPXAttemptRequestStatus{{
			Name: request.Name, AttemptDigest: request.Annotations[lpxAttemptDigestAnnotation], UID: request.UID,
		}},
	}
}

func deadlineTestRequest(dgd *nvidiacomv1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, name string, created time.Time, phase lpxv1alpha1.RequestPhase) *lpxv1alpha1.LPUPipelineRequest {
	generation := int64(1)
	request := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: dgd.Namespace, UID: types.UID(name + "-uid"), ResourceVersion: "1",
			Generation: generation, CreationTimestamp: metav1.NewTime(created),
			Labels: map[string]string{lpxOwnerUIDLabel: string(dgd.UID)},
			Annotations: map[string]string{
				lpxAttemptDigestAnnotation: "sha256:attempt", lpxDeploymentUIDAnnotation: string(dgd.UID),
				lpxPCSUIDAnnotation: string(pcs.UID),
			},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
		},
		Spec:   lpxv1alpha1.LPUPipelineRequestSpec{ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal},
		Status: &lpxv1alpha1.LPUPipelineRequestStatus{Phase: phase, ObservedGeneration: &generation},
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

func ownedLPXRequests(t *testing.T, ctx context.Context, reconciler *graphReconciler, dgd *nvidiacomv1alpha1.LPXGraphDeployment) []lpxv1alpha1.LPUPipelineRequest {
	t.Helper()
	requests, err := reconciler.listOwnedLPXRequests(ctx, dgd)
	require.NoError(t, err)
	return requests
}
