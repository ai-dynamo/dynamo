/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"testing"
	"time"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
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

	classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	require.Equal(t, lpxAttemptPreparedReason, lpxResult(classification).Reason)
	prepared := applyLPXAttemptTransition(t, dgd, classification)
	require.NotEmpty(t, prepared.PodCliqueSetUID)
	require.Empty(t, ownedLPXRequests(t, ctx, reconciler, dgd))

	_, err = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.NoError(t, err)
	requests := ownedLPXRequests(t, ctx, reconciler, dgd)
	require.Len(t, requests, 1)
	require.Contains(t, requests[0].Finalizers, lpxAttemptRecordingFinalizer)
	dgd.Status.Placement = nil
	_, classification, err = reconciler.reconcileSelectedLPXSafetyPreflight(ctx, dgd, source)
	require.NoError(t, err)
	require.NotNil(t, classification, "an existing complete LPR set must recover through preflight")
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: prepared}

	require.NoError(t, reconciler.Delete(ctx, &requests[0]))
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(ctx, dgd, source)
	require.NoError(t, err)
	armed := applyLPXAttemptTransition(t, dgd, classification)
	require.Equal(t, requests[0].UID, armed.Requests[0].UID)
	require.True(t, armed.DeadlineAt.Time.Equal(requests[0].CreationTimestamp.Add(30*time.Second)))
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(ctx, dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxAttemptAuthorityLostReason, lpxResult(classification).Reason)
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
		classification, err := reconciler.reconcileSelectedLPX(ctx, dgd, desired)
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
	_, _ = reconciler.reconcileSelectedLPX(ctx, dgd, desired)
	require.LessOrEqual(t, len(ownedLPXRequests(t, ctx, reconciler, dgd)), 1)
}

func TestLPXAttemptDeadlineRaceUsesOnlyDurableDisposition(t *testing.T) {
	t.Log("Seed an exact Bound request whose deadline has not yet been recovered")
	source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
	dgd := newLPXTestDeployment(t, source)
	source.Spec.Scheduling = deadlineTestScheduling()
	request := deadlineTestRequest(dgd, "attempt", time.Now().UTC().Truncate(time.Second), lpxv1alpha1.RequestPhaseBound)
	request.Finalizers = []string{lpxAttemptRecordingFinalizer, "scheduling.lpu.nvidia.com/lpx-cleanup"}
	pcs := deadlineTestPCS(dgd, source, "pcs-uid")
	attempt := deadlineTestAttempt(dgd, request, time.Now().Add(time.Minute))
	attempt.DeadlineAt, attempt.PodCliqueSetUID = nil, ""
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: attempt}
	reconciler := newLPXTestReconciler(t, nil, dgd, source, request, pcs)

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
	reconciler = newLPXTestReconciler(t, nil, dgd, source, request, pcs)
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	result := lpxResult(classification)
	require.Equal(t, lpxSchedulingDeadlineExceededReason, result.Reason)
	transition := classification.(*lpxDeadlineTransition)
	require.NotNil(t, transition.attempt.ExceededAt)
	require.Nil(t, transition.attempt.DisarmedAt)

	t.Log("Persist expiry before cleaning up the exact request and PCS")
	applyLPXAttemptTransition(t, dgd, classification)
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, request.Name).DeletionTimestamp.IsZero())

	t.Log("Start exact request deletion before holding the shared PCS at zero")
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	deleting := getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, request.Name)
	require.False(t, deleting.DeletionTimestamp.IsZero())
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	pcsKey := client.ObjectKeyFromObject(pcs)
	require.NoError(t, reconciler.Get(t.Context(), pcsKey, storedPCS))
	require.Zero(t, storedPCS.Spec.Replicas)

	t.Log("Release the recording finalizer only after scheduler cleanup finishes")
	deleting.Finalizers = []string{lpxAttemptRecordingFinalizer}
	require.NoError(t, reconciler.Update(t.Context(), deleting))
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	_, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.True(t, apierrors.IsNotFound(reconciler.Get(t.Context(), pcsKey, storedPCS)))
	require.Zero(t, classification.(*lpxDeadlineTransition).requeueAfter)

	t.Log("A completed old attempt cannot touch same-name replacement identities")
	replacement := request.DeepCopy()
	replacement.UID = "replacement-uid"
	replacement.ResourceVersion = "2"
	replacement.Annotations[lpxPCSUIDAnnotation] = "replacement-pcs-uid"
	pcs = deadlineTestPCS(dgd, source, "replacement-pcs-uid")
	reconciler = newLPXTestReconciler(t, nil, dgd, source, replacement, pcs)

	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxSchedulingDeadlineExceededReason, lpxResult(classification).Reason)
	require.True(t, getLPXRequest(t, t.Context(), reconciler.Client, dgd.Namespace, replacement.Name).DeletionTimestamp.IsZero())
	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(pcs), storedPCS))
	require.Equal(t, int32(1), storedPCS.Spec.Replicas)

	t.Log("A late Bound observation without a durable disposition still expires")
	late := deadlineTestAttempt(dgd, request, time.Now().Add(-time.Minute))
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{LPXAttempt: late}
	request.Status.Phase = lpxv1alpha1.RequestPhaseBound
	reconciler = newLPXTestReconciler(t, nil, dgd, source, request)
	classification, _, _, err = reconciler.reconcileLPXAttemptDeadline(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, lpxSchedulingDeadlineExceededReason, lpxResult(classification).Reason)
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

func deadlineTestRequest(dgd *nvidiacomv1alpha1.LPXGraphDeployment, name string, created time.Time, phase lpxv1alpha1.RequestPhase) *lpxv1alpha1.LPUPipelineRequest {
	generation := int64(1)
	request := &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: dgd.Namespace, UID: types.UID(name + "-uid"), ResourceVersion: "1",
			Generation: generation, CreationTimestamp: metav1.NewTime(created),
			Annotations: map[string]string{
				lpxAttemptDigestAnnotation: "sha256:attempt", lpxDeploymentUIDAnnotation: string(dgd.UID),
				lpxDeploymentGenerationAnnotation: "1", lpxPCSUIDAnnotation: "pcs-uid",
			},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, nvidiacomv1alpha1.LPXGraphDeploymentGVK)},
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

func deadlineTestPCS(dgd *nvidiacomv1alpha1.LPXGraphDeployment, source *nvidiacomv1beta1.DynamoGraphDeployment, uid types.UID) *grovev1alpha1.PodCliqueSet {
	return &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: dynamo.PCSNameForLPX(source), Namespace: dgd.Namespace, UID: uid, ResourceVersion: "1",
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
