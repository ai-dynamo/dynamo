// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestLPXExpiredScaleOutPreservesServingEngine(t *testing.T) {
	t.Log("Publish one bound engine and one expired scale-out request")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = ptr.To(int32(2))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	serving := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].requestName)
	serving.Status = deadlineTestRequest(child, pcs, serving.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
	require.NoError(t, r.Update(ctx, serving))
	expired := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[1].requestName)
	requests, err := r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)

	t.Log("Reject ambiguous replica ownership without changing the live engine")
	invalid := source.DeepCopy()
	invalid.Spec.Components = nil
	require.ErrorContains(t, r.retireExpiredLPXRequests(ctx, child, invalid, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{expired}), "without an LPX serving component")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(2), group.Spec.Replicas)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{}))

	t.Log("Lower Grove and delete only the expired scheduler intent in one reconciliation")
	err = r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{expired})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{})))
	require.True(t, apiequality.Semantic.DeepEqual(serving, getLPXRequest(t, ctx, r.Client, serving.Namespace, serving.Name)), "serving request changed")
}

func TestLPXExpiredModelRetiresItsEngineSiblings(t *testing.T) {
	t.Log("Publish the multiple model requests belonging to one speculative engine")
	ctx := t.Context()
	child, source, registry := newLPXSpecDecodeTestDGD(t)
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	require.Greater(t, len(desired.requests), 1)
	requests, err := r.listOwnedLPXRequests(ctx, child, findLPXTestPodCliqueSet(t, objects))
	require.NoError(t, err)
	expired := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[len(desired.requests)-1].requestName)

	t.Log("Map one expired model request to every request sharing its engine suffix")
	replicas, retiring, err := lpxExpiredEngineSuffix(requests, []*lpxv1alpha1.LPUPipelineRequest{expired}, 1)
	require.NoError(t, err)
	require.Zero(t, replicas)
	require.Len(t, retiring, len(requests))
}

func TestLPXExpiredModelPreservesExpiryProofUntilSiblingCleanupStarts(t *testing.T) {
	t.Log("Publish multiple model requests for one engine and expire one of them")
	ctx := t.Context()
	child, source, registry := newLPXSpecDecodeTestDGD(t)
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	requests, err := r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)
	require.Greater(t, len(requests), 1)
	expired := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].requestName)
	var sibling *lpxv1alpha1.LPUPipelineRequest
	for index := range requests {
		if requests[index].Name != expired.Name {
			sibling = &requests[index]
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
	err = r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{expired})
	require.ErrorIs(t, err, deleteErr)
	require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{}))

	t.Log("Retry from the retained expiry proof and remove the complete engine request set")
	r.Client = base
	requests, err = r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)
	expired = getLPXRequest(t, ctx, r.Client, child.Namespace, expired.Name)
	require.NoError(t, r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{expired}))
	for index := range desired.requests {
		requireLPXRequestNotFound(t, ctx, r.Client, child.Namespace, desired.requests[index].requestName)
	}
}

func TestLPXExpiredInteriorReplicaDoesNotScaleDown(t *testing.T) {
	t.Log("Publish three engine replicas and expire only the middle ordinal")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = ptr.To(int32(3))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	expired := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[1].requestName)
	child.Status.ExpiredRequestUIDs = []types.UID{expired.UID}
	require.NoError(t, r.Status().Update(ctx, child))

	t.Log("Wait for watched changes without polling or changing the live engines")
	result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.Zero(t, result)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	require.Equal(t, []types.UID{expired.UID}, child.Status.ExpiredRequestUIDs)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(3), group.Spec.Replicas)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{}))
	for _, object := range objects {
		if clique, ok := object.(*grovev1alpha1.PodClique); ok {
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), &grovev1alpha1.PodClique{}))
		}
	}
}

func TestLPXExpiredInteriorReplicaBlocksTrailingSuffix(t *testing.T) {
	t.Log("Publish three engines with both an interior failure and a failed tail")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = ptr.To(int32(3))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	pcs := findLPXTestPodCliqueSet(t, objects)
	interior := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].requestName)
	tail := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[2].requestName)
	requests, err := r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)

	t.Log("Do not partially mutate the group while an expired hole remains")
	require.NoError(t, r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{interior, tail}))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(3), group.Spec.Replicas)
	for _, request := range []*lpxv1alpha1.LPUPipelineRequest{interior, tail} {
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(request), &lpxv1alpha1.LPUPipelineRequest{}))
	}
}

func TestLPXExpiredMaximalTrailingRunScalesDown(t *testing.T) {
	t.Log("Publish four engines and expire the complete trailing pair")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = ptr.To(int32(4))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	pcs := findLPXTestPodCliqueSet(t, objects)
	requests, err := r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)
	expired := []*lpxv1alpha1.LPUPipelineRequest{
		getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[2].requestName),
		getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[3].requestName),
	}

	require.NoError(t, r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, expired))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(2), group.Spec.Replicas)
	for index, projection := range desired.requests {
		err := r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: projection.requestName}, &lpxv1alpha1.LPUPipelineRequest{})
		if index < 2 {
			require.NoError(t, err)
		} else {
			require.True(t, apierrors.IsNotFound(err))
		}
	}
}

func TestLPXDeadlineRetirementPreservesImplicitReplicaCapacity(t *testing.T) {
	t.Log("Publish one externally managed engine and expire its scheduler intent")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = nil
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	expired := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].requestName)
	requests, err := r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)

	t.Log("Delete the expired LPR without making observed zero capacity the new desired state")
	require.NoError(t, r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{expired}))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{})))

	t.Log("Continue deriving the implicit desired count from the preserved Grove capacity")
	selected, classification, err := r.prepareLPXMaterializing(ctx, child, source, pcs)
	require.NoError(t, err)
	require.Nil(t, classification)
	require.Equal(t, int32(1), selected.plan.Replicas)
}

func TestLPXExpiredRequestWaitsForMissingScalingGroup(t *testing.T) {
	t.Log("Keep an expired request while its live PCS has not materialized a scaling group")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	expired := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].requestName)
	require.NoError(t, r.Delete(ctx, group))
	requests, err := r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)
	err = r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{expired})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{}))

	t.Log("Once Grove creates the group, issue scale-down before deleting the LPR")
	group.ResourceVersion = ""
	group.UID = "replacement-group"
	group.DeletionTimestamp = nil
	group.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}
	require.NoError(t, r.Create(ctx, group))
	err = r.retireExpiredLPXRequests(ctx, child, source, pcs, requests, []*lpxv1alpha1.LPUPipelineRequest{expired})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Zero(t, group.Spec.Replicas)
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{})))
}

func TestLPXExpiredRequestsWaitForOwnerGarbageCollection(t *testing.T) {
	t.Log("Keep an expired request observable after its owner PCS disappears")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	expired := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].requestName)
	requests, err := r.listOwnedLPXRequests(ctx, child, pcs)
	require.NoError(t, err)
	require.NoError(t, r.Delete(ctx, pcs))

	err = r.retireExpiredLPXRequests(ctx, child, source, nil, requests, []*lpxv1alpha1.LPUPipelineRequest{expired})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(expired), &lpxv1alpha1.LPUPipelineRequest{}))
}
