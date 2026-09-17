// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestLPXScaleOutRetirementProtectsUnrelatedAuthority(t *testing.T) {
	const missingGroup = "missing group"
	for _, scenario := range []string{"foreign group", "updated child", "scale conflict", missingGroup, "replacement request", "shared replica"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Publish two engines and record only the second engine in the failed batch")
			ctx := t.Context()
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			lpx.ServingComponent(source).Replicas = ptr.To(int32(2))
			r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
			objects := lpxMaterializedObjects(t, r, child, source, desired)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			publishSelectedLPXForTest(t, ctx, r, child, desired)
			pcs := findLPXTestPodCliqueSet(t, objects)
			group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
			failed := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[1].requestName)
			attempt := deadlineTestAttempt(child, failed, time.Now().Add(-time.Second))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))

			t.Log("Change the observed authority before terminal cleanup attempts a destructive write")
			wantError := true
			switch scenario {
			case "foreign group":
				group.OwnerReferences[0].UID = "foreign-pcs"
				require.NoError(t, r.Update(ctx, group))
			case "updated child":
				newer := child.DeepCopy()
				newer.Generation++
				require.NoError(t, r.Update(ctx, newer))
			case "scale conflict":
				r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
					SubResourceUpdate: func(context.Context, client.Client, string, client.Object, ...client.SubResourceUpdateOption) error {
						return apierrors.NewConflict(consts.PodCliqueScalingGroupGVR.GroupResource(), group.Name, errors.New("stale scale"))
					},
				})
			case missingGroup:
				require.NoError(t, r.Delete(ctx, group))
				wantError = false
			case "replacement request":
				require.NoError(t, r.Delete(ctx, failed))
				failed.UID, failed.ResourceVersion = "replacement-uid", ""
				require.NoError(t, r.Create(ctx, failed))
				wantError = false
			case "shared replica":
				failed.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex = 0
				require.NoError(t, r.Update(ctx, failed))
			}

			t.Log("Preserve the PCS and request identities when scale-down is unsafe or not yet possible")
			requests := ownedLPXRequests(t, ctx, r, child)
			pending, err := r.retireLPXAttempt(ctx, child, attempt, requests)
			if wantError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Equal(t, scenario == missingGroup, pending)
			}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), &grovev1alpha1.PodCliqueSet{}))
			for _, before := range requests {
				require.Equal(t, &before, getLPXRequest(t, ctx, r.Client, before.Namespace, before.Name))
			}
			if scenario != missingGroup {
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
				require.Equal(t, int32(2), group.Spec.Replicas)
			}
		})
	}
}

func TestLPXFailedSpecDecodeScaleOutRetiresAllModels(t *testing.T) {
	t.Log("Publish one target and two draft requests for each of two engines")
	ctx := t.Context()
	child, source, registry := newLPXSpecDecodeTestDGD(t)
	lpx.ServingComponent(source).Replicas = nil
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	group := findLPXTestScalingGroup(t, objects, desired.plan.LPXScalingGroup)
	group.Spec.Replicas = 2
	createLPXTestObjects(t, ctx, r.Client, objects...)
	desired, rejected := requirePreparedLPX(t, r, ctx, child, source)
	require.Nil(t, rejected)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	requests := ownedLPXRequests(t, ctx, r, child)
	require.Len(t, requests, 6)
	attempt := currentLPXAttemptStatus(child).DeepCopy()
	attempt.Requests = nil
	for _, request := range requests {
		if request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex == 1 {
			attempt.Requests = append(attempt.Requests, deadlineTestAttempt(child, &request, time.Now()).Requests...)
		}
	}
	attempt.ExceededAt = ptr.To(metav1.Now())
	require.Len(t, attempt.Requests, 3)

	t.Log("Retire all requests in the failed engine while preserving the serving engine's models")
	for range 2 {
		_, err := r.retireLPXAttempt(ctx, child, attempt, requests)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	for _, request := range requests {
		if request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex == 0 {
			require.Equal(t, &request, getLPXRequest(t, ctx, r.Client, request.Namespace, request.Name))
		} else {
			require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(&request), &lpxv1alpha1.LPUPipelineRequest{})))
		}
	}
}

func TestLPXScaleOutRetirementWaitsForOwnerGarbageCollection(t *testing.T) {
	t.Log("Keep old requests observable after their shared PCS is already absent")
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = ptr.To(int32(2))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	failed := getLPXRequest(t, ctx, r.Client, child.Namespace, desired.requests[1].requestName)
	attempt := deadlineTestAttempt(child, failed, time.Now().Add(-time.Second))
	pcs := findLPXTestPodCliqueSet(t, objects)
	requests := ownedLPXRequests(t, ctx, r, child)
	require.Len(t, requests, 2)
	require.NoError(t, r.Delete(ctx, pcs))

	t.Log("Keep terminal cleanup pending until owner GC removes every old request")
	pending, err := r.retireLPXAttempt(ctx, child, attempt, requests)
	require.NoError(t, err)
	require.True(t, pending)
	for index := range requests {
		require.NoError(t, r.Delete(ctx, &requests[index]))
	}
	pending, err = r.retireLPXAttempt(ctx, child, attempt, nil)
	require.NoError(t, err)
	require.False(t, pending)
}
