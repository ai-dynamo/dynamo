// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/yaml"
)

func TestLPXWorkloadErrorDoesNotAcknowledgeGeneration(t *testing.T) {
	for _, tc := range []struct {
		name             string
		deadline         bool
		failStatus       bool
		workloadSucceeds bool
	}{
		{name: "ordinary error"},
		{name: "bounded deadline retry", deadline: true},
		{name: "status failure keeps both errors", deadline: true, failStatus: true},
		{name: "status failure after a pending pass", deadline: true, failStatus: true, workloadSucceeds: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Retain a pending request while processing a new child generation")
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			child.Generation, child.Status.ObservedGeneration = 2, 1
			if tc.deadline {
				dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](60)}
			}
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			publishSelectedLPXForTest(t, ctx, r, child, selected)
			pcs := findLPXTestPodCliqueSet(t, objects)
			pending := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
			pending.Status = newTestPipelineRequest(child, pcs, pending.Name, time.Now(), lpxv1alpha1.RequestPhasePending).Status
			require.NoError(t, r.Update(ctx, pending))

			t.Log("Fail runtime synchronization after the scheduling deadline is derived")
			readErr, statusErr := errors.New("ConfigMap read unavailable"), errors.New("status write unavailable")
			failReads, failStatus := !tc.workloadSucceeds, tc.failStatus
			reads := 0
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					if _, ok := object.(*corev1.ConfigMap); ok && failReads {
						reads++
						return readErr
					}
					return delegated.Get(ctx, key, object, opts...)
				},
				SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
					if subresource == "status" && failStatus {
						return statusErr
					}
					return delegated.SubResource(subresource).Update(ctx, object, opts...)
				},
			})
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			result, err := r.Reconcile(ctx, key)
			if !tc.workloadSucceeds {
				require.Positive(t, reads)
			}
			if tc.deadline && !tc.failStatus {
				require.NoError(t, err)
				require.Equal(t, pipelineRequestDeadlineRetryInterval, result.RequeueAfter)
			} else {
				if !tc.workloadSucceeds {
					require.ErrorIs(t, err, readErr)
				}
				require.Zero(t, result)
				if tc.failStatus {
					require.ErrorIs(t, err, statusErr)
				}
			}
			require.NoError(t, r.Get(ctx, key.NamespacedName, child))
			require.EqualValues(t, 1, child.Status.ObservedGeneration)
			if !tc.failStatus {
				ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
				require.NotNil(t, ready)
				require.Equal(t, v1alpha1.LPXReadyReasonFailed, ready.Reason)
				require.EqualValues(t, 2, ready.ObservedGeneration)
				require.Contains(t, ready.Message, readErr.Error())
			}

			t.Log("A successful pending pass acknowledges the generation and preserves its deadline wake")
			failReads, failStatus = false, false
			result, err = r.Reconcile(ctx, key)
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, key.NamespacedName, child))
			require.EqualValues(t, 2, child.Status.ObservedGeneration)
			if tc.deadline {
				require.Positive(t, result.RequeueAfter)
				require.LessOrEqual(t, result.RequeueAfter, time.Minute)
			}
		})
	}
}

func TestPipelineRequestDeadlineContinuesDuringRequestDeletion(t *testing.T) {
	t.Log("Publish two pending replicas and expire the tail first")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := findLPXTestPodCliqueSet(t, objects)
	prefix := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
	prefix.Status = newTestPipelineRequest(child, pcs, prefix.Name, time.Now(), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, prefix))
	tail := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].Name)
	tail.Status = newTestPipelineRequest(child, pcs, tail.Name, time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhasePending).Status
	tail.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
	require.NoError(t, r.Update(ctx, tail))

	t.Log("Retire the expired tail while its scheduler finalizer holds deletion")
	reconcileRequest := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 2 {
		_, err := r.Reconcile(ctx, reconcileRequest)
		require.NoError(t, err)
	}
	tail = getTestPipelineRequest(t, ctx, r.Client, child.Namespace, tail.Name)
	require.False(t, tail.DeletionTimestamp.IsZero())

	t.Log("Continue enforcing the surviving request's independent deadline")
	prefix = getTestPipelineRequest(t, ctx, r.Client, child.Namespace, prefix.Name)
	prefix.Status = newTestPipelineRequest(child, pcs, prefix.Name, time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, prefix))
	_, err := r.Reconcile(ctx, reconcileRequest)
	require.NoError(t, err)
	requirePipelineRequestNotFound(t, ctx, r.Client, child.Namespace, prefix.Name)
}

func TestPipelineRequestDeadlineFailureMustPersistBeforeCleanup(t *testing.T) {
	t.Log("Publish an expired request without a persisted deadline failure")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pending := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
	pending.Status = newTestPipelineRequest(child, pcs, pending.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, pending))

	t.Log("Reject the status write and forbid cleanup until the failure becomes durable")
	statusErr := errors.New("status write unavailable")
	failStatus := true
	groupMissing := true
	statusWrites, scaleWrites, deletes := 0, 0, 0
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			if _, ok := object.(*grovev1alpha1.PodCliqueScalingGroup); ok && groupMissing {
				return apierrors.NewNotFound(consts.PodCliqueScalingGroupGVR.GroupResource(), key.Name)
			}
			return delegated.Get(ctx, key, object, opts...)
		},
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if subresource == "status" {
				statusWrites++
				if failStatus {
					return statusErr
				}
			} else if subresource == "scale" {
				scaleWrites++
			}
			return delegated.SubResource(subresource).Update(ctx, object, opts...)
		},
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			deletes++
			return delegated.Delete(ctx, object, opts...)
		},
	})
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	result, err := r.Reconcile(ctx, request)
	require.ErrorIs(t, err, statusErr)
	require.Zero(t, result, "status errors use controller-runtime backoff, not the expiry requeue")
	require.Equal(t, 1, statusWrites)
	require.Zero(t, scaleWrites)
	require.Zero(t, deletes)
	require.NoError(t, r.Get(ctx, request.NamespacedName, child))
	require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, schedulingFailedCondition))

	t.Log("Record failure even without a PCSG observation, but leave the LPR intact")
	failStatus = false
	result, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Equal(t, time.Nanosecond, result.RequeueAfter, "status-only updates need an explicit follow-up")
	require.Equal(t, 2, statusWrites)
	require.Zero(t, scaleWrites)
	require.Zero(t, deletes)
	require.NoError(t, r.Get(ctx, request.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, schedulingFailedCondition))
	failed := meta.FindStatusCondition(child.Status.Conditions, "Ready").DeepCopy()

	t.Log("Persisted failure cannot authorize cleanup until the live group is observed")
	result, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Zero(t, result, "the PCSG watch resumes cleanup")
	require.Zero(t, scaleWrites)
	require.Zero(t, deletes)
	require.True(t, apiequality.Semantic.DeepEqual(pending, getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)))

	t.Log("Observing both the persisted failure and the group allows expired-replica cleanup")
	groupMissing = false
	result, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Zero(t, result, "request and group events drive cleanup")
	require.Equal(t, 1, scaleWrites)
	require.Equal(t, 1, deletes)
	requirePipelineRequestNotFound(t, ctx, r.Client, pending.Namespace, pending.Name)
	require.NoError(t, r.Get(ctx, request.NamespacedName, child))
	require.Equal(t, failed, meta.FindStatusCondition(child.Status.Conditions, "Ready"))

	t.Log("A sticky failure neither polls nor restores the expired capacity")
	result, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Zero(t, result)
	require.Equal(t, 1, scaleWrites)
	require.Equal(t, 1, deletes)
	requirePipelineRequestNotFound(t, ctx, r.Client, pending.Namespace, pending.Name)
}

func TestPipelineRequestDeadlineHoleWaitsForSchedulingChange(t *testing.T) {
	t.Log("Expire an interior engine while a higher ordinal is still Bound")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, desired.plan.LPXScalingGroup)
	requests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	for index := range requests {
		request := &requests[index]
		phase := lpxv1alpha1.RequestPhasePending
		if request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex == 1 {
			phase = lpxv1alpha1.RequestPhaseBound
		}
		request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now().Add(-time.Minute), phase).Status
		require.NoError(t, r.Update(ctx, request))
	}

	t.Log("Persist the failure once, then leave the hole unchanged without polling")
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	result, err := r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Equal(t, time.Nanosecond, result.RequeueAfter)
	for range 2 {
		result, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
		require.Zero(t, result)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
		require.Equal(t, int32(2), group.Spec.Replicas)
		current, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.True(t, apiequality.Semantic.DeepEqual(requests, current), "interior failures must not mutate requests")
	}

	t.Log("A scheduler update that expires the higher ordinal unblocks complete-suffix cleanup")
	higher := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, desired.requests[1].Name)
	higher.Status.Phase = lpxv1alpha1.RequestPhasePending
	require.NoError(t, r.Update(ctx, higher))
	result, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Zero(t, result)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Zero(t, group.Spec.Replicas)
	current, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	require.Empty(t, current)
}

func TestLPXDeletesOnlyStaleOwnedRuntimeConfigMaps(t *testing.T) {
	t.Log("Create current, stale, and foreign runtime ConfigMaps")
	dgd := loadTestDGD(t, lpx.PipelineSingle, "test-build")
	child := newLPXTestDeployment(t, dgd)
	owner := []metav1.OwnerReference{*metav1.NewControllerRef(child, v1alpha1.LPXGraphDeploymentGVK)}
	current := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: "runtime-current", Namespace: child.Namespace, UID: "current",
		Labels: map[string]string{deploymentUIDLabel: string(child.UID)}, OwnerReferences: owner,
	}}
	stale := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: "runtime-stale", Namespace: child.Namespace, UID: "stale",
		Labels: map[string]string{deploymentUIDLabel: string(child.UID)}, OwnerReferences: owner,
	}}
	foreign := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: "runtime-foreign", Namespace: child.Namespace, UID: "foreign",
		Labels:          map[string]string{deploymentUIDLabel: string(child.UID)},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)},
	}}
	r := newLPXTestReconciler(t, nil, child, dgd, current, stale, foreign)

	t.Log("Keep the previous runtime configuration while the replacement is not Ready")
	require.NoError(t, r.deleteUnusedConfigMaps(t.Context(), child, []client.Object{current}))
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(stale), &corev1.ConfigMap{}))

	t.Log("Delete only the stale ConfigMap owned by the current LPX child")
	setReadyCondition(child, v1beta1.DGDStateSuccessful, "Replacement is Ready")
	require.NoError(t, r.deleteUnusedConfigMaps(t.Context(), child, []client.Object{current}))

	t.Log("Preserve the current and foreign ConfigMaps")
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(current), &corev1.ConfigMap{}))
	require.True(t, apierrors.IsNotFound(r.Get(t.Context(), client.ObjectKeyFromObject(stale), &corev1.ConfigMap{})))
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(foreign), &corev1.ConfigMap{}))
}

func TestLPXDownloadsBeforePublication(t *testing.T) {
	for _, scenario := range []struct {
		name, componentName string
		sharedDraft         bool
		messages            []string
	}{
		{name: "disabled Grove", messages: []string{"Grove is disabled"}},
		{name: "long serving name", componentName: strings.Repeat("serving-", 7) + "engine"},
		{name: "reordered shared draft with long serving name", componentName: strings.Repeat("serving-", 7) + "engine", sharedDraft: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Reconcile authored LPX intent against cold remote builds")
			dgd := loadTestDGD(t, lpx.PipelineSingle, modelDownloadTestBuildID)
			if scenario.sharedDraft {
				dgd = loadTestSpecDecodeDGD(t)
				dgd.Spec.Components[0].LPX.BuildID = modelDownloadTestBuildID
				dgd.Spec.Components[1].LPX.BuildID = modelDownloadTestSecondBuildID
				dgd.Spec.Components[1].ComponentName = "shared-draft-name-is-not-materialized"
				dgd.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend},
					dgd.Spec.Components[1], dgd.Spec.Components[0],
				}
			}
			component := lpx.ServingComponent(dgd)
			if scenario.componentName != "" {
				component.ComponentName = scenario.componentName
			}
			registry, err := lpx.NewModelRegistry("", nil)
			require.NoError(t, err)
			observedRegistry := &fakeModelDownloadRegistry{ModelRegistry: registry}
			child := newLPXTestDeployment(t, dgd)
			r := newLPXTestReconciler(t, observedRegistry, child, dgd)
			if scenario.name == "disabled Grove" {
				r.runtimeConfig.Gate.Grove = false
				r.enabled = false
			}

			t.Log("Block downloads when Grove is disabled and wait for cold builds otherwise")
			before := dgd.DeepCopy()
			_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
			require.NoError(t, err)
			stored := &v1alpha1.LPXGraphDeployment{}
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), stored))
			failed := meta.FindStatusCondition(stored.Status.Conditions, "Ready")
			require.NotNil(t, failed)
			if len(scenario.messages) == 0 {
				require.Equal(t, metav1.ConditionFalse, failed.Status)
				require.Equal(t, v1alpha1.LPXReadyReasonPending, failed.Reason)
				wantCalls := []string{modelDownloadTestBuildID}
				if scenario.sharedDraft {
					wantCalls = append(wantCalls, modelDownloadTestSecondBuildID)
				}
				require.Equal(t, wantCalls, observedRegistry.calls)
			} else {
				require.Equal(t, metav1.ConditionFalse, failed.Status)
				require.Equal(t, v1alpha1.LPXReadyReasonFailed, failed.Reason)
				for _, message := range scenario.messages {
					require.Contains(t, failed.Message, message)
				}
				require.Empty(t, observedRegistry.calls)
			}
			require.Zero(t, observedRegistry.acquireBuildSnapshotCalls)
			pcs := &grovev1alpha1.PodCliqueSetList{}
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), pcs))
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, pcs.Items)
			require.Empty(t, requests.Items)
			require.Equal(t, before, dgd)
		})
	}
}

func TestLPXEditBeforeDeadlineFailureDoesNotAuthorizeRetry(t *testing.T) {
	t.Log("Publish a request, then advance the deployment before that request expires")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pending := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pending.Status = newTestPipelineRequest(child, pcs, pending.Name, time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, pending))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	child.Generation++
	require.NoError(t, r.Update(ctx, child))
	failureGeneration := child.Generation

	t.Log("Expire and remove the request at the already-edited generation")
	reconcileRequest := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(ctx, reconcileRequest)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pending), &lpxv1alpha1.LPUPipelineRequest{}))

	t.Log("After the failure fence is durable, scale down and remove the expired request")
	_, err = r.Reconcile(ctx, reconcileRequest)
	require.NoError(t, err)
	requirePipelineRequestNotFound(t, ctx, r.Client, pending.Namespace, pending.Name)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	failed := meta.FindStatusCondition(child.Status.Conditions, "Ready")
	require.NotNil(t, failed)
	require.Equal(t, metav1.ConditionFalse, failed.Status)
	require.Equal(t, v1alpha1.LPXReadyReasonFailed, failed.Reason)
	require.Equal(t, failureGeneration, failed.ObservedGeneration)
	require.Len(t, child.Status.Conditions, 2)
	schedulingFailed := meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition)
	require.NotNil(t, schedulingFailed)
	require.Equal(t, metav1.ConditionTrue, schedulingFailed.Status)
	require.Equal(t, failureGeneration, schedulingFailed.ObservedGeneration)

	t.Log("Reconciliation at the failure generation does not recreate scheduler intent")
	for range 3 {
		_, err = r.Reconcile(ctx, reconcileRequest)
		require.NoError(t, err)
		requirePipelineRequestNotFound(t, ctx, r.Client, pending.Namespace, pending.Name)
	}

	t.Log("An edit after the failure permits a fresh request")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	child.Generation++
	require.NoError(t, r.Update(ctx, child))
	for range 3 {
		_, err = r.Reconcile(ctx, reconcileRequest)
		require.NoError(t, err)
	}
	replacement := getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
	require.NotEqual(t, pending.UID, replacement.UID)
}

func TestLPXEndpointLifecycle(t *testing.T) {
	const uppercaseComponentName = "LPX"
	t.Log("Configure an uppercase LPX component and independently named materialization using Kubernetes discovery")
	dgd := loadTestDGD(t, lpx.PipelineSingle, "test-build")
	dgd.Spec.Components[0].ComponentName = uppercaseComponentName
	dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	dgd.Spec.Components[0].ModelRef = &v1beta1.ModelReference{Name: "test/model"}
	dgd.Spec.Annotations = map[string]string{"example.com/model-discovery": "enabled"}
	dgd.Spec.Labels = map[string]string{"example.com/policy": "enabled"}
	child := newLPXTestDeployment(t, dgd)
	child.Name = "independent-materialization"
	r := newLPXTestReconciler(t, nil, child, dgd)

	t.Log("Keep the ordinary model Service under the source DGD's ownership")
	modelService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: dynamo.GenerateServiceName("test/model"), Namespace: dgd.Namespace,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)},
			Labels:          map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
		},
		Spec: corev1.ServiceSpec{ClusterIP: corev1.ClusterIPNone,
			Selector: map[string]string{consts.KubeLabelDynamoBaseModelHash: dynamo.HashModelName("test/model")}},
	}
	require.NoError(t, r.Create(t.Context(), modelService))
	modelKey := client.ObjectKeyFromObject(modelService)
	require.NoError(t, r.Get(t.Context(), modelKey, modelService))
	beforeModelService := modelService.DeepCopy()

	t.Log("Publish the LPX-owned endpoint with its serving-role selector")
	require.NoError(t, r.reconcileEndpoint(t.Context(), child, dgd))
	service := &corev1.Service{}
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.PCSNameForLPX(child) + "-serve"}
	require.NoError(t, r.Get(t.Context(), key, service))
	require.True(t, metav1.IsControlledBy(service, child))
	require.Equal(t, consts.KubeLabelValueTrue, service.Spec.Selector[dynamo.LPXServingLabel])
	require.Equal(t, dynamo.PCSNameForLPX(child), service.Spec.Selector[grovecommon.LabelPartOfKey])

	t.Log("Converge propagated endpoint metadata, including removed keys")
	for _, value := range []string{"changed", ""} {
		if value == "" {
			dgd.Spec.Labels, dgd.Spec.Annotations = nil, nil
		} else {
			dgd.Spec.Labels["example.com/policy"] = value
			dgd.Spec.Annotations["example.com/model-discovery"] = value
		}
		require.NoError(t, r.reconcileEndpoint(t.Context(), child, dgd))
		updated := &corev1.Service{}
		require.NoError(t, r.Get(t.Context(), key, updated))
		require.Equal(t, value, updated.Labels["example.com/policy"])
		require.Equal(t, value, updated.Annotations["example.com/model-discovery"])
		if value == "" {
			require.NotContains(t, updated.Labels, "example.com/policy")
			require.NotContains(t, updated.Annotations, "example.com/model-discovery")
		}
		require.Equal(t, service.Labels[consts.KubeLabelDynamoDiscoveryEnabled], updated.Labels[consts.KubeLabelDynamoDiscoveryEnabled])
		require.Equal(t, service.Spec, updated.Spec)
		require.Equal(t, service.OwnerReferences, updated.OwnerReferences)
		for _, key := range []string{commoncontroller.NvidiaAnnotationHashKey, commoncontroller.NvidiaAnnotationGenerationKey} {
			require.NotEmpty(t, updated.Annotations[key])
			require.Equal(t, service.Annotations[key], updated.Annotations[key])
		}
		require.NoError(t, r.reconcileEndpoint(t.Context(), child, dgd))
		unchanged := &corev1.Service{}
		require.NoError(t, r.Get(t.Context(), key, unchanged))
		require.Equal(t, updated.ResourceVersion, unchanged.ResourceVersion)
	}

	t.Log("Preserve API-allocated Service fields when the endpoint selector changes")
	require.NoError(t, r.Get(t.Context(), key, service))
	service.Spec.ClusterIP = "10.96.0.10"
	service.Spec.ClusterIPs = []string{"10.96.0.10", "fd00::10"}
	service.Spec.IPFamilies = []corev1.IPFamily{corev1.IPv4Protocol, corev1.IPv6Protocol}
	service.Spec.IPFamilyPolicy = ptr.To(corev1.IPFamilyPolicyPreferDualStack)
	require.NoError(t, r.Update(t.Context(), service))
	allocated := service.DeepCopy()
	for _, global := range []bool{true, false} {
		dgd.Spec.Components[0].GlobalDynamoNamespace = global
		require.NoError(t, r.reconcileEndpoint(t.Context(), child, dgd))
		require.NoError(t, r.Get(t.Context(), key, service))
		allocated.Spec.Selector[consts.KubeLabelDynamoNamespace] = dgd.GetDynamoNamespaceForComponent(&dgd.Spec.Components[0])
		require.Equal(t, allocated.Spec, service.Spec)
		require.Equal(t, allocated.OwnerReferences, service.OwnerReferences)

		t.Log("Reconcile the unchanged endpoint without another write")
		version := service.ResourceVersion
		require.NoError(t, r.reconcileEndpoint(t.Context(), child, dgd))
		require.NoError(t, r.Get(t.Context(), key, service))
		require.Equal(t, version, service.ResourceVersion)
	}

	t.Log("Remove the LPX-owned discovery endpoint when switching back to non-Kubernetes discovery")
	delete(dgd.Annotations, consts.KubeAnnotationDynamoDiscoveryBackend)
	require.NoError(t, r.reconcileEndpoint(t.Context(), child, dgd))
	require.True(t, apierrors.IsNotFound(r.Get(t.Context(), key, service)))

	t.Log("Discovery cleanup leaves the ordinary DGD model service unchanged")
	require.NoError(t, r.Get(t.Context(), modelKey, modelService))
	require.Equal(t, beforeModelService, modelService)
}

func TestLPXEngineOrderAndUnrelatedEditsPreservePublication(t *testing.T) {
	t.Log("Publish a speculative engine using its frozen child identity")
	child, dgd, registry := newLPXSpecDecodeTestDGD(t)
	dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:old"}}}},
	}, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "prefill:old"}}}},
	})
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	createLPXTestObjects(t, t.Context(), r.Client, lpxMaterializedObjects(t, r, child, dgd, selected)...)
	publishSelectedLPXForTest(t, t.Context(), r, child, selected)
	beforeRequests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(t.Context(), beforeRequests))
	require.NotEmpty(t, beforeRequests.Items)
	beforePCS := renderLPXTestPodCliqueSet(t, t.Context(), r, child, dgd, selected)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	beforeChild := child.DeepCopy()

	t.Log("Reorder engines, update the frontend and enable an ordinary checkpoint without changing LPX")
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
	components := dgd.Spec.Components
	components[0], components[1] = components[1], components[0]
	dgd.GetComponentByName("frontend").PodTemplate.Spec.Containers[0].Image = "frontend:next"
	dgd.GetComponentByName("prefill").Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{
		Enabled: true, CheckpointRef: ptr.To("prefill-checkpoint"),
	}}
	dgd.Generation++
	require.NoError(t, r.Update(t.Context(), dgd))
	beforeDGD := dgd.DeepCopy()
	observedDGD, err := getDynamoGraphDeployment(t.Context(), r.Client, child)
	require.NoError(t, err)
	require.Equal(t, dgd, observedDGD)
	afterSelected := resolveLPXTestWorkload(t, r.modelRegistry, t.Context(), child, dgd)
	afterPCS := renderLPXTestPodCliqueSet(t, t.Context(), r, child, dgd, afterSelected)
	require.Equal(t, beforePCS, afterPCS)
	require.NotContains(t, afterPCS.Annotations, lpx.DGDGenerationAnnotation)
	_, missing, changed := resolvePipelineRequests(child, beforeRequests.Items, afterSelected.workload, afterSelected.plan)
	require.False(t, changed)
	require.Empty(t, missing)
	afterRequests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(t.Context(), afterRequests))
	require.Equal(t, beforeRequests, afterRequests)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.Equal(t, beforeChild, child)
	require.Equal(t, beforeDGD, dgd)
}

func TestLPXFailedScaleOutPreservesServingEngines(t *testing.T) {
	t.Run("deadline", func(t *testing.T) {
		t.Log("Observe a completed engine and a newer pending request in their shared PCS")
		ctx := t.Context()
		child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
		dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
		lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
		r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
		objects := lpxMaterializedObjects(t, r, child, dgd, selected)
		createLPXTestObjects(t, ctx, r.Client, objects...)
		publishSelectedLPXForTest(t, ctx, r, child, selected)
		pcs := findLPXTestPodCliqueSet(t, objects)
		serving := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
		serving.Status = newTestPipelineRequest(child, pcs, serving.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
		require.NoError(t, r.Update(ctx, serving))
		pending := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].Name)
		pending.Status = newTestPipelineRequest(child, pcs, pending.Name, time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhasePending).Status
		pending.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
		require.NoError(t, r.Update(ctx, pending))

		t.Log("Persist failure before changing Grove or deleting scheduler intent")
		request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
		group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
		_, err := r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
		require.Equal(t, int32(2), group.Spec.Replicas)
		pending = getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
		require.True(t, pending.DeletionTimestamp.IsZero())

		t.Log("Retire only the failed scale-out and keep Grove below the authored scale while cleanup is pending")
		for range 3 {
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), &grovev1alpha1.PodCliqueSet{}))
			require.True(t, apiequality.Semantic.DeepEqual(serving, getTestPipelineRequest(t, ctx, r.Client, serving.Namespace, serving.Name)), "serving request changed")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
			require.Equal(t, int32(1), group.Spec.Replicas)
		}
		t.Log("A later edit still waits for the failed request's scheduler finalizer")
		pending = getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
		require.False(t, pending.DeletionTimestamp.IsZero())
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
		child.Generation++
		require.NoError(t, r.Update(ctx, child))
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
		require.Equal(t, int32(1), group.Spec.Replicas)

		t.Log("After cleanup the later edit can retry without replacing the serving engine")
		retiredUID := pending.UID
		pending.Finalizers = nil
		require.NoError(t, r.Update(ctx, pending))
		for range 5 {
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
		}
		replacement := getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
		require.NotEqual(t, retiredUID, replacement.UID)
		require.True(t, replacement.DeletionTimestamp.IsZero())
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
		require.Equal(t, int32(2), group.Spec.Replicas)
		require.True(t, apiequality.Semantic.DeepEqual(serving, getTestPipelineRequest(t, ctx, r.Client, serving.Namespace, serving.Name)), "serving request changed")
	})
}

func TestLPXInvalidReplacementPreservesExistingWorkload(t *testing.T) {
	t.Log("Publish a speculative engine with two draft models")
	ctx := t.Context()
	child, dgd, registry := newLPXSpecDecodeTestDGD(t)
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := observedLPXTestPodCliqueSet(t, ctx, r, child, selected)
	beforeRequests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)

	t.Log("Change immutable composition and remove required runtime storage in the same edit")
	draft := dgd.GetComponentByName("draft")
	draft.Replicas = ptr.To(int32(1))
	agent := draft.ComponentRole(v1beta1.ComponentRoleLPXAgent)
	mounts := agent.PodTemplate.Spec.Containers[0].VolumeMounts
	agent.PodTemplate.Spec.Containers[0].VolumeMounts = nil
	replacement, err := lpx.ResolveSelectedWorkload(ctx, dgd, registry)
	require.NoError(t, err)
	require.NotEqual(t, selected.workload.Digest(), replacement.Digest())
	updateTestDGD(t, r, child, dgd)

	t.Log("Reject the replacement without deleting the serving PCS or its requests")
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 2 {
		_, err := r.Reconcile(ctx, request)
		require.ErrorContains(t, err, "model storage volume mount")
		observed, err := getPodCliqueSet(ctx, r.Client, child)
		require.NoError(t, err)
		require.Equal(t, pcs, observed)
		requests, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.Equal(t, beforeRequests, requests)
	}

	t.Log("Authorize replacement once the same desired composition can render successfully")
	agent.PodTemplate.Spec.Containers[0].VolumeMounts = mounts
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(pcs), &grovev1alpha1.PodCliqueSet{})))
}

func TestLPXExplicitScaleInDuringSchedulingFailure(t *testing.T) {
	t.Log("Publish four replicas with an expired interior request and a finalizer on the healthy tail")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	lpx.ServingComponent(dgd).Replicas = ptr.To(int32(4))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := findLPXTestPodCliqueSet(t, objects)
	group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
	before := make([]*lpxv1alpha1.LPUPipelineRequest, len(selected.requests))
	for index, request := range selected.requests {
		observed := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, request.Name)
		phase := lpxv1alpha1.RequestPhaseBound
		if index == 1 {
			phase = lpxv1alpha1.RequestPhasePending
		}
		observed.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now().Add(-time.Hour), phase).Status
		if index == 3 {
			observed.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
		}
		require.NoError(t, r.Update(ctx, observed))
		before[index] = observed
	}

	t.Log("Apply a four-to-three replica edit despite the retained interior expiry")
	lpx.ServingComponent(dgd).Replicas = ptr.To(int32(3))
	updateTestDGD(t, r, child, dgd)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 3 {
		_, err := r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
		require.EqualValues(t, 3, group.Spec.Replicas)
		tail := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, before[3].Name)
		require.False(t, tail.DeletionTimestamp.IsZero())
		for _, original := range before[:3] {
			observed := getTestPipelineRequest(t, ctx, r.Client, original.Namespace, original.Name)
			require.True(t, apiequality.Semantic.DeepEqual(original, observed))
		}
	}
	require.NoError(t, r.Get(ctx, request.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, schedulingFailedCondition))

	t.Log("Expire the remaining tail and clean its suffix while the removed request still awaits finalization")
	remainingTail := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, before[2].Name)
	remainingTail.Status = newTestPipelineRequest(child, pcs, remainingTail.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, remainingTail))
	_, err := r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.EqualValues(t, 1, group.Spec.Replicas)
	for _, original := range before[1:3] {
		requirePipelineRequestNotFound(t, ctx, r.Client, original.Namespace, original.Name)
	}
	observed := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, before[0].Name)
	require.True(t, apiequality.Semantic.DeepEqual(before[0], observed))
	tail := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, before[3].Name)
	require.False(t, tail.DeletionTimestamp.IsZero())
	observedPCS, err := getPodCliqueSet(ctx, r.Client, child)
	require.NoError(t, err)
	require.Equal(t, pcs.UID, observedPCS.UID)
}

func TestLPXExternalScaleRejectsInvalidReplicaCount(t *testing.T) {
	t.Log("Observe externally managed capacity above the materialization limit")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(dgd).Replicas = nil
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
	group.Spec.Replicas = 2497
	createLPXTestObjects(t, ctx, r.Client, objects...)

	t.Log("Reject the observed replica count before publishing requests and leave external capacity unchanged")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.ErrorContains(t, err, "between 0 and 2496")
	requests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(ctx, requests))
	require.Empty(t, requests.Items)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.EqualValues(t, 2497, group.Spec.Replicas)
}

func TestLPXExternalScaleInAfterSchedulingFailure(t *testing.T) {
	for _, tc := range []struct {
		name     string
		replicas int32
		expired  int
		retained []int
	}{
		{name: "scale to zero", replicas: 0, expired: 0},
		{name: "remove failed suffix", replicas: 2, expired: 2, retained: []int{0, 1}},
		{name: "retain an interior failure", replicas: 2, expired: 0, retained: []int{0, 1}},
		{name: "scale-in and remaining expired suffix", replicas: 1, expired: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Publish four externally managed replicas with one expired request")
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
			lpx.ServingComponent(dgd).Replicas = nil
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			selected.plan.Replicas = 4
			_, missing, changed := resolvePipelineRequests(child, nil, selected.workload, selected.plan)
			require.False(t, changed)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			pcs := findLPXTestPodCliqueSet(t, objects)
			group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
			require.NoError(t, r.reconcilePipelineRequests(ctx, child, pcs, missing))
			before := make([]*lpxv1alpha1.LPUPipelineRequest, len(missing))
			for index, request := range missing {
				observed := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, request.Name)
				phase := lpxv1alpha1.RequestPhaseBound
				if index == tc.expired {
					phase = lpxv1alpha1.RequestPhasePending
				}
				observed.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now().Add(-time.Hour), phase).Status
				require.NoError(t, r.Update(ctx, observed))
				before[index] = observed
			}
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			result, err := r.Reconcile(ctx, request)
			require.NoError(t, err)
			require.Equal(t, time.Nanosecond, result.RequeueAfter)
			require.NoError(t, r.Get(ctx, request.NamespacedName, child))
			failure := meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition).DeepCopy()
			require.Equal(t, metav1.ConditionTrue, failure.Status)

			t.Log("Lower capacity externally while keeping the same failed input revision")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
			group.Spec.Replicas = tc.replicas
			require.NoError(t, r.Update(ctx, group))
			beforeGroup := group.DeepCopy()
			deleted := make(map[client.ObjectKey]bool)
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
					key := client.ObjectKeyFromObject(object)
					require.False(t, deleted[key], "scale-in and deadline cleanup must not repeat a deletion in one pass")
					deleted[key] = true
					return delegated.Delete(ctx, object, opts...)
				},
			})
			for range 3 {
				clear(deleted)
				result, err = r.Reconcile(ctx, request)
				require.NoError(t, err)
				require.Zero(t, result)
			}

			t.Log("Finish scale-in and any eligible deadline suffix without scaling or replacing retained requests")
			for index, original := range before {
				if slices.Contains(tc.retained, index) {
					observed := getTestPipelineRequest(t, ctx, r.Client, original.Namespace, original.Name)
					require.True(t, apiequality.Semantic.DeepEqual(original, observed))
				} else {
					requirePipelineRequestNotFound(t, ctx, r.Client, original.Namespace, original.Name)
				}
			}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
			require.Equal(t, beforeGroup, group)
			require.NoError(t, r.Get(ctx, request.NamespacedName, child))
			require.Equal(t, failure, meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition))
			require.Equal(t, v1alpha1.LPXReadyReasonFailed, meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition).Reason)

			t.Log("A later external scale-out does not bypass the failed generation's publication gate")
			group.Spec.Replicas = 4
			require.NoError(t, r.Update(ctx, group))
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
			for index := int(tc.replicas); index < len(before); index++ {
				requirePipelineRequestNotFound(t, ctx, r.Client, before[index].Namespace, before[index].Name)
			}
		})
	}
}

func TestLPXInvalidEditsPreserveExistingWorkload(t *testing.T) {
	for _, scenario := range []struct {
		name, message string
		preserve      bool
	}{
		{name: "render", message: "model storage volume mount"},
		{name: "invalid source", message: "source"},
		{name: "transient snapshot", message: "temporary snapshot timeout", preserve: true},
		{name: "inconsistent snapshot", message: "immutable LPX build snapshot"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Stage a PCS and discovery endpoint before the API supplies a PCS UID")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
			r := newLPXTestReconciler(t, registry, child, dgd)
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			_, err := r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			pcs := &grovev1alpha1.PodCliqueSet{}
			pcsKey := client.ObjectKey{Namespace: child.Namespace, Name: dynamo.PCSNameForLPX(child)}
			require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
			pcs.UID = "staged-pcs"
			require.NoError(t, r.Update(t.Context(), pcs))
			endpoint := &corev1.Service{}
			endpointKey := client.ObjectKey{Namespace: child.Namespace, Name: pcsKey.Name + "-serve"}
			require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
			endpoint.UID = "staged-endpoint"
			require.NoError(t, r.Update(t.Context(), endpoint))
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items)

			beforePCS, beforeEndpoint := pcs.DeepCopy(), endpoint.DeepCopy()

			t.Log("Introduce a current terminal input failure, or a transient snapshot outage")
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			component := lpx.ServingComponent(dgd)
			switch scenario.name {
			case "render":
				component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0].VolumeMounts = nil
			case "invalid source":
				child.OwnerReferences[0].UID = "replaced-source"
			case "transient snapshot", "inconsistent snapshot":
				cause := errors.New("temporary snapshot timeout")
				if !scenario.preserve {
					cause = lpx.ErrBuildSnapshotInconsistent
				}
				r.modelRegistry = &snapshotFailureRegistry{ModelRegistry: registry, err: cause}
			}
			dgd.Generation++
			require.NoError(t, r.Update(t.Context(), dgd))
			child.Generation++
			child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
			require.NoError(t, err)
			require.NoError(t, r.Update(t.Context(), child))

			t.Log("Report the actionable failure without destroying existing resources")
			for range 3 {
				_, err = r.Reconcile(t.Context(), request)
				if err != nil {
					require.ErrorContains(t, err, scenario.message)
				} else {
					require.NoError(t, err)
				}
				require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
				require.Contains(t, meta.FindStatusCondition(child.Status.Conditions, "Ready").Message, scenario.message)
			}
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			failure := meta.FindStatusCondition(child.Status.Conditions, "Ready")
			require.NotNil(t, failure)
			require.Equal(t, metav1.ConditionFalse, failure.Status)
			require.Equal(t, v1alpha1.LPXReadyReasonFailed, failure.Reason)
			require.Contains(t, failure.Message, scenario.message)
			require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
			require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
			require.Equal(t, beforePCS, pcs)
			require.Equal(t, beforeEndpoint, endpoint)
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items)
		})
	}
}

func TestLPXMaterializationUsesDGDAndChildIdentity(t *testing.T) {
	t.Log("Render a materialization with a different name from its source DGD")
	_, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	child := newLPXTestDeployment(t, dgd)
	child.Name = "independent-materialization"
	child.UID = "independent-materialization-uid"
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	publishSelectedLPXForTest(t, t.Context(), r, child, selected)

	t.Log("Route Grove objects to the materialization while retaining the source provenance")
	pcs := objects[0].(*grovev1alpha1.PodCliqueSet)
	require.Equal(t, dgd.Name, pcs.Labels[consts.KubeLabelDynamoGraphDeploymentName])
	for _, object := range objects[1:] {
		require.Equal(t, []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(child)}}, mapChildToLPXGraphDeployment(t.Context(), object))
	}

	t.Log("Publish the runtime configuration and endpoint under the exact LGD owner")
	_, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)

	root := dynamo.PCSNameForLPX(child)
	for _, object := range []client.Object{
		&grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: root, Namespace: child.Namespace}},
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: lpx.LPUConfigMapName(root, pcs.Spec.Template.Cliques[0].Annotations[consts.AnnotationExtraResourcesHash]), Namespace: child.Namespace}},
		&corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: root + "-serve", Namespace: child.Namespace}},
	} {
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(object), object))
		require.True(t, metav1.IsControlledBy(object, child))
	}

	t.Log("Keep scheduler-request provenance rooted in the source DGD, not the independently named child")
	requests, err := r.getPipelineRequests(t.Context(), pcs)
	require.NoError(t, err)
	require.NotEmpty(t, requests)
	for _, request := range requests {
		require.Equal(t, dgd.Name, request.Labels[consts.KubeLabelDynamoGraphDeploymentName])
	}
}

func TestLPXMissingRequestGetsANewIndependentDeadline(t *testing.T) {
	t.Log("Publish two requests and remove one without a terminal deadline decision")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	createLPXTestObjects(t, ctx, r.Client, lpxMaterializedObjects(t, r, child, dgd, selected)...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)

	t.Log("Advance the deployment and remove one request before it fails")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	child.Generation++
	require.NoError(t, r.Update(ctx, child))

	missing := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].Name)
	previousUID := missing.UID
	require.NoError(t, r.Delete(ctx, missing))

	t.Log("Reconcile the PCS for the new generation before publishing the missing request")
	result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.Zero(t, result, "the PCS update drives the next reconcile")
	requirePipelineRequestNotFound(t, ctx, r.Client, missing.Namespace, missing.Name)

	t.Log("Reconciliation recreates deterministic scheduler intent without inheriting a scheduling clock")
	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	replacement := getTestPipelineRequest(t, ctx, r.Client, missing.Namespace, missing.Name)
	require.NotEqual(t, previousUID, replacement.UID)
	require.Nil(t, replacement.Status)
}

func TestLPXModelScaleDownRecreatesPodCliqueSet(t *testing.T) {
	ctx := t.Context()
	child, dgd, registry := newLPXSpecDecodeTestDGD(t)
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)

	scaledDGD := dgd.DeepCopy()
	require.Len(t, scaledDGD.Spec.Components, 2)
	scaledDGD.Spec.Components[1].Replicas = ptr.To(int32(1))
	scaled := resolveLPXTestWorkload(t, r.modelRegistry, ctx, child, scaledDGD)
	require.Len(t, scaled.requests, 2)

	desiredNames := make(map[string]struct{}, len(scaled.requests))
	for _, request := range scaled.requests {
		desiredNames[request.Name] = struct{}{}
	}
	var stale *lpxv1alpha1.LPUPipelineRequest
	for _, request := range selected.requests {
		if _, retained := desiredNames[request.Name]; retained {
			continue
		}
		stale = getTestPipelineRequest(t, ctx, r.Client, child.Namespace, request.Name)
		break
	}
	require.NotNil(t, stale)
	stale.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
	require.NoError(t, r.Update(ctx, stale))

	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(findLPXTestPodCliqueSet(t, objects)), pcs))
	pcs.Finalizers = []string{"test.example/observe-retirement"}
	require.NoError(t, r.Update(ctx, pcs))
	retiredTemplate := selected.plan.Agents[1].TemplateName
	originalCliqueCount := len(pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames)
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, group))
	require.Contains(t, group.Spec.CliqueNames, retiredTemplate)

	t.Log("Retire the immutable PCS without mutating Grove membership or directly deleting its LPRs")
	pcsDeleteObserved := false
	requestDeleteObserved := false
	wrapped := interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			switch object.(type) {
			case *grovev1alpha1.PodCliqueSet:
				pcsDeleteObserved = true
			case *lpxv1alpha1.LPUPipelineRequest:
				requestDeleteObserved = true
			}
			return delegated.Delete(ctx, object, opts...)
		},
	})
	r.Client = wrapped
	updateTestDGD(t, r, child, scaledDGD)
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
	require.True(t, pcsDeleteObserved)
	require.False(t, requestDeleteObserved)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
	require.False(t, pcs.DeletionTimestamp.IsZero())
	require.Contains(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames, retiredTemplate)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Contains(t, group.Spec.CliqueNames, retiredTemplate)
	stale = getTestPipelineRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	require.True(t, stale.DeletionTimestamp.IsZero())

	t.Log("Wait while foreground deletion keeps the PCS present")
	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)

	t.Log("Simulate foreground GC removing dependent requests before the PCS")
	requests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(ctx, requests))
	for index := range requests.Items {
		request := &requests.Items[index]
		request.Finalizers = nil
		require.NoError(t, r.Update(ctx, request))
		require.NoError(t, r.Delete(ctx, request))
	}
	pcs.Finalizers = nil
	require.NoError(t, r.Update(ctx, pcs))
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(pcs), &grovev1alpha1.PodCliqueSet{})))

	t.Log("Create the replacement PCS with the scaled-in immutable shape")
	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	synced, err := getPodCliqueSet(ctx, r.Client, child)
	require.NoError(t, err)
	require.NotNil(t, synced)
	require.Equal(t, []string{"cond", "agt0", "agt1"}, synced.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames)
	require.Len(t, synced.Spec.Template.Cliques, 3)
	require.Equal(t, scaled.workload.Digest().String(), synced.Annotations[lpx.WorkloadDigestAnnotation])
	require.Less(t, len(synced.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames), originalCliqueCount)
}

func TestLPXNativeScaleToZeroPreservesPodCliqueSetTemplate(t *testing.T) {
	t.Log("Publish one engine whose capacity is owned by the native Grove scaling group")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(dgd).Replicas = nil
	r, initial := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, initial)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, initial)
	pcs := observedLPXTestPodCliqueSet(t, ctx, r, child, initial)
	pcsUID := pcs.UID
	templateReplicas := ptr.Deref(pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas, 0)
	require.Positive(t, templateReplicas)
	request := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, initial.requests[0].Name)
	requestUID := request.UID

	t.Log("Scale the native Grove group to zero and derive empty scheduler intent")
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: initial.plan.LPXScalingGroup}, group))
	group.Spec.Replicas = 0
	require.NoError(t, r.Update(ctx, group))

	t.Log("Delete the stale LPR before reconciling the stable PCS template")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(request), &lpxv1alpha1.LPUPipelineRequest{})))
	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Zero(t, group.Spec.Replicas)
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), storedPCS))
	require.Equal(t, pcsUID, storedPCS.UID)
	require.Equal(t, templateReplicas, ptr.Deref(storedPCS.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas, 0))

	t.Log("Scale the native group back up and publish a fresh request through the same PCS")
	group.Spec.Replicas = 1
	require.NoError(t, r.Update(ctx, group))
	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	replacement := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, initial.requests[0].Name)
	require.NotEqual(t, requestUID, replacement.UID)
	require.Equal(t, pcsUID, metav1.GetControllerOf(replacement).UID)
}

func TestLPXReadableNamesSurviveServingComponentChanges(t *testing.T) {
	t.Log("Publish a speculative engine with long deployment and component names")
	_, dgd, registry := newLPXSpecDecodeTestDGD(t)
	dgd.Name = "test-models-gpt-oss-20b-lp20-b300"
	dgd.Spec.Components[0].ComponentName = "serving-component-with-name"
	dgd.Spec.Components[1].ComponentName = "draft-component-with-name"
	dgd.Spec.Components[1].Replicas = ptr.To(int32(1))
	dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	child := newLPXTestDeployment(t, dgd)
	r := newLPXTestReconciler(t, registry, child, dgd)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(t.Context(), request)
	require.NoError(t, err)
	root := dynamo.PCSNameForLPX(child)
	require.True(t, strings.HasPrefix(root, dgd.Name+"-"))
	require.Len(t, root, len(dgd.Name)+len("-ffff"))
	pcs := &grovev1alpha1.PodCliqueSet{}
	pcsKey := client.ObjectKey{Namespace: child.Namespace, Name: root}
	require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
	pcs.UID = "stable-pcs"
	require.NoError(t, r.Update(t.Context(), pcs))
	endpoint := &corev1.Service{}
	endpointKey := client.ObjectKey{Namespace: child.Namespace, Name: root + "-serve"}
	require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
	endpoint.UID = "stable-endpoint"
	require.NoError(t, r.Update(t.Context(), endpoint))
	require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
	beforeGroup := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
	beforeOwners := endpoint.OwnerReferences

	t.Log("Observe Grove's group and publish the original requests")
	selected := resolveLPXTestWorkload(t, r.modelRegistry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
	group.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}
	require.NoError(t, r.Create(t.Context(), group))
	_, err = r.Reconcile(t.Context(), request)
	require.NoError(t, err)

	for _, change := range []string{"rename serving component", "move conductor"} {
		t.Log(change)
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
		require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
		if change == "rename serving component" {
			dgd.Spec.Components[0].ComponentName = "renamed-serving-component"
		} else {
			conductor := *dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
			dgd.Spec.Components[0].Roles = slices.DeleteFunc(dgd.Spec.Components[0].Roles, func(role v1beta1.ComponentRoleSpec) bool {
				return role.Name == v1beta1.ComponentRoleLPXConductor
			})
			dgd.Spec.Components[1].Roles = append(dgd.Spec.Components[1].Roles, conductor)
		}
		dgd.Generation++
		require.NoError(t, r.Update(t.Context(), dgd))
		child.Generation++
		child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
		require.NoError(t, err)
		require.NoError(t, r.Update(t.Context(), child))

		t.Log("Preserve the PCS on rename and retire it when target and draft assignments change")
		_, err = r.Reconcile(t.Context(), request)
		require.NoError(t, err)
		if change == "move conductor" {
			require.True(t, apierrors.IsNotFound(r.Get(t.Context(), pcsKey, &grovev1alpha1.PodCliqueSet{})))
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
			require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
			require.Equal(t, "stable-endpoint", string(endpoint.UID))

			t.Log("Simulate dependent request garbage collection, which the fake client does not perform")
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), requests))
			require.NotEmpty(t, requests.Items)
			for index := range requests.Items {
				require.Equal(t, "stable-pcs", string(metav1.GetControllerOf(&requests.Items[index]).UID))
				require.NoError(t, r.Delete(t.Context(), &requests.Items[index]))
			}

			t.Log("Recreate the workload under the same readable names after retirement")
			_, err = r.Reconcile(t.Context(), request)
			require.NoError(t, err)
		}
		require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
		require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
		if change == "rename serving component" {
			require.Equal(t, "stable-pcs", string(pcs.UID))
		} else {
			require.NotEqual(t, "stable-pcs", string(pcs.UID))
		}
		require.Equal(t, "stable-endpoint", string(endpoint.UID))
		require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
		require.Equal(t, beforeGroup.Name, pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].Name)
		require.Equal(t, beforeGroup.CliqueNames, pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames)
		require.Equal(t, beforeOwners, endpoint.OwnerReferences)
		require.Equal(t, dgd.Name, pcs.Labels[consts.KubeLabelDynamoGraphDeploymentName])
		require.Equal(t, lpx.ServingComponent(dgd).ComponentName, endpoint.Spec.Selector[consts.KubeLabelDynamoComponent])
		require.Equal(t, root, endpoint.Spec.Selector[grovecommon.LabelPartOfKey])
		require.Equal(t, consts.KubeLabelValueTrue, endpoint.Spec.Selector[dynamo.LPXServingLabel])
		allPCS, allEndpoints := &grovev1alpha1.PodCliqueSetList{}, &corev1.ServiceList{}
		require.NoError(t, r.List(t.Context(), allPCS))
		require.NoError(t, r.List(t.Context(), allEndpoints))
		require.Len(t, allPCS.Items, 1)
		require.Len(t, allEndpoints.Items, 1)

		t.Log("Select only the conductor and share one runtime config across the clique roster")
		require.NotEmpty(t, pcs.Spec.Template.Cliques)
		configHash := pcs.Spec.Template.Cliques[0].Annotations[consts.AnnotationExtraResourcesHash]
		require.NotEmpty(t, configHash)
		cliqueNames := make([]string, 0, len(pcs.Spec.Template.Cliques))
		for _, clique := range pcs.Spec.Template.Cliques {
			cliqueNames = append(cliqueNames, clique.Name)
			require.Equal(t, configHash, clique.Annotations[consts.AnnotationExtraResourcesHash])
			for key, value := range clique.Labels {
				require.Empty(t, validation.IsValidLabelValue(value), key)
			}
			if clique.Name == "cond" {
				for key, value := range endpoint.Spec.Selector {
					if key != grovecommon.LabelPartOfKey {
						require.Equal(t, value, clique.Labels[key], key)
					}
				}
			} else {
				require.NotContains(t, clique.Labels, dynamo.LPXServingLabel)
			}
		}
		require.ElementsMatch(t, []string{"cond", "agt0", "agt1"}, cliqueNames)

		t.Log("Keep runtime allocation aligned with the clique names")
		config := &corev1.ConfigMap{}
		configKey := client.ObjectKey{Namespace: child.Namespace, Name: lpx.LPUConfigMapName(root, configHash)}
		require.NoError(t, r.Get(t.Context(), configKey, config))
		require.True(t, metav1.IsControlledBy(config, child))
		require.NotContains(t, config.Data, "datacenter.toml")
		for _, clique := range pcs.Spec.Template.Cliques {
			if clique.Name == "cond" {
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "LPX_ALLOCATION", Value: "agt0:agt1"})
			}
		}
	}
}

func TestLPXReconcileUsesOneInputSnapshot(t *testing.T) {
	for _, editDGD := range []bool{true, false} {
		name := "child metadata edit"
		if editDGD {
			name = "source scale-down"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("Materialize Grove for two replicas before publishing scheduler intent")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
			r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, t.Context(), r.Client, objects...)

			t.Log("Persist the PCS update and re-reconcile before testing LPR publication")
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			result, err := r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			require.Zero(t, result, "the PCS update drives the next reconcile")
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items)

			t.Log("Edit live input after the first LPR write without changing the reconcile snapshot")
			dgdReads, childReads, creates := 0, 0, 0
			wrapped := interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					switch object.(type) {
					case *v1beta1.DynamoGraphDeployment:
						dgdReads++
					case *v1alpha1.LPXGraphDeployment:
						childReads++
					}
					return delegated.Get(ctx, key, object, opts...)
				},
				Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
					if err := delegated.Create(ctx, object, opts...); err != nil {
						return err
					}
					if _, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok {
						creates++
						if creates == 1 {
							if editDGD {
								require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
								lpx.ServingComponent(dgd).Replicas = ptr.To(int32(1))
								dgd.Generation++
								return delegated.Update(ctx, dgd)
							}
							require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(child), child))
							child.Annotations[lpx.DGDGenerationAnnotation] = "999"
							return delegated.Update(ctx, child)
						}
					}
					return nil
				},
			})
			r.Client = wrapped
			_, err = r.Reconcile(t.Context(), request)
			if editDGD {
				require.NoError(t, err)
			} else {
				require.True(t, apierrors.IsConflict(err), "status writes still reject a changed child: %v", err)
			}
			require.Equal(t, 1, dgdReads)
			require.Equal(t, 1, childReads)
			require.Equal(t, 2, creates, "both requests use the validated two-replica snapshot")
			retained := getTestPipelineRequest(t, t.Context(), r.Client, child.Namespace, selected.requests[0].Name)

			t.Log("A subsequent reconcile waits for the updated handoff before retiring any LPR")
			_, err = r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			require.NoError(t, r.List(t.Context(), requests))
			require.Len(t, requests.Items, 2)
			group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(group), group))
			require.Equal(t, int32(2), group.Spec.Replicas, "an incomplete handoff must not scale Grove")
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			if editDGD {
				require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
				child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
				require.NoError(t, err)
				child.Generation++
				require.NoError(t, r.Update(t.Context(), child))
			}

			t.Log("Converge the new inputs without replacing the retained replica's request")
			_, err = r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(group), group))
			require.Equal(t, *lpx.ServingComponent(dgd).Replicas, group.Spec.Replicas)
			require.NoError(t, r.List(t.Context(), requests))
			require.Len(t, requests.Items, int(group.Spec.Replicas))
			require.Equal(t, retained.UID, getTestPipelineRequest(t, t.Context(), r.Client, retained.Namespace, retained.Name).UID)
		})
	}
}

func TestLPXReconcileWaitsForMatchingDGDRevision(t *testing.T) {
	for _, dgdAhead := range []bool{true, false} {
		name := "child ahead"
		if dgdAhead {
			name = "source ahead"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("Observe the source and child at different handoff revisions")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			r := newLPXTestReconciler(t, registry, child, dgd)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
			lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
			dgd.Generation++
			if dgdAhead {
				require.NoError(t, r.Update(t.Context(), dgd))
			} else {
				var err error
				child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
				require.NoError(t, err)
				child.Generation++
				require.NoError(t, r.Update(t.Context(), child))
			}

			t.Log("Wait without failing or publishing any workload from mismatched inputs")
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			result, err := r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			require.Zero(t, result, "source and child watches resume the handoff")
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			require.Len(t, child.Status.Conditions, 1)
			observed := meta.FindStatusCondition(child.Status.Conditions, "Ready")
			require.NotNil(t, observed)
			require.Equal(t, metav1.ConditionFalse, observed.Status)
			require.Equal(t, v1alpha1.LPXReadyReasonPending, observed.Reason)
			allPCS := &grovev1alpha1.PodCliqueSetList{}
			require.NoError(t, r.List(t.Context(), allPCS))
			require.Empty(t, allPCS.Items)
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items)

			t.Log("Resume publication when the matching source or child reaches the cache")
			if dgdAhead {
				child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
				require.NoError(t, err)
				child.Generation++
				require.NoError(t, r.Update(t.Context(), child))
			} else {
				require.NoError(t, r.Update(t.Context(), dgd))
			}
			_, err = r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			require.NoError(t, r.List(t.Context(), allPCS))
			require.Len(t, allPCS.Items, 1)
		})
	}
}

func TestScalePodCliqueScalingGroup(t *testing.T) {
	conflict := apierrors.NewConflict(consts.PodCliqueScalingGroupGVR.GroupResource(), "group", errors.New("stale version"))
	for _, tc := range []struct {
		name        string
		replicas    *int32
		err         error
		wantUpdates int
	}{
		{name: "scale out", replicas: ptr.To(int32(12)), wantUpdates: 1},
		{name: "scale in", replicas: ptr.To(int32(2)), wantUpdates: 1},
		{name: "scale to zero", replicas: ptr.To(int32(0)), wantUpdates: 1},
		{name: "unchanged", replicas: ptr.To(int32(9))},
		{name: "omitted replicas"},
		{name: "conflict", replicas: ptr.To(int32(12)), err: conflict, wantUpdates: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Use an already-owned group observation without rendering a workload")
			group := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "group", Namespace: "default", ResourceVersion: "7"},
				Spec:       grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 9},
			}
			updates := 0
			r := &graphReconciler{Client: interceptor.NewClient(newLPXTestClient(t), interceptor.Funcs{
				SubResourceUpdate: func(_ context.Context, _ client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
					require.Equal(t, "scale", subresource)
					require.Same(t, group, object)
					scale := (&client.SubResourceUpdateOptions{}).ApplyOptions(opts).SubResourceBody.(*autoscalingv1.Scale)
					require.Equal(t, group.ResourceVersion, scale.ResourceVersion)
					require.Equal(t, *tc.replicas, scale.Spec.Replicas)
					updates++
					return tc.err
				},
			})}

			t.Log("Only explicit changes write scale; successful writes update the in-pass observation")
			err := r.scalePodCliqueScalingGroup(t.Context(), group, tc.replicas)
			require.ErrorIs(t, err, tc.err)
			require.Equal(t, tc.wantUpdates, updates)
			want := int32(9)
			if tc.replicas != nil && tc.err == nil {
				want = *tc.replicas
			}
			require.Equal(t, want, group.Spec.Replicas)
		})
	}
}

func TestLPXRequestListFailureUsesControllerBackoff(t *testing.T) {
	t.Log("A request-list failure prevents deriving any active deadline")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	createLPXTestObjects(t, t.Context(), r.Client, lpxMaterializedObjects(t, r, child, dgd, desired)...)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	readError := errors.New("temporary request observation failure")
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, requests := list.(*lpxv1alpha1.LPUPipelineRequestList); requests {
				return readError
			}
			return delegated.List(ctx, list, opts...)
		},
	})
	result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.ErrorIs(t, err, readError)
	require.Zero(t, result)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.NotEqual(t, child.Generation, child.Status.ObservedGeneration)
	require.Contains(t, meta.FindStatusCondition(child.Status.Conditions, "Ready").Message, readError.Error())
}

func TestLPXScaleDownDoesNotRepeatWriteFromStaleCache(t *testing.T) {
	t.Log("Observe an externally inflated group through a cache that stays stale after scale")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	group := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
	group.Spec.Replicas = 9
	createLPXTestObjects(t, ctx, r.Client, objects...)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	stale := group.DeepCopy()
	scaled, updates := false, 0
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			if scaled && key == client.ObjectKeyFromObject(stale) {
				if observed, ok := object.(*grovev1alpha1.PodCliqueScalingGroup); ok {
					stale.DeepCopyInto(observed)
					return nil
				}
			}
			return delegated.Get(ctx, key, object, opts...)
		},
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if subresource != lpxTestScaleSubresource {
				return delegated.SubResource(subresource).Update(ctx, object, opts...)
			}
			updates++
			if err := delegated.SubResource(subresource).Update(ctx, object, opts...); err != nil {
				return err
			}
			scaled = true
			return nil
		},
	})

	t.Log("Continue reconciliation without issuing a second stale-resource-version write")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.Equal(t, 1, updates)
}

func TestLPXScaleDownUpdatesGroveBeforeDeletingStaleRequest(t *testing.T) {
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(dgd).Replicas = ptr.To(int32(2))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)

	stale := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].Name)
	stale.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
	require.NoError(t, r.Update(ctx, stale))

	liveDGD := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), liveDGD))
	lpx.ServingComponent(liveDGD).Replicas = ptr.To(int32(1))
	liveDGD.Generation++
	require.NoError(t, r.Update(ctx, liveDGD))
	liveChild := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), liveChild))
	liveChild.Generation++
	var err error
	liveChild.Spec.InputRevision, err = dynamo.LPXInputRevision(liveDGD, "")
	require.NoError(t, err)
	require.NoError(t, r.Update(ctx, liveChild))

	t.Log("A missing PCSG observation blocks scale-in without deleting existing requests")
	base := r.Client
	r.Client = interceptor.NewClient(base.(client.WithWatch), interceptor.Funcs{
		Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			if _, ok := object.(*grovev1alpha1.PodCliqueScalingGroup); ok {
				return apierrors.NewNotFound(consts.PodCliqueScalingGroupGVR.GroupResource(), key.Name)
			}
			return delegated.Get(ctx, key, object, opts...)
		},
	})
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	result, err := r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Zero(t, result, "the PCSG watch resumes scale-in")
	require.Equal(t, stale, getTestPipelineRequest(t, ctx, r.Client, stale.Namespace, stale.Name))
	r.Client = base

	t.Log("Persist the lower Grove scale before request retirement can block reconciliation")
	result, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Zero(t, result, "request deletion progresses through watches")
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	t.Log("Let request finalization proceed asynchronously after the lower scale is persisted")
	stale = getTestPipelineRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	require.False(t, stale.DeletionTimestamp.IsZero())
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(findLPXTestPodCliqueSet(t, objects)), &grovev1alpha1.PodCliqueSet{}))

	t.Log("Reverse the scale-down while the old request still protects its pods")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), liveDGD))
	lpx.ServingComponent(liveDGD).Replicas = ptr.To(int32(2))
	liveDGD.Generation++
	require.NoError(t, r.Update(ctx, liveDGD))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), liveChild))
	liveChild.Generation++
	liveChild.Spec.InputRevision, err = dynamo.LPXInputRevision(liveDGD, "")
	require.NoError(t, err)
	require.NoError(t, r.Update(ctx, liveChild))
	for range 2 {
		result, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.Zero(t, result, "terminating current names stop scale-out without polling")
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
		require.Equal(t, int32(1), group.Spec.Replicas, "Grove must not recreate pods protected by the retiring request")
	}

	t.Log("Allow scale-up and a fresh request once the scheduler finishes cleanup")
	stale = getTestPipelineRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	retiredUID := stale.UID
	stale.Finalizers = nil
	require.NoError(t, r.Update(ctx, stale))
	for range 4 {
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(2), group.Spec.Replicas)
	replacement := getTestPipelineRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	require.NotEqual(t, retiredUID, replacement.UID)
	require.True(t, replacement.DeletionTimestamp.IsZero())
}

func TestLPXStartupScalesMinimumSeedBeforePublishingRequests(t *testing.T) {
	for _, tc := range []struct {
		name    string
		minimum *int32
	}{
		{name: "default minimum"},
		{name: "explicit minimum", minimum: ptr.To(int32(2))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Start without Grove resources and create the owned PCS at its immutable minimum")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			component := lpx.ServingComponent(dgd)
			component.Replicas, component.MinAvailable = ptr.To(int32(3)), tc.minimum
			r := newLPXTestReconciler(t, registry, child, dgd)
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			result, err := r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.Zero(t, result, "Grove watches resume reconciliation")
			pcs, err := getPodCliqueSet(t.Context(), r.Client, child)
			require.NoError(t, err)
			require.NotNil(t, pcs)
			require.True(t, metav1.IsControlledBy(pcs, child))
			seed := ptr.Deref(tc.minimum, 1)
			groupTemplate := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
			require.Equal(t, seed, *groupTemplate.Replicas)

			t.Log("Wait without publishing requests while Grove has not created its scaling group")
			result, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.Zero(t, result)
			requests, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Empty(t, requests)

			t.Log("Observe Grove's group initialized from the PCS template")
			group := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:            grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, groupTemplate.Name),
					Namespace:       pcs.Namespace,
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: seed},
			}
			require.NoError(t, r.Create(t.Context(), group))

			t.Log("Publish requests in ordinal order only after live capacity reaches the explicit desired count")
			var published []int64
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
					if request, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok {
						require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(group), group))
						require.Equal(t, int32(3), group.Spec.Replicas)
						require.True(t, metav1.IsControlledBy(request, pcs))
						require.True(t, ptr.Deref(metav1.GetControllerOf(request).BlockOwnerDeletion, false))
						published = append(published, request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex)
					}
					return delegated.Create(ctx, object, opts...)
				},
			})
			result, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.Zero(t, result, "Request creation events resume reconciliation")
			require.Equal(t, []int64{0, 1, 2}, published)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
			require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition).Reason)

			t.Log("A later observation neither recreates requests nor copies live replicas into the template")
			_, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.Equal(t, []int64{0, 1, 2}, published)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, seed, *pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas)
		})
	}
}

func TestLPXTerminatingGroveResourcesBlockPublication(t *testing.T) {
	t.Log("Publish a workload whose PCS and scaling group are both terminating")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := observedLPXTestPodCliqueSet(t, ctx, r, child, selected)
	pending := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
	pending.Status = newTestPipelineRequest(child, pcs, pending.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, pending))
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, group))
	for _, object := range []client.Object{pcs, group} {
		object.SetFinalizers([]string{"example.com/cleanup"})
		require.NoError(t, r.Update(ctx, object))
		require.NoError(t, r.Delete(ctx, object))
	}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
	require.False(t, pcs.DeletionTimestamp.IsZero())

	t.Log("Let the publication fence report retirement before attempting scale")
	result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
	require.Zero(t, result, "PCS deletion is observed through its watch")

	t.Log("Whole-workload retirement does not expire or delete surviving requests")
	for range 2 {
		_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
		require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
		require.Nil(t, meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition))
		require.True(t, getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name).DeletionTimestamp.IsZero())
	}
}

const lpxTestScaleSubresource = "scale"

// updateTestDGD emulates the DGD controller publishing a new child revision.
func updateTestDGD(t *testing.T, r *graphReconciler, child *v1alpha1.LPXGraphDeployment, desired *v1beta1.DynamoGraphDeployment) {
	t.Helper()
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(desired), dgd))
	dgd.Spec = *desired.Spec.DeepCopy()
	dgd.Annotations = desired.DeepCopy().Annotations
	dgd.Generation++
	require.NoError(t, r.Update(t.Context(), dgd))
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	child.Generation++
	var err error
	child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, child.Annotations[dynamo.LPXRestartAnnotation])
	require.NoError(t, err)
	require.NoError(t, r.Update(t.Context(), child))
}

func TestSpecDecodeStatusCountsCompleteDraftInstances(t *testing.T) {
	t.Log("Materialize two draft instances and one target in their single shared scaling group")
	deployment, dgd, registry := newLPXSpecDecodeTestDGD(t)
	prepare, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), deployment, dgd)
	objects := lpxMaterializedObjects(t, prepare, deployment, dgd, selected)
	for _, object := range objects {
		switch live := object.(type) {
		case *grovev1alpha1.PodCliqueScalingGroup:
			live.Status.Replicas, live.Status.UpdatedReplicas = 1, 1
			live.Status.AvailableReplicas, live.Status.ScheduledReplicas = 1, 1
		case *grovev1alpha1.PodClique:
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
		}
	}
	pcs := findLPXTestPodCliqueSet(t, objects)

	t.Log("Observe each draft condition against an independent copy of the ready baseline")
	for _, scenario := range []string{"ready", "partial", "missing", "foreign", "old revision", "unobserved"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Seed isolated Grove observations and fetch the first draft")
			kubeClient := newLPXTestClient(t)
			createLPXTestObjects(t, t.Context(), kubeClient, objects...)
			firstDraft := getResource[*grovev1alpha1.PodClique](t, objects, selected.plan.Agents[0].CliqueName).DeepCopy()
			require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(firstDraft), firstDraft))
			require.Equal(t, "draft", firstDraft.Labels[lpx.StageLabel])

			t.Log("Make only the first draft incomplete while retaining a fully ready second draft and target")
			switch scenario {
			case "partial":
				firstDraft.Status.ReadyReplicas--
			case "foreign":
				firstDraft.OwnerReferences = []metav1.OwnerReference{{UID: "foreign", Controller: ptr.To(true)}}
			case "old revision":
				firstDraft.Status.CurrentPodCliqueSetGenerationHash = ptr.To("old")
			case "unobserved":
				firstDraft.Status.ObservedGeneration = ptr.To(firstDraft.Generation - 1)
			}
			if scenario == "missing" {
				require.NoError(t, kubeClient.Delete(t.Context(), firstDraft))
			} else {
				require.NoError(t, kubeClient.Update(t.Context(), firstDraft))
			}

			t.Log("Project logical draft counts and make both authored components await the complete pair")
			readiness, err := dynamo.EvaluateLPXGroveReadiness(t.Context(), kubeClient, dgd, pcs, getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup))
			require.NoError(t, err)
			require.Len(t, readiness.ComponentStatuses, 2)
			draft, target := readiness.ComponentStatuses["draft"], readiness.ComponentStatuses["lpx"]
			require.Equal(t, v1beta1.ComponentKindPodClique, draft.ComponentKind)
			require.Len(t, draft.ComponentNames, 2)
			require.Equal(t, int32(1), target.Replicas)
			if scenario == "ready" {
				require.True(t, readiness.Ready)
				require.Equal(t, int32(2), draft.Replicas)
				require.Equal(t, int32(2), draft.UpdatedReplicas)
				require.Equal(t, ptr.To(int32(2)), draft.ReadyReplicas)
				require.Equal(t, ptr.To(int32(2)), draft.ScheduledReplicas)
				require.Equal(t, ptr.To(int32(1)), target.AvailableReplicas)
			} else {
				require.False(t, readiness.Ready)
				require.Equal(t, ptr.To(int32(1)), draft.ReadyReplicas)
				require.Equal(t, ptr.To(int32(0)), target.AvailableReplicas)
			}
			require.Equal(t, readiness.Ready, draft.Ready)
			require.Equal(t, readiness.Ready, target.Ready)
		})
	}
}

func newLPXTestScheme(t testing.TB) *k8sruntime.Scheme {
	t.Helper()
	scheme := k8sruntime.NewScheme()
	for _, add := range []func(*k8sruntime.Scheme) error{
		corev1.AddToScheme, resourcev1.AddToScheme, v1alpha1.AddToScheme, v1beta1.AddToScheme,
		grovev1alpha1.AddToScheme, lpxv1alpha1.AddToScheme,
	} {
		require.NoError(t, add(scheme))
	}
	return scheme
}

type snapshotFailureRegistry struct {
	lpx.ModelRegistry
	err error
}

func (r *snapshotFailureRegistry) AcquireBuildSnapshot(context.Context, string) (*lpx.BuildSnapshot, error) {
	return nil, r.err
}

type downloadOrderedLPXRegistry struct {
	lpx.ModelRegistry
	buildURL   url.URL
	downloaded bool
	calls      []string
}

func (r *downloadOrderedLPXRegistry) AcquireBuildSnapshot(
	ctx context.Context,
	buildID string,
) (*lpx.BuildSnapshot, error) {
	r.calls = append(r.calls, "snapshot")
	if !r.downloaded {
		return nil, errors.New("LPX snapshot acquired before Model Express download")
	}
	return r.ModelRegistry.AcquireBuildSnapshot(ctx, buildID)
}

func (r *downloadOrderedLPXRegistry) BuildURL(string) (*url.URL, error) {
	buildURL := r.buildURL
	return &buildURL, nil
}

func (r *downloadOrderedLPXRegistry) EnsureDownloaded(context.Context, url.URL) (bool, error) {
	r.calls = append(r.calls, "download")
	if len(r.calls) == 1 {
		return false, nil
	}
	r.downloaded = true
	return true, nil
}

func TestSelectedLPXColdCacheDownloadsBeforeSnapshot(t *testing.T) {
	t.Log("Build a selected LPX deployment backed by an initially cold Model Express cache")
	child, dgd, baseRegistry := newLPXTestDGD(t, lpx.PipelineSingle)
	registry := &downloadOrderedLPXRegistry{
		ModelRegistry: baseRegistry,
		buildURL: url.URL{
			Scheme: lpx.BuildSchemeGCS,
			Host:   "test-bucket",
			Path:   "/build-v2",
		},
	}
	reconciler := newLPXTestReconciler(t, registry, child, dgd)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("Check Model Express before acquiring the initial build snapshot")
	result, err := reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, modelDownloadRequeueAfter, result.RequeueAfter)
	require.Equal(t, []string{"download"}, registry.calls)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, "Ready"))
	require.NotNil(t, child.Status.ModelDownload)
	require.Empty(t, child.Status.ModelDownload.Builds)
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, reconciler.List(t.Context(), sets))
	require.Empty(t, sets.Items)

	t.Log("Complete the download and reconcile the selected deployment again")
	_, err = reconciler.Reconcile(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, []string{"download", "download", "snapshot"}, registry.calls)
	require.NoError(t, reconciler.Get(t.Context(), request.NamespacedName, child))
	require.NotNil(t, child.Status.ModelDownload)
	require.Equal(t, []string{registry.buildURL.String()}, child.Status.ModelDownload.Builds)
}

func TestLPXPublishedWorkloadCleanupAfterFailure(t *testing.T) {
	for _, test := range []struct {
		name          string
		snapshotError error
	}{
		{name: "inconsistent snapshot", snapshotError: fmt.Errorf("%w: compiler metadata changed during duplicate reads", lpx.ErrBuildSnapshotInconsistent)},
		{name: "transient snapshot", snapshotError: errors.New("temporary object-store timeout")},
		{name: "render failure"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish a request backed by the rendered LPX workload")
			ctx := t.Context()
			deployment, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, dgd)
			createLPXTestObjects(t, ctx, reconciler.Client, lpxMaterializedObjects(t, reconciler, deployment, dgd, desired)...)
			condition := publishSelectedLPXForTest(t, ctx, reconciler, deployment, desired)
			require.NotNil(t, condition)
			require.Equal(t, v1alpha1.LPXReadyReasonPending, condition.Reason)
			requestBefore := getTestPipelineRequest(t, ctx, reconciler.Client, deployment.Namespace, desired.requests[0].Name)
			key := client.ObjectKeyFromObject(deployment)
			pcsKey := client.ObjectKey{Namespace: deployment.Namespace, Name: desired.plan.PodCliqueSetName}
			pcsBefore := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, reconciler.Get(ctx, pcsKey, pcsBefore))

			t.Log("Fail the actual snapshot dependency or break cross-role model storage")
			message := "must use the Conductor model-storage mount path"
			if test.snapshotError != nil {
				reconciler.modelRegistry = &snapshotFailureRegistry{ModelRegistry: registry, err: test.snapshotError}
				message = test.snapshotError.Error()
			} else {
				require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
				conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
				for index := range conductor.PodTemplate.Spec.Containers[0].VolumeMounts {
					mount := &conductor.PodTemplate.Spec.Containers[0].VolumeMounts[index]
					if mount.Name == consts.ModelStorageVolumeName {
						mount.MountPath = "/different-model-storage"
					}
				}
				require.NoError(t, reconciler.Update(ctx, dgd))
				require.NoError(t, reconciler.Get(ctx, key, deployment))
				revision, revisionErr := dynamo.LPXInputRevision(dgd, "")
				require.NoError(t, revisionErr)
				deployment.Spec.InputRevision = revision
				require.NoError(t, reconciler.Update(ctx, deployment))
			}
			_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			pcs := &grovev1alpha1.PodCliqueSet{}
			t.Log("Failed desired input must preserve the exact published request and PCS")
			require.Equal(t, requestBefore, getTestPipelineRequest(t, ctx, reconciler.Client, deployment.Namespace, requestBefore.Name))
			require.NoError(t, reconciler.Get(ctx, pcsKey, pcs))
			require.Equal(t, pcsBefore, pcs)

			t.Log("Persist the actual actionable failure and retain its wrapped cause")
			require.ErrorContains(t, err, message)
			if test.snapshotError != nil {
				require.ErrorIs(t, err, test.snapshotError)
				require.ErrorIs(t, err, lpx.ErrBuildSnapshotAcquisition)
			}
			require.NoError(t, reconciler.Get(ctx, key, deployment))
			failed := meta.FindStatusCondition(deployment.Status.Conditions, "Ready")
			require.NotNil(t, failed)
			require.Equal(t, metav1.ConditionFalse, failed.Status)
			require.Equal(t, deployment.Generation, failed.ObservedGeneration)
			require.Equal(t, err.Error(), failed.Message)
		})
	}
}

func TestLPXPodCliqueSetListOrder(t *testing.T) {
	t.Log("Create a PCS through the controller with ordered Agent init containers")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	agent := lpx.ServingComponent(dgd).ComponentRole(v1beta1.ComponentRoleLPXAgent)
	agent.PodTemplate.Spec.InitContainers = []corev1.Container{
		{Name: "setup", Image: "busybox"}, {Name: "migrate", Image: "busybox"},
	}
	r := newLPXTestReconciler(t, registry, child, dgd)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	pcs, err := getPodCliqueSet(t.Context(), r.Client, child)
	require.NoError(t, err)
	require.NotNil(t, pcs)
	originalUID := pcs.UID

	t.Log("Reverse authored init order and verify the controller opts into order-sensitive synchronization")
	slices.Reverse(agent.PodTemplate.Spec.InitContainers)
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	pcs, err = getPodCliqueSet(t.Context(), r.Client, child)
	require.NoError(t, err)
	require.Equal(t, originalUID, pcs.UID)
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleAgent {
			require.Equal(t, agent.PodTemplate.Spec.InitContainers, clique.Spec.PodSpec.InitContainers)
		}
	}

	t.Log("Unchanged desired input does not write the PCS again")
	before := pcs.DeepCopy()
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	pcs, err = getPodCliqueSet(t.Context(), r.Client, child)
	require.NoError(t, err)
	require.Equal(t, before, pcs)
}

func TestSelectedNodeLocalLPXReadinessUsesOneFixedScalingGroup(t *testing.T) {
	for _, pipeline := range []lpx.Pipeline{lpx.PipelineSingle, lpx.PipelineLPX} {
		t.Run(string(pipeline), func(t *testing.T) {
			ctx := t.Context()
			deployment, dgd, registry := newLPXTestDGD(t, pipeline)
			if pipeline == lpx.PipelineLPX {
				dgd.Spec.Components[0].Replicas = ptr.To(int32(1))
				dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(2))
			}
			reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, dgd)
			objects := lpxMaterializedObjects(t, reconciler, deployment, dgd, desired)
			for _, object := range objects {
				switch live := object.(type) {
				case *grovev1alpha1.PodCliqueScalingGroup:
					live.Status.Replicas, live.Status.UpdatedReplicas = 1, 1
					live.Status.AvailableReplicas, live.Status.ScheduledReplicas = 1, 1
				case *grovev1alpha1.PodClique:
					live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
					live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
					require.Equal(t, desired.workload.LPXComponentName(), live.Labels[consts.KubeLabelDynamoComponent])
				}
			}
			createLPXTestObjects(t, ctx, reconciler.Client, objects...)
			pcs := findLPXTestPodCliqueSet(t, objects)
			rootReads := 0
			base, ok := reconciler.Client.(client.WithWatch)
			require.True(t, ok)
			reader := interceptor.NewClient(base, interceptor.Funcs{Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
				if _, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
					rootReads++
				}
				return reader.Get(ctx, key, object, opts...)
			}})
			observe := func() dynamo.GroveReadiness {
				group, err := getPodCliqueScalingGroup(ctx, reconciler.Client, pcs)
				require.NoError(t, err)
				ready, err := dynamo.EvaluateLPXGroveReadiness(ctx, reader, dgd, pcs, group)
				require.NoError(t, err)
				require.Len(t, ready.ComponentStatuses, 1)
				require.Zero(t, rootReads, "reuse the synchronized PCS observation")
				return ready
			}
			t.Log("Report one complete engine, never separate GPU or Agent component capacity")
			ready := observe()
			require.True(t, ready.Ready)
			status := ready.ComponentStatuses[desired.workload.LPXComponentName()]
			require.Equal(t, v1beta1.ComponentKindPodCliqueScalingGroup, status.ComponentKind)
			require.Equal(t, []string{desired.plan.LPXScalingGroup}, status.ComponentNames)
			require.Equal(t, int32(1), status.Replicas)
			require.Equal(t, int32(1), status.UpdatedReplicas)
			require.Equal(t, ptr.To(int32(1)), status.AvailableReplicas)
			require.Equal(t, ptr.To(int32(1)), status.ScheduledReplicas)
			if desired.plan.CyborgClique != "" {
				t.Log("A stale GPU-role scale cannot satisfy complete-engine readiness")
				gpu := getResource[*grovev1alpha1.PodClique](t, objects, desired.plan.CyborgClique).DeepCopy()
				require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(gpu), gpu))
				gpu.Spec.Replicas, gpu.Status.Replicas, gpu.Status.UpdatedReplicas = 1, 1, 1
				gpu.Status.ReadyReplicas, gpu.Status.ScheduledReplicas = 1, 1
				require.NoError(t, reconciler.Update(ctx, gpu))
				ready = observe()
				require.False(t, ready.Ready)
				require.Equal(t, v1beta1.DGDReadyReasonUpdating, ready.Classification)
				t.Log("Partial GPU readiness contributes zero complete engine replicas")
				gpu.Spec.Replicas, gpu.Status.Replicas, gpu.Status.UpdatedReplicas = 2, 2, 2
				gpu.Status.ScheduledReplicas = 2
				require.NoError(t, reconciler.Update(ctx, gpu))
				ready = observe()
				require.False(t, ready.Ready)
				require.Equal(t, v1beta1.DGDReadyReasonPodsNotReady, ready.Classification)
				require.Equal(t, ptr.To(int32(0)), ready.ComponentStatuses[desired.workload.LPXComponentName()].AvailableReplicas)
				gpu.Status.ReadyReplicas = 2
				require.NoError(t, reconciler.Update(ctx, gpu))
				require.True(t, observe().Ready)
			}
			t.Log("Reject stale complete-engine capacity even when every existing role is ready")
			group := &grovev1alpha1.PodCliqueScalingGroup{}
			require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: desired.plan.LPXScalingGroup}, group))
			group.Spec.Replicas = 2
			require.NoError(t, reconciler.Update(ctx, group))
			ready = observe()
			require.False(t, ready.Ready)
			require.Equal(t, v1beta1.DGDReadyReasonUpdating, ready.Classification)
		})
	}
}

func TestLPXDisabledPreservesPublishedWorkloadUntilDeletion(t *testing.T) {
	for _, scenario := range []struct{ name, message string }{
		{"LPX", "LPX integration is disabled"},
		{"Grove", "Grove is disabled"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Publish a complete workload, discovery service and scheduling request")
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
			dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			createLPXTestObjects(t, ctx, r.Client, lpxMaterializedObjects(t, r, child, dgd, selected)...)
			key := client.ObjectKeyFromObject(child)
			for range 4 {
				_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
				require.NoError(t, err)
			}
			require.NoError(t, r.Get(ctx, key, child))
			before := []client.ObjectList{
				&grovev1alpha1.PodCliqueSetList{}, &corev1.ConfigMapList{},
				&corev1.ServiceList{}, &lpxv1alpha1.LPUPipelineRequestList{},
			}
			for _, list := range before {
				require.NoError(t, r.List(ctx, list))
				require.Positive(t, meta.LenList(list))
			}

			t.Log("Disable the provider before an input edit and an expired deadline can retire the workload")
			if scenario.name == "LPX" {
				r.runtimeConfig.Gate.LPX = false
			} else {
				r.runtimeConfig.Gate.Grove = false
			}
			r.enabled = false
			r.modelRegistry = nil
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
			originalSpec := dgd.Spec.DeepCopy()
			dgd.Spec.Components[0].LPX.BuildID = "edited-while-disabled"
			require.NoError(t, r.Update(ctx, dgd))
			for range 2 {
				result, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
				require.NoError(t, err)
				require.Zero(t, result)
			}
			require.NoError(t, r.Get(ctx, key, child))
			failed := meta.FindStatusCondition(child.Status.Conditions, "Ready")
			require.NotNil(t, failed)
			require.Equal(t, metav1.ConditionFalse, failed.Status)
			require.Equal(t, v1alpha1.LPXReadyReasonFailed, failed.Reason)
			require.Equal(t, scenario.message, failed.Message)
			require.Equal(t, child.Generation, failed.ObservedGeneration)
			require.Len(t, child.Status.Conditions, 1)
			for _, list := range before {
				current := list.DeepCopyObject().(client.ObjectList)
				require.NoError(t, r.List(ctx, current))
				require.Equal(t, list, current)
			}

			t.Log("Re-enable unchanged intent and resume the original PCS and request identities")
			dgd.Spec = *originalSpec
			require.NoError(t, r.Update(ctx, dgd))
			r.runtimeConfig.Gate.LPX, r.runtimeConfig.Gate.Grove = true, true
			r.enabled = true
			r.modelRegistry = registry
			_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, key, child))
			require.NotEqual(t, v1alpha1.LPXReadyReasonFailed, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
			for _, list := range before {
				current := list.DeepCopyObject().(client.ObjectList)
				require.NoError(t, r.List(ctx, current))
				require.Equal(t, list, current)
			}

			t.Log("Deletion remains available while disabled and needs no controller finalizer")
			if scenario.name == "LPX" {
				r.runtimeConfig.Gate.LPX = false
			} else {
				r.runtimeConfig.Gate.Grove = false
			}
			r.modelRegistry = nil
			require.Empty(t, child.Finalizers)
			require.NoError(t, r.Delete(ctx, child))
			_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
			require.NoError(t, err)
			require.True(t, apierrors.IsNotFound(r.Get(ctx, key, &v1alpha1.LPXGraphDeployment{})))
		})
	}
}

// lpxTestWorkload is a fixture DSL, not reconciliation state.
type lpxTestWorkload struct {
	workload *lpx.SelectedWorkload
	plan     *lpx.MaterializationPlan
	requests []lpxv1alpha1.LPUPipelineRequest
}

func resolveLPXTestWorkload(t *testing.T, registry lpx.ModelRegistry, ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment,
) *lpxTestWorkload {
	t.Helper()
	workload, err := lpx.ResolveSelectedWorkload(ctx, dgd, registry)
	require.NoError(t, err)
	plan, err := workload.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(deployment))
	require.NoError(t, err)
	_, missing, changed := resolvePipelineRequests(deployment, nil, workload, plan)
	require.False(t, changed)
	fixture := &lpxTestWorkload{workload: workload, plan: plan}
	for _, request := range missing {
		fixture.requests = append(fixture.requests, *request)
	}
	return fixture
}

func newPreparedLPXTestReconciler(
	t *testing.T,
	registry lpx.ModelRegistry,
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
) (*graphReconciler, *lpxTestWorkload) {
	t.Helper()

	// Construct the fake client before preparing the selected LPX plan.
	reconciler := newLPXTestReconciler(t, registry, deployment, dgd)
	desired := resolveLPXTestWorkload(t, reconciler.modelRegistry, ctx, deployment, dgd)
	return reconciler, desired
}

func findLPXTestPodCliqueSet(t *testing.T, objects []client.Object) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	for _, object := range objects {
		if pcs, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
			return pcs
		}
	}
	t.Fatal("PodCliqueSet not found")
	return nil
}

func newLPXTestDGD(t *testing.T, pipeline lpx.Pipeline) (*v1alpha1.LPXGraphDeployment, *v1beta1.DynamoGraphDeployment, lpx.ModelRegistry) {
	t.Helper()
	compilationMode := manifestcapnpv2.CompilationMode_lpuOnly
	if pipeline == lpx.PipelineLPX {
		compilationMode = manifestcapnpv2.CompilationMode_lpx
	}
	const buildID = "build-v2"
	registry := newLPXTestRegistryWithPartitionsAndMode(t, buildID, []int{7, 8}, compilationMode)
	dgd := loadTestDGD(t, pipeline, buildID)
	return newLPXTestDeployment(t, dgd), dgd, registry
}

func newLPXTestDeployment(t *testing.T, dgd *v1beta1.DynamoGraphDeployment) *v1alpha1.LPXGraphDeployment {
	t.Helper()
	revision, err := dynamo.LPXInputRevision(dgd, "")
	require.NoError(t, err)
	return &v1alpha1.LPXGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: v1alpha1.GroupVersion.String(), Kind: "LPXGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{Name: dgd.Name, Namespace: dgd.Namespace, UID: types.UID("lpx-" + string(dgd.UID)), Generation: 1,
			Annotations:     map[string]string{lpx.DGDGenerationAnnotation: strconv.FormatInt(dgd.Generation, 10)},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)},
		},
		Spec: v1alpha1.LPXGraphDeploymentSpec{InputRevision: revision},
	}
}

func loadTestDGD(t testing.TB, pipeline lpx.Pipeline, buildID string) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	payload, err := os.ReadFile("testdata/dgd.yaml")
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.UnmarshalStrict(payload, dgd))
	dgd.Spec.Components[0].LPX.BuildID = buildID

	// Hybrid pipelines author a GPU conductor while retaining the same Agent template.
	if pipeline == lpx.PipelineLPX {
		payload, err = os.ReadFile("testdata/hybrid-conductor.yaml")
		require.NoError(t, err)
		conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
		*conductor = v1beta1.ComponentRoleSpec{}
		require.NoError(t, yaml.UnmarshalStrict(payload, conductor))
	}
	return dgd
}

func newLPXSpecDecodeTestDGD(t *testing.T) (*v1alpha1.LPXGraphDeployment, *v1beta1.DynamoGraphDeployment, lpx.ModelRegistry) {
	t.Helper()
	dgd := loadTestSpecDecodeDGD(t)
	root := t.TempDir()
	writeLPXTestBuild(t, root, "draft-build", []int{7, 8}, manifestcapnpv2.CompilationMode_lpuOnly)
	writeLPXTestBuild(t, root, "target-build", []int{9, 10}, manifestcapnpv2.CompilationMode_lpuOnly)
	registryURL := (&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String()
	registry, err := lpx.NewModelRegistry(registryURL, nil)
	require.NoError(t, err)
	return newLPXTestDeployment(t, dgd), dgd, registry
}

func loadTestSpecDecodeDGD(t testing.TB) *v1beta1.DynamoGraphDeployment {
	dgd := loadTestDGD(t, lpx.PipelineSingle, "target-build")
	target := &dgd.Spec.Components[0]
	draft := target.DeepCopy()
	draft.ComponentName, draft.LPX.BuildID, draft.Replicas = "draft", "draft-build", ptr.To(int32(2))
	draft.Roles = []v1beta1.ComponentRoleSpec{*draft.ComponentRole(v1beta1.ComponentRoleLPXAgent)}
	dgd.Spec.Components = append(dgd.Spec.Components, *draft)
	return dgd
}

// newLPXTestClient shares schemes, indexes and the fake client's missing API-server behavior.
func newLPXTestClient(t testing.TB, objects ...client.Object) client.WithWatch {
	t.Helper()
	scheme := newLPXTestScheme(t)
	base := fake.NewClientBuilder().
		WithScheme(scheme).
		WithIndex(&lpxv1alpha1.LPUPipelineRequest{}, pipelineRequestPCSOwnerUIDIndex, pipelineRequestOwnerUID).
		WithIndex(&v1alpha1.LPXGraphDeployment{}, dgdControllerOwnerIndex, dgdControllerOwnerKey).
		WithStatusSubresource(&v1beta1.DynamoGraphDeployment{}, &v1alpha1.LPXGraphDeployment{}, &corev1.Pod{}).
		WithObjects(objects...).
		Build()
	nextLPRUID := 0
	return interceptor.NewClient(base, interceptor.Funcs{
		// The fake client checks resource versions but does not enforce Delete UID preconditions.
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			options := (&client.DeleteOptions{}).ApplyOptions(opts)
			if options.Preconditions != nil && options.Preconditions.UID != nil {
				observed := object.DeepCopyObject().(client.Object)
				if err := delegated.Get(ctx, client.ObjectKeyFromObject(object), observed); err != nil {
					return err
				}
				if observed.GetUID() != *options.Preconditions.UID {
					return apierrors.NewConflict(schema.GroupResource{Resource: object.GetObjectKind().GroupVersionKind().Kind}, object.GetName(), errors.New("UID precondition failed"))
				}
			}
			return delegated.Delete(ctx, object, opts...)
		},
		// The fake client does not implement scale for Grove custom resources.
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if group, ok := object.(*grovev1alpha1.PodCliqueScalingGroup); ok && subresource == "scale" {
				options := (&client.SubResourceUpdateOptions{}).ApplyOptions(opts)
				scale := options.SubResourceBody.(*autoscalingv1.Scale)
				group.ResourceVersion = scale.ResourceVersion
				group.Spec.Replicas = scale.Spec.Replicas
				return delegated.Update(ctx, group)
			}
			return delegated.SubResource(subresource).Update(ctx, object, opts...)
		},
		Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
			if pcs, ok := object.(*grovev1alpha1.PodCliqueSet); ok && pcs.UID == "" {
				pcs.UID = types.UID(pcs.Name + "-uid")
				pcs.Generation = 1
			}
			if request, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok {
				if request.UID == "" {
					nextLPRUID++
					request.UID = types.UID(fmt.Sprintf("lpr-%s-%d", request.Name, nextLPRUID))
				}
				if request.Generation == 0 {
					request.Generation = 1
				}
				if request.CreationTimestamp.IsZero() {
					request.CreationTimestamp = metav1.NewTime(time.Now().UTC().Truncate(time.Second))
				}
			}
			return delegated.Create(ctx, object, opts...)
		},
	})
}

func newLPXTestReconciler(
	t *testing.T,
	registry lpx.ModelRegistry,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	objects ...client.Object,
) *graphReconciler {
	t.Helper()
	revision, err := dynamo.LPXInputRevision(dgd, "")
	require.NoError(t, err)
	deployment.Spec.InputRevision = revision
	if deployment.ResourceVersion == "" {
		deployment.ResourceVersion = "1"
	}
	seed := append([]client.Object{deployment.DeepCopy(), dgd.DeepCopy()}, objects...)
	recorder := events.NewFakeRecorder(100)
	config := &configv1alpha1.OperatorConfiguration{
		LPX: configv1alpha1.LPXConfiguration{Enabled: true},
	}
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, DRA: true, LPX: true}}
	return &graphReconciler{
		Client:        newLPXTestClient(t, seed...),
		recorder:      recorder,
		enabled:       true,
		runtimeConfig: runtimeConfig,
		modelRegistry: registry,
		config:        config,
	}
}

func lpxMaterializedObjects(
	t *testing.T,
	reconciler *graphReconciler,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	desired *lpxTestWorkload,
) []client.Object {
	t.Helper()
	t.Log("Render LPX intent once and emulate the API-server and Grove observations")
	pcs := renderLPXTestPodCliqueSet(t, t.Context(), reconciler, deployment, dgd, desired)
	pcs.TypeMeta = metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueSet"}
	pcs.UID, pcs.Generation = "pcs-uid", 1
	pcs.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(deployment, v1alpha1.LPXGraphDeploymentGVK)}
	const generationHash = "pcs-generation-hash"
	pcs.Status = grovev1alpha1.PodCliqueSetStatus{
		ObservedGeneration: ptr.To(pcs.Generation), CurrentGenerationHash: ptr.To(generationHash),
	}

	// Emulate Grove materialization after the controller applies the desired live scale.
	groupTemplate := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].DeepCopy()
	group := &grovev1alpha1.PodCliqueScalingGroup{
		TypeMeta: metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup"},
		ObjectMeta: metav1.ObjectMeta{
			Name: desired.plan.LPXScalingGroup, Namespace: pcs.Namespace, UID: "lpu-group-uid",
			Generation: 1, Labels: grovecommon.GetDefaultLabelsForPodCliqueSetManagedResources(pcs.Name), Annotations: groupTemplate.Annotations,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
		},
		Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
			Replicas: desired.plan.Replicas, MinAvailable: groupTemplate.MinAvailable,
			CliqueNames: groupTemplate.CliqueNames,
		},
		Status: grovev1alpha1.PodCliqueScalingGroupStatus{
			ObservedGeneration: ptr.To(int64(1)), CurrentPodCliqueSetGenerationHash: ptr.To(generationHash),
		},
	}
	group.Labels[grovecommon.LabelPartOfKey] = pcs.Name
	group.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "0"
	objects := []client.Object{pcs, group}

	// Each Grove replica owns independent cliques.
	for replicaIndex := int32(0); replicaIndex < group.Spec.Replicas; replicaIndex++ {
		parent := grovecommon.ResourceNameReplica{Name: group.Name, Replica: int(replicaIndex)}
		for _, template := range pcs.Spec.Template.Cliques {
			rendered := template.DeepCopy()
			name := grovecommon.GeneratePodCliqueName(parent, rendered.Name)
			clique := &grovev1alpha1.PodClique{
				TypeMeta: metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodClique"},
				ObjectMeta: metav1.ObjectMeta{
					Name: name, Namespace: pcs.Namespace, UID: types.UID(name + "-uid"), Generation: 1,
					Labels: rendered.Labels, Annotations: rendered.Annotations,
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(group, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
				},
				Spec: rendered.Spec,
				Status: grovev1alpha1.PodCliqueStatus{
					ObservedGeneration: ptr.To(int64(1)), CurrentPodCliqueSetGenerationHash: ptr.To(generationHash),
					CurrentPodTemplateHash: ptr.To(rendered.Name + "-pod-template-hash"),
				},
			}
			clique.Labels[grovecommon.LabelPartOfKey] = pcs.Name
			clique.Labels[grovecommon.LabelPodCliqueScalingGroup] = group.Name
			clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] = strconv.FormatInt(int64(replicaIndex), 10)
			for index, dependency := range clique.Spec.StartsAfter {
				clique.Spec.StartsAfter[index] = grovecommon.GeneratePodCliqueName(parent, dependency)
			}
			objects = append(objects, clique)
		}
	}
	return objects
}

func createLPXTestObjects(t *testing.T, ctx context.Context, kubeClient client.Client, objects ...client.Object) {
	t.Helper()
	for _, object := range objects {
		require.NoError(t, kubeClient.Create(ctx, object.DeepCopyObject().(client.Object)))
	}
}

func observedLPXTestPodCliqueSet(
	t *testing.T,
	ctx context.Context,
	reconciler *graphReconciler,
	deployment *v1alpha1.LPXGraphDeployment,
	desired *lpxTestWorkload,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	if desired.plan == nil {
		return nil
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	err := reconciler.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: desired.plan.PodCliqueSetName}, pcs)
	if apierrors.IsNotFound(err) {
		return nil
	}
	require.NoError(t, err)
	return pcs
}

func TestLPXQueueEditSynchronizesPCSMetadata(t *testing.T) {
	t.Log("Keep PCS queue metadata and clique templates consistent across DGD queue edits")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Labels = map[string]string{consts.KubeLabelKaiSchedulerQueue: "queue-a"}
	for _, component := range lpx.Components(dgd) {
		for _, role := range component.Roles {
			metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, consts.KubeLabelKaiSchedulerQueue, "queue-a")
		}
	}
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	publishSelectedLPXForTest(t, t.Context(), r, child, selected)
	key := client.ObjectKeyFromObject(findLPXTestPodCliqueSet(t, objects))
	req := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(t.Context(), req)
	require.NoError(t, err)
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(t.Context(), key, pcs))
	metav1.SetMetaDataLabel(&pcs.ObjectMeta, "external.example/label", "preserved")
	metav1.SetMetaDataAnnotation(&pcs.ObjectMeta, "external.example/annotation", "preserved")
	require.NoError(t, r.Update(t.Context(), pcs))
	beforeUID := pcs.UID

	t.Log("Apply new queue metadata and templates in place while preserving untracked metadata")
	dgd.Spec.Labels[consts.KubeLabelKaiSchedulerQueue] = "queue-b"
	for _, component := range lpx.Components(dgd) {
		for _, role := range component.Roles {
			metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, consts.KubeLabelKaiSchedulerQueue, "queue-b")
		}
	}
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(t.Context(), req)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key, pcs))
	require.Equal(t, beforeUID, pcs.UID)
	require.Equal(t, "preserved", pcs.Labels["external.example/label"])
	require.Equal(t, "preserved", pcs.Annotations["external.example/annotation"])
	require.Equal(t, "queue-b", pcs.Labels[consts.KubeLabelKaiSchedulerQueue])
	for _, clique := range pcs.Spec.Template.Cliques {
		require.Equal(t, "queue-b", clique.Labels[consts.KubeLabelKaiSchedulerQueue])
	}

	t.Log("Remove an omitted managed queue from both PCS metadata and clique templates")
	delete(dgd.Spec.Labels, consts.KubeLabelKaiSchedulerQueue)
	for _, component := range lpx.Components(dgd) {
		for _, role := range component.Roles {
			delete(role.PodTemplate.Labels, consts.KubeLabelKaiSchedulerQueue)
		}
	}
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(t.Context(), req)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key, pcs))
	require.NotContains(t, pcs.Labels, consts.KubeLabelKaiSchedulerQueue)
	for _, clique := range pcs.Spec.Template.Cliques {
		require.NotContains(t, clique.Labels, consts.KubeLabelKaiSchedulerQueue)
	}

	t.Log("An unchanged reconciliation does not write the PCS again")
	beforeVersion := pcs.ResourceVersion
	_, err = r.Reconcile(t.Context(), req)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key, pcs))
	require.Equal(t, beforeVersion, pcs.ResourceVersion)
	require.Equal(t, "preserved", pcs.Labels["external.example/label"])
	require.Equal(t, "preserved", pcs.Annotations["external.example/annotation"])
}

func TestLPXEditBetweenFailureAndCleanupPreservesRetry(t *testing.T) {
	t.Log("Persist a deadline failure before the user edits the input to retry")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	publishSelectedLPXForTest(t, t.Context(), r, child, selected)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pending := getTestPipelineRequest(t, t.Context(), r.Client, child.Namespace, selected.requests[0].Name)
	pending.Status = newTestPipelineRequest(child, pcs, pending.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(t.Context(), pending))
	req := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(t.Context(), req)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), req.NamespacedName, child))
	oldFailure := meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition).DeepCopy()
	require.True(t, isSchedulingFailedConditionCurrent(child))

	t.Log("The edited generation uses the old failure to retire the already-expired cycle")
	dgd.Spec.Scheduling.AttemptDeadlineSeconds = ptr.To[int64](31)
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(t.Context(), req)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), req.NamespacedName, child))
	newFailure := meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition)
	require.Equal(t, oldFailure, newFailure)
	require.Greater(t, child.Generation, newFailure.ObservedGeneration)
	requirePipelineRequestNotFound(t, t.Context(), r.Client, pending.Namespace, pending.Name)
	t.Log("The same single edit permits republication after cleanup")
	for range 3 {
		_, err = r.Reconcile(t.Context(), req)
		require.NoError(t, err)
	}
	replacement := getTestPipelineRequest(t, t.Context(), r.Client, pending.Namespace, pending.Name)
	require.NotEqual(t, pending.UID, replacement.UID)
}

func TestLPXCommittedDeadlineCleanupPreservesExternalCapacity(t *testing.T) {
	t.Log("Publish an externally scaled request with a committed Pod and scheduler finalizer")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].Replicas = nil
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := findLPXTestPodCliqueSet(t, objects)
	request := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
	request.Finalizers = []string{"test.example/scheduler-release"}
	request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhaseBinding).Status
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: child.Namespace, Name: "committed-agent", UID: "agent-uid"}}
	request.Status.Committed = committedLPXTestPod(pod)
	require.NoError(t, r.Update(ctx, request))
	require.NoError(t, r.Create(ctx, pod))
	reconcileRequest := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("Persist failure without deleting the request or its Pod")
	_, err := r.Reconcile(ctx, reconcileRequest)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{}))
	request = getTestPipelineRequest(t, ctx, r.Client, request.Namespace, request.Name)
	require.True(t, request.DeletionTimestamp.IsZero())

	t.Log("Request deletion never authorizes direct Pod or PodClique deletion")
	failRequestDelete := true
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			switch object.(type) {
			case *corev1.Pod, *grovev1alpha1.PodClique:
				t.Fatalf("graph reconciliation must not delete %T", object)
			}
			if _, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok && failRequestDelete {
				return errors.New("request deletion temporarily unavailable")
			}
			return delegated.Delete(ctx, object, opts...)
		},
	})
	_, err = r.Reconcile(ctx, reconcileRequest)
	require.ErrorContains(t, err, "request deletion temporarily unavailable")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{}))
	request = getTestPipelineRequest(t, ctx, r.Client, request.Namespace, request.Name)
	require.True(t, request.DeletionTimestamp.IsZero())

	t.Log("Start request deletion and leave Pod cleanup to Grove and garbage collection")
	failRequestDelete = false
	_, err = r.Reconcile(ctx, reconcileRequest)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{}))
	request = getTestPipelineRequest(t, ctx, r.Client, request.Namespace, request.Name)
	require.False(t, request.DeletionTimestamp.IsZero())
	require.NotEmpty(t, request.Finalizers)

	t.Log("Registry failure does not authorize Pod cleanup for a terminating request")
	r.modelRegistry = &snapshotFailureRegistry{ModelRegistry: registry, err: errors.New("registry unavailable")}
	_, err = r.Reconcile(ctx, reconcileRequest)
	require.ErrorContains(t, err, "registry unavailable")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{}))
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, group))
	require.Equal(t, selected.plan.Replicas, group.Spec.Replicas)
	request = getTestPipelineRequest(t, ctx, r.Client, request.Namespace, request.Name)
	require.NotEmpty(t, request.Finalizers)

	t.Log("Wait for scheduler finalization without escalating cleanup or republishing")
	r.modelRegistry = registry
	_, err = r.Reconcile(ctx, reconcileRequest)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{}))
	request = getTestPipelineRequest(t, ctx, r.Client, request.Namespace, request.Name)
	require.False(t, request.DeletionTimestamp.IsZero())
	require.NotEmpty(t, request.Finalizers)
}

func TestLPXCommittedReleasePreservesPods(t *testing.T) {
	for _, release := range []bool{false, true} {
		t.Run(fmt.Sprintf("release=%t", release), func(t *testing.T) {
			t.Log("Observe a committed Degraded request with either repair or durable release authority")
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			publishSelectedLPXForTest(t, ctx, r, child, selected)
			request := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
			request.Status = newTestPipelineRequest(child, findLPXTestPodCliqueSet(t, objects), request.Name, time.Now(), lpxv1alpha1.RequestPhaseDegraded).Status
			pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: child.Namespace, Name: "committed-agent", UID: "agent-uid"}}
			request.Status.Committed = committedLPXTestPod(pod)
			if release {
				request.Status.Committed.Execution.Release = &lpxv1alpha1.ReleaseJournal{Reason: lpxv1alpha1.ReleaseReasonDependencyChanged, RequestedAtGeneration: request.Generation}
			}
			require.NoError(t, r.Update(ctx, request))
			require.NoError(t, r.Create(ctx, pod))

			t.Log("Neither repair nor release authorizes graph reconciliation to delete Pods")
			_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{}))
			request = getTestPipelineRequest(t, ctx, r.Client, request.Namespace, request.Name)
			require.True(t, request.DeletionTimestamp.IsZero())
			group := &grovev1alpha1.PodCliqueScalingGroup{}
			require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, group))
			require.Equal(t, selected.plan.Replicas, group.Spec.Replicas)
		})
	}
}

func committedLPXTestPod(pod *corev1.Pod) *lpxv1alpha1.Committed {
	return &lpxv1alpha1.Committed{
		Plan: lpxv1alpha1.CommittedPlan{Placement: lpxv1alpha1.PlanPlacement{ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal}},
		Execution: lpxv1alpha1.CommittedExecution{AcceptedGeneration: ptr.To[int64](1), NodeLocal: &lpxv1alpha1.NodeLocalExecution{
			PartitionSelections: []lpxv1alpha1.NodeLocalPartitionExecution{{SelectedRows: []lpxv1alpha1.NodeLocalRowExecution{{
				Current: &lpxv1alpha1.CurrentBinding{PodRef: lpxv1alpha1.ObjectReference{Namespace: pod.Namespace, Name: pod.Name, UID: string(pod.UID)}},
			}}}},
		}},
	}
}
