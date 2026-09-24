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
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/yaml"
)

func TestIndependentLPXRoleScalingAndReadiness(t *testing.T) {
	t.Log("Materialize two hybrid workloads, each with two backbones")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))
	second := dgd.Spec.Components[0].DeepCopy()
	second.ComponentName = "second"
	second.ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(3))
	dgd.Spec.Components = append(dgd.Spec.Components, *second)
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	pcs, _, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
	require.NoError(t, err)
	objects := materializeLPXTestPCS(t, child, pcs, plans["lpx"], plans[second.ComponentName])
	for _, object := range objects {
		switch live := object.(type) {
		case *grovev1alpha1.PodCliqueScalingGroup:
			live.Status.Replicas, live.Status.UpdatedReplicas = 2, 2
			live.Status.AvailableReplicas, live.Status.ScheduledReplicas = 2, 2
		case *grovev1alpha1.PodClique:
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
		}
	}
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 3 {
		_, err = r.Reconcile(t.Context(), key)
		require.NoError(t, err)
	}

	t.Log("Accept scheduler requests and observe each workload's initial Cyborg capacity")
	requests, err := r.getPipelineRequests(t.Context(), pcs)
	require.NoError(t, err)
	require.Len(t, requests, 4)
	requestUIDs := make(map[string]types.UID)
	for _, request := range requests {
		requestUIDs[request.Name] = request.UID
		request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
		require.NoError(t, r.Update(t.Context(), request))
	}
	for _, object := range objects {
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(object), object))
		if clique, ok := object.(*grovev1alpha1.PodClique); ok && clique.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleCyborgWorker {
			want := int32(1)
			if clique.Labels[lpx.StageLabel] == second.ComponentName {
				want = 3
			}
			require.Equal(t, want, clique.Spec.Replicas)
			clique.Status.ObservedGeneration = ptr.To(clique.Generation)
			clique.Status.Replicas, clique.Status.UpdatedReplicas = want, want
			clique.Status.ReadyReplicas, clique.Status.ScheduledReplicas = want, want
			require.NoError(t, r.Update(t.Context(), clique))
		}
	}
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, v1alpha1.LPXReadyCondition))

	for _, replicas := range []*int32{ptr.To(int32(3)), ptr.To(int32(1)), nil} {
		t.Log("Change only the first workload's Cyborg count, then wait for its workers")
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
		dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = replicas
		require.NoError(t, r.Update(t.Context(), dgd))
		child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
		require.NoError(t, err)
		child.Generation++
		require.NoError(t, r.Update(t.Context(), child))
		want := ptr.Deref(replicas, 4)
		if replicas == nil {
			t.Log("With an omitted count, external scaling may choose a different capacity on each backbone")
			for ordinal := range plans["lpx"].Replicas {
				clique := &grovev1alpha1.PodClique{}
				require.NoError(t, r.Get(t.Context(), client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(ordinal).CyborgClique}, clique))
				clique.Spec.Replicas = want + ordinal
				clique.Generation++
				require.NoError(t, r.Update(t.Context(), clique))
			}
		}
		_, err = r.Reconcile(t.Context(), key)
		require.NoError(t, err)

		t.Log("Observe the PCS template before scaling existing workers")
		_, err = r.Reconcile(t.Context(), key)
		require.NoError(t, err)

		t.Log("Observe the capacity write before calculating each workload's readiness")
		_, err = r.Reconcile(t.Context(), key)
		require.NoError(t, err)
		require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
		require.True(t, meta.IsStatusConditionFalse(child.Status.Components["lpx"].Conditions, v1alpha1.LPXReadyCondition))
		require.True(t, meta.IsStatusConditionTrue(child.Status.Components[second.ComponentName].Conditions, v1alpha1.LPXReadyCondition))

		t.Log("Persist the Cyborg template count without changing PCS identity or the other templates")
		expectedPCS := pcs.DeepCopy()
		for _, template := range expectedPCS.Spec.Template.Cliques {
			if template.Name == plans["lpx"].CyborgTemplate {
				template.Spec.Replicas = ptr.Deref(replicas, 1)
			}
		}
		actualPCS := &grovev1alpha1.PodCliqueSet{}
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), actualPCS))
		require.Equal(t, pcs.UID, actualPCS.UID)
		require.True(t, actualPCS.DeletionTimestamp.IsZero())
		require.Equal(t, pcs.Annotations[lpx.WorkloadDigestAnnotation], actualPCS.Annotations[lpx.WorkloadDigestAnnotation])
		require.Equal(t, expectedPCS.Spec, actualPCS.Spec)

		t.Log("Preserve both groups, all Agents, the other workload's workers and scheduler requests")
		for _, original := range objects {
			if _, ok := original.(*grovev1alpha1.PodCliqueSet); ok {
				continue
			}
			if clique, ok := original.(*grovev1alpha1.PodClique); ok && clique.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleCyborgWorker && clique.Labels[lpx.StageLabel] == "lpx" {
				continue
			}
			actual := original.DeepCopyObject().(client.Object)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(original), actual))
			require.Equal(t, original, actual)
		}
		requests, err = r.getPipelineRequests(t.Context(), pcs)
		require.NoError(t, err)
		require.Len(t, requests, 4)
		for _, request := range requests {
			require.Equal(t, requestUIDs[request.Name], request.UID)
		}

		t.Log("Observe the requested live capacity and prove Ready without repeating scale writes")
		var workers []*grovev1alpha1.PodClique
		for ordinal := range plans["lpx"].Replicas {
			clique := &grovev1alpha1.PodClique{}
			require.NoError(t, r.Get(t.Context(), client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(ordinal).CyborgClique}, clique))
			expected := want
			if replicas == nil {
				expected += ordinal
			}
			require.Equal(t, expected, clique.Spec.Replicas)
			require.Equal(t, ptr.To(int32(1)), clique.Spec.MinAvailable)
			clique.Status.ObservedGeneration = ptr.To(clique.Generation)
			clique.Status.Replicas, clique.Status.UpdatedReplicas = expected, expected
			clique.Status.ReadyReplicas, clique.Status.ScheduledReplicas = expected, expected
			require.NoError(t, r.Update(t.Context(), clique))
			workers = append(workers, clique)
		}
		_, err = r.Reconcile(t.Context(), key)
		require.NoError(t, err)
		require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
		require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, v1alpha1.LPXReadyCondition))
		for _, worker := range workers {
			actual := &grovev1alpha1.PodClique{}
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(worker), actual))
			require.Equal(t, worker, actual, "converged capacity must not write again")
		}
	}

	t.Log("Allow a user to resize an Agent clique while its new pods become ready")
	agent := &grovev1alpha1.PodClique{}
	agentKey := client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(1).Agents[0].CliqueName}
	require.NoError(t, r.Get(t.Context(), agentKey, agent))
	agent.Spec.Replicas++
	agent.Generation++
	require.NoError(t, r.Update(t.Context(), agent))
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionFalse(child.Status.Components["lpx"].Conditions, v1alpha1.LPXReadyCondition))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Components[second.ComponentName].Conditions, v1alpha1.LPXReadyCondition))
	observedAgent := &grovev1alpha1.PodClique{}
	require.NoError(t, r.Get(t.Context(), agentKey, observedAgent))
	require.Equal(t, agent, observedAgent, "reconciliation must preserve user-managed Agent capacity")

	t.Log("Accept the edited Agent capacity once Grove reports its live replica count ready")
	agent.Status.ObservedGeneration = ptr.To(agent.Generation)
	agent.Status.Replicas, agent.Status.UpdatedReplicas = agent.Spec.Replicas, agent.Spec.Replicas
	agent.Status.ReadyReplicas, agent.Status.ScheduledReplicas = agent.Spec.Replicas, agent.Spec.Replicas
	require.NoError(t, r.Update(t.Context(), agent))
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, v1alpha1.LPXReadyCondition))
	require.NoError(t, r.Get(t.Context(), agentKey, observedAgent))
	require.Equal(t, agent, observedAgent)
}

func TestLPXExternalCyborgCapacityValidation(t *testing.T) {
	t.Log("Use two explicitly managed backbones with externally managed workers in groups of four")
	root := t.TempDir()
	const buildID = "split-io"
	writeTestGraphBuild(t, root, buildID, testV2GraphManifestCapnp(t, buildID, testV2GraphManifestFixture{
		topology:       "URSA_V2_1__Q8__8C__G_96_25__KP_FEC__GHZ_1_0__DRACO_V1_1__G_106",
		partitionCount: 1, numChips: 8, devicesPerNode: 8,
		compilationMode:   manifestcapnpv2.CompilationMode_lpx,
		nonLPUDeviceTypes: []manifestcapnpv2.DeviceType{manifestcapnpv2.DeviceType_cuda},
		ioFPGACount:       2, ioFanoutFactor: 2,
	}))
	registry, err := lpx.NewModelRegistry(root, nil)
	require.NoError(t, err)
	dgd := loadTestDGD(t, lpx.PipelineLPX, buildID)
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))
	dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = nil
	child := newLPXTestDeployment(t, dgd)
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	pcs, _, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
	require.NoError(t, err)
	objects := materializeLPXTestPCS(t, child, pcs, plans["lpx"])
	for _, object := range objects {
		switch live := object.(type) {
		case *grovev1alpha1.PodCliqueScalingGroup:
			live.Status.Replicas, live.Status.UpdatedReplicas = 2, 2
			live.Status.AvailableReplicas, live.Status.ScheduledReplicas = 2, 2
		case *grovev1alpha1.PodClique:
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
		}
	}

	t.Log("Leave the second backbone's worker clique missing while scheduler requests become bound")
	worker := getResource[*grovev1alpha1.PodClique](t, objects, plans["lpx"].ForReplica(1).CyborgClique)
	objects = slices.DeleteFunc(objects, func(object client.Object) bool { return object.GetName() == worker.Name })
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 2 {
		_, err = r.Reconcile(t.Context(), key)
		require.NoError(t, err)
	}
	requests, err := r.getPipelineRequests(t.Context(), pcs)
	require.NoError(t, err)
	require.Len(t, requests, 2)
	for _, request := range requests {
		request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
		require.NoError(t, r.Update(t.Context(), request))
	}
	result, err := r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	require.Zero(t, result, "a missing clique waits for its watch")
	require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
	ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
	require.NotNil(t, ready)
	require.Equal(t, metav1.ConditionFalse, ready.Status)
	require.Equal(t, v1alpha1.LPXReadyReasonPending, ready.Reason)
	require.Contains(t, ready.Message, worker.Name)

	t.Log("Observe the missing clique and preserve the first backbone's independent worker capacity")
	require.NoError(t, r.Create(t.Context(), worker))
	firstWorker := &grovev1alpha1.PodClique{}
	firstWorkerKey := client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(0).CyborgClique}
	require.NoError(t, r.Get(t.Context(), firstWorkerKey, firstWorker))
	for _, step := range []struct {
		replicas  int32
		wantError string
	}{
		{replicas: 8},
		{replicas: 5, wantError: "must be divisible by ioFpgaCount 2"},
		{replicas: 6, wantError: "must provide fanoutFactor 2 clients"},
		{replicas: 12},
	} {
		t.Logf("Externally scale the second backbone to %d workers and report all of them ready", step.replicas)
		worker.Spec.Replicas = step.replicas
		worker.Generation++
		worker.Status.ObservedGeneration = ptr.To(worker.Generation)
		worker.Status.Replicas, worker.Status.UpdatedReplicas = step.replicas, step.replicas
		worker.Status.ReadyReplicas, worker.Status.ScheduledReplicas = step.replicas, step.replicas
		require.NoError(t, r.Update(t.Context(), worker))
		_, err = r.Reconcile(t.Context(), key)
		if step.wantError != "" {
			require.ErrorContains(t, err, step.wantError)
		} else {
			require.NoError(t, err)
		}
		require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
		ready = meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
		require.NotNil(t, ready)
		if step.wantError != "" {
			require.Equal(t, metav1.ConditionFalse, ready.Status)
			require.Equal(t, v1alpha1.LPXReadyReasonFailed, ready.Reason)
			require.Contains(t, ready.Message, step.wantError)
		} else {
			require.Equal(t, metav1.ConditionTrue, ready.Status)
		}

		t.Log("Validation must not write either externally managed worker clique")
		observed := &grovev1alpha1.PodClique{}
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(worker), observed))
		require.Equal(t, worker, observed)
		require.NoError(t, r.Get(t.Context(), firstWorkerKey, observed))
		require.Equal(t, firstWorker, observed)
	}
}

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
				dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](60)}
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

func TestLPXSharedWorkloadComponentDeadlines(t *testing.T) {
	for _, tc := range []struct {
		name          string
		target, draft *int64
		wantSeconds   int64
		wantFailed    bool
	}{
		{name: "target wakes first", target: ptr.To(int64(120)), draft: ptr.To(int64(300)), wantSeconds: 120},
		{name: "expanded drafts wake first", target: ptr.To(int64(300)), draft: ptr.To(int64(120)), wantSeconds: 120},
		{name: "unlimited target", draft: ptr.To(int64(120)), wantSeconds: 120},
		{name: "unlimited drafts", target: ptr.To(int64(120)), wantSeconds: 120},
		{name: "all unlimited"},
		{name: "only target expired", target: ptr.To(int64(30)), draft: ptr.To(int64(300)), wantFailed: true},
		{name: "only drafts expired", target: ptr.To(int64(300)), draft: ptr.To(int64(30)), wantFailed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Publish target and two draft requests with the same scheduling start")
			child, dgd, registry := newLPXSpecDecodeTestDGD(t)
			r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, t.Context(), r.Client, objects...)
			publishSelectedLPXForTest(t, t.Context(), r, child, selected)
			pcs := findLPXTestPodCliqueSet(t, objects)
			started := time.Now().UTC().Truncate(time.Second).Add(-time.Minute)
			requests, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Len(t, requests, 3)
			for _, request := range requests {
				request.Status = newTestPipelineRequest(child, pcs, request.Name, started, lpxv1alpha1.RequestPhasePending).Status
				require.NoError(t, r.Update(t.Context(), request))
			}

			t.Log("Edit component policies without replacing requests or resetting their clocks")
			dgd.GetComponentByName("lpx").LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: tc.target}
			dgd.GetComponentByName("draft").LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: tc.draft}
			dgd.Spec.Components[0], dgd.Spec.Components[1] = dgd.Spec.Components[1], dgd.Spec.Components[0]
			updateTestDGD(t, r, child, dgd)
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			result, err := r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
			require.Equal(t, tc.wantFailed, meta.IsStatusConditionTrue(child.Status.Conditions, schedulingFailedCondition))
			if tc.wantFailed {
				require.Positive(t, result.RequeueAfter)
			} else if tc.wantSeconds != 0 {
				require.WithinDuration(t, started.Add(time.Duration(tc.wantSeconds)*time.Second), time.Now().Add(result.RequeueAfter), time.Second)
			} else {
				require.Zero(t, result.RequeueAfter)
			}
			after, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.True(t, apiequality.Semantic.DeepEqual(requests, after), "deadline edits must preserve requests and their scheduling clocks")
		})
	}
}

func TestPipelineRequestDeadlineContinuesDuringRequestDeletion(t *testing.T) {
	t.Log("Publish two pending replicas and expire the tail first")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
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
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
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
	statusWrites, scaleWrites, deletes := 0, 0, 0
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if subresource == "status" {
				statusWrites++
				if failStatus {
					return statusErr
				}
			}
			if subresource == "scale" { //nolint:goconst // Keep API subresource names inline in tests.
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

	t.Log("Persist the failure while leaving capacity and requests intact for this pass")
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

	require.True(t, apiequality.Semantic.DeepEqual(pending, getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)))

	t.Log("Observing the persisted failure allows expired-replica cleanup")
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

func TestLPXDeadlineWaitsForScalingGroup(t *testing.T) {
	t.Log("Publish an expired request and hide its scaling group from the cache")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	expired := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, desired.requests[0].Name)
	expired.Status = newTestPipelineRequest(child, pcs, expired.Name, time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, expired))
	expired = getTestPipelineRequest(t, ctx, r.Client, expired.Namespace, expired.Name)
	base := r.Client
	r.Client = interceptor.NewClient(base.(client.WithWatch), interceptor.Funcs{
		List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if pcsgs, ok := list.(*grovev1alpha1.PodCliqueScalingGroupList); ok {
				pcsgs.Items = nil
				return nil
			}
			return delegated.List(ctx, list, opts...)
		},
	})

	t.Log("Wait for the group watch without evaluating deadlines or removing requests")
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	result, err := r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Zero(t, result)
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, schedulingFailedCondition))
	require.Equal(t, expired, getTestPipelineRequest(t, ctx, r.Client, expired.Namespace, expired.Name))

	t.Log("Once the group is observed, persist scheduling failure before cleanup")
	r.Client = base
	result, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Positive(t, result.RequeueAfter)
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, schedulingFailedCondition))
	require.Equal(t, expired, getTestPipelineRequest(t, ctx, r.Client, expired.Namespace, expired.Name))

	t.Log("A later pass lowers capacity and removes the expired request")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: desired.plan.LPXScalingGroup}, pcsg))
	require.Zero(t, pcsg.Spec.Replicas)
	requirePipelineRequestNotFound(t, ctx, r.Client, expired.Namespace, expired.Name)
}

func TestPipelineRequestDeadlineHoleWaitsForSchedulingChange(t *testing.T) {
	t.Log("Expire an interior workload while a higher ordinal is still Bound")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, desired.plan.LPXScalingGroup)
	requests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	for _, request := range requests {
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
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
		require.Equal(t, int32(2), pcsg.Spec.Replicas)
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
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	require.Zero(t, pcsg.Spec.Replicas)
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
			component := dgd.GetComponentByName("lpx")
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
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
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

func TestLPXServiceReconciliation(t *testing.T) {
	const uppercaseComponentName = "LPX"
	t.Log("Configure an uppercase LPX component and independently named materialization using Kubernetes discovery")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].ComponentName = uppercaseComponentName
	dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	dgd.Spec.Components[0].ModelRef = &v1beta1.ModelReference{Name: "test/model"}
	child.Name = "independent-materialization"
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)

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
	_, resources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
	require.NoError(t, err)
	require.NoError(t, r.reconcileRuntimeResources(t.Context(), child, resources))
	service := &corev1.Service{}
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.PCSNameForLPX(child) + "-serve"}
	require.NoError(t, r.Get(t.Context(), key, service))
	require.True(t, metav1.IsControlledBy(service, child))
	require.Equal(t, consts.KubeLabelValueTrue, service.Spec.Selector[dynamo.LPXServingLabel])
	require.Equal(t, dynamo.PCSNameForLPX(child), service.Spec.Selector[grovecommon.LabelPartOfKey])

	t.Log("Update the endpoint selector when the component's Dynamo namespace changes")
	require.NoError(t, r.Get(t.Context(), key, service))
	expected := service.DeepCopy()
	for _, global := range []bool{true, false} {
		dgd.Spec.Components[0].GlobalDynamoNamespace = global
		_, resources, err = r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
		require.NoError(t, err)
		require.NoError(t, r.reconcileRuntimeResources(t.Context(), child, resources))
		require.NoError(t, r.Get(t.Context(), key, service))
		expected.Spec.Selector[consts.KubeLabelDynamoNamespace] = dgd.GetDynamoNamespaceForComponent(&dgd.Spec.Components[0])
		require.Equal(t, expected.Spec, service.Spec)
		require.Equal(t, expected.OwnerReferences, service.OwnerReferences)

		t.Log("Reconcile the unchanged endpoint without another write")
		version := service.ResourceVersion
		require.NoError(t, r.reconcileRuntimeResources(t.Context(), child, resources))
		require.NoError(t, r.Get(t.Context(), key, service))
		require.Equal(t, version, service.ResourceVersion)
	}

	t.Log("Service reconciliation leaves the ordinary DGD model Service unchanged")
	require.NoError(t, r.Get(t.Context(), modelKey, modelService))
	require.Equal(t, beforeModelService, modelService)
}

func TestLPXWorkloadOrderAndUnrelatedEditsPreservePublication(t *testing.T) {
	t.Log("Publish a speculative workload using its frozen child identity")
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

	t.Log("Reorder workloads, update the frontend and enable an ordinary checkpoint without changing LPX")
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
	observed := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(beforeRequests.Items))
	for i := range beforeRequests.Items {
		request := &beforeRequests.Items[i]
		observed[request.Name] = request
	}
	_, missing, changed := resolvePipelineRequests(child, observed, afterSelected.workload, afterSelected.plan)
	require.False(t, changed)
	require.Empty(t, missing)
	afterRequests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(t.Context(), afterRequests))
	require.Equal(t, beforeRequests, afterRequests)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.Equal(t, beforeChild, child)
	require.Equal(t, beforeDGD, dgd)
}

func TestLPXFailedScaleOutPreservesServingWorkloads(t *testing.T) {
	t.Run("deadline", func(t *testing.T) {
		t.Log("Observe a completed workload and a newer pending request in their shared PCS")
		ctx := t.Context()
		child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
		dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
		dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
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
		pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
		_, err := r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
		require.Equal(t, int32(2), pcsg.Spec.Replicas)
		pending = getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
		require.True(t, pending.DeletionTimestamp.IsZero())

		t.Log("Retire only the failed scale-out and keep Grove below the authored scale while cleanup is pending")
		for range 3 {
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), &grovev1alpha1.PodCliqueSet{}))
			require.True(t, apiequality.Semantic.DeepEqual(serving, getTestPipelineRequest(t, ctx, r.Client, serving.Namespace, serving.Name)), "serving request changed")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
			require.Equal(t, int32(1), pcsg.Spec.Replicas)
		}
		t.Log("A later edit still waits for the failed request's scheduler finalizer")
		pending = getTestPipelineRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
		require.False(t, pending.DeletionTimestamp.IsZero())
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
		child.Generation++
		require.NoError(t, r.Update(ctx, child))
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
		require.Equal(t, int32(1), pcsg.Spec.Replicas)

		t.Log("After cleanup the later edit can retry without replacing the serving workload")
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
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
		require.Equal(t, int32(2), pcsg.Spec.Replicas)
		require.True(t, apiequality.Semantic.DeepEqual(serving, getTestPipelineRequest(t, ctx, r.Client, serving.Namespace, serving.Name)), "serving request changed")
	})
}

func TestLPXInvalidReplacementPreservesExistingWorkload(t *testing.T) {
	t.Log("Publish a speculative workload with two draft models")
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
	replacement, err := lpx.ResolveWorkload(ctx, dgd, singleGroupComponents(t, dgd), registry)
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
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(4))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
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
	dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(3))
	updateTestDGD(t, r, child, dgd)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 3 {
		_, err := r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
		require.EqualValues(t, 3, pcsg.Spec.Replicas)
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
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	require.EqualValues(t, 1, pcsg.Spec.Replicas)
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

func TestLPXNativeMinimumValidatedBeforePublishingPCS(t *testing.T) {
	for _, tc := range []struct {
		name    string
		shared  bool
		minimum int32
	}{
		{name: "single minimum", minimum: 2},
		{name: "single maximum", minimum: 2496},
		{name: "single oversized", minimum: 2497},
		{name: "shared minimum", shared: true, minimum: 2},
		{name: "shared maximum", shared: true, minimum: 2496},
		{name: "shared oversized", shared: true, minimum: 2497},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Start an externally managed workload at the conductor component's minimum")
			var (
				child    *v1alpha1.LPXGraphDeployment
				dgd      *v1beta1.DynamoGraphDeployment
				registry lpx.ModelRegistry
			)
			if tc.shared {
				child, dgd, registry = newLPXSpecDecodeTestDGD(t)
			} else {
				child, dgd, registry = newLPXTestDGD(t, lpx.PipelineSingle)
			}
			component := dgd.GetComponentByName("lpx")
			component.Replicas, component.MinAvailable = nil, ptr.To(tc.minimum)
			r := newLPXTestReconciler(t, registry, child, dgd)

			t.Log("Apply the existing replica bound before creating any Grove workload")
			_, reconcileErr := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
			pcs, err := getPodCliqueSet(t.Context(), r.Client, child)
			require.NoError(t, err)
			if tc.minimum > 2496 {
				require.ErrorContains(t, reconcileErr, "between 0 and 2496")
				require.Nil(t, pcs)
			} else {
				require.NoError(t, reconcileErr)
				require.NotNil(t, pcs)
				require.True(t, metav1.IsControlledBy(pcs, child))
				require.Equal(t, ptr.To(tc.minimum), pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas)
			}
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items, "requests wait for an observed Grove scaling group")
		})
	}
}

func TestLPXExternalScaleRejectsInvalidReplicaCount(t *testing.T) {
	t.Log("Observe externally managed capacity above the materialization limit")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.GetComponentByName("lpx").Replicas = nil
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
	pcsg.Spec.Replicas = 2497
	createLPXTestObjects(t, ctx, r.Client, objects...)

	t.Log("Reject the observed replica count before publishing requests and leave external capacity unchanged")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.ErrorContains(t, err, "between 0 and 2496")
	requests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(ctx, requests))
	require.Empty(t, requests.Items)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	require.EqualValues(t, 2497, pcsg.Spec.Replicas)
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
			dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
			dgd.GetComponentByName("lpx").Replicas = nil
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			selected.plan.Replicas = 4
			_, missing, changed := resolvePipelineRequests(child, nil, selected.workload, selected.plan)
			require.False(t, changed)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			pcs := findLPXTestPodCliqueSet(t, objects)
			pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
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
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
			pcsg.Spec.Replicas = tc.replicas
			require.NoError(t, r.Update(ctx, pcsg))
			beforeGroup := pcsg.DeepCopy()
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
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
			require.Equal(t, beforeGroup, pcsg)
			require.NoError(t, r.Get(ctx, request.NamespacedName, child))
			require.Equal(t, failure, meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition))
			require.Equal(t, v1alpha1.LPXReadyReasonFailed, meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition).Reason)

			t.Log("A later external scale-out does not bypass the failed generation's publication gate")
			pcsg.Spec.Replicas = 4
			require.NoError(t, r.Update(ctx, pcsg))
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
			for _, retired := range before[tc.replicas:] {
				requirePipelineRequestNotFound(t, ctx, r.Client, retired.Namespace, retired.Name)
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
			t.Log("Create a PCS, its scaling groups, and a discovery endpoint before publishing requests")
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
			createLPXTestScalingGroups(t, r.Client, pcs)
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
			component := dgd.GetComponentByName("lpx")
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
				require.ErrorContains(t, err, scenario.message)
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
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%s-lpu-%.16s", root, pcs.Spec.Template.Cliques[0].Annotations[v1alpha1.AnnotationExtraResourcesHash]), Namespace: child.Namespace}},
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
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
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
	require.Positive(t, result.RequeueAfter, "the surviving request's creation-time deadline remains active")
	require.LessOrEqual(t, result.RequeueAfter, 30*time.Second)
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
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, pcsg))
	require.Contains(t, pcsg.Spec.CliqueNames, retiredTemplate)

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
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	require.Contains(t, pcsg.Spec.CliqueNames, retiredTemplate)
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
	require.NotEqual(t, pcs.Annotations[lpx.WorkloadDigestAnnotation], synced.Annotations[lpx.WorkloadDigestAnnotation])
	require.Less(t, len(synced.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames), originalCliqueCount)
}

func TestLPXNativeScaleToZeroPreservesPodCliqueSetTemplate(t *testing.T) {
	t.Log("Publish one workload whose capacity is owned by the native Grove scaling group")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.GetComponentByName("lpx").Replicas = nil
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
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: initial.plan.LPXScalingGroup}, pcsg))
	pcsg.Spec.Replicas = 0
	require.NoError(t, r.Update(ctx, pcsg))

	t.Log("Delete the stale LPR before reconciling the stable PCS template")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(request), &lpxv1alpha1.LPUPipelineRequest{})))
	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	require.Zero(t, pcsg.Spec.Replicas)
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), storedPCS))
	require.Equal(t, pcsUID, storedPCS.UID)
	require.Equal(t, templateReplicas, ptr.Deref(storedPCS.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas, 0))

	t.Log("Scale the native group back up and publish a fresh request through the same PCS")
	pcsg.Spec.Replicas = 1
	require.NoError(t, r.Update(ctx, pcsg))
	_, err = r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	replacement := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, initial.requests[0].Name)
	require.NotEqual(t, requestUID, replacement.UID)
	require.Equal(t, pcsUID, metav1.GetControllerOf(replacement).UID)
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
			dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
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
								dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(1))
								dgd.Generation++
								return delegated.Update(ctx, dgd)
							}
							require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(child), child))
							metav1.SetMetaDataAnnotation(&child.ObjectMeta, "example.com/concurrent-edit", "changed")
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
			pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcsg), pcsg))
			require.Equal(t, int32(2), pcsg.Spec.Replicas, "an incomplete handoff must not scale Grove")
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
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcsg), pcsg))
			require.Equal(t, *dgd.GetComponentByName("lpx").Replicas, pcsg.Spec.Replicas)
			require.NoError(t, r.List(t.Context(), requests))
			require.Len(t, requests.Items, int(pcsg.Spec.Replicas))
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
			dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
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

func TestLPXRequestListFailureUsesControllerBackoff(t *testing.T) {
	t.Log("A request-list failure prevents deriving any active deadline")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
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

func TestLPXCorrectsTopLevelReplicaDrift(t *testing.T) {
	t.Log("Materialize an extra PCS ordinal after the top-level replica count drifts to two")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	pcs.Spec.Replicas = 2
	pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, desired.plan.LPXScalingGroup)
	extraGroup := pcsg.DeepCopy()
	extraGroup.Name = grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 1}, desired.plan.ScalingGroupTemplate)
	extraGroup.UID = types.UID(extraGroup.Name + "-uid")
	extraGroup.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "1"
	conductorName := grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: pcsg.Name, Replica: 0}, desired.plan.ConductorTemplate)
	extraClique := getResource[*grovev1alpha1.PodClique](t, objects, conductorName).DeepCopy()
	extraClique.Name = grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: extraGroup.Name, Replica: 0}, desired.plan.ConductorTemplate)
	extraClique.UID = types.UID(extraClique.Name + "-uid")
	extraClique.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "1"
	extraClique.Labels[grovecommon.LabelPodCliqueScalingGroup] = extraGroup.Name
	extraClique.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(extraGroup, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))}
	objects = append(objects, extraGroup, extraClique)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(extraClique), extraClique))

	t.Log("Ignore extra PCS ordinals so normal synchronization can restore one top-level replica")
	result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.Zero(t, result)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	require.EqualValues(t, 1, pcs.Spec.Replicas)
	observed := &grovev1alpha1.PodClique{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(extraClique), observed))
	require.Equal(t, extraClique, observed, "Grove owns cleanup of the extra ordinal")
}

func TestLPXReconcileRejectsForeignPodClique(t *testing.T) {
	t.Log("Materialize a workload whose workers need scaling")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	dgd.GetComponentByName("lpx").ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(3))
	r, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	worker := getResource[*grovev1alpha1.PodClique](t, objects, desired.plan.CyborgClique)
	worker.OwnerReferences[0].UID = "foreign"
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	writes := 0
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if subresource == "scale" {
				writes++
				return errors.New("unexpected capacity write")
			}
			return delegated.SubResource(subresource).Update(ctx, object, opts...)
		},
		Create: func(context.Context, client.WithWatch, client.Object, ...client.CreateOption) error {
			writes++
			return errors.New("unexpected resource publication")
		},
	})

	t.Log("Reject foreign cliques before scaling or publishing requests")
	_, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.ErrorContains(t, err, "is not controlled by")
	require.Zero(t, writes)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	condition := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
	require.NotNil(t, condition)
	require.Equal(t, v1alpha1.LPXReadyReasonFailed, condition.Reason)
}

func TestLPXScalingWaitsForObservedCapacity(t *testing.T) {
	for _, tc := range []struct {
		name             string
		groupReplicas    int32
		wantGroupUpdates int
	}{
		{name: "group scale down", groupReplicas: 9, wantGroupUpdates: 1},
		{name: "group scale up", groupReplicas: 1, wantGroupUpdates: 1},
		{name: "clique scale", groupReplicas: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe a hybrid workload with group or clique capacity still to apply")
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineLPX)
			dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup)
			for _, object := range objects {
				if pclq, ok := object.(*grovev1alpha1.PodClique); ok {
					pclq.Status.Replicas, pclq.Status.UpdatedReplicas = pclq.Spec.Replicas, pclq.Spec.Replicas
					pclq.Status.ReadyReplicas, pclq.Status.ScheduledReplicas = pclq.Spec.Replicas, pclq.Spec.Replicas
				}
			}
			createLPXTestObjects(t, ctx, r.Client, objects...)
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			_, err := r.Reconcile(ctx, key)
			require.NoError(t, err)
			publishSelectedLPXForTest(t, ctx, r, child, selected)
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
			pcs := findLPXTestPodCliqueSet(t, objects)
			for _, desired := range selected.requests {
				request := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, desired.Name)
				request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
				require.NoError(t, r.Update(ctx, request))
			}

			t.Log("Request three Cyborg replicas while the PCS template and live cliques still have one")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
			dgd.GetComponentByName("lpx").ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(3))
			require.NoError(t, r.Update(ctx, dgd))
			require.NoError(t, r.Get(ctx, key.NamespacedName, child))
			child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
			require.NoError(t, err)
			child.Generation++
			require.NoError(t, r.Update(ctx, child))
			pcsg.Spec.Replicas = tc.groupReplicas
			pcsg.Status.Replicas, pcsg.Status.UpdatedReplicas = tc.groupReplicas, tc.groupReplicas
			pcsg.Status.AvailableReplicas, pcsg.Status.ScheduledReplicas = tc.groupReplicas, tc.groupReplicas
			require.NoError(t, r.Update(ctx, pcsg))
			cachedPclqs := &grovev1alpha1.PodCliqueList{}
			require.NoError(t, r.List(ctx, cachedPclqs, client.InNamespace(pcs.Namespace), client.MatchingLabels{grovecommon.LabelPartOfKey: pcs.Name}))
			cacheStale := true
			groupUpdates, cliqueUpdates, cliqueLists := 0, 0, 0
			base := r.Client.(client.WithWatch)
			r.Client = interceptor.NewClient(base, interceptor.Funcs{
				Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					if _, ok := object.(*grovev1alpha1.PodClique); ok {
						t.Fatal("PodCliques must be listed once at the observation boundary")
					}
					return delegated.Get(ctx, key, object, opts...)
				},
				List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if observed, ok := list.(*grovev1alpha1.PodCliqueList); ok {
						cliqueLists++
						options := (&client.ListOptions{}).ApplyOptions(opts)
						require.Equal(t, pcs.Namespace, options.Namespace)
						require.Equal(t, grovecommon.LabelPartOfKey+"="+pcs.Name+","+grovecommon.LabelPodCliqueSetReplicaIndex+"=0", options.LabelSelector.String())
						if cacheStale {
							*observed = *cachedPclqs.DeepCopy()
							return nil
						}
					}
					return delegated.List(ctx, list, opts...)
				},
				SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
					if subresource == "scale" {
						switch object.(type) {
						case *grovev1alpha1.PodCliqueScalingGroup:
							groupUpdates++
						case *grovev1alpha1.PodClique:
							cliqueUpdates++
						}
					}
					return delegated.SubResource(subresource).Update(ctx, object, opts...)
				},
				Patch: func(ctx context.Context, delegated client.WithWatch, object client.Object, patch client.Patch, opts ...client.PatchOption) error {
					switch object.(type) {
					case *grovev1alpha1.PodCliqueScalingGroup, *grovev1alpha1.PodClique:
						t.Fatal("capacity writes must use the scale subresource")
					}
					return delegated.Patch(ctx, object, patch, opts...)
				},
			})

			if tc.groupReplicas > 2 {
				t.Log("Scale down the group before updating its worker template")
				_, err = r.Reconcile(ctx, key)
				require.NoError(t, err)
				require.Equal(t, 1, groupUpdates)
				require.Zero(t, cliqueUpdates)
			}

			t.Log("Persist the PCS worker template before scaling existing cliques")
			cliqueLists = 0
			result, err := r.Reconcile(ctx, key)
			require.NoError(t, err)
			require.Zero(t, result, "PCS observations resume reconciliation through watches")
			require.Zero(t, cliqueUpdates)
			require.Equal(t, 1, cliqueLists)
			require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
			cyborgIndex := slices.IndexFunc(pcs.Spec.Template.Cliques, func(clique *grovev1alpha1.PodCliqueTemplateSpec) bool {
				return clique.Name == selected.plan.CyborgTemplate
			})
			require.NotEqual(t, -1, cyborgIndex)
			require.EqualValues(t, 3, pcs.Spec.Template.Cliques[cyborgIndex].Spec.Replicas)
			require.NoError(t, r.Get(ctx, key.NamespacedName, child))
			require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition).Reason)
			if tc.groupReplicas < 2 {
				t.Log("Scale up the group and wait to observe it before updating its workers")
				_, err = r.Reconcile(ctx, key)
				require.NoError(t, err)
				require.Equal(t, 1, groupUpdates)
				require.Zero(t, cliqueUpdates)
			}

			t.Log("Scaling the workers stops reconciliation before the cached ready 1/1 state can be reported")
			cliqueLists = 0
			result, err = r.Reconcile(ctx, key)
			require.NoError(t, err)
			require.Zero(t, result, "capacity observations resume reconciliation through watches")
			require.Equal(t, 1, cliqueLists)
			require.Equal(t, tc.wantGroupUpdates, groupUpdates)
			require.Equal(t, 2, cliqueUpdates)
			require.NoError(t, r.Get(ctx, key.NamespacedName, child))
			require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, v1alpha1.LPXReadyCondition))

			t.Log("Once the cache observes capacity, wait for the new pods without repeating writes")
			cacheStale = false
			_, err = r.Reconcile(ctx, key)
			require.NoError(t, err)
			require.Equal(t, tc.wantGroupUpdates, groupUpdates)
			require.Equal(t, 2, cliqueUpdates)
			require.NoError(t, r.Get(ctx, key.NamespacedName, child))
			require.True(t, meta.IsStatusConditionFalse(child.Status.Components["lpx"].Conditions, v1alpha1.LPXReadyCondition))
			require.Equal(t, ptr.To(int32(0)), child.Status.Components["lpx"].AvailableReplicas)

			t.Log("Report readiness only after Grove observes the new generation and all requested replicas")
			require.NoError(t, base.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
			pcsg.Status.ObservedGeneration = ptr.To(pcsg.Generation)
			pcsg.Status.Replicas, pcsg.Status.UpdatedReplicas = 2, 2
			pcsg.Status.AvailableReplicas, pcsg.Status.ScheduledReplicas = 2, 2
			require.NoError(t, base.Update(ctx, pcsg))
			liveCliques := &grovev1alpha1.PodCliqueList{}
			require.NoError(t, base.List(ctx, liveCliques))
			for i := range liveCliques.Items {
				clique := &liveCliques.Items[i]
				clique.Status.ObservedGeneration = ptr.To(clique.Generation)
				clique.Status.Replicas, clique.Status.UpdatedReplicas = clique.Spec.Replicas, clique.Spec.Replicas
				clique.Status.ReadyReplicas, clique.Status.ScheduledReplicas = clique.Spec.Replicas, clique.Spec.Replicas
				require.NoError(t, base.Update(ctx, clique))
			}
			_, err = r.Reconcile(ctx, key)
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, key.NamespacedName, child))
			require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, v1alpha1.LPXReadyCondition))
		})
	}
}

func TestLPXScaleDownUpdatesGroveBeforeDeletingStaleRequest(t *testing.T) {
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)

	stale := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].Name)
	stale.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
	require.NoError(t, r.Update(ctx, stale))

	liveDGD := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), liveDGD))
	liveDGD.GetComponentByName("lpx").Replicas = ptr.To(int32(1))
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
		List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if pcsgs, ok := list.(*grovev1alpha1.PodCliqueScalingGroupList); ok {
				pcsgs.Items = nil
				return nil
			}
			return delegated.List(ctx, list, opts...)
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
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, pcsg))
	require.Equal(t, int32(1), pcsg.Spec.Replicas)
	t.Log("Let request finalization proceed asynchronously after the lower scale is persisted")
	stale = getTestPipelineRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	require.False(t, stale.DeletionTimestamp.IsZero())
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	require.Equal(t, int32(1), pcsg.Spec.Replicas)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(findLPXTestPodCliqueSet(t, objects)), &grovev1alpha1.PodCliqueSet{}))

	t.Log("Reverse the scale-down while the old request still protects its pods")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), liveDGD))
	liveDGD.GetComponentByName("lpx").Replicas = ptr.To(int32(2))
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
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
		require.Equal(t, int32(1), pcsg.Spec.Replicas, "Grove must not recreate pods protected by the retiring request")
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
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
	require.Equal(t, int32(2), pcsg.Spec.Replicas)
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
			component := dgd.GetComponentByName("lpx")
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
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:            grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, groupTemplate.Name),
					Namespace:       pcs.Namespace,
					Labels:          map[string]string{grovecommon.LabelPartOfKey: pcs.Name},
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: seed},
			}
			require.NoError(t, r.Create(t.Context(), pcsg))

			t.Log("Publish requests in ordinal order only after live capacity reaches the explicit desired count")
			var published []int64
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
					if request, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok {
						require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(pcsg), pcsg))
						require.Equal(t, int32(3), pcsg.Spec.Replicas)
						require.True(t, metav1.IsControlledBy(request, pcs))
						require.True(t, ptr.Deref(metav1.GetControllerOf(request).BlockOwnerDeletion, false))
						published = append(published, request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex)
					}
					return delegated.Create(ctx, object, opts...)
				},
			})
			result, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.Zero(t, result, "the PCSG watch resumes reconciliation")
			require.Empty(t, published, "the scale response cannot be used as a complete PCSG observation")

			t.Log("Publish after the PCSG watch supplies its updated capacity")
			_, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
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
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	pcs := observedLPXTestPodCliqueSet(t, ctx, r, child, selected)
	pending := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
	pending.Status = newTestPipelineRequest(child, pcs, pending.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending).Status
	require.NoError(t, r.Update(ctx, pending))
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, pcsg))
	for _, object := range []client.Object{pcs, pcsg} {
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
	for _, scenario := range []string{"ready", "partial", "missing", "old revision", "unobserved"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Copy the observed cliques and select the first draft")
			pclqs := make(map[string]*grovev1alpha1.PodClique)
			for _, object := range objects {
				if pclq, ok := object.(*grovev1alpha1.PodClique); ok {
					pclqs[pclq.Name] = pclq.DeepCopy()
				}
			}
			firstDraft := pclqs[selected.plan.Agents[0].CliqueName]
			require.Equal(t, "draft", firstDraft.Labels[lpx.StageLabel])

			t.Log("Make only the first draft incomplete while retaining a fully ready second draft and target")
			switch scenario {
			case "partial":
				firstDraft.Status.ReadyReplicas--
			case "old revision":
				firstDraft.Status.CurrentPodCliqueSetGenerationHash = ptr.To("old")
			case "unobserved":
				firstDraft.Status.ObservedGeneration = ptr.To(firstDraft.Generation - 1)
			}
			if scenario == "missing" {
				delete(pclqs, firstDraft.Name)
			}

			t.Log("Project logical draft counts and make both authored components await the complete pair")
			readiness := dynamo.EvaluateLPXGroveReadiness(t.Context(), dgd, dgd.GetComponentByName("lpx").ComponentName, singleGroupComponents(t, dgd), pcs, getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, selected.plan.LPXScalingGroup), pclqs)
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
					if mount.Name == v1alpha1.ModelStorageVolumeName {
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
	agent := dgd.GetComponentByName("lpx").ComponentRole(v1beta1.ComponentRoleLPXAgent)
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
	createLPXTestScalingGroups(t, r.Client, pcs)

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
			dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
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
	workload *lpx.Workload
	plan     *lpx.MaterializationPlan
	requests []lpxv1alpha1.LPUPipelineRequest
}

func resolveLPXTestWorkload(t *testing.T, registry lpx.ModelRegistry, ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment,
) *lpxTestWorkload {
	t.Helper()
	workload, err := lpx.ResolveWorkload(ctx, dgd, singleGroupComponents(t, dgd), registry)
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
			Annotations:     map[string]string{},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)},
		},
		Spec: v1alpha1.LPXGraphDeploymentSpec{InputRevision: revision},
	}
}

func loadTestDGD(t testing.TB, pipeline lpx.Pipeline, buildID string) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	payload, err := os.ReadFile("testdata/dgd.yaml")
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}}}
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
		// The fake client implements /scale only for built-in Kubernetes workloads.
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if subresource != "scale" {
				return delegated.SubResource(subresource).Update(ctx, object, opts...)
			}
			scale := (&client.SubResourceUpdateOptions{}).ApplyOptions(opts).SubResourceBody.(*autoscalingv1.Scale)
			live := object.DeepCopyObject().(client.Object)
			if err := delegated.Get(ctx, client.ObjectKeyFromObject(object), live); err != nil {
				return err
			}
			switch resource := live.(type) {
			case *grovev1alpha1.PodCliqueScalingGroup:
				resource.Spec.Replicas = scale.Spec.Replicas
			case *grovev1alpha1.PodClique:
				resource.Spec.Replicas = scale.Spec.Replicas
			default:
				return fmt.Errorf("unsupported test scale resource %T", object)
			}
			live.SetResourceVersion(scale.ResourceVersion)
			live.SetGeneration(live.GetGeneration() + 1)
			if err := delegated.Update(ctx, live); err != nil {
				return err
			}
			scale.ResourceVersion = live.GetResourceVersion()
			return nil
		},
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
	return materializeLPXTestPCS(t, deployment, pcs, desired.plan)
}

// materializeLPXTestPCS constructs native Grove observations for each independent workload.
func materializeLPXTestPCS(t *testing.T, deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, plans ...*lpx.MaterializationPlan) []client.Object {
	t.Helper()
	pcs.TypeMeta = metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueSet"}
	pcs.UID, pcs.Generation = "pcs-uid", 1
	pcs.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(deployment, v1alpha1.LPXGraphDeploymentGVK)}
	const generationHash = "pcs-generation-hash"
	pcs.Status = grovev1alpha1.PodCliqueSetStatus{
		ObservedGeneration: ptr.To(pcs.Generation), CurrentGenerationHash: ptr.To(generationHash),
	}

	// Emulate Grove materialization after the controller applies the desired live scale.
	objects := []client.Object{pcs}
	for _, plan := range plans {
		groupIndex := slices.IndexFunc(pcs.Spec.Template.PodCliqueScalingGroupConfigs, func(config grovev1alpha1.PodCliqueScalingGroupConfig) bool {
			return config.Name == plan.ScalingGroupTemplate
		})
		require.NotEqual(t, -1, groupIndex)
		groupTemplate := pcs.Spec.Template.PodCliqueScalingGroupConfigs[groupIndex].DeepCopy()
		pcsg := &grovev1alpha1.PodCliqueScalingGroup{
			TypeMeta: metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup"},
			ObjectMeta: metav1.ObjectMeta{
				Name: plan.LPXScalingGroup, Namespace: pcs.Namespace, UID: types.UID(plan.LPXScalingGroup + "-uid"),
				Generation: 1, Labels: grovecommon.GetDefaultLabelsForPodCliqueSetManagedResources(pcs.Name), Annotations: groupTemplate.Annotations,
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
			},
			Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
				Replicas: plan.Replicas, MinAvailable: groupTemplate.MinAvailable,
				CliqueNames: groupTemplate.CliqueNames,
			},
			Status: grovev1alpha1.PodCliqueScalingGroupStatus{
				ObservedGeneration: ptr.To(int64(1)), CurrentPodCliqueSetGenerationHash: ptr.To(generationHash),
			},
		}
		pcsg.Labels[grovecommon.LabelPartOfKey] = pcs.Name
		pcsg.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "0"
		objects = append(objects, pcsg)

		// Each Grove replica owns independent cliques.
		for replicaIndex := range pcsg.Spec.Replicas {
			parent := grovecommon.ResourceNameReplica{Name: pcsg.Name, Replica: int(replicaIndex)}
			for _, template := range pcs.Spec.Template.Cliques {
				if !slices.Contains(groupTemplate.CliqueNames, template.Name) {
					continue
				}
				rendered := template.DeepCopy()
				name := grovecommon.GeneratePodCliqueName(parent, rendered.Name)
				clique := &grovev1alpha1.PodClique{
					TypeMeta: metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodClique"},
					ObjectMeta: metav1.ObjectMeta{
						Name: name, Namespace: pcs.Namespace, UID: types.UID(name + "-uid"), Generation: 1,
						Labels: rendered.Labels, Annotations: rendered.Annotations,
						OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcsg, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
					},
					Spec: rendered.Spec,
					Status: grovev1alpha1.PodCliqueStatus{
						ObservedGeneration: ptr.To(int64(1)), CurrentPodCliqueSetGenerationHash: ptr.To(generationHash),
						CurrentPodTemplateHash: ptr.To(rendered.Name + "-pod-template-hash"),
					},
				}
				clique.Labels[grovecommon.LabelPartOfKey] = pcs.Name
				clique.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "0"
				clique.Labels[grovecommon.LabelPodCliqueScalingGroup] = pcsg.Name
				clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] = strconv.FormatInt(int64(replicaIndex), 10)
				for index, dependency := range clique.Spec.StartsAfter {
					clique.Spec.StartsAfter[index] = grovecommon.GeneratePodCliqueName(parent, dependency)
				}
				objects = append(objects, clique)
			}
		}
	}
	return objects
}

// createLPXTestScalingGroups emulates Grove creating each group at its template seed.
func createLPXTestScalingGroups(t *testing.T, kubeClient client.Client, pcs *grovev1alpha1.PodCliqueSet) {
	t.Helper()
	for _, config := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		pcsg := &grovev1alpha1.PodCliqueScalingGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:            grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, config.Name),
				Namespace:       pcs.Namespace,
				Labels:          map[string]string{grovecommon.LabelPartOfKey: pcs.Name},
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
			},
			Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
				Replicas: ptr.Deref(config.Replicas, 1), MinAvailable: config.MinAvailable, CliqueNames: config.CliqueNames,
			},
		}
		require.NoError(t, kubeClient.Create(t.Context(), pcsg))
	}
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

func TestLPXEditBetweenFailureAndCleanupPreservesRetry(t *testing.T) {
	t.Log("Persist a deadline failure before the user edits the input to retry")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
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
	dgd.Spec.Components[0].LPX.Scheduling.AttemptDeadlineSeconds = ptr.To[int64](31)
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
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To[int64](30)}
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
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, pcsg))
	require.Equal(t, selected.plan.Replicas, pcsg.Spec.Replicas)
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
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
			require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, pcsg))
			require.Equal(t, selected.plan.Replicas, pcsg.Spec.Replicas)
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

func TestLPXWaitsForAllScalingGroups(t *testing.T) {
	for _, tc := range []struct {
		name     string
		pcsg     string
		deleting bool
	}{
		{name: "missing explicit group", pcsg: "lpx"},
		{name: "missing external group", pcsg: "external"},
		{name: "deleting group", pcsg: "external", deleting: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Create a PCS with explicit and externally managed workloads")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			dgd.Spec.Components[0].Replicas = ptr.To(int32(1))
			second := dgd.Spec.Components[0].DeepCopy()
			second.ComponentName, second.Replicas = "external", nil
			dgd.Spec.Components = append(dgd.Spec.Components, *second)
			child.Status.Components = map[string]v1alpha1.LPXComponentStatus{
				"lpx":      {Conditions: []metav1.Condition{readyCondition(child.Generation, v1beta1.DGDStateSuccessful, "Previous observation")}},
				"external": {Conditions: []metav1.Condition{readyCondition(child.Generation, v1beta1.DGDStateSuccessful, "Previous observation")}},
			}
			r := newLPXTestReconciler(t, registry, child, dgd)
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
			require.NoError(t, err)
			pcs, _, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
			require.NoError(t, err)
			createLPXTestObjects(t, t.Context(), r.Client, materializeLPXTestPCS(t, child, pcs, plans["lpx"], plans["external"])...)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			before := pcs.DeepCopy()

			t.Log("Observe one configured group as missing or deleting")
			base := r.Client
			r.Client = interceptor.NewClient(base.(client.WithWatch), interceptor.Funcs{
				List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if err := delegated.List(ctx, list, opts...); err != nil {
						return err
					}
					if pcsgs, ok := list.(*grovev1alpha1.PodCliqueScalingGroupList); ok {
						for i := range pcsgs.Items {
							if pcsgs.Items[i].Name == plans[tc.pcsg].LPXScalingGroup {
								if tc.deleting {
									pcsgs.Items[i].DeletionTimestamp = ptr.To(metav1.Now())
								} else {
									pcsgs.Items = slices.Delete(pcsgs.Items, i, i+1)
								}
								break
							}
						}
					}
					return nil
				},
			})
			r.modelRegistry = &snapshotFailureRegistry{ModelRegistry: registry, err: errors.New("workload resolution must wait for all groups")}

			t.Log("Return Pending before resolving workloads, updating the PCS, or publishing requests")
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			result, err := r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.Zero(t, result, "the PCSG watch resumes reconciliation")
			require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
			ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
			require.NotNil(t, ready)
			require.Equal(t, v1alpha1.LPXReadyReasonPending, ready.Reason)
			require.Equal(t, "Waiting for all LPX scaling groups", ready.Message)
			require.Len(t, child.Status.Components, 2)
			for _, component := range child.Status.Components {
				condition := meta.FindStatusCondition(component.Conditions, v1alpha1.LPXReadyCondition)
				require.NotNil(t, condition)
				require.Equal(t, metav1.ConditionUnknown, condition.Status)
				require.Equal(t, "NotObserved", condition.Reason)
				require.Equal(t, "Component readiness has not been observed", condition.Message)
				require.Equal(t, child.Generation, condition.ObservedGeneration)
			}
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, before, pcs)
			requests, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Empty(t, requests)

			t.Log("Observing every group allows PCS synchronization and request publication")
			r.Client, r.modelRegistry = base, registry
			for range 2 {
				_, err = r.Reconcile(t.Context(), key)
				require.NoError(t, err)
			}
			requests, err = r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Len(t, requests, 2)
		})
	}
}

func TestIndependentLPXWorkloadsScaleAndReportReadiness(t *testing.T) {
	t.Log("Author two independent workloads with explicit and external replica ownership")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))
	second := dgd.Spec.Components[0].DeepCopy()
	second.ComponentName, second.Replicas = "second-engine", nil
	dgd.Spec.Components = append(dgd.Spec.Components, *second)
	dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	require.Len(t, workloads, 2)
	pcs, resources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
	require.NoError(t, err)
	require.Len(t, resources, 4)
	names := make(map[string]bool)
	for _, resource := range resources {
		require.NotContains(t, names, resource.GetName())
		names[resource.GetName()] = true
	}

	t.Log("Reordering authored components preserves rendering and immutable identities")
	reordered := dgd.DeepCopy()
	slices.Reverse(reordered.Spec.Components)
	reorderedWorkloads, reorderedPlans, err := r.resolveWorkloads(t.Context(), child, reordered)
	require.NoError(t, err)
	reorderedPCS, reorderedResources, err := r.renderPodCliqueSet(t.Context(), child, reordered, reorderedWorkloads, reorderedPlans)
	require.NoError(t, err)
	require.Equal(t, pcs, reorderedPCS)
	require.Equal(t, resources, reorderedResources)
	require.Equal(t, plans, reorderedPlans)

	t.Log("Observe two explicit replicas and three externally managed replicas in Grove")
	firstPlan, secondPlan := plans["lpx"], plans["second-engine"]
	secondPlan.Replicas = 3
	objects := materializeLPXTestPCS(t, child, pcs, firstPlan, secondPlan)
	for _, object := range objects {
		switch live := object.(type) {
		case *grovev1alpha1.PodCliqueScalingGroup:
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.AvailableReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
		case *grovev1alpha1.PodClique:
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
		}
	}
	createLPXTestObjects(t, t.Context(), r.Client, objects...)

	t.Log("Record request publication order across both workloads")
	var publishedTargets []lpxv1alpha1.PodCliqueScalingGroupReference
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
			if request, ok := object.(*lpxv1alpha1.LPUPipelineRequest); ok {
				publishedTargets = append(publishedTargets, *request.Spec.MaterializationTarget.PodCliqueScalingGroupRef)
			}
			return delegated.Create(ctx, object, opts...)
		},
	})
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	t.Log("Observe the synchronized PCS before publishing scheduler requests")
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	requests, err := r.getPipelineRequests(t.Context(), pcs)
	require.NoError(t, err)
	require.Len(t, requests, 5)
	require.Equal(t, []lpxv1alpha1.PodCliqueScalingGroupReference{
		{Name: firstPlan.LPXScalingGroup, ReplicaIndex: 0},
		{Name: firstPlan.LPXScalingGroup, ReplicaIndex: 1},
		{Name: secondPlan.LPXScalingGroup, ReplicaIndex: 0},
		{Name: secondPlan.LPXScalingGroup, ReplicaIndex: 1},
		{Name: secondPlan.LPXScalingGroup, ReplicaIndex: 2},
	}, publishedTargets)
	original := make(map[string]types.UID)
	for _, request := range requests {
		original[request.Name] = request.UID
		require.Equal(t, "default", request.Annotations[pipelineRequestModelAnnotation])
		request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
		require.NoError(t, r.Update(t.Context(), request))
	}

	t.Log("Publish disjoint Services and aggregate both workloads' readiness")
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, v1alpha1.LPXReadyCondition))
	for groupName, plan := range plans {
		status := child.Status.Components[workloads[groupName].ServingComponentName()]
		require.True(t, meta.IsStatusConditionTrue(status.Conditions, v1alpha1.LPXReadyCondition))
		ready := meta.FindStatusCondition(status.Conditions, v1alpha1.LPXReadyCondition)
		require.Equal(t, v1alpha1.LPXReadyReasonReady, ready.Reason)
		require.Equal(t, child.Generation, ready.ObservedGeneration)
		require.Equal(t, plan.Replicas, status.Replicas)
		require.Equal(t, []string{plan.LPXScalingGroup}, status.ComponentNames)
		service := &corev1.Service{}
		require.NoError(t, r.Get(t.Context(), client.ObjectKey{Namespace: child.Namespace, Name: plan.ResourcePrefix + "-serve"}, service))
		require.Equal(t, workloads[groupName].ServingComponentName(), service.Spec.Selector[consts.KubeLabelDynamoComponent])
		require.Equal(t, consts.KubeLabelValueTrue, service.Spec.Selector[dynamo.LPXServingLabel])
	}

	t.Log("Scheduler progress and failure affect only their own workload, preserving its healthy neighbor")
	healthy := child.Status.Components["lpx"]
	bound := meta.FindStatusCondition(child.Status.Components["second-engine"].Conditions, v1alpha1.LPXReadyCondition).DeepCopy()
	var secondRequest *lpxv1alpha1.LPUPipelineRequest
	for _, request := range requests {
		if target := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef; target.Name == secondPlan.LPXScalingGroup && target.ReplicaIndex == 0 {
			secondRequest = request
		}
	}
	require.NotNil(t, secondRequest)
	for _, observation := range []struct {
		phase   lpxv1alpha1.RequestPhase
		status  metav1.ConditionStatus
		reason  string
		message string
	}{
		{lpxv1alpha1.RequestPhasePending, metav1.ConditionFalse, v1alpha1.LPXReadyReasonPending, "LPX scheduler is waiting to plan the current request"},
		{lpxv1alpha1.RequestPhaseUnsupported, metav1.ConditionFalse, v1alpha1.LPXReadyReasonFailed, "LPX scheduler cannot support the current request"},
		{lpxv1alpha1.RequestPhaseBound, metav1.ConditionTrue, v1alpha1.LPXReadyReasonReady, bound.Message},
	} {
		secondRequest.Status.Phase = observation.phase
		require.NoError(t, r.Update(t.Context(), secondRequest))
		_, err = r.Reconcile(t.Context(), key)
		require.NoError(t, err)
		require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
		condition := meta.FindStatusCondition(child.Status.Components["second-engine"].Conditions, v1alpha1.LPXReadyCondition)
		require.NotNil(t, condition)
		require.Equal(t, observation.status, condition.Status)
		require.Equal(t, observation.reason, condition.Reason)
		require.Equal(t, observation.message, condition.Message)
		require.Equal(t, child.Generation, condition.ObservedGeneration)
		require.Equal(t, healthy, child.Status.Components["lpx"])
		require.Equal(t, observation.status, meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition).Status)
	}

	t.Log("A missing role in the second workload leaves the first workload's readiness intact")
	clique := &grovev1alpha1.PodClique{}
	conductorName := grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: secondPlan.LPXScalingGroup, Replica: int(secondPlan.ReplicaIndex)}, secondPlan.ConductorTemplate)
	require.NoError(t, r.Get(t.Context(), client.ObjectKey{Namespace: child.Namespace, Name: conductorName}, clique))
	require.NoError(t, r.Delete(t.Context(), clique))
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Components["lpx"].Conditions, v1alpha1.LPXReadyCondition))
	require.True(t, meta.IsStatusConditionFalse(child.Status.Components["second-engine"].Conditions, v1alpha1.LPXReadyCondition))
	require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, v1alpha1.LPXReadyCondition))

	t.Log("Scale down the explicit workload without replacing the PCS or its neighbor's requests")
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
	dgd.Spec.Components[0].Replicas = ptr.To(int32(1))
	require.NoError(t, r.Update(t.Context(), dgd))
	child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, "")
	require.NoError(t, err)
	require.NoError(t, r.Update(t.Context(), child))
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	pcsgs, err := getPodCliqueScalingGroups(t.Context(), r.Client, pcs)
	require.NoError(t, err)
	require.EqualValues(t, 1, pcsgs[firstPlan.LPXScalingGroup].Spec.Replicas)
	require.EqualValues(t, 3, pcsgs[secondPlan.LPXScalingGroup].Spec.Replicas)
	requests, err = r.getPipelineRequests(t.Context(), pcs)
	require.NoError(t, err)
	require.Len(t, requests, 4)
	for _, request := range requests {
		require.Equal(t, original[request.Name], request.UID)
	}
	observedPCS, err := getPodCliqueSet(t.Context(), r.Client, child)
	require.NoError(t, err)
	require.Equal(t, pcs.UID, observedPCS.UID)
	require.Equal(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, observedPCS.Spec.Template.PodCliqueScalingGroupConfigs)

	t.Log("External scale-out publishes only the new ordinal and preserves the explicit workload")
	pcsg := pcsgs[secondPlan.LPXScalingGroup]
	pcsg.Spec.Replicas = 4
	require.NoError(t, r.Update(t.Context(), pcsg))
	_, err = r.Reconcile(t.Context(), key)
	require.NoError(t, err)
	requests, err = r.getPipelineRequests(t.Context(), pcs)
	require.NoError(t, err)
	require.Len(t, requests, 5)
	for _, request := range requests {
		if uid, exists := original[request.Name]; exists {
			require.Equal(t, uid, request.UID)
		} else {
			require.Equal(t, secondPlan.LPXScalingGroup, request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name)
			require.EqualValues(t, 3, request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex)
		}
	}
}

func TestIndependentLPXDeadlineCleanup(t *testing.T) {
	for _, tc := range []struct {
		name           string
		firstExpired   int64
		secondDeadline *int64
		secondExternal bool
		secondHealthy  bool
		wantFirst      int32
		wantSecond     int32
		wantRequests   int
	}{
		{name: "interior failure leaves the other engine's suffix removable", firstExpired: 0, secondDeadline: ptr.To(int64(60)), wantFirst: 3, wantSecond: 1, wantRequests: 4},
		{name: "healthy neighbor survives failure cleanup", firstExpired: 2, secondHealthy: true, secondDeadline: ptr.To(int64(60)), wantFirst: 2, wantSecond: 2, wantRequests: 4},
		{name: "independent expired suffixes", firstExpired: 2, secondDeadline: ptr.To(int64(60)), wantFirst: 2, wantSecond: 1, wantRequests: 3},
		{name: "external engine retains capacity ownership", firstExpired: 2, secondExternal: true, secondDeadline: ptr.To(int64(60)), wantFirst: 2, wantSecond: 2, wantRequests: 3},
		{name: "pending neighbor has a longer deadline", firstExpired: 2, secondDeadline: ptr.To(int64(300)), wantFirst: 2, wantSecond: 2, wantRequests: 4},
		{name: "pending neighbor has no deadline", firstExpired: 2, wantFirst: 2, wantSecond: 2, wantRequests: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Publish two workloads with different replica axes and active scheduling clocks")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			dgd.Spec.Components[0].Replicas = ptr.To(int32(3))
			second := dgd.Spec.Components[0].DeepCopy()
			second.ComponentName, second.Replicas = "timeout-engine", ptr.To(int32(2))
			second.LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: tc.secondDeadline}
			if tc.secondExternal {
				second.Replicas = nil
			}
			dgd.Spec.Components = append(dgd.Spec.Components, *second)
			dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To(int64(60))}
			r := newLPXTestReconciler(t, registry, child, dgd)
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
			require.NoError(t, err)
			pcs, _, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
			require.NoError(t, err)
			firstPlan, secondPlan := plans["lpx"], plans["timeout-engine"]
			secondPlan.Replicas = 2
			createLPXTestObjects(t, t.Context(), r.Client, materializeLPXTestPCS(t, child, pcs, firstPlan, secondPlan)...)
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			_, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			t.Log("Observe the synchronized PCS before publishing scheduler requests")
			_, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			requests, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Len(t, requests, 5)
			for _, request := range requests {
				target := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
				phase := lpxv1alpha1.RequestPhaseBound
				if target.Name == firstPlan.LPXScalingGroup && target.ReplicaIndex == tc.firstExpired || !tc.secondHealthy && target.Name == secondPlan.LPXScalingGroup && target.ReplicaIndex == 1 {
					phase = lpxv1alpha1.RequestPhasePending
				}
				request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now().Add(-2*time.Minute), phase).Status
				require.NoError(t, r.Update(t.Context(), request))
			}

			t.Log("Persist scheduling failure before deleting either workload's requests")
			result, err := r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			require.Positive(t, result.RequeueAfter)
			requests, err = r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Len(t, requests, 5)

			t.Log("Apply each workload's suffix rule and capacity ownership independently")
			_, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			pcsgs, err := getPodCliqueScalingGroups(t.Context(), r.Client, pcs)
			require.NoError(t, err)
			require.Equal(t, tc.wantFirst, pcsgs[firstPlan.LPXScalingGroup].Spec.Replicas)
			require.Equal(t, tc.wantSecond, pcsgs[secondPlan.LPXScalingGroup].Spec.Replicas)
			requests, err = r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Len(t, requests, tc.wantRequests)

			t.Log("A failed generation must not republish either workload's retired requests")
			_, err = r.Reconcile(t.Context(), key)
			require.NoError(t, err)
			again, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Equal(t, requests, again)
		})
	}
}
