// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"slices"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestLPXPriorityEditsPreservePublishedRequests(t *testing.T) {
	for _, scenario := range []struct {
		name    string
		initial *string
		phase   lpxv1alpha1.RequestPhase
	}{
		{"omitted pending", nil, lpxv1alpha1.RequestPhasePending},
		{"explicit pending", new("high"), lpxv1alpha1.RequestPhasePending},
		{"omitted committed", nil, lpxv1alpha1.RequestPhaseBound},
		{"explicit committed", new("high"), lpxv1alpha1.RequestPhaseBound},
	} {
		name, initial := scenario.name, scenario.initial
		t.Run(name, func(t *testing.T) {
			t.Log("Author a class and supply compiler and Grove observations without seeding LPRs")
			ctx := t.Context()
			_, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			component := dgd.GetComponentByName("lpx")
			component.LPX.Scheduling = &v1beta1.SchedulingSpec{PriorityClassName: initial, AttemptDeadlineSeconds: new(int64(3600))}
			child := newLPXTestDeployment(t, dgd)
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

			t.Log("Let production Reconcile publish the authored class")
			for range 2 {
				_, err := r.Reconcile(ctx, key)
				require.NoError(t, err)
			}
			pcs := observedLPXTestPodCliqueSet(t, ctx, r, child, selected)
			requests, err := r.getPipelineRequests(ctx, pcs)
			require.NoError(t, err)
			require.Len(t, requests, 1)
			for _, request := range requests {
				require.Equal(t, initial, request.Spec.PriorityClassName)
				require.NotEmpty(t, request.UID)
			}
			t.Log("Supply scheduler lifecycle observations solely to prove producer retention")
			for _, request := range requests {
				request.Status = newTestPipelineRequest(child, pcs, request.Name, time.Now().Add(-time.Minute), scenario.phase).Status
				require.NoError(t, r.Update(ctx, request))
			}
			requests, err = r.getPipelineRequests(ctx, pcs)
			require.NoError(t, err)
			initialTemplate := pcs.Spec.Template.DeepCopy()

			for _, next := range []*string{new("higher"), nil, new("new-class")} {
				t.Log("Advance only the priority input revision and retain exact published intent")
				previousRevision := child.Spec.InputRevision
				component.LPX.Scheduling.PriorityClassName = next
				updateTestDGD(t, r, child, dgd)
				require.NotEqual(t, previousRevision, child.Spec.InputRevision)
				for range 2 {
					_, err := r.Reconcile(ctx, key)
					require.NoError(t, err)
				}
				current, err := getPodCliqueSet(ctx, r.Client, child)
				require.NoError(t, err)
				require.NotNil(t, current)
				require.Equal(t, pcs.UID, current.UID)
				require.True(t, current.DeletionTimestamp.IsZero())
				require.Equal(t, initialTemplate, current.Spec.Template.DeepCopy())
				retained, err := r.getPipelineRequests(ctx, current)
				require.NoError(t, err)
				require.Equal(t, requests, retained, "preserve UID, class, spec, status and clocks")
			}
		})
	}
}

func TestLPXPriorityPublicationGroups(t *testing.T) {
	for _, shape := range []string{"independent", "shared fanout", "agent runtime", "all local"} {
		t.Run(shape, func(t *testing.T) {
			t.Log("Construct authored group policies and compiler inputs")
			pipeline := lpx.PipelineSingle
			if shape == "all local" {
				pipeline = lpx.PipelineHybrid
			}
			_, dgd, registry := newLPXTestDGD(t, pipeline)
			if shape == "shared fanout" {
				_, dgd, registry = newLPXSpecDecodeTestDGD(t)
			}
			target := dgd.GetComponentByName("lpx")
			target.LPX.Scheduling = &v1beta1.SchedulingSpec{PriorityClassName: new("high")}
			want := 1
			switch shape {
			case "independent":
				second := target.DeepCopy()
				second.ComponentName = "second"
				second.LPX.Scheduling.PriorityClassName = new("low")
				dgd.Spec.Components = append(dgd.Spec.Components, *second)
				want = 2
			case "shared fanout":
				want = 3
			case "agent runtime":
				target.Roles = slices.DeleteFunc(target.Roles, func(role v1beta1.ComponentRoleSpec) bool { return role.Name == v1beta1.ComponentRoleLPXConductor })
				target.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.InitContainers = []corev1.Container{{Name: consts.RuntimeContainerName, Image: "runtime", RestartPolicy: new(corev1.ContainerRestartPolicyAlways)}}
			case "all local":
				target.LPX.Experimental = &v1beta1.LPXExperimentalSpec{LocalPartitions: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll}}
				want = 0
			}
			child := newLPXTestDeployment(t, dgd)
			r := newLPXTestReconciler(t, registry, child, dgd)
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
			require.NoError(t, err)
			pcs, _, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
			require.NoError(t, err)
			orderedPlans := []*lpx.MaterializationPlan{plans["lpx"]}
			if shape == "independent" {
				orderedPlans = append(orderedPlans, plans["second"])
			}
			objects := materializeLPXTestPCS(t, child, pcs, orderedPlans...)
			createLPXTestObjects(t, t.Context(), r.Client, objects...)

			t.Log("Production publication follows serving-group policy without creating all-local requests")
			for range 2 {
				_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
				require.NoError(t, err)
			}
			requests, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			require.Len(t, requests, want)
			for _, request := range requests {
				expected := "high"
				if shape == "independent" && request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name == plans["second"].LPXScalingGroup {
					expected = "low"
				}
				require.Equal(t, new(expected), request.Spec.PriorityClassName)
			}
			require.Empty(t, pcs.Spec.Template.PriorityClassName, "LPX policy does not become Grove Pod priority")
		})
	}
}

func TestLPXPriorityResolutionPreservesInputs(t *testing.T) {
	t.Log("Resolve a new class independently of the caller's pointer")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	selected := resolveLPXTestWorkload(t, registry, t.Context(), child, dgd)
	priority := "high"
	_, created := resolvePipelineRequests(child, nil, selected.workload, selected.plan, &priority)
	require.Len(t, created, 1)
	priority = "changed"
	require.Equal(t, new("high"), created[0].Spec.PriorityClassName)

	t.Log("A class edit returns the actual observed identity without mutating either input")
	observed := created[0]
	observed.UID = types.UID("retained-uid")
	before := observed.DeepCopy()
	resolved, missing := resolvePipelineRequests(child, map[string]*lpxv1alpha1.LPUPipelineRequest{observed.Name: observed}, selected.workload, selected.plan, &priority)
	require.Empty(t, missing)
	require.Same(t, observed, resolved[observed.Name])
	require.Equal(t, before, observed)
	require.Equal(t, "changed", priority)
}

func TestLPXPriorityDeadlineRetryPreservesExternalCapacity(t *testing.T) {
	t.Log("Publish two externally managed replicas through production reconciliation")
	ctx := t.Context()
	_, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	component := dgd.GetComponentByName("lpx")
	component.Replicas, component.MinAvailable = nil, new(int32(2))
	component.LPX.Scheduling = &v1beta1.SchedulingSpec{PriorityClassName: new("old"), AttemptDeadlineSeconds: new(int64(30))}
	child := newLPXTestDeployment(t, dgd)
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 2 {
		_, err := r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	pcs := observedLPXTestPodCliqueSet(t, ctx, r, child, selected)
	survivor := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].Name)
	failed := getTestPipelineRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].Name)
	survivor.Status = newTestPipelineRequest(child, pcs, survivor.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhaseBound).Status
	require.NoError(t, r.Update(ctx, survivor))
	failed.Status = newTestPipelineRequest(child, pcs, failed.Name, time.Now().Add(-time.Hour), lpxv1alpha1.RequestPhasePending).Status
	failed.Finalizers = []string{"test.example/scheduler-release"}
	require.NoError(t, r.Update(ctx, failed))
	survivor = getTestPipelineRequest(t, ctx, r.Client, survivor.Namespace, survivor.Name)

	t.Log("Persist deadline failure and start retirement without reducing external capacity")
	for range 3 {
		_, err := r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, "SchedulingFailed"))
	failed = getTestPipelineRequest(t, ctx, r.Client, failed.Namespace, failed.Name)
	require.False(t, failed.DeletionTimestamp.IsZero())
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	groupKey := client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}
	require.NoError(t, r.Get(ctx, groupKey, group))
	require.Equal(t, int32(2), group.Spec.Replicas)

	t.Log("A priority revision waits for scheduler finalization with the old class intact")
	component.LPX.Scheduling.PriorityClassName = new("new")
	updateTestDGD(t, r, child, dgd)
	for range 2 {
		_, err := r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	require.Equal(t, failed, getTestPipelineRequest(t, ctx, r.Client, failed.Namespace, failed.Name))

	t.Log("The retry uses current policy without rewriting the survivor or external capacity")
	failed.Finalizers = nil
	require.NoError(t, r.Update(ctx, failed))
	for range 4 {
		_, err := r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	replacement := getTestPipelineRequest(t, ctx, r.Client, failed.Namespace, failed.Name)
	require.NotEqual(t, failed.UID, replacement.UID)
	require.Equal(t, new("new"), replacement.Spec.PriorityClassName)
	require.Equal(t, survivor, getTestPipelineRequest(t, ctx, r.Client, survivor.Namespace, survivor.Name))
	require.NoError(t, r.Get(ctx, groupKey, group))
	require.Equal(t, int32(2), group.Spec.Replicas)
	currentPCS := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), currentPCS))
	require.Equal(t, pcs.UID, currentPCS.UID)
	require.Equal(t, pcs.Spec.Template, currentPCS.Spec.Template)
}
