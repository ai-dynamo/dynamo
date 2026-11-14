// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"fmt"
	"maps"
	"net/url"
	"slices"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestCommittedFabricPods(t *testing.T) {
	const group = "pcs-0-lpx"
	bound, degraded := lpxv1alpha1.RequestPhaseBound, lpxv1alpha1.RequestPhaseDegraded

	for _, tc := range []struct {
		name     string
		requests []*lpxv1alpha1.LPUPipelineRequest
		want     []string
		settled  bool
	}{
		{name: "a replica without requests is not settled"},
		{
			name:     "one bound plan lists its fabric pods one time, sorted",
			requests: []*lpxv1alpha1.LPUPipelineRequest{newTestPlacedPipelineRequest("target", group, 0, bound, "b", "a", "a")},
			want:     []string{"a", "b"}, settled: true,
		},
		{
			name: "all models of the replica contribute their fabric pods; a Degraded request is settled",
			requests: []*lpxv1alpha1.LPUPipelineRequest{
				newTestPlacedPipelineRequest("draft", group, 0, bound, "a"),
				newTestPlacedPipelineRequest("target", group, 0, degraded, "c"),
			},
			want: []string{"a", "c"}, settled: true,
		},
		{
			name: "one model without a realized plan keeps the replica unsettled",
			requests: []*lpxv1alpha1.LPUPipelineRequest{
				newTestPlacedPipelineRequest("draft", group, 0, bound, "a"),
				newTestPlacedPipelineRequest("target", group, 0, lpxv1alpha1.RequestPhaseBinding, "a"),
			},
		},
		{
			name:     "a plan that the scheduler releases can move",
			requests: []*lpxv1alpha1.LPUPipelineRequest{newTestPlacedPipelineRequest("target", group, 0, lpxv1alpha1.RequestPhaseReleasing, "a")},
		},
		{
			name: "other replicas and other groups are ignored",
			requests: []*lpxv1alpha1.LPUPipelineRequest{
				newTestPlacedPipelineRequest("target", group, 0, bound, "a"),
				newTestPlacedPipelineRequest("target-1", group, 1, lpxv1alpha1.RequestPhasePending),
				newTestPlacedPipelineRequest("other", "pcs-0-other", 0, bound, "z"),
			},
			want: []string{"a"}, settled: true,
		},
		{
			name:     "a bound plan without rows gives no fabric pod",
			requests: []*lpxv1alpha1.LPUPipelineRequest{newTestPlacedPipelineRequest("target", group, 0, bound)},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requests := make(map[string]*lpxv1alpha1.LPUPipelineRequest)
			for _, request := range tc.requests {
				requests[request.Name] = request
			}
			// The requests have no order, so the result must not depend on one.
			for range 64 {
				fabricPods, settled := committedFabricPods(requests, group, 0)
				require.Equal(t, tc.settled, settled)
				require.Equal(t, tc.want, fabricPods)
			}
		})
	}

	t.Log("A request that is being deleted no longer fixes the fabric pods")
	deleting := newTestPlacedPipelineRequest("target", group, 0, bound, "a")
	deleting.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	_, settled := committedFabricPods(map[string]*lpxv1alpha1.LPUPipelineRequest{deleting.Name: deleting}, group, 0)
	require.False(t, settled)
}

func TestLPXCyborgFabricPodPlacement(t *testing.T) {
	ctx := t.Context()

	t.Log("Select per-fabric-pod placement for one hybrid workload in a cluster with three fabric pods")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: lpx.CyborgPlacementPerFabricPod}
	r := newLPXTestReconciler(t, registry, child, dgd,
		newTestFabricPodNode("gpu-a", "a"), newTestFabricPodNode("gpu-b", "b"), newTestFabricPodNode("gpu-c", "c"),
		&corev1.Node{Name: "unlabeled"},
	)
	workloads, plans, err := r.resolveWorkloads(ctx, child, dgd)
	require.NoError(t, err)
	pcs, _, err := r.renderPodCliqueSet(ctx, child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	createLPXTestObjects(t, ctx, r.Client, materializeLPXTestPCS(t, child, pcs, plans["lpx"])...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	cliqueKey := client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(0).CyborgClique}

	t.Log("Provision: before the scheduler commits a plan, keep one Cyborg position in every fabric pod")
	for range 3 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	clique := &grovev1alpha1.PodClique{}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, lpx.CyborgPlacementPerFabricPod, clique.Labels[lpx.CyborgPlacementLabel], "Pods inherit the label that Pod admission and the Pod watch select")
	require.Equal(t, int32(3), clique.Spec.Replicas)
	requests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	require.Len(t, requests, 1, "placement does not delay request publication")

	t.Log("Hold each new Pod while Grove's gate is on it: Grove removes its gate with a list of the gates that stay")
	createLPXTestObjects(t, ctx, r.Client, newTestCyborgPod(clique, 0), newTestCyborgPod(clique, 1), newTestCyborgPod(clique, 2))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"", "", ""}, testCyborgFabricPods(pods))

	t.Log("After Grove releases the Pods, pin each one to the fabric pod of its Grove index and remove the last gate in the same write")
	for _, pod := range pods {
		pod.Spec.SchedulingGates = []corev1.PodSchedulingGate{{Name: lpx.CyborgPlacementGate}}
		require.NoError(t, r.Update(ctx, pod))
	}
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a", "b", "c"}, testCyborgFabricPods(pods))
	require.Equal(t, []string{"0/0", "0/1", "0/2"}, testCyborgRanks(pods), "one Pod for each fabric pod has rank 0, and its Grove index is its replica index")
	for _, pod := range pods {
		require.Equal(t, map[string]string{fabricPodLabel: pod.Labels[cyborgFabricPodLabel]}, pod.Spec.NodeSelector)
		require.Empty(t, pod.Spec.SchedulingGates)
	}

	t.Log("A converged provision writes nothing")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Equal(t, pods, listTestCyborgPods(t, ctx, r.Client, clique))

	t.Log("Prune: after the scheduler binds the LPUs in fabric pod b, delete the other Cyborg Pods before lowering capacity")
	for _, pod := range pods {
		pod.Spec.NodeName = "gpu-" + pod.Labels[cyborgFabricPodLabel]
		require.NoError(t, r.Update(ctx, pod))
	}
	placeTestPipelineRequests(t, ctx, r, pcs, plans["lpx"].LPXScalingGroup, lpxv1alpha1.RequestPhaseBound, "b")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"b"}, testCyborgFabricPods(pods))
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("A capacity that waits for Grove is a replica change that is not done: a PodCliqueSet write in this time reports deferred scaling")
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, dynamo.LPXRestartAnnotation, "during-prune")
	require.NoError(t, r.Update(ctx, child))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	require.True(t, meta.IsStatusConditionTrue(child.Status.Conditions, "ScalingDeferred"))

	t.Log("The capacity stays while Grove has not created a Pod for each removed one: Grove must not observe the lower capacity before the removals")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("When Grove has created those Pods, lower the capacity; Grove then removes its Pods, which have no Node")
	replaceTestCyborgPods(t, ctx, r.Client, clique)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)
	removeTestExcessCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []int{1}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))

	t.Log("The Pod that stays keeps its Grove index, its rank and its replica index: the controller adds and removes no Pod for it")
	stays := listTestCyborgPods(t, ctx, r.Client, clique)
	for range 2 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, stays, pods)
	require.Equal(t, []string{"0/1"}, testCyborgRanks(pods))
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)

	t.Log("Scale: two replicas mean two Cyborg Pods in each occupied fabric pod")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = new(int32(2))
	updateTestDGD(t, r, child, dgd)
	for range 3 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"b", "b"}, testCyborgFabricPods(pods))
	require.Equal(t, []string{"1/3", "0/1"}, testCyborgRanks(pods), "the new Pod takes the free rank, and the lowest free replica index that is congruent to it")

	t.Log("Provision again when the scheduler releases the plan, and keep the Pods that are in place")
	requests, err = r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	for _, request := range requests {
		request.Status.Phase = lpxv1alpha1.RequestPhaseReleasing
		require.NoError(t, r.Update(ctx, request))
	}
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(6), clique.Spec.Replicas)
	createLPXTestObjects(t, ctx, r.Client,
		newTestReleasedCyborgPod(clique, 2), newTestReleasedCyborgPod(clique, 3), newTestReleasedCyborgPod(clique, 4), newTestReleasedCyborgPod(clique, 5))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"b", "b", "a", "a", "c", "c"}, testCyborgFabricPods(pods))
	require.Equal(t, []string{"1/3", "0/1", "0/2", "1/5", "0/4", "1/7"}, testCyborgRanks(pods), "each fabric pod has the ranks 0 and 1, and no replica index occurs two times")
}

func TestLPXCyborgPinDoesNotWaitForDesiredState(t *testing.T) {
	ctx := t.Context()

	t.Log("Run one hybrid workload with per-fabric-pod placement in a cluster with one fabric pod")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: lpx.CyborgPlacementPerFabricPod}
	r := newLPXTestReconciler(t, registry, child, dgd, newTestFabricPodNode("gpu-a", "a"))
	workloads, plans, err := r.resolveWorkloads(ctx, child, dgd)
	require.NoError(t, err)
	pcs, _, err := r.renderPodCliqueSet(ctx, child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	createLPXTestObjects(t, ctx, r.Client, materializeLPXTestPCS(t, child, pcs, plans["lpx"])...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	for range 3 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	clique := &grovev1alpha1.PodClique{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(0).CyborgClique}, clique))

	t.Log("A build download for a new revision holds every pass before the desired workload is resolved")
	r.modelRegistry = &downloadOrderedLPXRegistry{ModelRegistry: registry, buildURL: url.URL{Scheme: lpx.BuildSchemeGCS, Host: "bucket", Path: "/build"}}

	t.Log("Grove replaces a Cyborg Pod of the running workload, and admission gates the new Pod")
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	result, err := r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Positive(t, result.RequeueAfter, "the pass waits for the download")

	t.Log("The pin does not wait: the placement of the running workload is known from observed objects")
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(pods))
	require.Empty(t, pods[0].Spec.SchedulingGates)

	t.Log("A pass that waits only pins: a second Pod in the same fabric pod stays, and the capacity is not written")
	extra := newTestReleasedCyborgPod(clique, 1)
	extra.Labels[cyborgFabricPodLabel], extra.Spec.NodeName = "a", "gpu-a"
	createLPXTestObjects(t, ctx, r.Client, extra)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	clique.Spec.Replicas = 2
	require.NoError(t, r.Update(ctx, clique))
	r.modelRegistry = &downloadOrderedLPXRegistry{ModelRegistry: registry, buildURL: url.URL{Scheme: lpx.BuildSchemeGCS, Host: "bucket", Path: "/other-build"}}
	result, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Positive(t, result.RequeueAfter)
	require.Len(t, listTestCyborgPods(t, ctx, r.Client, clique), 2)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)
	require.NoError(t, r.Delete(ctx, extra))

	t.Log("The DGD changes, and its controller does not deliver the new input revision to the LPX deployment")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	dgd.Spec.Env = append(dgd.Spec.Env, corev1.EnvVar{Name: "CHANGED", Value: "after-the-handoff"})
	require.NoError(t, r.Update(ctx, dgd))

	t.Log("Grove replaces the Cyborg Pod again, and the pin does not wait for the handoff")
	require.NoError(t, r.Delete(ctx, pods[0]))
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
	require.Contains(t, ready.Message, "Waiting for the matching DynamoGraphDeployment")
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(pods))
	require.Empty(t, pods[0].Spec.SchedulingGates)

	t.Log("The API server refuses the pin of the next Pod: the pass reports it, also when it stops before the workloads")
	require.NoError(t, r.Delete(ctx, pods[0]))
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Patch: func(ctx context.Context, delegated client.WithWatch, object client.Object, patch client.Patch, opts ...client.PatchOption) error {
			if _, pod := object.(*corev1.Pod); pod {
				return apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, object.GetName(), fmt.Errorf("denied"))
			}
			return delegated.Patch(ctx, object, patch, opts...)
		},
	})
	_, err = r.Reconcile(ctx, key)
	require.ErrorContains(t, err, "place Cyborg Pod")
}

func TestLPXCyborgPlacementErrorDoesNotStopItsCorrection(t *testing.T) {
	ctx := t.Context()

	t.Log("Select per-fabric-pod placement with a Cyborg node selector that matches no Node")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: lpx.CyborgPlacementPerFabricPod}
	conductor.PodTemplate.Spec.NodeSelector = map[string]string{"pool": "wrong"}
	r := newLPXTestReconciler(t, registry, child, dgd, newTestFabricPodNode("gpu-a", "a", "pool", "gpu"))
	workloads, plans, err := r.resolveWorkloads(ctx, child, dgd)
	require.NoError(t, err)
	pcs, _, err := r.renderPodCliqueSet(ctx, child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	createLPXTestObjects(t, ctx, r.Client, materializeLPXTestPCS(t, child, pcs, plans["lpx"])...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	cliqueKey := client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(0).CyborgClique}

	t.Log("Report the PodClique that has no selectable Node")
	_, err = r.Reconcile(ctx, key)
	require.ErrorContains(t, err, "no Node with label")
	require.ErrorContains(t, err, cliqueKey.Name)

	t.Log("The user corrects the selector: the error must not stop the PodCliqueSet write")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec.NodeSelector = map[string]string{"pool": "gpu"}
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(ctx, key)
	require.ErrorContains(t, err, "no Node with label", "Grove has not replaced the PodClique yet")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), live))
	index := slices.IndexFunc(live.Spec.Template.Cliques, func(template *grovev1alpha1.PodCliqueTemplateSpec) bool {
		return template.Name == plans["lpx"].CyborgTemplate
	})
	require.Equal(t, map[string]string{"pool": "gpu"}, live.Spec.Template.Cliques[index].Spec.PodSpec.NodeSelector)

	t.Log("The PodClique still has the old selector, so the placement still has no Node: it reads the PodClique, not the template")
	_, err = r.Reconcile(ctx, key)
	require.ErrorContains(t, err, "no Node with label")

	t.Log("After Grove gives the PodClique the corrected template, the placement provisions")
	clique := &grovev1alpha1.PodClique{}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	clique.Spec.PodSpec.NodeSelector = map[string]string{"pool": "gpu"}
	require.NoError(t, r.Update(ctx, clique))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
}

func TestLPXCyborgPlacementContinuesPastOnePodClique(t *testing.T) {
	ctx := t.Context()

	t.Log("Two PodCliques have the placement label: the first one selects no Node, the second one selects fabric pod a")
	pcs := &grovev1alpha1.PodCliqueSet{Name: "pcs", Namespace: "default"}
	pcs.Spec.Template.Cliques = []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "cond", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}}}
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{{Name: "lpx", CliqueNames: []string{"cond"}}}
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{Name: "pcs-0-lpx", Namespace: "default"}
	pcsg.Spec.Replicas = 2
	placed := map[string]string{lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod}
	unselectable := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-0", ResourceVersion: "1", Labels: placed}
	unselectable.Spec.Replicas, unselectable.Spec.PodSpec.NodeSelector = 1, map[string]string{"pool": "wrong"}
	selectable := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-1-cond", Namespace: "default", UID: "clique-1", ResourceVersion: "1", Labels: placed}
	selectable.Spec.Replicas = 1
	pod := newTestReleasedCyborgPod(selectable, 0)
	r := &graphReconciler{
		Client: newLPXTestClient(t, unselectable, selectable, pod, newTestFabricPodNode("gpu-a", "a", "pool", "gpu")),
		config: &configv1alpha1.OperatorConfiguration{},
	}
	pcsgs := map[string]*grovev1alpha1.PodCliqueScalingGroup{pcsg.Name: pcsg}
	pclqs := map[string]*grovev1alpha1.PodClique{unselectable.Name: unselectable, selectable.Name: selectable}

	t.Log("Name the PodClique without a Node, and still give the other PodClique its placement")
	placements, err := r.resolveCyborgPlacements(ctx, pcs, pcsgs, pclqs, nil)
	require.ErrorContains(t, err, unselectable.Name)
	require.Equal(t, map[string]cyborgPlacement{selectable.Name: {fabricPods: []string{"a"}, replicas: 1}}, placements)

	t.Log("Pin the Pod of the PodClique that has a placement")
	require.NoError(t, r.pinCyborgPods(ctx, pclqs, placements))
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, selectable)))
}

func TestLPXCyborgPlacementStepsContinuePastOnePodClique(t *testing.T) {
	ctx := t.Context()

	t.Log("Given two placed PodCliques that need one Pod more capacity, each with one Pod to pin, and an API server that refuses the pin for the first PodClique")
	refused := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-0", ResourceVersion: "1", Labels: map[string]string{
		lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod,
	}}
	accepted := refused.DeepCopy()
	accepted.Name, accepted.UID = "pcs-0-lpx-1-cond", "clique-1"
	refused.Spec.Replicas, accepted.Spec.Replicas = 1, 1
	refusedPod := newTestReleasedCyborgPod(refused, 0)
	base := newLPXTestClient(t, refused, accepted, refusedPod, newTestReleasedCyborgPod(accepted, 0))
	r := &graphReconciler{Client: interceptor.NewClient(base, interceptor.Funcs{
		Patch: func(ctx context.Context, delegated client.WithWatch, object client.Object, patch client.Patch, opts ...client.PatchOption) error {
			if object.GetName() == refusedPod.Name {
				return apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, object.GetName(), fmt.Errorf("denied"))
			}
			return delegated.Patch(ctx, object, patch, opts...)
		},
	})}
	pclqs := map[string]*grovev1alpha1.PodClique{refused.Name: refused, accepted.Name: accepted}
	placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 2}
	placements := map[string]cyborgPlacement{refused.Name: placement, accepted.Name: placement}

	t.Log("When the pin step runs, it names the refused Pod and still pins the Pod of the second PodClique")
	err := r.pinCyborgPods(ctx, pclqs, placements)
	require.ErrorContains(t, err, refusedPod.Name)
	require.Equal(t, []string{""}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, refused)))
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, accepted)))

	t.Log("When the placement step runs, it names the refused Pod again and still writes the capacity of the second PodClique")
	err = r.reconcileCyborgPlacements(ctx, pclqs, placements, false)
	require.ErrorContains(t, err, refusedPod.Name)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(refused), refused))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(accepted), accepted))
	require.Equal(t, int32(1), refused.Spec.Replicas)
	require.Equal(t, int32(2), accepted.Spec.Replicas)
}

func TestLPXCyborgRefusedPinKeepsItsPosition(t *testing.T) {
	ctx := t.Context()

	t.Log("One position is open in fabric pod a, and Grove released two Pods")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 2
	first, second := newTestReleasedCyborgPod(clique, 0), newTestReleasedCyborgPod(clique, 1)
	first.Spec.SchedulingGates = []corev1.PodSchedulingGate{{Name: lpx.CyborgPlacementGate}}
	second.Spec.SchedulingGates = []corev1.PodSchedulingGate{{Name: lpx.CyborgPlacementGate}}
	base := newLPXTestClient(t, clique, first, second)
	r := &graphReconciler{Client: interceptor.NewClient(base, interceptor.Funcs{
		// An earlier step of the same pass pinned the first Pod; the cache does not show it yet.
		Patch: func(ctx context.Context, delegated client.WithWatch, object client.Object, patch client.Patch, opts ...client.PatchOption) error {
			if object.GetName() == first.Name {
				return apierrors.NewConflict(schema.GroupResource{Resource: "pods"}, object.GetName(), fmt.Errorf("the object has been modified"))
			}
			return delegated.Patch(ctx, object, patch, opts...)
		},
	})}

	t.Log("The refused pin keeps the position: the second Pod does not take it")
	live, err := r.liveCyborgPods(ctx, clique)
	require.NoError(t, err)
	surplus, err := r.fillCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, live)
	require.NoError(t, err)
	require.Empty(t, surplus)
	require.Equal(t, []string{"", ""}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
}

func TestLPXCyborgPlacementOutlivesItsAnnotation(t *testing.T) {
	ctx := t.Context()

	t.Log("Run one hybrid workload with one explicit Cyborg replica for each of three fabric pods")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: lpx.CyborgPlacementPerFabricPod}
	conductor.Replicas = new(int32(1))
	r := newLPXTestReconciler(t, registry, child, dgd,
		newTestFabricPodNode("gpu-a", "a"), newTestFabricPodNode("gpu-b", "b"), newTestFabricPodNode("gpu-c", "c"))
	workloads, plans, err := r.resolveWorkloads(ctx, child, dgd)
	require.NoError(t, err)
	pcs, _, err := r.renderPodCliqueSet(ctx, child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	createLPXTestObjects(t, ctx, r.Client, materializeLPXTestPCS(t, child, pcs, plans["lpx"])...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	cliqueKey := client.ObjectKey{Namespace: pcs.Namespace, Name: plans["lpx"].ForReplica(0).CyborgClique}
	for range 3 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	clique := &grovev1alpha1.PodClique{}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("The user removes the annotation; Grove has not replaced this PodClique yet, so it keeps the placement label")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Annotations = nil
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, "ScalingDeferred"), "the explicit replica count is not a pending capacity change for a placed PodClique")
	for range 2 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, lpx.CyborgPlacementPerFabricPod, clique.Labels[lpx.CyborgPlacementLabel])

	t.Log("Admission still gates the Pods of that PodClique, so the placement keeps its capacity and pins them")
	require.Equal(t, int32(3), clique.Spec.Replicas, "the explicit replica count does not replace the placement capacity")
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(pods))
	require.Empty(t, pods[0].Spec.SchedulingGates)

	t.Log("After Grove replaces the PodClique without the label, the explicit replica count applies")
	delete(clique.Labels, lpx.CyborgPlacementLabel)
	require.NoError(t, r.Update(ctx, clique))
	for range 2 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)
}

func TestLPXCyborgPlacementKeepsIndexBlocks(t *testing.T) {
	ctx := t.Context()

	t.Log("Provision two Cyborg Pods in each of three fabric pods")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1", Labels: map[string]string{
		lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod,
	}}
	clique.Spec.Replicas = 6
	placement := cyborgPlacement{fabricPods: []string{"a", "b", "c"}, replicas: 2}
	r := &graphReconciler{Client: newLPXTestClient(t, clique)}

	t.Log("Grove creates and releases the Pods concurrently, so the controller observes them in no fixed order")
	for _, indexes := range [][]int{{4, 1}, {3}, {0, 5, 2}} {
		for _, index := range indexes {
			require.NoError(t, r.Create(ctx, newTestReleasedCyborgPod(clique, index)))
		}
		err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
		require.NoError(t, err)
	}

	t.Log("Each fabric pod holds the contiguous index block of its position in name order, and each Pod has the rank and the replica index of its Grove index")
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a", "a", "b", "b", "c", "c"}, testCyborgFabricPods(pods))
	require.Equal(t, []string{"0/0", "1/1", "0/2", "1/3", "0/4", "1/5"}, testCyborgRanks(pods))
	require.Equal(t, []int{0, 1, 2, 3, 4, 5}, testCyborgPodIndexes(pods))
}

func TestLPXCyborgPlacementAfterPrune(t *testing.T) {
	ctx := t.Context()

	t.Log("Given a provision of two Cyborg Pods in each of four fabric pods, of which the pruned placement keeps b and d")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1", Labels: map[string]string{
		lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod,
	}}
	clique.Spec.Replicas = 4
	r := &graphReconciler{Client: newLPXTestClient(t, clique)}
	for _, index := range []int{2, 3, 6, 7} {
		pod := rankTestCyborgPod(newTestBoundCyborgPod(clique, index, []string{"a", "b", "c", "d"}[index/2]), index%2, index)
		pod.Status.Conditions = newTestCyborgPodConditions(true)
		require.NoError(t, r.Create(ctx, pod))
	}
	placement := cyborgPlacement{fabricPods: []string{"b", "d"}, replicas: 2}

	t.Log("When the controller reconciles the final placement, each Pod stays as it is: its Grove index is above the total, and its rank and replica index do not depend on that")
	before := listTestCyborgPods(t, ctx, r.Client, clique)
	for range 2 {
		require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	}
	require.Equal(t, before, listTestCyborgPods(t, ctx, r.Client, clique))
	require.Equal(t, int32(4), clique.Spec.Replicas)

	t.Log("When fabric pod d loses the Pod with rank 1, Grove creates a Pod with its lowest free index, which is 0")
	require.NoError(t, r.Delete(ctx, before[3]))
	replaceTestCyborgPods(t, ctx, r.Client, clique)
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))

	t.Log("Then the new Pod is in fabric pod d with rank 1, and with the lowest free replica index that is congruent to its rank")
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []int{2, 3, 6, 0}, testCyborgPodIndexes(pods))
	require.Equal(t, []string{"b", "b", "d", "d"}, testCyborgFabricPods(pods))
	require.Equal(t, []string{"0/2", "1/3", "0/6", "1/1"}, testCyborgRanks(pods))
	require.Equal(t, map[string]string{fabricPodLabel: "d"}, pods[3].Spec.NodeSelector)
	require.Equal(t, before[:3], pods[:3], "the other Pods do not change")
}

func TestLPXCyborgPodWithoutItsOwnRankHoldsNoPosition(t *testing.T) {
	for _, tc := range []struct {
		name string
		// rank and index are the labels of the Pod with Grove index 1; the Pod with
		// Grove index 0 has rank 0 and replica index 0. An empty value is no label.
		rank, index string
	}{
		{name: "two passes gave one rank to two Pods of a fabric pod", rank: "0", index: "1"},
		{name: "two passes gave one replica index to two Pods", rank: "1", index: "0"},
		{name: "a placed Pod has no rank", index: "1"},
		{name: "a placed Pod has no replica index", rank: "1"},
		{name: "a rank is not a number", rank: "one", index: "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given two placed Cyborg Pods in fabric pod a, which has two positions")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1", Labels: map[string]string{
				lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod,
			}}
			clique.Spec.Replicas = 2
			other := newTestBoundCyborgPod(clique, 1, "a")
			delete(other.Labels, cyborgFabricPodRankLabel)
			delete(other.Labels, cyborgReplicaIndexLabel)
			if tc.rank != "" {
				other.Labels[cyborgFabricPodRankLabel] = tc.rank
			}
			if tc.index != "" {
				other.Labels[cyborgReplicaIndexLabel] = tc.index
			}
			r := &graphReconciler{Client: newLPXTestClient(t, clique, newTestBoundCyborgPod(clique, 0, "a"), other)}
			placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 2}

			t.Log("When the controller reconciles the placement, it removes the Pod that has no rank and replica index of its own")
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
			require.Equal(t, []int{0}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
			require.Equal(t, int32(2), clique.Spec.Replicas)

			t.Log("Then the Pod that Grove creates gets the rank and the replica index that are free")
			replaceTestCyborgPods(t, ctx, r.Client, clique)
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
			pods := listTestCyborgPods(t, ctx, r.Client, clique)
			require.Equal(t, []string{"a", "a"}, testCyborgFabricPods(pods))
			require.Equal(t, []string{"0/0", "1/1"}, testCyborgRanks(pods))
		})
	}
}

func TestCyborgRanks(t *testing.T) {
	type held struct {
		fabricPod   string
		rank, index int
	}

	for _, tc := range []struct {
		name     string
		replicas int32
		// held are the ranks and replica indexes of the Pods that keep a position.
		held []held
		// fabricPod and podIndex are the open position and the Grove index of the new Pod.
		fabricPod   string
		podIndex    int
		rank, index int
	}{
		{name: "a provision gives the rank and the replica index of the Grove index", replicas: 2, fabricPod: "b", podIndex: 3, rank: 1, index: 3},
		{
			name: "the result of a provision does not depend on the Pods that are placed", replicas: 2,
			held: []held{{"a", 0, 0}, {"b", 0, 2}, {"c", 1, 5}}, fabricPod: "b", podIndex: 3, rank: 1, index: 3,
		},
		{
			name: "a replacement takes the free rank of its fabric pod and the lowest free replica index of that rank", replicas: 2,
			held: []held{{"b", 0, 2}, {"b", 1, 3}, {"d", 0, 6}}, fabricPod: "d", podIndex: 0, rank: 1, index: 1,
		},
		{
			name: "a replica index of a Pod in a different fabric pod is not free", replicas: 2,
			held: []held{{"b", 0, 2}, {"b", 1, 1}, {"d", 0, 6}}, fabricPod: "d", podIndex: 0, rank: 1, index: 3,
		},
		{
			name: "the Grove index is the replica index when it is congruent to the free rank", replicas: 2,
			held: []held{{"d", 1, 7}}, fabricPod: "d", podIndex: 4, rank: 0, index: 4,
		},
		{
			name: "three Pods for each fabric pod", replicas: 3,
			held: []held{{"a", 0, 0}, {"a", 2, 2}, {"b", 1, 1}}, fabricPod: "a", podIndex: 9, rank: 1, index: 4,
		},
		{name: "one Pod for each fabric pod has rank 0", replicas: 1, held: []held{{"a", 0, 0}}, fabricPod: "b", podIndex: 5, rank: 0, index: 5},
		{name: "a Pod without a Grove index takes the lowest free numbers", replicas: 2, held: []held{{"a", 0, 0}}, fabricPod: "a", podIndex: -1, rank: 1, index: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given the ranks and the replica indexes that the Pods of a PodClique hold")
			ranks := newCyborgRanks(tc.replicas)
			for _, pod := range tc.held {
				require.True(t, ranks.hold(pod.fabricPod, pod.rank, pod.index))
			}

			t.Log("When a Pod takes an open position, then it gets a rank that is free in its fabric pod and a replica index that is free in the PodClique")
			rank, index := ranks.take(tc.fabricPod, tc.podIndex)
			require.Equal(t, tc.rank, rank)
			require.Equal(t, tc.index, index)
			require.Equal(t, rank, index%int(tc.replicas), "the replica index selects the FPGA route of the rank")

			t.Log("And no other Pod can hold that rank in the fabric pod, or that replica index")
			require.False(t, ranks.hold(tc.fabricPod, rank, index+int(tc.replicas)*100))
			require.False(t, ranks.hold("other", rank, index))
			require.True(t, ranks.hold("other", rank, index+int(tc.replicas)*100), "a rank is a number of one fabric pod")
		})
	}
}

func TestLPXCyborgCapacityWaitsForGrove(t *testing.T) {
	for _, tc := range []struct {
		name string
		// created is the number of Pods without a position that Grove created.
		created int
		// removed adds a Pod in deletion, and surplus a Pod that the placement does not keep.
		removed, surplus bool
		// replicas is the count of the placement, and capacity is the capacity before the pass.
		replicas, capacity int32
		want               int32
	}{
		{name: "a lower capacity waits while the PodClique has fewer live Pods than its capacity: Grove has not created a Pod for a removed one", replicas: 1, capacity: 2, want: 2},
		{name: "a Pod in deletion is not a live Pod", removed: true, replicas: 1, capacity: 2, want: 2},
		{name: "a lower capacity is written when the PodClique has its capacity of live Pods", created: 1, replicas: 1, capacity: 2, want: 1},
		{name: "a lower capacity does not wait for a removed Pod that still stops: Grove has observed its removal", created: 1, removed: true, replicas: 1, capacity: 2, want: 1},
		{name: "a lower capacity waits in the pass that removes a Pod, also with the capacity of live Pods", surplus: true, replicas: 1, capacity: 2, want: 2},
		{name: "a lower capacity waits while the PodClique has more live Pods than its capacity: Grove has not removed its Pods", created: 2, replicas: 1, capacity: 2, want: 2},
		{name: "a higher capacity does not wait", replicas: 2, capacity: 1, want: 2},
		{name: "a higher capacity does not wait in the pass that removes a Pod", surplus: true, replicas: 3, capacity: 2, want: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given a Cyborg PodClique with one Pod in fabric pod a, and possibly Pods that Grove created, a removed Pod, or a Pod outside the placement")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
			clique.Spec.Replicas = tc.capacity
			objects := []client.Object{clique, newTestBoundCyborgPod(clique, 0, "a")}
			for index := range tc.created {
				objects = append(objects, newTestReleasedCyborgPod(clique, 4+index))
			}
			if tc.removed {
				objects = append(objects, newTestStoppingCyborgPod(clique, 1))
			}
			if tc.surplus {
				objects = append(objects, newTestBoundCyborgPod(clique, 2, "b"))
			}
			r := &graphReconciler{Client: newLPXTestClient(t, objects...)}

			t.Log("When the controller reconciles the placement")
			require.NoError(t, r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: tc.replicas}, false))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))

			t.Log("Then the capacity is written or waits, and the Pod in fabric pod a and the Pods that Grove created stay")
			require.Equal(t, tc.want, clique.Spec.Replicas)
			require.Len(t, listTestCyborgPods(t, ctx, r.Client, clique), 1+tc.created)
		})
	}
}

func TestLPXCyborgPlacementIgnoresAnEarlierPodClique(t *testing.T) {
	ctx := t.Context()

	t.Log("Given a Cyborg PodClique that Grove created again with the same name, with one Pod in fabric pod a, and a bound Pod of the earlier PodClique in fabric pod b")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 2
	earlier := clique.DeepCopy()
	earlier.UID = "earlier-clique-uid"
	r := &graphReconciler{Client: newLPXTestClient(t, clique, newTestBoundCyborgPod(clique, 0, "a"), newTestBoundCyborgPod(earlier, 5, "b"))}

	t.Log("When the controller reconciles a placement of one Pod in fabric pod a")
	require.NoError(t, r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, false))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))

	t.Log("Then the Pod of the earlier PodClique stays, and it does not count as a live Pod of this one: the capacity waits")
	require.Equal(t, []int{0, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)
}

func TestLPXCyborgTerminalPodHoldsNoPosition(t *testing.T) {
	for _, tc := range []struct {
		name  string
		phase corev1.PodPhase
	}{
		{name: "a Failed Pod, for example after an eviction", phase: corev1.PodFailed},
		{name: "a Succeeded Pod", phase: corev1.PodSucceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given two Cyborg Pods in fabric pod a, one of them in a terminal phase")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1", Labels: map[string]string{
				lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod,
			}}
			clique.Spec.Replicas = 2
			terminal := newTestBoundCyborgPod(clique, 1, "a")
			terminal.Status.Phase = tc.phase
			r := &graphReconciler{Client: newLPXTestClient(t, clique, newTestBoundCyborgPod(clique, 0, "a"), terminal)}
			placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 2}

			t.Log("When the controller reconciles the placement, it removes that Pod: it cannot run again, and Grove creates no Pod in its place while it is there")
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
			require.Equal(t, []int{0}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
			require.Equal(t, int32(2), clique.Spec.Replicas)

			t.Log("Then the Pod that Grove creates takes the position")
			require.NoError(t, r.Create(ctx, newTestReleasedCyborgPod(clique, 1)))
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
			pods := listTestCyborgPods(t, ctx, r.Client, clique)
			require.Equal(t, []string{"a", "a"}, testCyborgFabricPods(pods))
			require.Equal(t, []string{"0/0", "1/1"}, testCyborgRanks(pods))
		})
	}
}

func TestLPXCyborgScaleInRetention(t *testing.T) {
	type pod struct {
		unbound bool
		// waiting is a bound Pod that the kubelet has not started.
		waiting bool
		ready   bool
	}

	for _, tc := range []struct {
		name string
		// pods are the Pods with the indexes 0 and 1 of one fabric pod. Each has its
		// index as its rank, or the rank of the other Pod when swapped is true.
		pods    [2]pod
		swapped bool
		want    []int
	}{
		{name: "a bound Pod stays before an unbound Pod", pods: [2]pod{{unbound: true}, {}}, want: []int{1}},
		{name: "a Pod that the kubelet started stays before a Pod that it can still refuse", pods: [2]pod{{waiting: true}, {}}, want: []int{1}},
		{name: "a Ready Pod stays before a Pod that is not Ready", pods: [2]pod{{}, {ready: true}}, want: []int{1}},
		{name: "the lower rank stays among equal Pods", pods: [2]pod{{}, {}}, want: []int{0}},
		{name: "the lower rank stays, also with the higher Grove index", pods: [2]pod{{}, {}}, swapped: true, want: []int{1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given two placed Cyborg Pods in one fabric pod whose count goes from two to one")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
			clique.Spec.Replicas = 2
			r := &graphReconciler{Client: newLPXTestClient(t, clique)}
			for index, spec := range tc.pods {
				pod := newTestBoundCyborgPod(clique, index, "a")
				pod.Status.Conditions = newTestCyborgPodConditions(spec.ready)
				if spec.unbound {
					pod.Spec.NodeName = ""
				}
				if spec.waiting {
					pod.Status.InitContainerStatuses[0].State = corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "PodInitializing"}}
				}
				if tc.swapped {
					rankTestCyborgPod(pod, 1-index, index)
				}
				require.NoError(t, r.Create(ctx, pod))
			}

			t.Log("When the controller reconciles the placement, Grove creates a Pod for the removed one, and the controller reconciles again")
			placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
			kept := listTestCyborgPodIndexes(t, ctx, r.Client, clique)
			replaceTestCyborgPods(t, ctx, r.Client, clique)
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
			removeTestExcessCyborgPods(t, ctx, r.Client, clique)

			t.Log("Then the first pass keeps the Pod that the placement loses least with, and the second pass lowers the capacity to one")
			require.Equal(t, tc.want, kept)
			require.Equal(t, tc.want, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
			require.Equal(t, int32(1), clique.Spec.Replicas)
		})
	}
}

func TestCyborgPodStarted(t *testing.T) {
	waiting := corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "PodInitializing"}}
	running := corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}
	ended := corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{Reason: "Completed"}}

	for _, tc := range []struct {
		name string
		// init and main are the statuses that the kubelet reports for the init
		// containers and for the containers of the Pod.
		init, main []corev1.ContainerStatus
		want       bool
	}{
		{name: "the kubelet has reported no container"},
		{name: "each container waits for its first start", init: []corev1.ContainerStatus{{State: waiting}}, main: []corev1.ContainerStatus{{State: waiting}}},
		{name: "an init container runs", init: []corev1.ContainerStatus{{State: running}}, main: []corev1.ContainerStatus{{State: waiting}}, want: true},
		{name: "an init container ended", init: []corev1.ContainerStatus{{State: ended}}, main: []corev1.ContainerStatus{{State: waiting}}, want: true},
		{name: "a container of a Pod without init containers runs", main: []corev1.ContainerStatus{{State: waiting}, {State: running}}, want: true},
		{name: "a container that waits for a restart does not count by itself: the init containers of a Cyborg Pod keep their state", main: []corev1.ContainerStatus{{State: waiting, LastTerminationState: ended}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given a bound Cyborg Pod as the operator Pod cache keeps it")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid"}
			pod := newTestBoundCyborgPod(clique, 0, "a")
			pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses = tc.init, tc.main

			t.Log("Then the Pod counts as started when one of its container statuses has no waiting state")
			require.Equal(t, tc.want, cyborgPodStarted(podcache.Project(pod)))
		})
	}
}

func TestCompareCyborgRetention(t *testing.T) {
	type pod struct {
		index   int
		unbound bool
		// waiting is a bound Pod that the kubelet has not started.
		waiting bool
		ready   bool
		// rank is the rank of the Pod; nil is the index.
		rank *int
	}

	for _, tc := range []struct {
		name string
		pods []pod
		want []int
	}{
		{name: "a bound Pod goes before an unbound Pod", pods: []pod{{index: 0, unbound: true}, {index: 1}}, want: []int{1, 0}},
		{name: "a Pod that the kubelet started goes before a Pod that it can still refuse", pods: []pod{{index: 0, waiting: true}, {index: 1}}, want: []int{1, 0}},
		{name: "a bound Pod that the kubelet has not started goes before an unbound Pod", pods: []pod{{index: 0, unbound: true}, {index: 1, waiting: true}}, want: []int{1, 0}},
		{name: "a Ready Pod goes before a Pod that is not Ready", pods: []pod{{index: 0}, {index: 1, ready: true}}, want: []int{1, 0}},
		{name: "a Pod that the kubelet started goes before a Pod with a lower rank that it can still refuse", pods: []pod{{index: 0, waiting: true}, {index: 5}}, want: []int{5, 0}},
		{name: "the lower rank goes first, also with the higher index", pods: []pod{{index: 1, ready: true, rank: new(1)}, {index: 3, ready: true, rank: new(0)}}, want: []int{3, 1}},
		{name: "the lower index goes first among equal ranks, also when its Pod is newer", pods: []pod{{index: 3, ready: true, rank: new(0)}, {index: 1, ready: true, rank: new(0)}}, want: []int{1, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given Pods that compete for the positions of one fabric pod; a Pod with a higher index is older here")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid"}
			var pods []*corev1.Pod
			for _, spec := range tc.pods {
				pod := newTestBoundCyborgPod(clique, spec.index, "a")
				pod.CreationTimestamp = metav1.NewTime(pod.CreationTimestamp.Add(-2 * time.Duration(spec.index) * time.Second))
				pod.Status.Conditions = newTestCyborgPodConditions(spec.ready)
				if spec.unbound {
					pod.Spec.NodeName, pod.Status.InitContainerStatuses = "", nil
				}
				if spec.waiting {
					pod.Status.InitContainerStatuses[0].State = corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "PodInitializing"}}
				}
				if spec.rank != nil {
					rankTestCyborgPod(pod, *spec.rank, spec.index)
				}
				pods = append(pods, podcache.Project(pod))
			}

			t.Log("When the controller orders them by retention")
			slices.SortStableFunc(pods, compareCyborgRetention)

			t.Log("Then the Pods that keep their positions come first")
			require.Equal(t, tc.want, testCyborgPodIndexes(pods))
		})
	}
}

func TestLPXCyborgPlacementGateOrder(t *testing.T) {
	grove, placement := corev1.PodSchedulingGate{Name: grovePodGangGate}, corev1.PodSchedulingGate{Name: lpx.CyborgPlacementGate}
	other := corev1.PodSchedulingGate{Name: "example.com/quota"}

	for _, tc := range []struct {
		name   string
		gates  []corev1.PodSchedulingGate
		pinned bool
		want   []corev1.PodSchedulingGate
	}{
		{name: "the placement gate holds a Pod that Grove has not released", gates: []corev1.PodSchedulingGate{grove, placement}, want: []corev1.PodSchedulingGate{grove, placement}},
		{name: "the pin removes the placement gate after Grove's gate", gates: []corev1.PodSchedulingGate{placement}, pinned: true},
		{name: "the pin keeps the gate of another controller", gates: []corev1.PodSchedulingGate{other, placement}, pinned: true, want: []corev1.PodSchedulingGate{other}},
		{name: "a Pod that admission did not gate is pinned while Grove's gate holds it", gates: []corev1.PodSchedulingGate{grove}, pinned: true, want: []corev1.PodSchedulingGate{grove}},
		{name: "a Pod without a gate cannot take a selector"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given one unbound Cyborg Pod with these scheduling gates and one open position")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
			clique.Spec.Replicas = 1
			pod := newTestCyborgPod(clique, 0)
			pod.Spec.SchedulingGates = tc.gates
			r := &graphReconciler{Client: newLPXTestClient(t, clique, pod)}

			t.Log("When the controller reconciles the placement")
			err := r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, false)
			require.NoError(t, err)

			t.Log("Then the Pod has its pin and no placement gate, or is unchanged")
			pods := listTestCyborgPods(t, ctx, r.Client, clique)
			require.Len(t, pods, 1)
			require.Equal(t, tc.want, pods[0].Spec.SchedulingGates)
			if tc.pinned {
				require.Equal(t, map[string]string{fabricPodLabel: "a"}, pods[0].Spec.NodeSelector)
				require.Equal(t, "a", pods[0].Labels[cyborgFabricPodLabel])
				require.Equal(t, []string{"0/0"}, testCyborgRanks(pods))
			} else {
				require.Empty(t, pods[0].Spec.NodeSelector)
				require.Empty(t, pods[0].Labels[cyborgFabricPodLabel])
				require.Equal(t, []string{"/"}, testCyborgRanks(pods))
			}
		})
	}
}

func TestLPXCyborgPlacementOfReleasedPods(t *testing.T) {
	ctx := t.Context()

	t.Log("Keep one Cyborg Pod in each of two fabric pods")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 2
	placement := cyborgPlacement{fabricPods: []string{"a", "b"}, replicas: 1}

	t.Log("Pod admission added no gate, and Grove lifted its gate from three Pods before the controller observed them: two are bound in fabric pod a, one is not bound")
	older, newer, unbound := newTestCyborgPod(clique, 0), newTestCyborgPod(clique, 1), newTestCyborgPod(clique, 2)
	older.Spec.NodeName, newer.Spec.NodeName = "gpu-a", "gpu-a"
	older.Spec.SchedulingGates, newer.Spec.SchedulingGates, unbound.Spec.SchedulingGates = nil, nil, nil
	r := &graphReconciler{Client: newLPXTestClient(t, clique, older, newer, unbound, newTestFabricPodNode("gpu-a", "a"), newTestFabricPodNode("gpu-b", "b"))}

	t.Log("Adopt the first bound Pod, remove the second, and leave the unbound Pod for its own binding: Kubernetes refuses a selector on a Pod without a gate")
	err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Len(t, pods, 2)
	require.Equal(t, older.Name, pods[0].Name)
	require.Equal(t, "a", pods[0].Labels[cyborgFabricPodLabel])
	require.Equal(t, []string{"0/0", "/"}, testCyborgRanks(pods), "the record of an adopted Pod keeps the next Pods apart from it; its containers started without it")
	require.Empty(t, pods[0].Spec.NodeSelector, "a bound Pod needs no selector")
	require.Equal(t, unbound.Name, pods[1].Name)
	require.Empty(t, pods[1].Labels[cyborgFabricPodLabel])
	require.Empty(t, pods[1].Spec.NodeSelector)

	t.Log("A Pod on a Node without a fabric pod cannot serve any placement")
	stray := newTestCyborgPod(clique, 3)
	stray.Spec.NodeName = "gone"
	require.NoError(t, r.Create(ctx, stray))
	err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	require.Len(t, listTestCyborgPods(t, ctx, r.Client, clique), 2)
}

func TestLPXCyborgPlacementWaitsForScaling(t *testing.T) {
	ctx := t.Context()

	t.Log("Grove defers replica writes while the placement shrinks from two fabric pods to one")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 2
	kept, pruned := newTestCyborgPod(clique, 0), newTestCyborgPod(clique, 1)
	kept.Labels[cyborgFabricPodLabel], pruned.Labels[cyborgFabricPodLabel] = "a", "b"
	r := &graphReconciler{Client: newLPXTestClient(t, clique, rankTestCyborgPod(kept, 0, 0), rankTestCyborgPod(pruned, 0, 1))}
	placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}

	t.Log("Keep both Pods: a removal without the lower capacity makes Grove create a replacement")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, true))
	require.Len(t, listTestCyborgPods(t, ctx, r.Client, clique), 2)
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("When Grove permits, remove the Pod outside the placement; the capacity stays in that pass")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Lower the capacity when Grove has created a Pod for the removed one")
	replaceTestCyborgPods(t, ctx, r.Client, clique)
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(1), clique.Spec.Replicas)
}

func TestLPXCyborgPruneDoesNotLoopOnFailedScale(t *testing.T) {
	ctx := t.Context()

	t.Log("The scheduler selected fabric pod a of three, and the API server refuses the lower capacity")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 3
	kept, prunedB, prunedC := newTestCyborgPod(clique, 0), newTestCyborgPod(clique, 1), newTestCyborgPod(clique, 2)
	kept.Labels[cyborgFabricPodLabel], prunedB.Labels[cyborgFabricPodLabel], prunedC.Labels[cyborgFabricPodLabel] = "a", "b", "c"
	rankTestCyborgPod(kept, 0, 0)
	rankTestCyborgPod(prunedB, 0, 1)
	rankTestCyborgPod(prunedC, 0, 2)
	kept.Spec.NodeName = "gpu-a"
	refuseScale := true
	base := newLPXTestClient(t, clique, kept, prunedB, prunedC, newTestFabricPodNode("gpu-a", "a"))
	r := &graphReconciler{Client: interceptor.NewClient(base, interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			if refuseScale {
				return apierrors.NewServiceUnavailable("scale is unavailable")
			}
			return delegated.SubResource(subresource).Update(ctx, object, opts...)
		},
	})}
	placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}

	t.Log("Remove the two placed Pods outside fabric pod a")
	err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))

	t.Log("Grove replaces both Pods because its capacity is still three; one replacement binds in fabric pod a")
	floating, gated := newTestCyborgPod(clique, 1), newTestCyborgPod(clique, 2)
	floating.Name, floating.UID, floating.Spec.NodeName = floating.Name+"-r", floating.UID+"-r", "gpu-a"
	gated.Name, gated.UID = gated.Name+"-r", gated.UID+"-r"
	createLPXTestObjects(t, ctx, r.Client, floating, gated)

	t.Log("Report each failed capacity write, and leave the replacements to Grove's scale-in: a second removal only makes Grove create them again")
	for range 3 {
		err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
		require.ErrorContains(t, err, "scale Cyborg PodClique")
		require.Equal(t, []string{"a", "", ""}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
	}

	t.Log("Write the capacity when the API server accepts it, still without a removal")
	refuseScale = false
	err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)
	require.Len(t, listTestCyborgPods(t, ctx, r.Client, clique), 3)
}

func TestEligibleFabricPods(t *testing.T) {
	nodes := []metav1.PartialObjectMetadata{
		{Name: "gpu-a", Labels: map[string]string{fabricPodLabel: "a", "pool": "gpu"}},
		{Name: "gpu-b", Labels: map[string]string{fabricPodLabel: "b", "pool": "gpu"}},
		{Name: "lpu-c", Labels: map[string]string{fabricPodLabel: "c", "pool": "lpu"}},
		{Name: "unlabeled", Labels: map[string]string{"pool": "gpu"}},
	}

	for _, tc := range []struct {
		name    string
		podSpec corev1.PodSpec
		want    []string
		wantErr string
	}{
		{name: "every labeled Node counts without constraints", want: []string{"a", "b", "c"}},
		{name: "the node selector limits the fabric pods", podSpec: corev1.PodSpec{NodeSelector: map[string]string{"pool": "gpu"}}, want: []string{"a", "b"}},
		{name: "a selector on the fabric pod label keeps one fabric pod", podSpec: corev1.PodSpec{NodeSelector: map[string]string{fabricPodLabel: "b"}}, want: []string{"b"}},
		{
			name: "required node affinity limits the fabric pods",
			podSpec: corev1.PodSpec{Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchExpressions: []corev1.NodeSelectorRequirement{
					{Key: fabricPodLabel, Operator: corev1.NodeSelectorOpNotIn, Values: []string{"a"}},
				}}},
			}}}},
			want: []string{"b", "c"},
		},
		{name: "no selectable Node is an error", podSpec: corev1.PodSpec{NodeSelector: map[string]string{"pool": "none"}}, wantErr: "no Node with label"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Spec: grovev1alpha1.PodCliqueSpec{PodSpec: tc.podSpec}}
			fabricPods, err := eligibleFabricPods(nodes, clique)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.want, fabricPods)
		})
	}
}

func TestCyborgPlacementApplies(t *testing.T) {
	agents, noAgents := []lpx.ExpectedAgent{{Replicas: 2}}, []lpx.ExpectedAgent{{Replicas: 0}}

	for _, tc := range []struct {
		name    string
		mode    string
		plan    lpx.MaterializationPlan
		want    bool
		wantErr bool
	}{
		{name: "placement is opt-in", plan: lpx.MaterializationPlan{CyborgTemplate: "cond", Agents: agents}},
		{name: "a hybrid workload with Agents follows its fabric pods", mode: lpx.CyborgPlacementPerFabricPod, plan: lpx.MaterializationPlan{CyborgTemplate: "cond", Agents: agents}, want: true},
		{name: "an LPU-only conductor has no Cyborg", mode: lpx.CyborgPlacementPerFabricPod, plan: lpx.MaterializationPlan{ConductorTemplate: "cond", Agents: agents}},
		{name: "a Cyborg with only local partitions has no fabric pod", mode: lpx.CyborgPlacementPerFabricPod, plan: lpx.MaterializationPlan{CyborgTemplate: "cond", Agents: noAgents}},
		{name: "an unknown mode is rejected", mode: "per-rack", plan: lpx.MaterializationPlan{CyborgTemplate: "cond", Agents: agents}, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "lpx", Roles: []v1beta1.ComponentRoleSpec{{
				Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: &corev1.PodTemplateSpec{},
			}}}
			if tc.mode != "" {
				component.Roles[0].PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: tc.mode}
			}
			placed, err := cyborgPlacementApplies(component, &tc.plan)
			require.Equal(t, tc.wantErr, err != nil)
			require.Equal(t, tc.want, placed)
		})
	}
}

func TestPlacedCyborgPodPredicate(t *testing.T) {
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", Labels: map[string]string{
		lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod,
	}}
	unplaced := newTestCyborgPod(clique, 0)
	unselected := unplaced.DeepCopy()
	delete(unselected.Labels, lpx.CyborgPlacementLabel)
	placed := unplaced.DeepCopy()
	placed.Labels[cyborgFabricPodLabel] = "a"
	bound := placed.DeepCopy()
	bound.Spec.NodeName = "gpu-a"
	started := bound.DeepCopy()
	started.Status.InitContainerStatuses = newTestStartedContainerStatuses()
	running := started.DeepCopy()
	running.Status.Phase = corev1.PodRunning
	ready := running.DeepCopy()
	ready.Status.Conditions = newTestCyborgPodConditions(true)
	failed := started.DeepCopy()
	failed.Status.Phase = corev1.PodFailed
	deleting := bound.DeepCopy()
	deleting.DeletionTimestamp = new(metav1.Now())

	t.Log("Given the Pod watch of the controller: a new Pod and a removed Pod with the placement label start a pass, and no other Pod does")
	filter := placedCyborgPodPredicate()
	require.True(t, filter.Create(event.CreateEvent{Object: unplaced}))
	require.False(t, filter.Create(event.CreateEvent{Object: unselected}))
	require.True(t, filter.Delete(event.DeleteEvent{Object: bound}))
	require.False(t, filter.Delete(event.DeleteEvent{Object: unselected}))

	for _, tc := range []struct {
		name     string
		old, new *corev1.Pod
		want     bool
	}{
		{name: "each version of an unplaced Pod can accept a placement", old: unplaced, new: unplaced, want: true},
		{name: "a Pod gets its placement", old: unplaced, new: placed, want: true},
		{name: "a placed Pod gets its Node", old: placed, new: bound, want: true},
		{name: "the kubelet starts a placed Pod: the placement does not change", old: bound, new: started},
		{name: "other status changes of a placed Pod start no pass", old: started, new: running},
		{name: "a placed Pod that becomes Ready starts no pass", old: running, new: ready},
		{name: "a placed Pod that loses Ready starts no pass", old: ready, new: running},
		{name: "a placed Pod gets a terminal phase: it holds no position", old: started, new: failed, want: true},
		{name: "a placed Pod gets its deletion mark", old: bound, new: deleting, want: true},
		{name: "a Pod without the placement label starts no pass", old: unselected, new: unselected},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("When the Pod changes, then the watch starts a pass only for a change that the placement can use")
			require.Equal(t, tc.want, filter.Update(event.UpdateEvent{ObjectOld: tc.old, ObjectNew: tc.new}))
		})
	}
}

// newTestPlacedPipelineRequest returns one model's request for a workload replica, with a
// committed NodeLocal plan that has one selected row in each of fabricPods.
func newTestPlacedPipelineRequest(name, scalingGroup string, replica int64, phase lpxv1alpha1.RequestPhase, fabricPods ...string) *lpxv1alpha1.LPUPipelineRequest {
	rows := make([]lpxv1alpha1.NodeLocalSelectedRow, 0, len(fabricPods))
	for _, fabricPod := range fabricPods {
		rows = append(rows, lpxv1alpha1.NodeLocalSelectedRow{Endpoint: lpxv1alpha1.NodeLocalEndpoint{FabricID: fabricPod}})
	}
	generation := int64(1)
	return &lpxv1alpha1.LPUPipelineRequest{
		Name: name, Namespace: "default", Generation: generation,
		Spec: lpxv1alpha1.LPUPipelineRequestSpec{
			ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal,
			MaterializationTarget: lpxv1alpha1.MaterializationTarget{
				PodCliqueScalingGroupRef: &lpxv1alpha1.PodCliqueScalingGroupReference{Name: scalingGroup, ReplicaIndex: replica},
			},
		},
		Status: &lpxv1alpha1.LPUPipelineRequestStatus{
			Phase: phase, ObservedGeneration: &generation, LastPlanRevision: 1,
			Committed: &lpxv1alpha1.Committed{
				Execution: lpxv1alpha1.CommittedExecution{AcceptedGeneration: &generation, NodeLocal: &lpxv1alpha1.NodeLocalExecution{}},
				Plan: lpxv1alpha1.CommittedPlan{
					PlannedFromGeneration: generation, Revision: 1, PlanDigest: "sha256:plan",
					Placement: lpxv1alpha1.PlanPlacement{
						ExecutionBackend: lpxv1alpha1.ExecutionBackendNodeLocal,
						NodeLocal:        &lpxv1alpha1.NodeLocalPlacement{PartitionSelections: []lpxv1alpha1.NodeLocalPartitionSelection{{SelectedRows: rows}}},
					},
				},
			},
		},
	}
}

// newTestFabricPodNode returns a Node in fabricPod with the additional label pairs.
func newTestFabricPodNode(name, fabricPod string, labels ...string) *corev1.Node {
	node := &corev1.Node{Name: name, Labels: map[string]string{fabricPodLabel: fabricPod}}
	for index := 0; index+1 < len(labels); index += 2 {
		node.Labels[labels[index]] = labels[index+1]
	}
	return node
}

// newTestCyborgPod returns the Pod that Grove creates for clique at index: held by
// Grove's scheduling gate, unbound, without a placement, and younger at a higher
// index. For a clique with the placement label, the Pod also has that label and
// the gate from Pod admission.
func newTestCyborgPod(clique *grovev1alpha1.PodClique, index int) *corev1.Pod {
	created := time.Date(2026, time.November, 11, 0, 0, index, 0, time.UTC)
	pod := &corev1.Pod{
		Name: fmt.Sprintf("%s-%d-%d", clique.Name, index, created.Unix()), Namespace: clique.Namespace,
		UID:               types.UID(fmt.Sprintf("%s-pod-%d-%d", clique.Name, index, created.Unix())),
		CreationTimestamp: metav1.NewTime(created),
		Labels: map[string]string{
			grovecommon.LabelPodClique:         clique.Name,
			grovecommon.LabelPodCliquePodIndex: strconv.Itoa(index),
		},
		Annotations:     maps.Clone(clique.Annotations),
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(clique, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))},
		Spec:            corev1.PodSpec{SchedulingGates: []corev1.PodSchedulingGate{{Name: grovePodGangGate}}},
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if mode := clique.Labels[lpx.CyborgPlacementLabel]; mode != "" {
		pod.Labels[lpx.CyborgPlacementLabel] = mode
		pod.Spec.SchedulingGates = append(pod.Spec.SchedulingGates, corev1.PodSchedulingGate{Name: lpx.CyborgPlacementGate})
	}
	return pod
}

// newTestBoundCyborgPod returns the Pod of newTestCyborgPod after the controller
// placed it in fabricPod, the scheduler bound it to Node gpu-<fabricPod>, and the
// kubelet started its first init container. The Pod is not Ready. Its rank and
// its replica index are its Grove index, which no other Pod of the PodClique has;
// rankTestCyborgPod sets other values. An empty fabricPod leaves the Pod without
// a placement.
func newTestBoundCyborgPod(clique *grovev1alpha1.PodClique, index int, fabricPod string) *corev1.Pod {
	pod := newTestCyborgPod(clique, index)
	pod.Spec.SchedulingGates, pod.Spec.NodeName = nil, "gpu-"+fabricPod
	pod.Status.Conditions = newTestCyborgPodConditions(false)
	pod.Status.InitContainerStatuses = newTestStartedContainerStatuses()
	if fabricPod != "" {
		pod.Labels[cyborgFabricPodLabel] = fabricPod
		pod.Spec.NodeSelector = map[string]string{fabricPodLabel: fabricPod}
		rankTestCyborgPod(pod, index, index)
	}
	return pod
}

// rankTestCyborgPod records the rank and the replica index on the placed Pod, and returns it.
func rankTestCyborgPod(pod *corev1.Pod, rank, index int) *corev1.Pod {
	pod.Labels[cyborgFabricPodRankLabel], pod.Labels[cyborgReplicaIndexLabel] = strconv.Itoa(rank), strconv.Itoa(index)
	return pod
}

// newTestStoppingCyborgPod returns the Pod of newTestBoundCyborgPod in fabric pod a
// after its removal, while its containers stop. A test finalizer keeps it in the
// fake client, which accepts such a Pod only as an initial object.
func newTestStoppingCyborgPod(clique *grovev1alpha1.PodClique, index int) *corev1.Pod {
	pod := newTestBoundCyborgPod(clique, index, "a")
	pod.DeletionTimestamp, pod.Finalizers = new(metav1.Now()), []string{testCyborgPodHold}
	return pod
}

// testCyborgPodSerial numbers the Pods that replaceTestCyborgPods creates.
var testCyborgPodSerial atomic.Int64

// replaceTestCyborgPods plays the part of Grove while clique has fewer live Pods
// than its capacity: for each missing Pod it creates a Pod with the lowest index
// that no live Pod has. Grove has released the new Pod, so only the gate from Pod
// admission holds it, and it is newer than each earlier Pod.
func replaceTestCyborgPods(t *testing.T, ctx context.Context, c client.Client, clique *grovev1alpha1.PodClique) {
	t.Helper()
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	live := listTestCyborgPods(t, ctx, c, clique)
	for index := 0; len(live) < int(clique.Spec.Replicas); index++ {
		if slices.ContainsFunc(live, func(pod *corev1.Pod) bool { return podCliquePodIndex(pod) == index }) {
			continue
		}
		pod, serial := newTestReleasedCyborgPod(clique, index), testCyborgPodSerial.Add(1)
		pod.Name, pod.UID = fmt.Sprintf("%s-%d", pod.Name, serial), types.UID(fmt.Sprintf("%s-%d", pod.UID, serial))
		pod.CreationTimestamp = metav1.NewTime(pod.CreationTimestamp.Add(time.Duration(serial) * time.Hour))
		require.NoError(t, c.Create(ctx, pod))
		live = append(live, pod)
	}
}

// removeTestExcessCyborgPods plays the part of Grove after a lower capacity: it
// removes the newest Pods of clique that have no Node until clique has its
// capacity of live Pods. Grove selects a Pod without a Node first; the test fails
// when Grove would have to select a bound Pod.
func removeTestExcessCyborgPods(t *testing.T, ctx context.Context, c client.Client, clique *grovev1alpha1.PodClique) {
	t.Helper()
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	live := listTestCyborgPods(t, ctx, c, clique)
	excess := len(live) - int(clique.Spec.Replicas)
	for _, pod := range slices.Backward(live) {
		if excess > 0 && pod.Spec.NodeName == "" {
			require.NoError(t, c.Delete(ctx, pod))
			excess--
		}
	}
	require.LessOrEqual(t, excess, 0, "Grove would remove a bound Cyborg Pod")
}

// placeTestPipelineRequests plays the part of the LPX scheduler: it gives every
// request of the PodCliqueSet the phase and a committed plan in fabricPods for
// replica 0 of the scaling group.
func placeTestPipelineRequests(
	t *testing.T,
	ctx context.Context,
	r *graphReconciler,
	pcs *grovev1alpha1.PodCliqueSet,
	scalingGroup string,
	phase lpxv1alpha1.RequestPhase,
	fabricPods ...string,
) {
	t.Helper()
	requests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	for _, request := range requests {
		request.Status = newTestPlacedPipelineRequest(request.Name, scalingGroup, 0, phase, fabricPods...).Status
		require.NoError(t, r.Update(ctx, request))
	}
}

// reconcileTestCyborgPlacement runs the placement steps of one complete controller
// pass for clique: the pin step, and the step that gives a Pod a lower index,
// removes Pods and writes the capacity. It does not run the other parts of a
// pass, such as the scaling state. It refreshes clique afterwards.
func reconcileTestCyborgPlacement(
	t *testing.T,
	ctx context.Context,
	r *graphReconciler,
	clique *grovev1alpha1.PodClique,
	placement cyborgPlacement,
	scalingBlocked bool,
) error {
	t.Helper()
	pclqs := map[string]*grovev1alpha1.PodClique{clique.Name: clique}
	placements := map[string]cyborgPlacement{clique.Name: placement}
	require.NoError(t, r.pinCyborgPods(ctx, pclqs, placements))
	err := r.reconcileCyborgPlacements(ctx, pclqs, placements, scalingBlocked)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	return err
}

// testCyborgPodHold is the finalizer that keeps a removed Pod in the fake client.
const testCyborgPodHold = "lpx.nvidia.com/test-hold"

// newTestStartedContainerStatuses returns the init container status of a Pod whose
// first init container the kubelet started.
func newTestStartedContainerStatuses() []corev1.ContainerStatus {
	return []corev1.ContainerStatus{{Name: "grove-initc", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}}}
}

// newTestReleasedCyborgPod returns the Pod of newTestCyborgPod after Grove removed its scheduling gate.
func newTestReleasedCyborgPod(clique *grovev1alpha1.PodClique, index int) *corev1.Pod {
	pod := newTestCyborgPod(clique, index)
	pod.Spec.SchedulingGates = slices.DeleteFunc(pod.Spec.SchedulingGates, func(gate corev1.PodSchedulingGate) bool { return gate.Name == grovePodGangGate })
	return pod
}

// listTestCyborgPods returns the Pods of clique that are not being deleted, oldest first.
func listTestCyborgPods(t *testing.T, ctx context.Context, reader client.Reader, clique *grovev1alpha1.PodClique) []*corev1.Pod {
	t.Helper()
	list := &corev1.PodList{}
	require.NoError(t, reader.List(ctx, list, client.InNamespace(clique.Namespace), client.MatchingLabels{grovecommon.LabelPodClique: clique.Name}))
	var pods []*corev1.Pod
	for index := range list.Items {
		if list.Items[index].DeletionTimestamp.IsZero() {
			pods = append(pods, &list.Items[index])
		}
	}
	slices.SortFunc(pods, func(a, b *corev1.Pod) int { return a.CreationTimestamp.Compare(b.CreationTimestamp.Time) })
	return pods
}

// listTestCyborgPodIndexes returns the sorted Grove indexes of the Pods of clique
// that are not being deleted.
func listTestCyborgPodIndexes(t *testing.T, ctx context.Context, reader client.Reader, clique *grovev1alpha1.PodClique) []int {
	t.Helper()
	indexes := testCyborgPodIndexes(listTestCyborgPods(t, ctx, reader, clique))
	slices.Sort(indexes)
	return indexes
}

// newTestCyborgPodConditions returns the conditions that the kubelet reports for a
// bound Pod, in its order. The operator Pod cache keeps only the Ready condition:
// a test that gives Pods to a function directly projects them with podcache.Project.
func newTestCyborgPodConditions(ready bool) []corev1.PodCondition {
	status := corev1.ConditionFalse
	if ready {
		status = corev1.ConditionTrue
	}
	return []corev1.PodCondition{
		{Type: corev1.PodInitialized, Status: corev1.ConditionTrue},
		{Type: corev1.PodReady, Status: status},
		{Type: corev1.ContainersReady, Status: status},
		{Type: corev1.PodScheduled, Status: corev1.ConditionTrue},
	}
}

// testCyborgPodIndexes returns the Grove index of every Pod, in the given order.
func testCyborgPodIndexes(pods []*corev1.Pod) []int {
	indexes := make([]int, 0, len(pods))
	for _, pod := range pods {
		indexes = append(indexes, podCliquePodIndex(pod))
	}
	return indexes
}

// testCyborgRanks returns the recorded rank and replica index of every Pod as
// "rank/index", in the given order. A Pod without a record gives "/".
func testCyborgRanks(pods []*corev1.Pod) []string {
	ranks := make([]string, 0, len(pods))
	for _, pod := range pods {
		ranks = append(ranks, pod.Labels[cyborgFabricPodRankLabel]+"/"+pod.Labels[cyborgReplicaIndexLabel])
	}
	return ranks
}

// testCyborgFabricPods returns the recorded fabric pod of every Pod, in the given order.
func testCyborgFabricPods(pods []*corev1.Pod) []string {
	fabricPods := make([]string, 0, len(pods))
	for _, pod := range pods {
		fabricPods = append(fabricPods, pod.Labels[cyborgFabricPodLabel])
	}
	return fabricPods
}
