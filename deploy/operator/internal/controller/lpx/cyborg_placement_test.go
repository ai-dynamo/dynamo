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
			name: "all models of the replica contribute their fabric pods",
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
			fabricPods, settled := committedFabricPods(requests, group, 0)
			require.Equal(t, tc.settled, settled)
			require.Equal(t, tc.want, fabricPods)
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
	for _, request := range requests {
		request.Status = newTestPlacedPipelineRequest(request.Name, plans["lpx"].LPXScalingGroup, 0, lpxv1alpha1.RequestPhaseBound, "b").Status
		require.NoError(t, r.Update(ctx, request))
	}
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"b"}, testCyborgFabricPods(pods))
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)

	t.Log("The Pod that stays has index 1 and the new total is one. The controller waits until Grove has synchronized the lower capacity: Grove can still hold Pods that it created for the removed ones")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)

	t.Log("Renumber: fabric pod b gets one more position, so the capacity is two, and the Pod with index 1 stays")
	settleTestCyborgClique(t, ctx, r.Client, clique)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)
	require.Equal(t, []int{1}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))

	t.Log("Grove gives the new Pod its lowest free index, and the controller pins it to fabric pod b; the old Pod stays until the new one is bound")
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	for range 2 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []int{0, 1}, testCyborgPodIndexes(pods))
	require.Equal(t, []string{"b", "b"}, testCyborgFabricPods(pods))
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("The added position is not a capacity change that waits: a PodCliqueSet write in this time reports no deferred scaling")
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, dynamo.LPXRestartAnnotation, "during-renumber")
	require.NoError(t, r.Update(ctx, child))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	updateTestDGD(t, r, child, dgd)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, key.NamespacedName, child))
	require.False(t, meta.IsStatusConditionTrue(child.Status.Conditions, "ScalingDeferred"))

	t.Log("When the new Pod is bound, the controller removes the Pod with index 1 and then lowers the capacity")
	bindTestCyborgPod(t, ctx, r.Client, clique, 0, "gpu-b")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []int{0}, testCyborgPodIndexes(pods))
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)

	t.Log("A Pod with an index below the total stays")
	settleTestCyborgClique(t, ctx, r.Client, clique)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Equal(t, pods, listTestCyborgPods(t, ctx, r.Client, clique))
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
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 1))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "b"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))

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
	require.Equal(t, []string{"b", "b", "a", "a", "c", "c"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
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
	_, err = r.pinCyborgPods(ctx, pclqs, placements)
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, selectable)))
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
	surplus, changed, err := r.fillCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, live)
	require.NoError(t, err)
	require.False(t, changed)
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
		_, err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
		require.NoError(t, err)
	}

	t.Log("Each fabric pod holds the contiguous index block of its position in name order")
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a", "a", "b", "b", "c", "c"}, testCyborgFabricPods(pods))

	require.Equal(t, []int{0, 1, 2, 3, 4, 5}, testCyborgPodIndexes(pods))
}

func TestLPXCyborgPlacementRenumbersAfterPrune(t *testing.T) {
	ctx := t.Context()

	t.Log("Two bound Cyborg Pods are in each of three fabric pods, and the scheduler binds the LPUs in fabric pod c")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1", Labels: map[string]string{
		lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod,
	}}
	clique.Spec.Replicas = 6
	r := &graphReconciler{Client: newLPXTestClient(t, clique)}
	for index, fabricPod := range []string{"a", "a", "b", "b", "c", "c"} {
		require.NoError(t, r.Create(ctx, newTestBoundCyborgPod(clique, index, fabricPod)))
	}
	placement := cyborgPlacement{fabricPods: []string{"c"}, replicas: 2, settled: true}

	t.Log("Prune first: the Pods of fabric pod c keep the indexes 4 and 5 in the pass that removes the other Pods and lowers the capacity")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Wait until Grove has synchronized the lower capacity: Grove can still hold Pods that it created for the removed ones")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Then give fabric pod c one more position: the capacity is three, and no Pod goes")
	settleTestCyborgClique(t, ctx, r.Client, clique)
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(3), clique.Spec.Replicas)
	require.Equal(t, []int{4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))

	t.Log("Grove gives the new Pod its lowest free index, and the controller pins it to the added position")
	first := newTestReleasedCyborgPod(clique, 0)
	first.CreationTimestamp = metav1.NewTime(first.CreationTimestamp.Add(time.Hour))
	require.NoError(t, r.Create(ctx, first))
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{0, 4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, []string{"c", "c", "c"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))

	t.Log("The old Pods stay while the new Pod is not bound: fabric pod c keeps its two bound Cyborg Pods, also when the new Pod gets no GPU")
	for range 2 {
		require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	}
	require.Equal(t, []int{0, 4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("When the new Pod is bound, remove one Pod above the total, and then lower the capacity; the removed Pod terminates for some time")
	holdTestCyborgPod(t, ctx, r.Client, clique, 5, true)
	bindTestCyborgPod(t, ctx, r.Client, clique, 0, "gpu-c")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{0, 4}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Wait again for Grove, and then give the second Pod a lower index in the same way")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(2), clique.Spec.Replicas)
	settleTestCyborgClique(t, ctx, r.Client, clique)
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(3), clique.Spec.Replicas)
	holdTestCyborgPod(t, ctx, r.Client, clique, 5, false)
	require.NoError(t, r.Create(ctx, newTestReleasedCyborgPod(clique, 1)))
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	bindTestCyborgPod(t, ctx, r.Client, clique, 1, "gpu-c")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))

	t.Log("The Pods of fabric pod c now have the indexes 0 and 1: the capacity stays, and the controller removes no more Pods")
	settleTestCyborgClique(t, ctx, r.Client, clique)
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{0, 1}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, []string{"c", "c"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Scale in to one Pod: the Pod with the lowest index stays, although it is the newer one, so its index stays below the total")
	placement.replicas = 1
	for range 2 {
		require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	}
	require.Equal(t, int32(1), clique.Spec.Replicas)
	require.Equal(t, []int{0}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
}

func TestLPXCyborgSurgeConditions(t *testing.T) {
	for _, tc := range []struct {
		name           string
		settled        bool
		scalingBlocked bool
		want           int32
	}{
		{name: "a settled placement gets one more position for a Pod above the total", settled: true, want: 3},
		{name: "a placement that is not settled gets none: its fabric pods can still change", want: 2},
		{name: "the capacity write waits while Grove defers replica writes", settled: true, scalingBlocked: true, want: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given two bound Cyborg Pods with the indexes 0 and 2 in a placement of two Pods, and a PodClique that Grove has synchronized")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
			clique.Spec.Replicas = 2
			r := &graphReconciler{Client: newLPXTestClient(t, clique, newTestBoundCyborgPod(clique, 0, "a"), newTestBoundCyborgPod(clique, 2, "a"))}
			settleTestCyborgClique(t, ctx, r.Client, clique)

			t.Log("When the controller reconciles the placement")
			placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 2, settled: tc.settled}
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, tc.scalingBlocked))

			t.Log("Then the capacity has the added position, or not, and each Pod stays")
			require.Equal(t, tc.want, clique.Spec.Replicas)
			require.Equal(t, []int{0, 2}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
		})
	}
}

func TestLPXCyborgSurgeRemoval(t *testing.T) {
	for _, tc := range []struct {
		name string
		// refusal is the answer of the API server to the removal; nil accepts it.
		refusal      error
		wantErr      string
		wantPods     []int
		wantCapacity int32
	}{
		{name: "a removal that the API server accepts", wantPods: []int{0, 1}, wantCapacity: 2},
		{
			name:    "a refused removal is an error, and the capacity stays",
			refusal: apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, "pod", fmt.Errorf("denied")),
			wantErr: "outside its placement", wantPods: []int{0, 1, 2}, wantCapacity: 3,
		},
		{
			name:     "a Pod that is already gone is not an error",
			refusal:  apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, "pod"),
			wantPods: []int{0, 1, 2}, wantCapacity: 2,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given a placement of two Pods with one more position, whose new Pod with index 1 is bound")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
			clique.Spec.Replicas = 3
			base := newLPXTestClient(t, clique, newTestBoundCyborgPod(clique, 0, "a"), newTestBoundCyborgPod(clique, 1, "a"), newTestBoundCyborgPod(clique, 2, "a"))
			r := &graphReconciler{Client: interceptor.NewClient(base, interceptor.Funcs{
				Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
					if tc.refusal != nil {
						return tc.refusal
					}
					return delegated.Delete(ctx, object, opts...)
				},
			})}

			t.Log("When the controller reconciles the placement, it removes the Pod with index 2")
			err := reconcileTestCyborgPlacement(t, ctx, r, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 2, settled: true}, false)

			t.Log("Then the result follows the answer of the API server")
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.wantPods, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
			require.Equal(t, tc.wantCapacity, clique.Spec.Replicas)
		})
	}
}

func TestLPXCyborgServedRecord(t *testing.T) {
	ctx := t.Context()

	t.Log("After a scale-in, two fabric pods have two Cyborg Pods each; the Pods of fabric pod d have indexes above the total of four, and one Pod of fabric pod b is not Ready")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 4
	var (
		records int
		refused error
	)
	r := &graphReconciler{Client: interceptor.NewClient(newLPXTestClient(t, clique), interceptor.Funcs{
		Patch: func(ctx context.Context, delegated client.WithWatch, object client.Object, patch client.Patch, opts ...client.PatchOption) error {
			if refused != nil {
				return refused
			}
			records++
			return delegated.Patch(ctx, object, patch, opts...)
		},
	})}
	for index, ready := range map[int]bool{2: true, 3: false, 6: true, 7: true} {
		pod := newTestBoundCyborgPod(clique, index, []string{"b", "d"}[index/4])
		pod.Status.Conditions = newTestCyborgPodConditions(ready)
		require.NoError(t, r.Create(ctx, pod))
	}
	settleTestCyborgClique(t, ctx, r.Client, clique)
	placement := cyborgPlacement{fabricPods: []string{"b", "d"}, replicas: 2, settled: true}

	t.Log("The controller records on each Ready Pod that it served, also below the total; a Pod that serves gets no new Pod")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	served := map[int]string{}
	for _, pod := range listTestCyborgPods(t, ctx, r.Client, clique) {
		served[podCliquePodIndex(pod)] = pod.Labels[cyborgServedLabel]
	}
	require.Equal(t, map[int]string{2: "true", 3: "", 6: "true", 7: "true"}, served)
	require.Equal(t, 3, records)
	require.Equal(t, int32(4), clique.Spec.Replicas)

	t.Log("A second pass writes no record again")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, 3, records)

	t.Log("The Pods lose Ready: a Pod that served keeps its index, because a new Pod in its place would stop a Pod of a workload in service")
	for _, pod := range listTestCyborgPods(t, ctx, r.Client, clique) {
		pod.Status.Conditions = newTestCyborgPodConditions(false)
		require.NoError(t, r.Status().Update(ctx, pod))
	}
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(4), clique.Spec.Replicas)
	require.Equal(t, []int{2, 3, 6, 7}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))

	t.Log("A record that the API server refuses is an error")
	bindTestCyborgPod(t, ctx, r.Client, clique, 3, "gpu-b")
	for _, pod := range listTestCyborgPods(t, ctx, r.Client, clique) {
		pod.Status.Conditions = newTestCyborgPodConditions(true)
		require.NoError(t, r.Status().Update(ctx, pod))
	}
	refused = apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, "pod", fmt.Errorf("denied"))
	_, err := r.pinCyborgPods(ctx, map[string]*grovev1alpha1.PodClique{clique.Name: clique}, map[string]cyborgPlacement{clique.Name: placement})
	require.ErrorContains(t, err, "record that Cyborg Pod")
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
			require.Equal(t, []string{"a", "a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
		})
	}
}
func TestLPXCyborgScaleInRetention(t *testing.T) {
	type pod struct {
		unbound bool
		ready   bool
		served  bool
	}

	for _, tc := range []struct {
		name string
		// pods are the Pods with the indexes 0 and 1 of one fabric pod.
		pods [2]pod
		want []int
	}{
		{name: "a bound Pod stays before an unbound Pod", pods: [2]pod{{unbound: true}, {}}, want: []int{1}},
		{name: "a Ready Pod stays before a Pod that is not Ready", pods: [2]pod{{}, {ready: true}}, want: []int{1}},
		{name: "a Pod that served stays before a Pod that did not", pods: [2]pod{{}, {served: true}}, want: []int{1}},
		{name: "the lower index stays among equal Pods", pods: [2]pod{{}, {}}, want: []int{0}},
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
				if spec.served {
					pod.Labels[cyborgServedLabel] = "true"
				}
				require.NoError(t, r.Create(ctx, pod))
			}

			t.Log("When the controller reconciles the placement")
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, false))

			t.Log("Then the Pod that the placement loses least with stays, and the capacity is one")
			require.Equal(t, tc.want, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
			require.Equal(t, int32(1), clique.Spec.Replicas)
		})
	}
}

func TestCyborgSurgeFabricPod(t *testing.T) {
	type pod struct {
		index     int
		fabricPod string
		unbound   bool
		ready     bool
		served    bool
		failed    bool
	}

	for _, tc := range []struct {
		name       string
		fabricPods []string
		replicas   int32
		pods       []pod
		// unsettled is a placement whose fabric pods are not those of committed plans.
		unsettled bool
		// added is the capacity above the placement total.
		added int32
		// notObserved leaves Grove's observed generation behind the PodClique generation.
		notObserved bool
		// neverObserved leaves the PodClique without an observed generation.
		neverObserved bool
		// uncounted is the number of Pods that Grove counts above the placement total.
		uncounted int32
		want      string
	}{
		{
			name:       "a Pod above the total gets one more position in its fabric pod",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			want: "b",
		},
		{
			name:       "Pods below the total need no position",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 0, fabricPod: "b"}, {index: 1, fabricPod: "b"}},
		},
		{
			name:       "an index equal to the total is above the range",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods: []pod{{index: 0, fabricPod: "b"}, {index: 2, fabricPod: "d"}},
			want: "d",
		},
		{
			name:       "the total counts every fabric pod: the indexes of the first block of two fabric pods are in the range",
			fabricPods: []string{"b", "d"}, replicas: 2,
			pods: []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}, {index: 6, fabricPod: "d"}, {index: 7, fabricPod: "d"}},
			want: "d",
		},
		{
			name:       "the first fabric pod in name order is complete before the next one starts",
			fabricPods: []string{"b", "d"}, replicas: 2,
			pods: []pod{{index: 1, fabricPod: "b"}, {index: 5, fabricPod: "d"}, {index: 6, fabricPod: "d"}, {index: 7, fabricPod: "b"}},
			want: "b",
		},
		{
			name:       "one Cyborg Pod for each fabric pod also gets a position: the old Pod stays until the new one is bound",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods: []pod{{index: 1, fabricPod: "b"}, {index: 3, fabricPod: "d"}},
			want: "d",
		},
		{
			name:       "a Ready Pod uses the index that it has",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods: []pod{{index: 2, fabricPod: "b", ready: true}, {index: 3, fabricPod: "d"}},
			want: "d",
		},
		{
			name:       "a Ready Pod below the total does not stop the step",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 0, fabricPod: "b", ready: true}, {index: 3, fabricPod: "b"}},
			want: "b",
		},
		{
			name:       "no position when each Pod above the total is Ready",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 2, fabricPod: "b", ready: true}, {index: 3, fabricPod: "b", ready: true}},
		},
		{
			name:       "a Pod that served keeps its index while it is not Ready",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods: []pod{{index: 2, fabricPod: "b", served: true}, {index: 3, fabricPod: "d"}},
			want: "d",
		},
		{
			name:       "a placement that is not settled gets no position",
			fabricPods: []string{"b"}, replicas: 2,
			pods:      []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			unsettled: true,
		},
		{
			name:       "wait while a Pod of the placement is not bound",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 0, fabricPod: "b", unbound: true}, {index: 3, fabricPod: "b"}},
		},
		{
			name:       "wait while the PodClique has too few Pods",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods: []pod{{index: 3, fabricPod: "d"}},
		},
		{
			name:       "wait while a Pod that the placement does not keep is still there",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 0, fabricPod: "c"}, {index: 1, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
		},
		{
			name:       "wait while a Pod has no position",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 0}, {index: 1, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
		},
		{
			name:       "a Pod above the total outside the placement gets no position",
			fabricPods: []string{"b"}, replicas: 1,
			pods:  []pod{{index: 0, fabricPod: "b"}, {index: 3, fabricPod: "c"}},
			added: 1,
		},
		{
			name:       "wait until Grove has synchronized the capacity",
			fabricPods: []string{"b"}, replicas: 2,
			pods:        []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			notObserved: true,
		},
		{
			name:       "wait until Grove has synchronized the PodClique one time",
			fabricPods: []string{"b"}, replicas: 2,
			pods:          []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			neverObserved: true,
		},
		{
			name:       "wait while Grove counts a Pod that the controller does not observe",
			fabricPods: []string{"b"}, replicas: 2,
			pods:      []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			uncounted: 1,
		},
		{
			name:       "wait while a different capacity is in effect, for example during a prune",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 4,
		},
		{
			name:       "the added position stays while Grove has not created the new Pod, also before its status follows",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 1, notObserved: true,
			want: "b",
		},
		{
			name:       "the added position stays while the new Pod is not bound",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b", unbound: true}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 1, uncounted: 1,
			want: "b",
		},
		{
			name:       "a Pod in phase Failed above the total gets no position: the controller removes it",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods: []pod{{index: 0, fabricPod: "b"}, {index: 3, fabricPod: "d", failed: true}},
		},
		{
			name:       "wait while a Pod of the placement is in phase Failed",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 0, fabricPod: "b", failed: true}, {index: 3, fabricPod: "b"}},
		},
		{
			name:       "the added position stays when the new Pod fails after its bind: the scheduler does not accept a Pod in phase Failed",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b", failed: true}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 1, uncounted: 1,
			want: "b",
		},
		{
			name:       "the added position goes away when the new Pod is bound",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b"}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 1, uncounted: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given the live Pods of one Cyborg PodClique in Grove index order, its capacity, and Grove's status of it")
			total := tc.replicas * int32(len(tc.fabricPods))
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", Generation: 2}
			clique.Spec.Replicas = total + tc.added
			clique.Status.ObservedGeneration, clique.Status.Replicas = new(clique.Generation), total+tc.uncounted
			if tc.notObserved {
				clique.Status.ObservedGeneration = new(clique.Generation - 1)
			}
			if tc.neverObserved {
				clique.Status.ObservedGeneration = nil
			}
			var live []*corev1.Pod
			for _, spec := range tc.pods {
				pod := newTestBoundCyborgPod(clique, spec.index, spec.fabricPod)
				pod.Status.Conditions = newTestCyborgPodConditions(spec.ready)
				if spec.unbound {
					pod.Spec.NodeName = ""
				}
				if spec.served {
					pod.Labels[cyborgServedLabel] = "true"
				}
				if spec.failed {
					pod.Status.Phase = corev1.PodFailed
				}
				live = append(live, podcache.Project(pod))
			}
			slices.SortFunc(live, func(a, b *corev1.Pod) int { return podCliquePodIndex(a) - podCliquePodIndex(b) })

			t.Log("When the controller selects the fabric pod that gets one more position")
			surge := cyborgSurgeFabricPod(clique, live, cyborgPlacement{fabricPods: tc.fabricPods, replicas: tc.replicas, settled: !tc.unsettled})

			t.Log("Then it selects this fabric pod, or none")
			require.Equal(t, tc.want, surge)
		})
	}
}

func TestCompareCyborgRetention(t *testing.T) {
	type pod struct {
		index   int
		unbound bool
		ready   bool
		served  bool
	}

	for _, tc := range []struct {
		name string
		pods []pod
		want []int
	}{
		{name: "a bound Pod goes before an unbound Pod", pods: []pod{{index: 0, unbound: true}, {index: 1}}, want: []int{1, 0}},
		{name: "a Ready Pod goes before a Pod that is not Ready", pods: []pod{{index: 0}, {index: 1, ready: true}}, want: []int{1, 0}},
		{name: "a Ready Pod goes before a Pod that served and is not Ready", pods: []pod{{index: 0, served: true}, {index: 1, ready: true}}, want: []int{1, 0}},
		{name: "a Pod that served goes before a Pod that did not", pods: []pod{{index: 0}, {index: 5, served: true}}, want: []int{5, 0}},
		{name: "an unbound Pod is last, also with the served record", pods: []pod{{index: 0, unbound: true, served: true}, {index: 1}}, want: []int{1, 0}},
		{name: "the lower index goes first, also when its Pod is newer", pods: []pod{{index: 3, ready: true}, {index: 1, ready: true}}, want: []int{1, 3}},
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
					pod.Spec.NodeName = ""
				}
				if spec.served {
					pod.Labels[cyborgServedLabel] = "true"
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
			changed, err := r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, false)
			require.NoError(t, err)
			require.Equal(t, tc.pinned, changed)

			t.Log("Then the Pod has its pin and no placement gate, or is unchanged")
			pods := listTestCyborgPods(t, ctx, r.Client, clique)
			require.Len(t, pods, 1)
			require.Equal(t, tc.want, pods[0].Spec.SchedulingGates)
			if tc.pinned {
				require.Equal(t, map[string]string{fabricPodLabel: "a"}, pods[0].Spec.NodeSelector)
				require.Equal(t, "a", pods[0].Labels[cyborgFabricPodLabel])
			} else {
				require.Empty(t, pods[0].Spec.NodeSelector)
				require.Empty(t, pods[0].Labels[cyborgFabricPodLabel])
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
	changed, err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	require.True(t, changed)
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Len(t, pods, 2)
	require.Equal(t, older.Name, pods[0].Name)
	require.Equal(t, "a", pods[0].Labels[cyborgFabricPodLabel])
	require.Empty(t, pods[0].Spec.NodeSelector, "a bound Pod needs no selector")
	require.Equal(t, unbound.Name, pods[1].Name)
	require.Empty(t, pods[1].Labels[cyborgFabricPodLabel])
	require.Empty(t, pods[1].Spec.NodeSelector)

	t.Log("A Pod on a Node without a fabric pod cannot serve any placement")
	stray := newTestCyborgPod(clique, 3)
	stray.Spec.NodeName = "gone"
	require.NoError(t, r.Create(ctx, stray))
	_, err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
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
	r := &graphReconciler{Client: newLPXTestClient(t, clique, kept, pruned)}
	placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}

	t.Log("Keep both Pods: a removal without the lower capacity makes Grove create a replacement")
	changed, err := r.reconcileCyborgPlacement(ctx, clique, placement, true)
	require.NoError(t, err)
	require.False(t, changed)
	require.Len(t, listTestCyborgPods(t, ctx, r.Client, clique), 2)

	t.Log("Remove the Pod outside the placement, then lower the capacity, when Grove permits")
	changed, err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	require.True(t, changed)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)
}

func TestLPXCyborgPruneDoesNotLoopOnFailedScale(t *testing.T) {
	ctx := t.Context()

	t.Log("The scheduler selected fabric pod a of three, and the API server refuses the lower capacity")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 3
	kept, prunedB, prunedC := newTestCyborgPod(clique, 0), newTestCyborgPod(clique, 1), newTestCyborgPod(clique, 2)
	kept.Labels[cyborgFabricPodLabel], prunedB.Labels[cyborgFabricPodLabel], prunedC.Labels[cyborgFabricPodLabel] = "a", "b", "c"
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

	t.Log("Remove the two placed Pods outside fabric pod a, then report the failed capacity write")
	_, err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.ErrorContains(t, err, "scale Cyborg PodClique")
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))

	t.Log("Grove replaces both Pods because its capacity is still three; one replacement binds in fabric pod a")
	floating, gated := newTestCyborgPod(clique, 1), newTestCyborgPod(clique, 2)
	floating.Name, floating.UID, floating.Spec.NodeName = floating.Name+"-r", floating.UID+"-r", "gpu-a"
	gated.Name, gated.UID = gated.Name+"-r", gated.UID+"-r"
	createLPXTestObjects(t, ctx, r.Client, floating, gated)

	t.Log("Leave the replacements to Grove's scale-in: a second removal only makes Grove create them again")
	for range 3 {
		_, err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
		require.ErrorContains(t, err, "scale Cyborg PodClique")
		require.Equal(t, []string{"a", "", ""}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
	}

	t.Log("Write the capacity when the API server accepts it, still without a removal")
	refuseScale = false
	changed, err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	require.True(t, changed)
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
	placed := unplaced.DeepCopy()
	placed.Labels[cyborgFabricPodLabel] = "a"
	bound := placed.DeepCopy()
	bound.Spec.NodeName = "gpu-a"
	ready := bound.DeepCopy()
	ready.Status.Phase = corev1.PodRunning
	unselected := unplaced.DeepCopy()
	delete(unselected.Labels, lpx.CyborgPlacementLabel)
	filter := placedCyborgPodPredicate()

	require.True(t, filter.Create(event.CreateEvent{Object: unplaced}))
	require.False(t, filter.Create(event.CreateEvent{Object: unselected}), "other Cyborg Pods never wake the controller")
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: unplaced, ObjectNew: unplaced}), "every version of an unplaced Pod can accept a placement")
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: unplaced, ObjectNew: placed}))
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: placed, ObjectNew: bound}))
	require.False(t, filter.Update(event.UpdateEvent{ObjectOld: bound, ObjectNew: ready}), "status traffic of a placed Pod is ignored")
	require.True(t, filter.Delete(event.DeleteEvent{Object: bound}))
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
// placed it in fabricPod and the scheduler bound it to Node gpu-<fabricPod>. The
// Pod is not Ready. An empty fabricPod leaves the Pod without a placement.
func newTestBoundCyborgPod(clique *grovev1alpha1.PodClique, index int, fabricPod string) *corev1.Pod {
	pod := newTestCyborgPod(clique, index)
	pod.Spec.SchedulingGates, pod.Spec.NodeName = nil, "gpu-"+fabricPod
	pod.Status.Conditions = newTestCyborgPodConditions(false)
	if fabricPod != "" {
		pod.Labels[cyborgFabricPodLabel] = fabricPod
		pod.Spec.NodeSelector = map[string]string{fabricPodLabel: fabricPod}
	}
	return pod
}

// reconcileTestCyborgPlacement runs the two placement steps of one controller pass
// for clique: the pin step, which decides the added position, and the step that
// removes Pods and writes the capacity. It refreshes clique afterwards.
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
	placements, err := r.pinCyborgPods(ctx, pclqs, map[string]cyborgPlacement{clique.Name: placement})
	require.NoError(t, err)
	err = r.reconcileCyborgPlacements(ctx, pclqs, placements, scalingBlocked)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	return err
}

// settleTestCyborgClique plays the part of Grove after it synchronized the
// PodClique: the status records the current generation and the number of Pods
// that are not being deleted. It refreshes clique.
func settleTestCyborgClique(t *testing.T, ctx context.Context, c client.Client, clique *grovev1alpha1.PodClique) {
	t.Helper()
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	clique.Status.ObservedGeneration = new(clique.Generation)
	clique.Status.Replicas = int32(len(listTestCyborgPods(t, ctx, c, clique)))
	require.NoError(t, c.Update(ctx, clique))
}

// holdTestCyborgPod adds a finalizer to the Pod of clique that has the Grove index,
// or removes it. With the finalizer, a deleted Pod stays in the API server with a
// deletion timestamp, as a Pod does while its containers stop.
func holdTestCyborgPod(t *testing.T, ctx context.Context, c client.Client, clique *grovev1alpha1.PodClique, index int, hold bool) {
	t.Helper()
	list := &corev1.PodList{}
	require.NoError(t, c.List(ctx, list, client.InNamespace(clique.Namespace), client.MatchingLabels{grovecommon.LabelPodClique: clique.Name}))
	for i := range list.Items {
		if pod := &list.Items[i]; podCliquePodIndex(pod) == index {
			pod.Finalizers = nil
			if hold {
				pod.Finalizers = []string{"lpx.nvidia.com/test-hold"}
			}
			require.NoError(t, c.Update(ctx, pod))
		}
	}
}

// bindTestCyborgPod plays the part of the scheduler: it binds the live Pod of clique
// that has the Grove index to the Node.
func bindTestCyborgPod(t *testing.T, ctx context.Context, c client.Client, clique *grovev1alpha1.PodClique, index int, node string) {
	t.Helper()
	for _, pod := range listTestCyborgPods(t, ctx, c, clique) {
		if podCliquePodIndex(pod) == index {
			pod.Spec.NodeName = node
			require.NoError(t, c.Update(ctx, pod))
			return
		}
	}
	require.Failf(t, "no live Cyborg Pod", "index %d", index)
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

// testCyborgFabricPods returns the recorded fabric pod of every Pod, in the given order.
func testCyborgFabricPods(pods []*corev1.Pod) []string {
	fabricPods := make([]string, 0, len(pods))
	for _, pod := range pods {
		fabricPods = append(fabricPods, pod.Labels[cyborgFabricPodLabel])
	}
	return fabricPods
}
