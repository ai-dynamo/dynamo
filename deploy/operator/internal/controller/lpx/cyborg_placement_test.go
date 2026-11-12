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
	require.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("Lower the capacity when the removed Pods are gone: Grove must not observe it before their deletion marks")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
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

	t.Log("The Pod with index 1 also stays while the new Pod is only bound: the kubelet can still refuse it")
	bindTestCyborgPod(t, ctx, r.Client, clique, 0, "gpu-b")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Equal(t, []int{0, 1}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))

	t.Log("When the kubelet has started the new Pod, the controller removes the Pod with index 1")
	startTestCyborgPod(t, ctx, r.Client, clique, 0)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []int{0}, testCyborgPodIndexes(pods))
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("It lowers the capacity when that Pod is gone")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
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
	require.NoError(t, r.pinCyborgPods(ctx, pclqs, placements))
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
	live, _, err := r.liveCyborgPods(ctx, clique)
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
	pclqs, placements := map[string]*grovev1alpha1.PodClique{clique.Name: clique}, map[string]cyborgPlacement{clique.Name: placement}

	t.Log("Prune first: remove the Pods of the fabric pods a and b. The Pods of fabric pod c keep the indexes 4 and 5")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, int32(6), clique.Spec.Replicas)

	t.Log("Lower the capacity when the removed Pods are gone")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Wait until Grove has synchronized the lower capacity: Grove can still hold Pods that it created for the removed ones")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Then give fabric pod c one more position: the capacity is three, and no Pod goes")
	settleTestCyborgClique(t, ctx, r.Client, clique)
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(3), clique.Spec.Replicas)
	require.Equal(t, []int{4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))

	t.Log("Grove gives the new Pod its lowest free index. A pass that stops before the placement step does not pin it: the step starts and ends in complete passes only")
	first := newTestReleasedCyborgPod(clique, 0)
	first.CreationTimestamp = metav1.NewTime(first.CreationTimestamp.Add(time.Hour))
	require.NoError(t, r.Create(ctx, first))
	require.NoError(t, r.pinCyborgPods(ctx, pclqs, placements))
	require.Equal(t, []string{"c", "c", ""}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))

	t.Log("A complete pass pins the new Pod to the added position")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{0, 4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, []string{"c", "c", "c"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))

	t.Log("The old Pods stay while the new Pod is not bound, also when it gets no GPU, and while the kubelet has not started it")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	bindTestCyborgPod(t, ctx, r.Client, clique, 0, "gpu-c")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{0, 4, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("When the new Pod runs, remove the Pod with the lowest index above the total. The added position stays for the second Pod above the total, also while the removed Pod stops")
	holdTestCyborgPod(t, ctx, r.Client, clique, 4, true)
	startTestCyborgPod(t, ctx, r.Client, clique, 0)
	for range 2 {
		require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
		require.Equal(t, []int{0, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
		require.Equal(t, int32(3), clique.Spec.Replicas)
	}

	t.Log("When the removed Pod is gone, Grove creates the next Pod, and the second Pod gets a lower index in the same way")
	holdTestCyborgPod(t, ctx, r.Client, clique, 4, false)
	require.NoError(t, r.Create(ctx, newTestReleasedCyborgPod(clique, 1)))
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	bindTestCyborgPod(t, ctx, r.Client, clique, 1, "gpu-c")
	startTestCyborgPod(t, ctx, r.Client, clique, 1)
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []int{0, 1}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
	require.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("No Pod is above the total now: lower the capacity when the removed Pod is gone")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("The Pods of fabric pod c have the indexes 0 and 1: the capacity stays, and the controller removes no more Pods")
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
		refusal  error
		wantErr  string
		wantPods []int
	}{
		{name: "a removal that the API server accepts", wantPods: []int{0, 1}},
		{
			name:    "a refused removal is an error",
			refusal: apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, "pod", fmt.Errorf("denied")),
			wantErr: "outside its placement", wantPods: []int{0, 1, 2},
		},
		{
			name:     "a Pod that is already gone is not an error",
			refusal:  apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, "pod"),
			wantPods: []int{0, 1, 2},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given a placement of two Pods with one more position, whose new Pod with index 1 runs")
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

			t.Log("Then the result follows the answer of the API server, and the capacity stays in the pass that removes a Pod")
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.wantPods, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
			require.Equal(t, int32(3), clique.Spec.Replicas)
		})
	}
}

func TestLPXCyborgCapacityWaitsForRemovedPods(t *testing.T) {
	for _, tc := range []struct {
		name string
		// stops and alsoStops are the ends of the grace periods of two removed Pods,
		// from now. Zero is no such Pod.
		stops, alsoStops time.Duration
		// surplus adds a Pod that the placement does not keep.
		surplus bool
		// replicas is the count of the placement, and capacity is the capacity before the pass.
		replicas, capacity int32
		want               int32
		wantRetry          bool
	}{
		{name: "a lower capacity waits while a removed Pod is in its grace period", stops: time.Hour, replicas: 1, capacity: 2, want: 2, wantRetry: true},
		{name: "a Pod whose grace period is over does not hold the capacity: its Node can be gone", stops: -time.Minute, replicas: 1, capacity: 2, want: 1},
		{name: "the wait is for the removed Pod that stops last", stops: -time.Minute, alsoStops: time.Hour, replicas: 1, capacity: 2, want: 2, wantRetry: true},
		{name: "the wait is for the removed Pod that stops last, in each order of the Pods", stops: time.Hour, alsoStops: -time.Minute, replicas: 1, capacity: 2, want: 2, wantRetry: true},
		{name: "a lower capacity does not wait when no Pod is in deletion", replicas: 1, capacity: 2, want: 1},
		{name: "a lower capacity waits in the pass that removes a Pod", surplus: true, replicas: 1, capacity: 2, want: 2},
		{name: "a higher capacity does not wait", stops: time.Hour, replicas: 2, capacity: 1, want: 2},
		{name: "a higher capacity does not wait in the pass that removes a Pod", surplus: true, replicas: 3, capacity: 2, want: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given a Cyborg PodClique with one Pod in fabric pod a, and possibly a removed Pod or a Pod outside the placement")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
			clique.Spec.Replicas = tc.capacity
			objects := []client.Object{clique, newTestBoundCyborgPod(clique, 0, "a")}
			if tc.stops != 0 {
				objects = append(objects, newTestStoppingCyborgPod(clique, 1, time.Now().Add(tc.stops)))
			}
			if tc.alsoStops != 0 {
				objects = append(objects, newTestStoppingCyborgPod(clique, 3, time.Now().Add(tc.alsoStops)))
			}
			if tc.surplus {
				objects = append(objects, newTestBoundCyborgPod(clique, 2, "b"))
			}
			r := &graphReconciler{Client: newLPXTestClient(t, objects...)}

			t.Log("When the controller reconciles the placement")
			retryAt, err := r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: tc.replicas}, false)
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))

			t.Log("Then the capacity is written or waits, and a wait for a grace period asks for a pass at its end")
			require.Equal(t, tc.want, clique.Spec.Replicas)
			if tc.wantRetry {
				require.WithinDuration(t, time.Now().Add(max(tc.stops, tc.alsoStops)), retryAt, time.Minute)
			} else {
				require.Zero(t, retryAt)
			}
			require.Equal(t, []int{0}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
		})
	}
}

func TestLPXCyborgPlacementIgnoresAnEarlierPodClique(t *testing.T) {
	ctx := t.Context()

	t.Log("Given a Cyborg PodClique that Grove created again with the same name, and two Pods of the earlier one: a bound Pod in a different fabric pod, and a removed Pod in its grace period")
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
	clique.Spec.Replicas = 2
	earlier := clique.DeepCopy()
	earlier.UID = "earlier-clique-uid"
	r := &graphReconciler{Client: newLPXTestClient(t, clique,
		newTestBoundCyborgPod(clique, 0, "a"), newTestBoundCyborgPod(earlier, 5, "b"), newTestStoppingCyborgPod(earlier, 6, time.Now().Add(time.Hour)),
	)}

	t.Log("When the controller reconciles a placement of one Pod in fabric pod a")
	retryAt, err := r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, false)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))

	t.Log("Then it lowers the capacity at once and keeps the Pod of the earlier PodClique: those Pods are not Pods of this one")
	require.Zero(t, retryAt)
	require.Equal(t, int32(1), clique.Spec.Replicas)
	require.Equal(t, []int{0, 5}, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
}

func TestLPXCyborgCapacityWaitOfTwoPodCliques(t *testing.T) {
	ctx := t.Context()

	t.Log("Given three Cyborg PodCliques with one Pod more capacity than their placement: the removed Pod of the first stops in two hours, that of the second in one hour, and the third has none")
	placements, pclqs, objects := map[string]cyborgPlacement{}, map[string]*grovev1alpha1.PodClique{}, []client.Object{}
	for index, stops := range []time.Duration{2 * time.Hour, time.Hour, 0} {
		clique := &grovev1alpha1.PodClique{Name: fmt.Sprintf("pcs-0-lpx-%d-cond", index), Namespace: "default", UID: types.UID(fmt.Sprintf("clique-uid-%d", index)), ResourceVersion: "1"}
		clique.Spec.Replicas = 2
		objects = append(objects, clique, newTestBoundCyborgPod(clique, 0, "a"))
		if stops != 0 {
			objects = append(objects, newTestStoppingCyborgPod(clique, 1, time.Now().Add(stops)))
		}
		placements[clique.Name], pclqs[clique.Name] = cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, clique
	}
	r := &graphReconciler{Client: newLPXTestClient(t, objects...)}

	t.Log("When the controller reconciles the placements")
	retryAt, err := r.reconcileCyborgPlacements(ctx, pclqs, placements, false)
	require.NoError(t, err)

	t.Log("Then it asks for a pass at the end of the first grace period, and only the third PodClique has the lower capacity")
	require.WithinDuration(t, time.Now().Add(time.Hour), retryAt, time.Minute)
	for index, want := range []int32{2, 2, 1} {
		clique := pclqs[fmt.Sprintf("pcs-0-lpx-%d-cond", index)]
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
		require.Equal(t, want, clique.Spec.Replicas)
	}
}

func TestLPXCyborgCapacityWaitEndsWithTheGracePeriod(t *testing.T) {
	ctx := t.Context()

	t.Log("Given a workload with per-fabric-pod placement and one fabric pod, whose Cyborg PodClique has one Pod more capacity than its placement and a removed Pod in its grace period")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: lpx.CyborgPlacementPerFabricPod}
	node := newTestFabricPodNode("gpu-a", "a")
	renderer := newLPXTestReconciler(t, registry, child, dgd, node)
	workloads, plans, err := renderer.resolveWorkloads(ctx, child, dgd)
	require.NoError(t, err)
	pcs, _, err := renderer.renderPodCliqueSet(ctx, child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	objects := materializeLPXTestPCS(t, child, pcs, plans["lpx"])
	cliqueIndex := slices.IndexFunc(objects, func(object client.Object) bool { return object.GetName() == plans["lpx"].ForReplica(0).CyborgClique })
	clique := objects[cliqueIndex].(*grovev1alpha1.PodClique)
	clique.Spec.Replicas = 2
	stopsAt := time.Now().Add(time.Hour)
	objects = append(objects, node, newTestBoundCyborgPod(clique, 0, "a"), newTestStoppingCyborgPod(clique, 1, stopsAt))
	r := newLPXTestReconciler(t, registry, child, dgd, objects...)
	key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("When the controller reconciles after it synchronized the PodCliqueSet, the capacity stays, and the controller asks for a pass at the end of the grace period")
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	result, err := r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)
	require.Positive(t, result.RequeueAfter)
	require.LessOrEqual(t, result.RequeueAfter, time.Until(stopsAt))

	t.Log("When the removed Pod is gone, the controller lowers the capacity")
	holdTestCyborgPod(t, ctx, r.Client, clique, 1, false)
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)
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
	err := r.pinCyborgPods(ctx, map[string]*grovev1alpha1.PodClique{clique.Name: clique}, map[string]cyborgPlacement{clique.Name: placement})
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

			t.Log("When the controller reconciles the placement two times")
			placement := cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
			kept := listTestCyborgPodIndexes(t, ctx, r.Client, clique)
			require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))

			t.Log("Then the first pass keeps the Pod that the placement loses least with, and the second pass lowers the capacity to one")
			require.Equal(t, tc.want, kept)
			require.Equal(t, tc.want, listTestCyborgPodIndexes(t, ctx, r.Client, clique))
			require.Equal(t, int32(1), clique.Spec.Replicas)
		})
	}
}

func TestCyborgRenumberStep(t *testing.T) {
	type pod struct {
		index     int
		fabricPod string
		unbound   bool
		// waiting is a bound Pod that the kubelet has not started.
		waiting bool
		ready   bool
		served  bool
		failed  bool
		// added is a Pod that Grove created after the provision, for a step.
		added bool
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
		// surge is the fabric pod that gets one more position.
		surge string
		// retire is the index of the Pod that gives up its position; nil is no Pod.
		retire *int
	}{
		{
			name:       "a Pod above the total gets one more position in its fabric pod",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			surge: "b",
		},
		{
			name:       "Pods below the total need no position",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 0, fabricPod: "b"}, {index: 1, fabricPod: "b"}},
		},
		{
			name:       "an index equal to the total is above the range",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods:  []pod{{index: 0, fabricPod: "b"}, {index: 2, fabricPod: "d"}},
			surge: "d",
		},
		{
			name:       "the total counts every fabric pod: the indexes of the first block of two fabric pods are in the range",
			fabricPods: []string{"b", "d"}, replicas: 2,
			pods:  []pod{{index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}, {index: 6, fabricPod: "d"}, {index: 7, fabricPod: "d"}},
			surge: "d",
		},
		{
			name:       "the first fabric pod in name order is complete before the next one starts",
			fabricPods: []string{"b", "d"}, replicas: 2,
			pods:  []pod{{index: 1, fabricPod: "b"}, {index: 5, fabricPod: "d"}, {index: 6, fabricPod: "d"}, {index: 7, fabricPod: "b"}},
			surge: "b",
		},
		{
			name:       "one Cyborg Pod for each fabric pod also gets a position: the old Pod stays until the new one runs",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods:  []pod{{index: 1, fabricPod: "b"}, {index: 3, fabricPod: "d"}},
			surge: "d",
		},
		{
			name:       "a Ready Pod uses the index that it has",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods:  []pod{{index: 2, fabricPod: "b", ready: true}, {index: 3, fabricPod: "d"}},
			surge: "d",
		},
		{
			name:       "a Ready Pod below the total does not stop the step",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b", ready: true}, {index: 3, fabricPod: "b"}},
			surge: "b",
		},
		{
			name:       "no position when each Pod above the total is Ready",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 2, fabricPod: "b", ready: true}, {index: 3, fabricPod: "b", ready: true}},
		},
		{
			name:       "a Pod that served keeps its index while it is not Ready",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods:  []pod{{index: 2, fabricPod: "b", served: true}, {index: 3, fabricPod: "d"}},
			surge: "d",
		},
		{
			name:       "the Pods above a Pod that keeps its index keep theirs: a new Pod would get the FPGA route of that Pod",
			fabricPods: []string{"b", "d"}, replicas: 2,
			pods:  []pod{{index: 4, fabricPod: "b", served: true}, {index: 5, fabricPod: "b"}, {index: 6, fabricPod: "d"}, {index: 7, fabricPod: "d"}},
			surge: "d",
		},
		{
			name:       "a Ready Pod above the range also keeps the Pods above it at their indexes",
			fabricPods: []string{"b"}, replicas: 2,
			pods: []pod{{index: 2, fabricPod: "b", ready: true}, {index: 3, fabricPod: "b"}},
		},
		{
			name:       "the Pod that a step added goes when the Pod above the range served in that time: the two have the same FPGA route",
			fabricPods: []string{"b"}, replicas: 3,
			pods:   []pod{{index: 0, fabricPod: "b", added: true}, {index: 1, fabricPod: "b", added: true}, {index: 4, fabricPod: "b", served: true}, {index: 5, fabricPod: "b"}},
			added:  1,
			retire: new(1),
		},
		{
			name:       "the Pod that a step added also goes when the kubelet has not started it",
			fabricPods: []string{"b"}, replicas: 2,
			pods:   []pod{{index: 0, fabricPod: "b", added: true, waiting: true}, {index: 2, fabricPod: "b", ready: true}, {index: 3, fabricPod: "b"}},
			added:  1,
			retire: new(0),
		},
		{
			name:       "the Pod that a step added also goes when it is Ready: the Pod that served first keeps its place",
			fabricPods: []string{"b"}, replicas: 2,
			pods:   []pod{{index: 0, fabricPod: "b", added: true, ready: true}, {index: 2, fabricPod: "b", served: true}, {index: 3, fabricPod: "b"}},
			added:  1,
			retire: new(0),
		},
		{
			name:       "a fabric pod without a Pod too many loses no Pod",
			fabricPods: []string{"b"}, replicas: 3,
			pods: []pod{{index: 0, fabricPod: "b", added: true}, {index: 4, fabricPod: "b", served: true}, {index: 5, fabricPod: "b"}},
		},
		{
			name:       "a scale-in by one Pod is not a step: no Pod is newer than the Pod above the range, and the retention order selects the Pod that goes",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b"}, {index: 1, fabricPod: "b"}, {index: 2, fabricPod: "b", served: true}},
			added: 1,
		},
		{
			name:       "only the Pod with the highest index below the range can be the Pod that a step added",
			fabricPods: []string{"b"}, replicas: 3,
			pods:  []pod{{index: 0, fabricPod: "b", added: true}, {index: 1, fabricPod: "b"}, {index: 4, fabricPod: "b", served: true}, {index: 5, fabricPod: "b"}},
			added: 1,
		},
		{
			name:       "a scale-in by more Pods is not a step",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b", added: true}, {index: 1, fabricPod: "b", added: true}, {index: 2, fabricPod: "b", served: true}, {index: 3, fabricPod: "b"}},
			added: 2,
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
			surge: "b",
		},
		{
			name:       "the added position stays while the new Pod is not bound",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b", unbound: true}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 1, uncounted: 1,
			surge: "b",
		},
		{
			name:       "the added position stays while the kubelet has not started the new Pod: it can still refuse the Pod",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b", waiting: true}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 1, uncounted: 1,
			surge: "b",
		},
		{
			name:       "the added position stays when the new Pod fails after its bind: the scheduler does not accept a Pod in phase Failed",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b", failed: true}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added: 1, uncounted: 1,
			surge: "b",
		},
		{
			name:       "when the new Pod runs, the Pod with the lowest index above the total gives up its position: the indexes that stay are consecutive modulo the count",
			fabricPods: []string{"b"}, replicas: 2,
			pods:   []pod{{index: 0, fabricPod: "b"}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b"}},
			added:  1,
			retire: new(2),
		},
		{
			name:       "the Pod gives up its position when the other Pods of its fabric pod run, also when the kubelet has not started it",
			fabricPods: []string{"b"}, replicas: 2,
			pods:   []pod{{index: 0, fabricPod: "b"}, {index: 2, fabricPod: "b", waiting: true}, {index: 3, fabricPod: "b"}},
			added:  1,
			retire: new(2),
		},
		{
			name:       "the added position stays while an old Pod that stays does not run",
			fabricPods: []string{"b"}, replicas: 2,
			pods:  []pod{{index: 0, fabricPod: "b"}, {index: 2, fabricPod: "b"}, {index: 3, fabricPod: "b", waiting: true}},
			added: 1, uncounted: 1,
			surge: "b",
		},
		{
			name:       "the Pods that run in a different fabric pod do not count",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods:  []pod{{index: 0, fabricPod: "b"}, {index: 1, fabricPod: "d", waiting: true}, {index: 3, fabricPod: "d"}},
			added: 1, uncounted: 1,
			surge: "d",
		},
		{
			name:       "the Pod gives up its position also when the capacity is already the total",
			fabricPods: []string{"b", "d"}, replicas: 1,
			pods:   []pod{{index: 0, fabricPod: "d"}, {index: 1, fabricPod: "b"}, {index: 3, fabricPod: "d"}},
			retire: new(3),
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
					pod.Spec.NodeName, pod.Status.InitContainerStatuses = "", nil
				}
				if spec.waiting {
					pod.Status.InitContainerStatuses[0].State = corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "PodInitializing"}}
				}
				if spec.served {
					pod.Labels[cyborgServedLabel] = "true"
				}
				if spec.failed {
					pod.Status.Phase = corev1.PodFailed
				}
				if spec.added {
					pod.CreationTimestamp = metav1.NewTime(pod.CreationTimestamp.Add(time.Hour))
				}
				live = append(live, podcache.Project(pod))
			}
			slices.SortFunc(live, func(a, b *corev1.Pod) int { return podCliquePodIndex(a) - podCliquePodIndex(b) })

			t.Log("When the controller decides the step that gives a Pod a lower index")
			surge, retire := cyborgRenumberStep(clique, live, cyborgPlacement{fabricPods: tc.fabricPods, replicas: tc.replicas, settled: !tc.unsettled})

			t.Log("Then a fabric pod gets one more position, or a Pod gives up its position, or neither")
			require.Equal(t, tc.surge, surge)
			var retired *int
			for _, pod := range live {
				if retire != "" && pod.UID == retire {
					retired = new(podCliquePodIndex(pod))
				}
			}
			require.Equal(t, tc.retire, retired)
			require.Equal(t, tc.retire == nil, retire == "")
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
		{name: "a container that ended waits for its restart", main: []corev1.ContainerStatus{{State: waiting, LastTerminationState: ended}}, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Given a bound Cyborg Pod as the operator Pod cache keeps it")
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid"}
			pod := newTestBoundCyborgPod(clique, 0, "a")
			pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses = tc.init, tc.main

			t.Log("Then the Pod counts as started when the kubelet started one of its containers at some time")
			require.Equal(t, tc.want, cyborgPodStarted(podcache.Project(pod)))
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
			_, err := r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, false)
			require.NoError(t, err)

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
	_, err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
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
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, true))
	require.Len(t, listTestCyborgPods(t, ctx, r.Client, clique), 2)
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("When Grove permits, remove the Pod outside the placement; the capacity stays in that pass")
	require.NoError(t, reconcileTestCyborgPlacement(t, ctx, r, clique, placement, false))
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
	require.Equal(t, int32(2), clique.Spec.Replicas)

	t.Log("Lower the capacity when the Pod is gone")
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
	_, err := r.reconcileCyborgPlacement(ctx, clique, placement, false)
	require.NoError(t, err)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))

	t.Log("Grove replaces both Pods because its capacity is still three; one replacement binds in fabric pod a")
	floating, gated := newTestCyborgPod(clique, 1), newTestCyborgPod(clique, 2)
	floating.Name, floating.UID, floating.Spec.NodeName = floating.Name+"-r", floating.UID+"-r", "gpu-a"
	gated.Name, gated.UID = gated.Name+"-r", gated.UID+"-r"
	createLPXTestObjects(t, ctx, r.Client, floating, gated)

	t.Log("Report each failed capacity write, and leave the replacements to Grove's scale-in: a second removal only makes Grove create them again")
	for range 3 {
		_, err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
		require.ErrorContains(t, err, "scale Cyborg PodClique")
		require.Equal(t, []string{"a", "", ""}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
	}

	t.Log("Write the capacity when the API server accepts it, still without a removal")
	refuseScale = false
	_, err = r.reconcileCyborgPlacement(ctx, clique, placement, false)
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
	started := bound.DeepCopy()
	started.Status.InitContainerStatuses = newTestStartedContainerStatuses()
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: bound, ObjectNew: started}), "the Pod that takes the place of a Pod above the range runs")
	failed := started.DeepCopy()
	failed.Status.Phase = corev1.PodFailed
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: started, ObjectNew: failed}), "a Pod in a terminal phase holds no position")
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
// placed it in fabricPod, the scheduler bound it to Node gpu-<fabricPod>, and the
// kubelet started its first init container. The Pod is not Ready. An empty
// fabricPod leaves the Pod without a placement.
func newTestBoundCyborgPod(clique *grovev1alpha1.PodClique, index int, fabricPod string) *corev1.Pod {
	pod := newTestCyborgPod(clique, index)
	pod.Spec.SchedulingGates, pod.Spec.NodeName = nil, "gpu-"+fabricPod
	pod.Status.Conditions = newTestCyborgPodConditions(false)
	pod.Status.InitContainerStatuses = newTestStartedContainerStatuses()
	if fabricPod != "" {
		pod.Labels[cyborgFabricPodLabel] = fabricPod
		pod.Spec.NodeSelector = map[string]string{fabricPodLabel: fabricPod}
	}
	return pod
}

// newTestStoppingCyborgPod returns the Pod of newTestBoundCyborgPod in fabric pod a
// after its removal: the API server marks it with the end of its grace period,
// stopsAt, and keeps it while its containers stop. A test finalizer keeps it in
// the fake client, which accepts such a Pod only as an initial object.
func newTestStoppingCyborgPod(clique *grovev1alpha1.PodClique, index int, stopsAt time.Time) *corev1.Pod {
	pod := newTestBoundCyborgPod(clique, index, "a")
	pod.DeletionTimestamp, pod.Finalizers = new(metav1.NewTime(stopsAt)), []string{testCyborgPodHold}
	return pod
}

// reconcileTestCyborgPlacement runs the placement steps of one complete controller
// pass for clique: the pin step, and the step that gives a Pod a lower index,
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
	placements := map[string]cyborgPlacement{clique.Name: placement}
	require.NoError(t, r.pinCyborgPods(ctx, pclqs, placements))
	_, err := r.reconcileCyborgPlacements(ctx, pclqs, placements, scalingBlocked)
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

// testCyborgPodHold is the finalizer that keeps a removed Pod in the fake client.
const testCyborgPodHold = "lpx.nvidia.com/test-hold"

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
				pod.Finalizers = []string{testCyborgPodHold}
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

// newTestStartedContainerStatuses returns the init container status of a Pod whose
// first init container the kubelet started.
func newTestStartedContainerStatuses() []corev1.ContainerStatus {
	return []corev1.ContainerStatus{{Name: "grove-initc", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}}}
}

// startTestCyborgPod plays the part of the kubelet: it starts the first init
// container of the live Pod of clique that has the Grove index.
func startTestCyborgPod(t *testing.T, ctx context.Context, c client.Client, clique *grovev1alpha1.PodClique, index int) {
	t.Helper()
	for _, pod := range listTestCyborgPods(t, ctx, c, clique) {
		if podCliquePodIndex(pod) == index {
			pod.Status.InitContainerStatuses = newTestStartedContainerStatuses()
			require.NoError(t, c.Status().Update(ctx, pod))
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
