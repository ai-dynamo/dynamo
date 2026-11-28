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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
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
)

func TestLPXCyborgFabricPodPlacement(t *testing.T) {
	ctx := t.Context()

	t.Log("Select per-fabric-pod placement for one hybrid workload in a cluster with three fabric pods")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: cyborgPlacementPerFabricPod}
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
	require.Equal(t, cyborgPlacementPerFabricPod, clique.Labels[cyborgPlacementLabel], "Pods inherit the label that the Pod watch and the placement select")
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
		pod.Spec.SchedulingGates = []corev1.PodSchedulingGate{{Name: cyborgPlacementGate}}
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

	t.Log("The Pod that stays keeps its rank and its replica index: the controller adds and removes no Pod for it")
	for range 2 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"1 b 0/1"}, describeTestCyborgPods(pods))
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)

	t.Log("Scale: two replicas mean two Cyborg Pods in each occupied fabric pod; the Pod with replica index 1 and rank 0 loses its position, and Grove creates a Pod in its place")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = new(int32(2))
	updateTestDGD(t, r, child, dgd)
	for range 3 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(2), clique.Spec.Replicas)
	require.Empty(t, listTestCyborgPods(t, ctx, r.Client, clique))
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0), newTestReleasedCyborgPod(clique, 1))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods = listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"b", "b"}, testCyborgFabricPods(pods))
	require.Equal(t, []string{"0/0", "1/1"}, testCyborgRanks(pods), "each replica index is congruent to its rank modulo the new count")

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
	require.Equal(t, []string{"0/0", "1/1", "0/2", "1/3", "0/4", "1/5"}, testCyborgRanks(pods), "each fabric pod has the ranks 0 and 1, and no replica index occurs two times")
}

func TestLPXCyborgPlacementDoesNotWaitForDesiredState(t *testing.T) {
	ctx := t.Context()

	t.Log("Run one hybrid workload with per-fabric-pod placement in a cluster with one fabric pod")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: cyborgPlacementPerFabricPod}
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

	t.Log("Grove replaces a Cyborg Pod of the running workload, and the new Pod has the placement gate of its template")
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	result, err := r.Reconcile(ctx, key)
	require.NoError(t, err)
	require.Positive(t, result.RequeueAfter, "the pass waits for the download")

	t.Log("The placement does not wait: it is known from observed objects")
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(pods))
	require.Empty(t, pods[0].Spec.SchedulingGates)

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
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: cyborgPlacementPerFabricPod}
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

func TestLPXCyborgPlacementOutlivesItsAnnotation(t *testing.T) {
	ctx := t.Context()

	t.Log("Run one hybrid workload with one explicit Cyborg replica for each of three fabric pods")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: cyborgPlacementPerFabricPod}
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
	require.Equal(t, cyborgPlacementPerFabricPod, clique.Labels[cyborgPlacementLabel])

	t.Log("The template of that PodClique still gates its Pods, so the placement keeps its capacity and pins them")
	require.Equal(t, int32(3), clique.Spec.Replicas, "the explicit replica count does not replace the placement capacity")
	createLPXTestObjects(t, ctx, r.Client, newTestReleasedCyborgPod(clique, 0))
	_, err = r.Reconcile(ctx, key)
	require.NoError(t, err)
	pods := listTestCyborgPods(t, ctx, r.Client, clique)
	require.Equal(t, []string{"a"}, testCyborgFabricPods(pods))
	require.Empty(t, pods[0].Spec.SchedulingGates)

	t.Log("After Grove replaces the PodClique without the label, the explicit replica count applies")
	delete(clique.Labels, cyborgPlacementLabel)
	require.NoError(t, r.Update(ctx, clique))
	for range 2 {
		_, err = r.Reconcile(ctx, key)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, cliqueKey, clique))
	require.Equal(t, int32(1), clique.Spec.Replicas)
}

func TestLPXCyborgPlacementOfPods(t *testing.T) {
	placement, grove := corev1.PodSchedulingGate{Name: cyborgPlacementGate}, corev1.PodSchedulingGate{Name: grovePodGangGate}
	quota := corev1.PodSchedulingGate{Name: "example.com/quota"}
	bound, degraded := lpxv1alpha1.RequestPhaseBound, lpxv1alpha1.RequestPhaseDegraded

	// The Cyborg PodClique of workload replica 0, with the placement label of its template.
	clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1", Labels: map[string]string{
		cyborgPlacementLabel: cyborgPlacementPerFabricPod,
	}}
	// A Pod that no Node has, with these scheduling gates.
	gated := func(index int, gates ...corev1.PodSchedulingGate) *corev1.Pod {
		pod := newTestCyborgPod(clique, index)
		pod.Spec.SchedulingGates = gates
		return pod
	}
	// A Pod that the scheduler bound before the controller pinned it.
	scheduled := func(index int, node string) *corev1.Pod {
		pod := gated(index)
		pod.Spec.NodeName = node
		return pod
	}
	// A Pod that the controller pinned and the scheduler bound in fabricPod. Its replica index is its Grove index.
	placed := func(index int, fabricPod string, rank int) *corev1.Pod {
		pod := scheduled(index, "gpu-"+fabricPod)
		pod.Labels[cyborgFabricPodLabel], pod.Spec.NodeSelector = fabricPod, map[string]string{fabricPodLabel: fabricPod}
		pod.Labels[cyborgFabricPodRankLabel], pod.Labels[cyborgReplicaIndexLabel] = strconv.Itoa(rank), strconv.Itoa(index)
		return pod
	}
	unranked := placed(2, "a", 0)
	delete(unranked.Labels, cyborgFabricPodRankLabel)
	delete(unranked.Labels, cyborgReplicaIndexLabel)
	// One GPU Node in each fabric pod.
	nodes := func(fabricPods ...string) (nodes []*corev1.Node) {
		for _, fabricPod := range fabricPods {
			nodes = append(nodes, newTestFabricPodNode("gpu-"+fabricPod, fabricPod))
		}
		return nodes
	}
	// The requests of workload replica 0, each with a committed plan in fabricPods.
	request := func(name string, phase lpxv1alpha1.RequestPhase, fabricPods ...string) *lpxv1alpha1.LPUPipelineRequest {
		return newTestPlacedPipelineRequest(name, "pcs-0-lpx", 0, phase, fabricPods...)
	}

	for _, tc := range []struct {
		name string
		// replicas is the Cyborg count of the PodCliqueSet template: the count for each fabric pod.
		replicas int32
		// capacity is the replica count of the PodClique before the pass.
		capacity       int32
		nodes          []*corev1.Node
		nodeSelector   map[string]string
		requests       []*lpxv1alpha1.LPUPipelineRequest
		coherentUpdate bool
		pods           []*corev1.Pod
		// want describes each Pod that stays, in Grove index order (see describeTestCyborgPods).
		want []string
		// wantGates are the scheduling gates of the first Pod that stays, when not nil.
		wantGates    []corev1.PodSchedulingGate
		wantCapacity int32
	}{
		{
			name:     "in a provision, a Pod takes the fabric pod, the rank and the replica index of its Grove index, also before the Pods with lower indexes exist",
			replicas: 2, capacity: 6, nodes: nodes("a", "b", "c"),
			pods:         []*corev1.Pod{gated(4, placement), gated(1, placement), gated(3, placement)},
			want:         []string{"1 a 1/1", "3 b 1/3", "4 c 0/4"},
			wantCapacity: 6,
		},
		{
			name:     "a Pod that Grove has not released keeps the placement gate: Grove removes its gate with a list of the gates that stay",
			replicas: 1, capacity: 1, nodes: nodes("a"),
			pods: []*corev1.Pod{gated(0, placement, grove)},
			want: []string{"0 -"}, wantGates: []corev1.PodSchedulingGate{placement, grove}, wantCapacity: 1,
		},
		{
			name:     "the pin keeps the gate of another owner",
			replicas: 1, capacity: 1, nodes: nodes("a"),
			pods: []*corev1.Pod{gated(0, quota, placement)},
			want: []string{"0 a 0/0"}, wantGates: []corev1.PodSchedulingGate{quota}, wantCapacity: 1,
		},
		{
			name:     "a Pod without the gate of its template is pinned while Grove's gate holds it",
			replicas: 1, capacity: 1, nodes: nodes("a"),
			pods: []*corev1.Pod{gated(0, grove)},
			want: []string{"0 a 0/0"}, wantGates: []corev1.PodSchedulingGate{grove}, wantCapacity: 1,
		},
		{
			name:     "a Pod without a gate gets no pin: Kubernetes accepts no node selector on it",
			replicas: 1, capacity: 1, nodes: nodes("a"),
			pods: []*corev1.Pod{gated(0)},
			want: []string{"0 -"}, wantCapacity: 1,
		},
		{
			name:     "a Pod that was scheduled before its pin keeps the position of its Node, and a Pod without a position is removed",
			replicas: 1, capacity: 2, nodes: nodes("a", "b"),
			pods: []*corev1.Pod{scheduled(0, "gpu-b"), scheduled(1, "gpu-b"), scheduled(2, "node-without-fabric-pod")},
			want: []string{"0 b 0/0"}, wantCapacity: 2,
		},
		{
			name:     "only the fabric pods that the node selector of the PodClique can select get Pods",
			replicas: 1, capacity: 2,
			nodes:        []*corev1.Node{newTestFabricPodNode("gpu-a", "a", "pool", "gpu"), newTestFabricPodNode("gpu-b", "b", "pool", "cpu")},
			nodeSelector: map[string]string{"pool": "gpu"},
			pods:         []*corev1.Pod{gated(0, placement), gated(1, placement)},
			want:         []string{"0 a 0/0", "1 -"}, wantCapacity: 1,
		},
		{
			name:     "one model without a bound plan keeps the provision in every fabric pod",
			replicas: 1, capacity: 3, nodes: nodes("a", "b", "c"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("draft", bound, "b"), request("target", lpxv1alpha1.RequestPhaseBinding, "b")},
			pods:     []*corev1.Pod{placed(0, "a", 0), placed(1, "b", 0), placed(2, "c", 0)},
			want:     []string{"0 a 0/0", "1 b 0/1", "2 c 0/2"}, wantCapacity: 3,
		},
		{
			name:     "when every request is Bound or Degraded, only the fabric pods of the plans keep their Pods; the capacity waits until Grove has observed the removals",
			replicas: 1, capacity: 3, nodes: nodes("a", "b", "c"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("draft", bound, "b"), request("target", degraded, "b")},
			pods:     []*corev1.Pod{placed(0, "a", 0), placed(1, "b", 0), placed(2, "c", 0)},
			want:     []string{"1 b 0/1"}, wantCapacity: 3,
		},
		{
			name:     "the capacity follows when Grove has created a Pod for each removed one",
			replicas: 1, capacity: 3, nodes: nodes("a", "b", "c"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("target", bound, "b")},
			pods:     []*corev1.Pod{gated(0, placement), placed(1, "b", 0), gated(2, placement)},
			want:     []string{"0 -", "1 b 0/1", "2 -"}, wantCapacity: 1,
		},
		{
			name:     "a replacement Pod takes the rank that its fabric pod lost, and the lowest free replica index of that rank",
			replicas: 2, capacity: 4, nodes: nodes("a", "b", "c", "d"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("target", bound, "b", "d")},
			pods:     []*corev1.Pod{gated(0, placement), placed(2, "b", 0), placed(3, "b", 1), placed(6, "d", 0)},
			want:     []string{"0 d 1/1", "2 b 0/2", "3 b 1/3", "6 d 0/6"}, wantCapacity: 4,
		},
		{
			name:     "a scale-in keeps the Pod with the lower rank in each fabric pod",
			replicas: 1, capacity: 4, nodes: nodes("a", "b", "c", "d"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("target", bound, "b", "d")},
			pods:     []*corev1.Pod{placed(2, "b", 0), placed(3, "b", 1), placed(6, "d", 0), placed(7, "d", 1)},
			want:     []string{"2 b 0/2", "6 d 0/6"}, wantCapacity: 4,
		},
		{
			name:     "a scale-in does not keep a rank that is not below the new count: the new Pod takes rank 0",
			replicas: 1, capacity: 2, nodes: nodes("a", "b"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("target", bound, "b")},
			pods:     []*corev1.Pod{gated(2, placement), placed(3, "b", 1)},
			want:     []string{"2 b 0/2"}, wantCapacity: 2,
		},
		{
			name:     "a scale-out does not keep a replica index that is not congruent to its rank",
			replicas: 2, capacity: 2, nodes: nodes("a", "b"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("target", bound, "b")},
			pods:     []*corev1.Pod{gated(0, placement), placed(1, "b", 0)},
			want:     []string{"0 b 0/0"}, wantCapacity: 2,
		},
		{
			name:     "a placed Pod without its own rank and replica index has no position",
			replicas: 2, capacity: 2, nodes: nodes("a"),
			pods: []*corev1.Pod{placed(0, "a", 0), placed(1, "a", 0), unranked},
			want: []string{"0 a 0/0"}, wantCapacity: 2,
		},
		{
			name:     "during a Coherent update the pin continues, and the removals and the capacity wait",
			replicas: 1, capacity: 2, nodes: nodes("a", "b"), coherentUpdate: true,
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("target", bound, "a")},
			pods:     []*corev1.Pod{gated(0, placement), placed(1, "b", 0)},
			want:     []string{"0 a 0/0", "1 b 0/1"}, wantCapacity: 2,
		},
		{
			name:     "while a lower capacity is pending, a scheduled Pod without a position stays for Grove's scale-in: Grove would replace it",
			replicas: 1, capacity: 2, nodes: nodes("a", "b"),
			requests: []*lpxv1alpha1.LPUPipelineRequest{request("target", bound, "a")},
			pods:     []*corev1.Pod{placed(0, "a", 0), scheduled(1, "gpu-b")},
			want:     []string{"0 a 0/0", "1 -"}, wantCapacity: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			t.Log("Given the observed PodCliqueSet, PodClique, Nodes, requests and Cyborg Pods of one workload replica")
			pcs := &grovev1alpha1.PodCliqueSet{Name: "pcs", Namespace: "default", Generation: 1}
			pcs.Spec.Template.Cliques = []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "cond", Spec: grovev1alpha1.PodCliqueSpec{Replicas: tc.replicas}}}
			pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{{Name: "lpx", CliqueNames: []string{"cond"}}}
			if tc.coherentUpdate {
				pcs.Spec.UpdateStrategy = &grovev1alpha1.PodCliqueSetUpdateStrategy{Type: grovev1alpha1.CoherentStrategy}
			}
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{Name: "pcs-0-lpx", Namespace: "default"}
			pcsg.Spec.Replicas = 1
			observed := clique.DeepCopy()
			observed.Spec.Replicas, observed.Spec.PodSpec.NodeSelector = tc.capacity, tc.nodeSelector
			objects := []client.Object{observed}
			for _, node := range tc.nodes {
				objects = append(objects, node)
			}
			for _, pod := range tc.pods {
				objects = append(objects, pod)
			}
			requests := make(map[string]*lpxv1alpha1.LPUPipelineRequest)
			for _, request := range tc.requests {
				requests[request.Name] = request
			}
			r := &graphReconciler{Client: newLPXTestClient(t, objects...), config: &configv1alpha1.OperatorConfiguration{}}

			t.Log("When the controller reconciles the placement")
			err := r.reconcileCyborgPlacement(ctx,
				pcs, map[string]*grovev1alpha1.PodCliqueScalingGroup{pcsg.Name: pcsg}, map[string]*grovev1alpha1.PodClique{observed.Name: observed}, requests)
			require.NoError(t, err)

			t.Log("Then the Pods and the capacity of the PodClique are these")
			pods := listTestCyborgPods(t, ctx, r.Client, observed)
			require.Equal(t, tc.want, describeTestCyborgPods(pods))
			for _, pod := range pods {
				if pod.Labels[cyborgFabricPodLabel] != "" && pod.Spec.NodeName == "" {
					require.Equal(t, map[string]string{fabricPodLabel: pod.Labels[cyborgFabricPodLabel]}, pod.Spec.NodeSelector, "the pin of %s", pod.Name)
					require.NotContains(t, pod.Spec.SchedulingGates, placement, "the pin of %s removes the placement gate in the same write", pod.Name)
				}
			}
			if tc.wantGates != nil {
				require.Equal(t, tc.wantGates, pods[0].Spec.SchedulingGates)
			}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(observed), observed))
			require.Equal(t, tc.wantCapacity, observed.Spec.Replicas)
		})
	}
}

func TestLPXCyborgPlacementContinuesPastOnePodClique(t *testing.T) {
	ctx := t.Context()

	t.Log("Two workload replicas have a placed PodClique: the first one selects no Node, the second one selects fabric pod a")
	pcs := &grovev1alpha1.PodCliqueSet{Name: "pcs", Namespace: "default"}
	pcs.Spec.Template.Cliques = []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "cond", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}}}
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{{Name: "lpx", CliqueNames: []string{"cond"}}}
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{Name: "pcs-0-lpx", Namespace: "default"}
	pcsg.Spec.Replicas = 2
	placed := map[string]string{cyborgPlacementLabel: cyborgPlacementPerFabricPod}
	unselectable := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-0", ResourceVersion: "1", Labels: placed}
	unselectable.Spec.Replicas, unselectable.Spec.PodSpec.NodeSelector = 1, map[string]string{"pool": "wrong"}
	selectable := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-1-cond", Namespace: "default", UID: "clique-1", ResourceVersion: "1", Labels: placed}
	selectable.Spec.Replicas = 1
	r := &graphReconciler{
		Client: newLPXTestClient(t, unselectable, selectable, newTestReleasedCyborgPod(selectable, 0), newTestFabricPodNode("gpu-a", "a", "pool", "gpu")),
		config: &configv1alpha1.OperatorConfiguration{},
	}

	t.Log("Name the PodClique without a Node, and still pin the Pod of the other PodClique")
	err := r.reconcileCyborgPlacement(ctx, pcs,
		map[string]*grovev1alpha1.PodCliqueScalingGroup{pcsg.Name: pcsg},
		map[string]*grovev1alpha1.PodClique{unselectable.Name: unselectable, selectable.Name: selectable}, nil)
	require.ErrorContains(t, err, unselectable.Name)
	require.Equal(t, []string{"0 a 0/0"}, describeTestCyborgPods(listTestCyborgPods(t, ctx, r.Client, selectable)))
}

func TestLPXCyborgPlacementDoesNotScaleFromAnOldObservation(t *testing.T) {
	ctx := t.Context()

	t.Log("The pass observes a placed PodClique with capacity for one of two fabric pods")
	pcs := &grovev1alpha1.PodCliqueSet{Name: "pcs", Namespace: "default"}
	pcs.Spec.Template.Cliques = []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "cond", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}}}
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = []grovev1alpha1.PodCliqueScalingGroupConfig{{Name: "lpx", CliqueNames: []string{"cond"}}}
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{Name: "pcs-0-lpx", Namespace: "default"}
	pcsg.Spec.Replicas = 1
	observed := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1", Labels: map[string]string{
		cyborgPlacementLabel: cyborgPlacementPerFabricPod,
	}}
	observed.Spec.Replicas = 1
	r := &graphReconciler{
		Client: newLPXTestClient(t, observed.DeepCopy(), newTestFabricPodNode("gpu-a", "a"), newTestFabricPodNode("gpu-b", "b")),
		config: &configv1alpha1.OperatorConfiguration{},
	}

	t.Log("The role drops the placement, and the PodClique of that name no longer has the label")
	current := observed.DeepCopy()
	current.Labels = nil
	require.NoError(t, r.Update(ctx, current))

	t.Log("The placement does not scale the PodClique that it no longer owns, and reports no error: the new version starts a new pass")
	err := r.reconcileCyborgPlacement(ctx, pcs,
		map[string]*grovev1alpha1.PodCliqueScalingGroup{pcsg.Name: pcsg}, map[string]*grovev1alpha1.PodClique{observed.Name: observed}, nil)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(current), current))
	require.Equal(t, int32(1), current.Spec.Replicas)
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
// index. For a clique with the placement label, the Pod also has that label and,
// before Grove's gate, the placement gate that the template of such a clique has.
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
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if mode := clique.Labels[cyborgPlacementLabel]; mode != "" {
		pod.Labels[cyborgPlacementLabel] = mode
		pod.Spec.SchedulingGates = []corev1.PodSchedulingGate{{Name: cyborgPlacementGate}}
	}
	pod.Spec.SchedulingGates = append(pod.Spec.SchedulingGates, corev1.PodSchedulingGate{Name: grovePodGangGate})
	return pod
}

// newTestReleasedCyborgPod returns the Pod of newTestCyborgPod after Grove removed its scheduling gate.
func newTestReleasedCyborgPod(clique *grovev1alpha1.PodClique, index int) *corev1.Pod {
	pod := newTestCyborgPod(clique, index)
	pod.Spec.SchedulingGates = slices.DeleteFunc(pod.Spec.SchedulingGates, func(gate corev1.PodSchedulingGate) bool { return gate.Name == grovePodGangGate })
	return pod
}

// testCyborgPodSerial numbers the Pods that replaceTestCyborgPods creates.
var testCyborgPodSerial atomic.Int64

// replaceTestCyborgPods plays the part of Grove while clique has fewer live Pods
// than its capacity: for each missing Pod it creates a Pod with the lowest index
// that no live Pod has. Grove has released the new Pod, so only the placement
// gate of its template holds it, and it is newer than each earlier Pod.
func replaceTestCyborgPods(t *testing.T, ctx context.Context, c client.Client, clique *grovev1alpha1.PodClique) {
	t.Helper()
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(clique), clique))
	live := listTestCyborgPods(t, ctx, c, clique)
	for index := 0; len(live) < int(clique.Spec.Replicas); index++ {
		if slices.ContainsFunc(live, func(pod *corev1.Pod) bool {
			return pod.Labels[grovecommon.LabelPodCliquePodIndex] == strconv.Itoa(index)
		}) {
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

// describeTestCyborgPods returns "<Grove index> <fabric pod> <rank>/<replica index>"
// for each Pod that has a placement and "<Grove index> -" for each other Pod, in
// the given order.
func describeTestCyborgPods(pods []*corev1.Pod) (described []string) {
	for _, pod := range pods {
		placement := "-"
		if fabricPod := pod.Labels[cyborgFabricPodLabel]; fabricPod != "" {
			placement = fabricPod + " " + pod.Labels[cyborgFabricPodRankLabel] + "/" + pod.Labels[cyborgReplicaIndexLabel]
		}
		described = append(described, pod.Labels[grovecommon.LabelPodCliquePodIndex]+" "+placement)
	}
	return described
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
