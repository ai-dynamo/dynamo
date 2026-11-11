// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"testing"
	"time"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func TestLPXCyborgPlacementLabel(t *testing.T) {
	for _, tc := range []struct {
		name        string
		annotations map[string]string
		labels      map[string]string
		want        string
	}{
		{name: "a workload that does not select the placement has no label"},
		{
			name:        "the annotation on the conductor role marks the Cyborg Pods",
			annotations: map[string]string{cyborgPlacementAnnotation: lpx.CyborgPlacementPerFabricPod},
			want:        lpx.CyborgPlacementPerFabricPod,
		},
		{
			name:   "an authored label does not mark Pods that this controller does not place",
			labels: map[string]string{lpx.CyborgPlacementLabel: lpx.CyborgPlacementPerFabricPod, "authored": "kept"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
			conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
			conductor.PodTemplate.Annotations, conductor.PodTemplate.Labels = tc.annotations, tc.labels
			r := newLPXTestReconciler(t, registry, child, dgd)
			workloads, plans, err := r.resolveWorkloads(ctx, child, dgd)
			require.NoError(t, err)

			pcs, _, err := r.renderPodCliqueSet(ctx, child, dgd, workloads, plans, nil)
			require.NoError(t, err)

			for _, clique := range pcs.Spec.Template.Cliques {
				want := ""
				if clique.Name == plans["lpx"].CyborgTemplate {
					want = tc.want
					require.Equal(t, tc.labels["authored"], clique.Labels["authored"], "authored labels reach the Cyborg template")
				}
				require.Equal(t, want, clique.Labels[lpx.CyborgPlacementLabel], clique.Name)
			}
		})
	}
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
	require.Equal(t, []string{"a", "a", "b", "b", "c", "c"}, testCyborgFabricPods(listTestCyborgPods(t, ctx, r.Client, clique)))
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
			clique := &grovev1alpha1.PodClique{Name: "pcs-0-lpx-0-cond", Namespace: "default", UID: "clique-uid", ResourceVersion: "1"}
			clique.Spec.Replicas = 1
			pod := newTestCyborgPod(clique, 0)
			pod.Spec.SchedulingGates = tc.gates
			r := &graphReconciler{Client: newLPXTestClient(t, clique, pod)}

			changed, err := r.reconcileCyborgPlacement(ctx, clique, cyborgPlacement{fabricPods: []string{"a"}, replicas: 1}, false)
			require.NoError(t, err)
			require.Equal(t, tc.pinned, changed)

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
	nodes := []client.Object{
		newTestFabricPodNode("gpu-a", "a", "pool", "gpu"),
		newTestFabricPodNode("gpu-b", "b", "pool", "gpu"),
		newTestFabricPodNode("lpu-c", "c", "pool", "lpu"),
		&corev1.Node{Name: "unlabeled", Labels: map[string]string{"pool": "gpu"}},
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
			r := &graphReconciler{Client: newLPXTestClient(t, nodes...), config: newLPXTestReconciler(t, nil, &v1alpha1.LPXGraphDeployment{}, loadTestDGD(t, lpx.PipelineHybrid, "build")).config}
			pcs := &grovev1alpha1.PodCliqueSet{}
			pcs.Spec.Template.Cliques = []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "cond", Spec: grovev1alpha1.PodCliqueSpec{PodSpec: tc.podSpec}}}
			fabricPods, err := r.eligibleFabricPods(t.Context(), pcs, "cond")
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.want, fabricPods)
		})
	}
}

func TestLPXCyborgPlacementRejectsNamespaceRestrictedMode(t *testing.T) {
	ctx := t.Context()

	t.Log("Select per-fabric-pod placement while the operator can read only one namespace")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate.Annotations = map[string]string{cyborgPlacementAnnotation: lpx.CyborgPlacementPerFabricPod}
	r := newLPXTestReconciler(t, registry, child, dgd, newTestFabricPodNode("gpu-a", "a"))
	r.config.Namespace.Restricted = child.Namespace

	t.Log("Report the missing Node access before a PodCliqueSet can create a Pod with the placement label")
	_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.ErrorContains(t, err, "namespace-restricted")
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, r.List(ctx, sets))
	require.Empty(t, sets.Items)
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

// testCyborgFabricPods returns the recorded fabric pod of every Pod, in the given order.
func testCyborgFabricPods(pods []*corev1.Pod) []string {
	fabricPods := make([]string, 0, len(pods))
	for _, pod := range pods {
		fabricPods = append(fabricPods, pod.Labels[cyborgFabricPodLabel])
	}
	return fabricPods
}
