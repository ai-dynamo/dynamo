/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestRemoveReleasedAgentPods(t *testing.T) {
	for _, tc := range []struct {
		name, change                                  string
		reason                                        lpxv1alpha1.ReleaseReason
		remove, wantError, deleteError, missingClique bool
	}{
		{name: "dependency changed", remove: true},
		{name: "request changed", reason: lpxv1alpha1.ReleaseReasonRequestChanged, remove: true},
		{name: "binding failed", reason: lpxv1alpha1.ReleaseReasonBindingFailed, remove: true},
		{name: "unbound target", change: "target", remove: true},
		{name: "old-only predecessor", change: "old-only", remove: true},
		{name: "request deleted", reason: lpxv1alpha1.ReleaseReasonRequestDeleted},
		{name: "unknown reason", reason: "unknown"},
		{name: "no status", change: "scheduler-status"},
		{name: "no commitment", change: "committed"},
		{name: "no node-local execution", change: "nodeLocal"},
		{name: "no release journal", change: "release"},
		{name: "bound request", change: "bound"},
		{name: "deleting request", change: "deleting"},
		{name: "cross-namespace reference", change: "namespace", wantError: true},
		{name: "empty journal UID", change: "uid", wantError: true},
		{name: "same-name successor", change: "successor"},
		{name: "foreign graph in same namespace", change: "foreign-group", wantError: true},
		{name: "replaced PodClique", change: "stale-clique", wantError: true},
		{name: "replaced scaling group", change: "stale-group", wantError: true},
		{name: "ownerless Pod", change: "pod-owner", wantError: true},
		{name: "ownerless PodClique", change: "clique-owner", wantError: true},
		{name: "Pod owner is not a controller", change: "controller", wantError: true},
		{name: "Pod owner has wrong kind", change: "kind", wantError: true},
		{name: "PodClique owner has foreign API", change: "api", wantError: true},
		{name: "terminating PodClique", change: "terminating", remove: true},
		{name: "PodClique cache miss retries", missingClique: true, wantError: true},
		{name: "delete failure retries", deleteError: true, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe one released Agent Pod and its ownership beneath a trusted scaling group")
			pcsgs, pclq, pod := newLPXTestAgentHierarchy()
			request := newLPXTestReleaseRequest(pod)
			execution := &request.Status.Committed.Execution
			if tc.reason != "" {
				execution.Release.Reason = tc.reason
			}
			switch tc.change {
			case "target":
				row := &execution.NodeLocal.PartitionSelections[0].SelectedRows[0]
				row.Target, row.Current = &lpxv1alpha1.BindingTarget{PodRef: row.Current.PodRef}, nil
			case "old-only":
				execution.NodeLocal.PlanReplacement = &lpxv1alpha1.NodeLocalPlanReplacement{RealizeMembers: []lpxv1alpha1.RealizeMember{{
					Predecessor: lpxv1alpha1.RealizeMemberPredecessor{OldOnly: &lpxv1alpha1.OldOnlyMember{Current: *execution.NodeLocal.PartitionSelections[0].SelectedRows[0].Current}},
				}}}
				execution.NodeLocal.PartitionSelections = nil
			case "scheduler-status":
				request.Status = nil
			case "committed":
				request.Status.Committed = nil
			case "nodeLocal":
				execution.NodeLocal = nil
			case "release":
				execution.Release = nil
			case "bound":
				request.Status.Phase = lpxv1alpha1.RequestPhaseBound
			case "deleting":
				request.DeletionTimestamp = ptr.To(metav1.Now())
			case "namespace":
				execution.NodeLocal.PartitionSelections[0].SelectedRows[0].Current.PodRef.Namespace = "other"
			case "uid":
				execution.NodeLocal.PartitionSelections[0].SelectedRows[0].Current.PodRef.UID = ""
			case "successor":
				pod.UID = "successor-uid"
			case "foreign-group":
				pclq.OwnerReferences[0].Name = "other-graph-engines"
			case "stale-clique":
				pclq.UID = "replacement-clique-uid"
			case "stale-group":
				pclq.OwnerReferences[0].UID = "previous-group-uid"
			case "pod-owner":
				pod.OwnerReferences = nil
			case "clique-owner":
				pclq.OwnerReferences = nil
			case "controller":
				pod.OwnerReferences[0].Controller = ptr.To(false)
			case "kind":
				pod.OwnerReferences[0].Kind = "PodCliqueSet"
			case "api":
				pclq.OwnerReferences[0].APIVersion = "other.example.com/v1alpha1"
			case "terminating":
				pclq.DeletionTimestamp, pclq.Finalizers = ptr.To(metav1.Now()), []string{"example.com/cleanup"}
			}
			objects := []client.Object{pod}
			if !tc.missingClique {
				objects = append(objects, pclq)
			}
			base := newLPXTestClient(t, objects...)
			r := &graphReconciler{Client: interceptor.NewClient(base, interceptor.Funcs{
				Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
					if tc.deleteError {
						return errors.New("delete temporarily unavailable")
					}
					return delegated.Delete(ctx, object, opts...)
				},
			})}
			requests := map[string]*lpxv1alpha1.LPUPipelineRequest{request.Name: request}

			t.Log("Remove only an eligible, exact, owned Pod and preserve it when observation fails")
			err := r.removeReleasedAgentPods(t.Context(), pcsgs, requests)
			if tc.wantError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			err = base.Get(t.Context(), client.ObjectKeyFromObject(pod), &corev1.Pod{})
			if tc.remove {
				require.True(t, apierrors.IsNotFound(err))
			} else {
				require.NoError(t, err)
			}

			if tc.deleteError || tc.missingClique {
				t.Log("Complete cleanup after the transient failure clears")
				r.Client = base
				if tc.missingClique {
					require.NoError(t, base.Create(t.Context(), pclq))
				}
				require.NoError(t, r.removeReleasedAgentPods(t.Context(), pcsgs, requests))
				require.True(t, apierrors.IsNotFound(base.Get(t.Context(), client.ObjectKeyFromObject(pod), &corev1.Pod{})))
			}
		})
	}
}

func TestRemoveReleasedAgentPodsContinuesAfterMissingPod(t *testing.T) {
	t.Log("Keep a disappeared current Pod before a live target in the release journal")
	pcsgs, pclq, pod := newLPXTestAgentHierarchy()
	request := newLPXTestReleaseRequest(pod)
	row := &request.Status.Committed.Execution.NodeLocal.PartitionSelections[0].SelectedRows[0]
	row.Target = &lpxv1alpha1.BindingTarget{PodRef: row.Current.PodRef}
	row.Current.PodRef.Name, row.Current.PodRef.UID = "already-gone", "already-gone-uid"
	r := &graphReconciler{Client: newLPXTestClient(t, pclq, pod)}

	t.Log("Request a retry for the missing observation while removing the remaining owned target")
	err := r.removeReleasedAgentPods(t.Context(), pcsgs, map[string]*lpxv1alpha1.LPUPipelineRequest{request.Name: request})
	require.Error(t, err)
	require.True(t, apierrors.IsNotFound(r.Get(t.Context(), client.ObjectKeyFromObject(pod), &corev1.Pod{})))
}

func TestRemoveReleasedAgentPodsConcurrentOwnershipChange(t *testing.T) {
	t.Log("Observe an owned Pod, then change its owner immediately before deletion")
	pcsgs, pclq, pod := newLPXTestAgentHierarchy()
	request := newLPXTestReleaseRequest(pod)
	requests := map[string]*lpxv1alpha1.LPUPipelineRequest{request.Name: request}
	base := newLPXTestClient(t, pclq, pod)
	r := &graphReconciler{Client: interceptor.NewClient(base, interceptor.Funcs{
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			live := &corev1.Pod{}
			require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(object), live))
			live.OwnerReferences[0].UID = "foreign-clique-uid"
			require.NoError(t, delegated.Update(ctx, live))
			return delegated.Delete(ctx, object, opts...)
		},
	})}

	t.Log("The resource-version precondition refuses deletion and returns the conflict for retry")
	err := r.removeReleasedAgentPods(t.Context(), pcsgs, requests)
	require.True(t, apierrors.IsConflict(err), "expected delete conflict, got %v", err)
	require.NoError(t, base.Get(t.Context(), client.ObjectKeyFromObject(pod), &corev1.Pod{}))
}

func newLPXTestAgentHierarchy() (map[string]*grovev1alpha1.PodCliqueScalingGroup, *grovev1alpha1.PodClique, *corev1.Pod) {
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "engines", UID: "engines-uid"}}
	pclq := &grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{
		Namespace: pcsg.Namespace, Name: "old-agent-clique", UID: "old-agent-clique-uid",
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcsg, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
	}}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace: pcsg.Namespace, Name: "agent", UID: "agent-uid",
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pclq, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))},
	}}
	return map[string]*grovev1alpha1.PodCliqueScalingGroup{pcsg.Name: pcsg}, pclq, pod
}

func newLPXTestReleaseRequest(pod *corev1.Pod) *lpxv1alpha1.LPUPipelineRequest {
	committed := committedLPXTestPod(pod)
	committed.Execution.Release = &lpxv1alpha1.ReleaseJournal{Reason: lpxv1alpha1.ReleaseReasonDependencyChanged, RequestedAtGeneration: 1}
	return &lpxv1alpha1.LPUPipelineRequest{
		ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: "request", Generation: 1},
		Status:     &lpxv1alpha1.LPUPipelineRequestStatus{Phase: lpxv1alpha1.RequestPhaseReleasing, Committed: committed},
	}
}

func TestGetPipelineRequests(t *testing.T) {
	for _, tc := range []struct {
		name      string
		namespace string
		uid       types.UID
		want      []string
	}{
		{name: "owned requests", namespace: "default", uid: "pcs-uid", want: []string{"first", "last"}},
		{name: "different PCS with the same name", namespace: "default", uid: "other-uid", want: []string{"other-owner"}},
		{name: "different namespace", namespace: "other", uid: "pcs-uid", want: []string{"other-namespace"}},
		{name: "no owned requests", namespace: "default", uid: "missing-uid"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Seed only pipeline requests, with separate owners and namespaces")
			var objects []client.Object
			for _, request := range []struct {
				name, namespace string
				uid             types.UID
			}{
				{"last", "default", "pcs-uid"},
				{"first", "default", "pcs-uid"},
				{"other-owner", "default", "other-uid"},
				{"other-namespace", "other", "pcs-uid"},
			} {
				objects = append(objects, &lpxv1alpha1.LPUPipelineRequest{ObjectMeta: metav1.ObjectMeta{
					Name: request.name, Namespace: request.namespace,
					OwnerReferences: []metav1.OwnerReference{{
						APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueSet",
						Name: "pcs", UID: request.uid, Controller: ptr.To(true),
					}},
				}})
			}
			r := &graphReconciler{Client: newLPXTestClient(t, objects...)}
			pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "pcs", Namespace: tc.namespace, UID: tc.uid}}

			t.Log("Return only this PCS's requests")
			requests, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			var names []string
			for _, request := range requests {
				names = append(names, request.Name)
			}
			require.ElementsMatch(t, tc.want, names)
		})
	}
}

func TestImplicitV2LPXConductorlessGroveIdentityPublishesRequest(t *testing.T) {
	t.Log("Publish the implicit hybrid runtime")
	ctx := t.Context()
	deployment, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, dgd)
	require.Empty(t, desired.plan.ConductorTemplate)
	require.NotEmpty(t, desired.plan.CyborgClique)

	objects := lpxMaterializedObjects(t, reconciler, deployment, dgd, desired)
	for _, object := range objects {
		require.NotEmpty(t, object.GetName())
	}
	pcsg := getResource[*grovev1alpha1.PodCliqueScalingGroup](t, objects, desired.plan.LPXScalingGroup)
	require.NotContains(t, pcsg.Spec.CliqueNames, "")
	cyborg := getResource[*grovev1alpha1.PodClique](t, objects, desired.plan.CyborgClique)
	require.NotContains(t, cyborg.Spec.StartsAfter, "")
	require.Equal(t, v1alpha1.LPXSchedulerName, cyborg.Spec.PodSpec.SchedulerName)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	condition := publishSelectedLPXForTest(t, ctx, reconciler, deployment, desired)
	require.NotNil(t, condition)
	require.Equal(t, v1alpha1.LPXReadyReasonPending, condition.Reason)
	request := getTestPipelineRequest(t, ctx, reconciler.Client, deployment.Namespace, desired.requests[0].Name)
	pcs := findLPXTestPodCliqueSet(t, objects)
	require.True(t, metav1.IsControlledBy(request, pcs))
	require.True(t, *metav1.GetControllerOf(request).BlockOwnerDeletion)
	require.Empty(t, request.Finalizers)
	require.NotNil(t, request.Spec.CyborgPodCliqueRef)
	require.Equal(t, desired.plan.CyborgClique, request.Spec.CyborgPodCliqueRef.Name)
}

func TestPipelineRequestIdentityDigest(t *testing.T) {
	t.Log("Every identity field distinguishes requests")
	base := pipelineRequestIdentityDigest("ns", "dgd", "uid-a", "", "default", 0)
	firstGroup := pipelineRequestIdentityDigest("ns", "dgd", "uid-a", "first", "default", 0)
	secondGroup := pipelineRequestIdentityDigest("ns", "dgd", "uid-a", "second", "default", 0)
	require.NotEqual(t, base, firstGroup)
	require.NotEqual(t, firstGroup, secondGroup)
	for _, test := range []struct {
		name       string
		namespace  string
		deployment string
		uid        types.UID
		model      string
		replica    int32
		same       bool
	}{
		{"unchanged", "ns", "dgd", "uid-a", "default", 0, true},
		{"namespace", "other-ns", "dgd", "uid-a", "default", 0, false},
		{"deployment", "ns", "other-dgd", "uid-a", "default", 0, false},
		{"UID", "ns", "dgd", "uid-b", "default", 0, false},
		{"model", "ns", "dgd", "uid-a", "other-model", 0, false},
		{"replica", "ns", "dgd", "uid-a", "default", 1, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			digest := pipelineRequestIdentityDigest(test.namespace, test.deployment, test.uid, "", test.model, test.replica)
			require.NotEmpty(t, digest)
			require.Equal(t, test.same, base == digest)
		})
	}
}

func TestPipelineRequestName(t *testing.T) {
	t.Log("Keep request names readable and within the DNS label limit")
	digest := pipelineRequestIdentityDigest("ns", "dgd", "uid", "", "default", 0)
	for _, test := range []struct{ name, prefix string }{
		{"dgd", "dgd"},
		{"a-very-long-but-readable-dynamo-graph-deployment-name", "a-very-long-but-readabl"},
		{"graph.name", "graph-name"},
	} {
		t.Run(test.name, func(t *testing.T) {
			actual := pipelineRequestName(test.name, digest)
			require.LessOrEqual(t, len(actual), 63)
			require.Equal(t, "lpx-"+test.prefix+"-"+strings.TrimPrefix(digest, "sha256:")[:32], actual)
		})
	}
}

func TestResolvePipelineRequestsPreservesImmutableIntent(t *testing.T) {
	t.Log("Immutable intent compares desired annotations and spec, not scheduler-owned metadata")
	deployment, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	desired := resolveLPXTestWorkload(t, registry, t.Context(), deployment, dgd)
	for _, test := range []struct {
		name       string
		annotation string
		value      string
		changed    bool
	}{
		{"scheduler annotation", "scheduler.example/status", "observed", false},
		{"workload changed", lpx.WorkloadDigestAnnotation, "another-build", true},
		{"compiler snapshot changed", lpxv1alpha1.CompilerSnapshotDigestAnnotation, "another-snapshot", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			current := desired.requests[0].DeepCopy()
			current.Annotations[test.annotation] = test.value
			require.Equal(t, !test.changed, pipelineRequestMatches(current, &desired.requests[0]))
			requests, missing := resolvePipelineRequests(deployment, map[string]*lpxv1alpha1.LPUPipelineRequest{current.Name: current}, desired.workload, desired.plan)
			if !test.changed {
				require.Empty(t, missing)
				require.Equal(t, current, requests[current.Name])
			} else {
				require.Equal(t, &desired.requests[0], requests[current.Name])
				require.Equal(t, []*lpxv1alpha1.LPUPipelineRequest{requests[current.Name]}, missing)
			}
		})
	}
}

func TestNodeLocalSpecDecodePublishesOneRequestAndAgentCliquePerModelProjection(t *testing.T) {
	ctx := t.Context()
	deployment, dgd, registry := newLPXSpecDecodeTestDGD(t)

	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, dgd)
	require.Len(t, desired.requests, 3)
	require.Len(t, desired.plan.Agents, 3)
	objects := lpxMaterializedObjects(t, reconciler, deployment, dgd, desired)
	createLPXTestObjects(t, ctx, reconciler.Client, objects...)

	condition := publishSelectedLPXForTest(t, ctx, reconciler, deployment, desired)
	require.NotNil(t, condition)
	require.Equal(t, v1alpha1.LPXReadyReasonPending, condition.Reason)
	requests, err := reconciler.getPipelineRequests(ctx, findLPXTestPodCliqueSet(t, objects))
	require.NoError(t, err)
	require.Len(t, requests, 3)

	requestByModel := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(requests))
	for _, request := range requests {
		model := request.Annotations[pipelineRequestModelAnnotation]
		requestByModel[model] = request
		require.Equal(t, string(deployment.UID), request.Labels[deploymentUIDLabel])
		require.Equal(t, deployment.Name, request.Annotations[lpx.DeploymentNameAnnotation])
		require.NotContains(t, request.Annotations, "lpx.nvidia.com/deployment-uid")
		require.NotContains(t, request.Annotations, "scheduling.lpu.nvidia.com/dgd-uid")
		require.Equal(t, model, request.Spec.NodeLocal.Model)
		require.Equal(t, desired.plan.LPXScalingGroup, request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name)
	}
	for _, projection := range desired.requests {
		request, found := requestByModel[projection.Annotations[pipelineRequestModelAnnotation]]
		require.True(t, found)
		require.Equal(t, projection.Annotations[lpx.WorkloadDigestAnnotation], request.Annotations[lpx.WorkloadDigestAnnotation])
		require.Equal(t, projection.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation], request.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation])
	}

	for index, expected := range desired.plan.Agents {
		projection := &desired.requests[index]
		clique := getResource[*grovev1alpha1.PodClique](t, objects, expected.CliqueName)
		require.Equal(t, int32(expected.Replicas), clique.Spec.Replicas)
		require.Equal(t, ptr.To(int32(expected.Replicas)), clique.Spec.MinAvailable)
		require.Equal(t, projection.Annotations[lpx.WorkloadDigestAnnotation], clique.Annotations[lpx.WorkloadDigestAnnotation])
		require.Equal(t, projection.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation], clique.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation])
		require.Equal(t, projection.Annotations[pipelineRequestModelAnnotation], clique.Annotations[lpxv1alpha1.PodModelAnnotation])
		require.NotContains(t, clique.Annotations, lpxv1alpha1.PodPartitionIDAnnotation)
		require.NotContains(t, clique.Annotations, lpxv1alpha1.PodRankInPartitionAnnotation)
	}
}

func TestResolvePipelineRequestsCollectsMissingInOrder(t *testing.T) {
	for _, tc := range []struct {
		name          string
		replicas      int32
		published     []int
		wantMissing   []int
		intentChanged bool
	}{
		{name: "all missing", replicas: 2, wantMissing: []int{0, 1, 2, 3, 4, 5}},
		{name: "partial publication", replicas: 2, published: []int{4, 0, 2}, wantMissing: []int{1, 3, 5}},
		{name: "all published", replicas: 2, published: []int{5, 4, 3, 2, 1, 0}},
		{name: "scaled down", replicas: 1, published: []int{5, 4, 3, 2, 1, 0}},
		{name: "zero replicas", published: []int{5, 4, 3, 2, 1, 0}},
		{name: "mismatch retains ordered pending requests", replicas: 2, published: []int{5}, wantMissing: []int{0, 1, 2, 3, 4, 5}, intentChanged: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve two workload replicas with draft0, draft1 and target models")
			deployment, dgd, registry := newLPXSpecDecodeTestDGD(t)
			desired := resolveLPXTestWorkload(t, registry, t.Context(), deployment, dgd)
			desired.plan.Replicas = 2
			_, rendered := resolvePipelineRequests(deployment, nil, desired.workload, desired.plan)
			require.Len(t, rendered, 6)
			for index, request := range rendered {
				require.Equal(t, int64(index/3), request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex)
				require.Equal(t, []string{"draft0", "draft1", "target"}[index%3], request.Annotations[pipelineRequestModelAnnotation])
			}

			t.Log("Keep observed receipts and collect only absent desired names without sorting")
			observed := make(map[string]*lpxv1alpha1.LPUPipelineRequest)
			for _, index := range tc.published {
				request := rendered[index].DeepCopy()
				request.UID = types.UID(request.Name)
				request.Status = &lpxv1alpha1.LPUPipelineRequestStatus{Phase: lpxv1alpha1.RequestPhaseBound}
				if tc.intentChanged {
					request.Annotations[lpxv1alpha1.CompilerSnapshotDigestAnnotation] = "another-snapshot"
				}
				observed[request.Name] = request
			}
			desired.plan.Replicas = tc.replicas
			requests, missing := resolvePipelineRequests(deployment, observed, desired.workload, desired.plan)
			require.Len(t, requests, int(tc.replicas)*3)
			require.Len(t, missing, len(tc.wantMissing))
			for index, renderedIndex := range tc.wantMissing {
				require.Equal(t, rendered[renderedIndex], missing[index])
				require.Same(t, requests[missing[index].Name], missing[index])
			}
			for _, request := range observed {
				if !tc.intentChanged && request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex < int64(tc.replicas) {
					require.Same(t, request, requests[request.Name])
				}
			}
		})
	}
}

func TestResolveNamedPipelineRequestsUsesResolvedGroupIdentity(t *testing.T) {
	for _, test := range []struct {
		name  string
		count int
	}{
		{name: "sole workload", count: 1},
		{name: "multiple workloads", count: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve complete workloads through the public resolver")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			if test.count == 2 {
				second := dgd.Spec.Components[0].DeepCopy()
				second.ComponentName = "independent"
				dgd.Spec.Components = append(dgd.Spec.Components, *second)
			}
			r := newLPXTestReconciler(t, registry, child, dgd)
			workloads, err := lpx.ResolveWorkloads(t.Context(), r.Client, dgd, dynamo.PCSNameForLPX(child), registry)
			require.NoError(t, err)
			require.Len(t, workloads, test.count)

			t.Log("Publish each workload independently using the group identity fixed during resolution")
			names := make(map[string]bool)
			for _, workload := range workloads {
				group := ""
				if test.count == 2 {
					group = workload.ServingComponentName()
				}
				require.Equal(t, group, workload.GroupName())
				_, missing := resolveNamedPipelineRequests(child, nil, workload, 1)
				require.Len(t, missing, 1)
				request := missing[0]
				digest := pipelineRequestIdentityDigest(child.Namespace, child.Name, child.UID, group, "default", 0)
				require.Equal(t, pipelineRequestName(child.Name, digest), request.Name)
				require.Equal(t, workload.ScalingGroup(), request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name)
				require.False(t, names[request.Name])
				names[request.Name] = true
			}
		})
	}
}

func TestPipelineRequestsPendingDeletion(t *testing.T) {
	for _, tc := range []struct {
		name        string
		desired     []string
		terminating []string
		want        []string
	}{
		{"all current", []string{"first", "second"}, nil, nil},
		{"removed request", []string{"first"}, nil, []string{"second"}},
		{"terminating removed request", []string{"first"}, []string{"second"}, []string{"second"}},
		{"terminating current name", []string{"first", "second"}, []string{"first"}, []string{"first"}},
		{"both pending", []string{"first"}, []string{"first"}, []string{"first", "second"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Block publication until removed and already-terminating request names disappear")
			requests := map[string]*lpxv1alpha1.LPUPipelineRequest{
				"first":  {ObjectMeta: metav1.ObjectMeta{Name: "first"}},
				"second": {ObjectMeta: metav1.ObjectMeta{Name: "second"}},
			}
			desired := make(map[string]*lpxv1alpha1.LPUPipelineRequest)
			for _, request := range requests {
				if slices.Contains(tc.desired, request.Name) {
					desired[request.Name] = request
				}
				if slices.Contains(tc.terminating, request.Name) {
					request.DeletionTimestamp = ptr.To(metav1.Now())
				}
			}

			var names []string
			for _, request := range pipelineRequestsPendingDeletion(requests, desired) {
				names = append(names, request.Name)
			}
			require.ElementsMatch(t, tc.want, names)
		})
	}
}

func getTestPipelineRequest(t *testing.T, ctx context.Context, kubeClient client.Reader, namespace, name string) *lpxv1alpha1.LPUPipelineRequest {
	t.Helper()
	request := &lpxv1alpha1.LPUPipelineRequest{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, request))
	return request
}

// publishSelectedLPXForTest seeds PCS-owned requests for controller scenarios.
// It does not simulate reconciliation, readiness or deletion decisions.
func publishSelectedLPXForTest(
	t *testing.T,
	ctx context.Context,
	reconciler *graphReconciler,
	deployment *v1alpha1.LPXGraphDeployment,
	desired *lpxTestWorkload,
) *metav1.Condition {
	t.Helper()
	pcs := observedLPXTestPodCliqueSet(t, ctx, reconciler, deployment, desired)
	require.NotNil(t, pcs)
	for index := range desired.requests {
		request := desired.requests[index].DeepCopy()
		request.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}
		require.NoError(t, reconciler.Create(ctx, request))
	}
	setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for seeded LPX requests")
	return meta.FindStatusCondition(deployment.Status.Conditions, v1alpha1.LPXReadyCondition)
}

func requirePipelineRequestNotFound(
	t *testing.T,
	ctx context.Context,
	kubeClient client.Reader,
	namespace string,
	name string,
) {
	t.Helper()
	request := &lpxv1alpha1.LPUPipelineRequest{}
	err := kubeClient.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, request)
	require.True(t, apierrors.IsNotFound(err), "expected LPX request %s/%s to be absent, got %v", namespace, name, err)
}

func TestResolvePipelineRequestsRequestsOnlyRemotePartitions(t *testing.T) {
	for _, tc := range []struct {
		name            string
		localPartitions *v1beta1.LPXLocalPartitions
		wantPartitions  [][]int64
	}{
		{name: "all-local selection requests no LPU placement", localPartitions: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll}, wantPartitions: [][]int64{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve a hybrid workload with the selected local partitions")
			deployment, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
			dgd.Spec.Components[0].LPX.Experimental = &v1beta1.LPXExperimentalSpec{LocalPartitions: tc.localPartitions}
			desired := resolveLPXTestWorkload(t, registry, t.Context(), deployment, dgd)

			t.Log("Publish one request per remote workload replica, listing only remote compiler partitions")
			_, missing := resolvePipelineRequests(deployment, nil, desired.workload, desired.plan)
			partitions := make([][]int64, 0, len(missing))
			for _, request := range missing {
				compilerIDs := make([]int64, 0, len(request.Spec.Partitions))
				for _, partition := range request.Spec.Partitions {
					compilerIDs = append(compilerIDs, partition.CompilerPartitionID)
				}
				partitions = append(partitions, compilerIDs)
			}
			require.Equal(t, tc.wantPartitions, partitions)
		})
	}
}
