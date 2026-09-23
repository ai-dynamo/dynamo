/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"slices"
	"strings"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

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
	deployment, dgd, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	reconciler, desired := newPreparedLPXTestReconciler(t, registry, ctx, deployment, dgd)
	require.Empty(t, desired.plan.ConductorTemplate)
	require.Empty(t, desired.plan.ConductorClique)
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
			requests, missing, changed := resolvePipelineRequests(deployment, map[string]*lpxv1alpha1.LPUPipelineRequest{current.Name: current}, desired.workload, desired.plan)
			require.Equal(t, test.changed, changed)
			require.Empty(t, missing)
			if !changed {
				require.Equal(t, current, requests[current.Name])
			} else {
				require.Nil(t, requests)
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
		{name: "mismatch discards earlier missing requests", replicas: 2, published: []int{5}, intentChanged: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve two workload replicas with draft0, draft1 and target models")
			deployment, dgd, registry := newLPXSpecDecodeTestDGD(t)
			desired := resolveLPXTestWorkload(t, registry, t.Context(), deployment, dgd)
			desired.plan.Replicas = 2
			_, rendered, changed := resolvePipelineRequests(deployment, nil, desired.workload, desired.plan)
			require.False(t, changed)
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
			requests, missing, changed := resolvePipelineRequests(deployment, observed, desired.workload, desired.plan)
			require.Equal(t, tc.intentChanged, changed)
			if changed {
				require.Nil(t, requests)
				require.Nil(t, missing)
				return
			}
			require.Len(t, requests, int(tc.replicas)*3)
			require.Len(t, missing, len(tc.wantMissing))
			for index, renderedIndex := range tc.wantMissing {
				require.Equal(t, rendered[renderedIndex], missing[index])
				require.Same(t, requests[missing[index].Name], missing[index])
			}
			for _, request := range observed {
				if request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex < int64(tc.replicas) {
					require.Same(t, request, requests[request.Name])
				}
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

// publishSelectedLPXForTest seeds requests through the production publication helper.
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
	requests := make([]*lpxv1alpha1.LPUPipelineRequest, len(desired.requests))
	for index := range desired.requests {
		requests[index] = desired.requests[index].DeepCopy()
	}
	require.NoError(t, reconciler.reconcilePipelineRequests(ctx, deployment, pcs, requests))
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
