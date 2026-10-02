/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package grovecapacity

import (
	"context"
	"errors"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGrowthScalesMemberCliqueAndObservesGroveSlots(t *testing.T) {
	t.Log("start from a Ready primary in Grove slot zero")
	fixture := newCapacityFixture(t)
	ctx := t.Context()
	initial, err := fixture.adapter.Observe(ctx, "group")
	require.NoError(t, err)
	require.Len(t, initial.Allocations, 1)
	target := growthTarget(initial.Allocations[0].Incarnation)
	originalTemplate := fixture.clique.Spec.PodSpec.DeepCopy()

	t.Log("accept the target durably and scale only the generated member clique")
	result, err := fixture.adapter.Apply(ctx, "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	assert.Equal(t, 1, fixture.scale.updates)
	assert.Equal(t, int64(1), fixture.scale.acceptedRevisionAtWrite)
	require.NoError(t, fixture.client.Get(ctx, fixture.adapter.Clique, fixture.clique))
	assert.Equal(t, int32(2), fixture.clique.Spec.Replicas)
	assert.Equal(t, *originalTemplate, fixture.clique.Spec.PodSpec)
	assert.Equal(t, "PodCliqueScalingGroup", metav1.GetControllerOf(fixture.clique).Kind)
	pods := &corev1.PodList{}
	require.NoError(t, fixture.client.List(ctx, pods))
	assert.Len(t, pods.Items, 1, "only Grove, not this adapter, may create capacity Pods")

	t.Log("wait for Grove's asynchronous joiner without claiming count convergence")
	observation, err := fixture.adapter.Observe(ctx, "group")
	require.NoError(t, err)
	assert.Equal(t, target.ControlRevision, observation.AppliedRevision)
	assert.Len(t, observation.Allocations, 1)
	joiner := grovePod(fixture.clique, "joiner", "joiner-uid", "1", false)
	require.NoError(t, fixture.client.Create(ctx, joiner))
	observation, err = fixture.adapter.Observe(ctx, "group")
	require.NoError(t, err)
	require.Len(t, observation.Allocations, 2)
	assert.Equal(t, enginegroup.ReplicaID("replica-1"), observation.Allocations[1].Incarnation.ReplicaID)
	assert.Equal(t, enginegroup.CapacitySlotID("slot-1"), observation.Allocations[1].Incarnation.SlotID)
	assert.Equal(t, enginegroup.PodUID("joiner-uid"), observation.Allocations[1].Incarnation.CapacityRefs[0].UID)
	assert.False(t, observation.Allocations[1].Available)

	t.Log("report availability only after the exact joining Pod becomes Ready")
	joiner.Status.Conditions[0].Status = corev1.ConditionTrue
	require.NoError(t, fixture.client.Status().Update(ctx, joiner))
	observation, err = fixture.adapter.Observe(ctx, "group")
	require.NoError(t, err)
	assert.True(t, observation.Allocations[1].Available)

	t.Log("pin the committed joining incarnation and reject a new UID in the same slot")
	pinned := target
	pinned.ControlRevision = 2
	pinned.Replicas = append([]enginegroup.CapacityReplicaTarget(nil), target.Replicas...)
	pinned.Replicas[1].Bootstrap = nil
	pinned.Replicas[1].Incarnation = &observation.Allocations[1].Incarnation
	result, err = fixture.adapter.Apply(ctx, "group", pinned)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, fixture.client.Delete(ctx, joiner))
	replacement := grovePod(fixture.clique, "replacement", "replacement-uid", "1", true)
	require.NoError(t, fixture.client.Create(ctx, replacement))
	result, err = fixture.adapter.Apply(ctx, "group", pinned)
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "MissingExactIncarnation", result.Rejection.Reason)
	assert.Equal(t, 1, fixture.scale.updates)
}

func TestGrowthReplayRepairsDriftAndAmbiguousScaleWrite(t *testing.T) {
	t.Log("persist acceptance before an intentionally inconclusive scale write")
	fixture := newCapacityFixture(t)
	observation, err := fixture.adapter.Observe(t.Context(), "group")
	require.NoError(t, err)
	target := growthTarget(observation.Allocations[0].Incarnation)
	fixture.scale.writeError = errors.New("scale response lost")
	_, err = fixture.adapter.Apply(t.Context(), "group", target)
	require.ErrorContains(t, err, "scale response lost")
	observation, err = fixture.adapter.Observe(t.Context(), "group")
	require.NoError(t, err)
	assert.Equal(t, target.ControlRevision, observation.AppliedRevision)

	t.Log("restart the adapter and replay the identical accepted target")
	restarted := *fixture.adapter
	fixture.scale.writeError = nil
	result, err := restarted.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	assert.Equal(t, 2, fixture.scale.updates)
	result, err = restarted.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	assert.Equal(t, 2, fixture.scale.updates, "already converged scaling is a no-op")

	t.Log("repair a lowered native count without resetting acceptance or changing the payload")
	require.NoError(t, fixture.client.Get(t.Context(), fixture.adapter.Clique, fixture.clique))
	fixture.clique.Spec.Replicas = 1
	require.NoError(t, fixture.client.Update(t.Context(), fixture.clique))
	result, err = restarted.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	assert.Equal(t, 3, fixture.scale.updates)

	t.Log("preserve acceptance when Grove resets child annotations from its template")
	require.NoError(t, fixture.client.Get(t.Context(), fixture.adapter.Clique, fixture.clique))
	fixture.clique.Annotations = map[string]string{"template-annotation": "value"}
	require.NoError(t, fixture.client.Update(t.Context(), fixture.clique))
	observation, err = restarted.Observe(t.Context(), "group")
	require.NoError(t, err)
	assert.Equal(t, target.ControlRevision, observation.AppliedRevision)

	t.Log("refuse conflicting and stale targets after a restart")
	conflicting := target
	conflicting.TransitionID = "other-operation"
	result, err = restarted.Apply(t.Context(), "group", conflicting)
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "ConflictingCapacityRevision", result.Rejection.Reason)
	newer := target
	newer.ControlRevision = 2
	result, err = restarted.Apply(t.Context(), "group", newer)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	result, err = restarted.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "StaleCapacityRevision", result.Rejection.Reason)
}

func TestGrowthRejectsUnsafeCapacityTargets(t *testing.T) {
	// Every target changes one invariant on a new, independently owned fixture.
	cases := []struct {
		name string
		mode string
		want string
	}{
		{name: "UID-bound release is not implemented", mode: "release", want: "ReleaseUnsupported"},
		{name: "sparse slot growth is not representable by a count", mode: "sparse", want: "UnsupportedSlotMapping"},
		{name: "recovery bootstrap is not append growth", mode: "recover", want: "UnsupportedBootstrap"},
		{name: "an exact missing UID must not be recreated", mode: "missing", want: "MissingExactIncarnation"},
		{name: "growth cannot lower a clique count", mode: "shrink", want: "ReleaseUnsupported"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("build an absolute target with one unsafe capacity change")
			fixture := newCapacityFixture(t)
			observation, err := fixture.adapter.Observe(t.Context(), "group")
			require.NoError(t, err)
			target := growthTarget(observation.Allocations[0].Incarnation)
			switch tc.mode {
			case "release":
				target.ReleaseFences = []enginegroup.ReleaseFence{{TransitionID: "release"}}
			case "sparse":
				target.Replicas[1].SlotID = "slot-3"
			case "recover":
				target.Replicas[1].Bootstrap.Mode = enginegroup.BootstrapModeRestoreFixedSlot
			case "missing":
				require.NoError(t, fixture.client.Delete(t.Context(), fixture.primary))
			case "shrink":
				fixture.clique.Spec.Replicas = 3
				require.NoError(t, fixture.client.Update(t.Context(), fixture.clique))
			}

			t.Log("reject before accepting the target or writing the scale subresource")
			result, err := fixture.adapter.Apply(t.Context(), "group", target)
			require.NoError(t, err)
			require.NotNil(t, result.Rejection)
			assert.Equal(t, tc.want, result.Rejection.Reason)
			assert.Zero(t, fixture.scale.updates)
			observation, err = fixture.adapter.Observe(t.Context(), "group")
			require.NoError(t, err)
			assert.Zero(t, observation.AppliedRevision)
		})
	}
}

func TestObservationRejectsInvalidNativeCapacityBindings(t *testing.T) {
	// Binding failures cannot be hidden by a selector that omits the unexpected Pod.
	cases := []string{"duplicate-slot", "invalid-index", "unlabelled-group", "wrong-owner", "recreated-clique"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			t.Log("introduce one invalid identity into an otherwise healthy Grove clique")
			fixture := newCapacityFixture(t)
			if name == "recreated-clique" {
				fixture.adapter.CliqueUID = "old-clique-uid"
			} else {
				pod := grovePod(fixture.clique, "unexpected", "unexpected-uid", "1", true)
				switch name {
				case "duplicate-slot":
					pod.Labels[grovecommon.LabelPodCliquePodIndex] = "0"
				case "invalid-index":
					pod.Labels[grovecommon.LabelPodCliquePodIndex] = "01"
				case "unlabelled-group":
					delete(pod.Labels, consts.KubeLabelDynamoEngineGroup)
				case "wrong-owner":
					pod.OwnerReferences[0].UID = "other-clique-uid"
				}
				require.NoError(t, fixture.client.Create(t.Context(), pod))
			}

			t.Log("fail closed instead of reporting a partial or reassigned allocation set")
			_, err := fixture.adapter.Observe(t.Context(), "group")
			require.Error(t, err)
		})
	}
}

type capacityFixture struct {
	client  client.Client
	adapter *Adapter
	clique  *grovev1alpha1.PodClique
	primary *corev1.Pod
	scale   *testScaleAPI
}

func newCapacityFixture(t *testing.T) capacityFixture {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))
	clique := &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test", Name: "world-0-members", UID: "clique-uid",
			Labels: map[string]string{consts.KubeLabelDynamoEngineGroup: "group"},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup",
				Name: "world", UID: "world-uid", Controller: ptr.To(true),
			}},
		},
		Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1, MinAvailable: ptr.To(int32(1)),
			PodSpec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine-image"}}}},
	}
	primary := grovePod(clique, "primary", "primary-uid", "0", true)
	scale := &testScaleAPI{}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(clique, primary).
		WithStatusSubresource(&corev1.Pod{}).
		WithInterceptorFuncs(interceptor.Funcs{SubResourceGet: scale.Get, SubResourceUpdate: scale.Update}).Build()
	journal := kubejournal.NewStore(kubeClient, "test", "group", "group-uid", "grove-capacity")
	scale.journal = journal
	adapter := &Adapter{Client: kubeClient, Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID,
		GroupName: "group", Journal: journal}
	return capacityFixture{client: kubeClient, adapter: adapter, clique: clique, primary: primary, scale: scale}
}

func grovePod(clique *grovev1alpha1.PodClique, name string, uid types.UID, index string, ready bool) *corev1.Pod {
	status := corev1.ConditionFalse
	if ready {
		status = corev1.ConditionTrue
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: clique.Namespace, Name: name, UID: uid,
			Labels: map[string]string{
				grovecommon.LabelPodClique: clique.Name, grovecommon.LabelPodCliquePodIndex: index,
				consts.KubeLabelDynamoEngineGroup: "group", consts.KubeLabelDynamoScaleRepresentative: "true",
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "PodClique",
				Name: clique.Name, UID: clique.UID, Controller: ptr.To(true),
			}},
		},
		Spec:   *clique.Spec.PodSpec.DeepCopy(),
		Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: status}}},
	}
}

func growthTarget(primary enginegroup.ReplicaIncarnation) enginegroup.CapacityTarget {
	return enginegroup.CapacityTarget{
		ControlRevision: 1, TransitionID: "grow", ProfileFingerprint: "profile",
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		Replicas: []enginegroup.CapacityReplicaTarget{
			{ReplicaID: "replica-0", SlotID: "slot-0", Incarnation: &primary},
			{ReplicaID: "replica-1", SlotID: "slot-1", Bootstrap: &enginegroup.CapacityBootstrap{
				Mode: enginegroup.BootstrapModeJoin, BaseTopologyGeneration: 1, NativeMembers: []enginegroup.NativeMemberID{"dp-1"},
			}},
		},
	}
}

// testScaleAPI supplies the native Scale behavior absent from controller-runtime's fake client.
type testScaleAPI struct {
	journal                 kubejournal.Store
	updates                 int
	acceptedRevisionAtWrite int64
	writeError              error
}

func (s *testScaleAPI) Get(ctx context.Context, reader client.Client, name string, object client.Object,
	subresource client.Object, _ ...client.SubResourceGetOption) error {
	if name != "scale" {
		return errors.New("unexpected subresource")
	}
	clique := &grovev1alpha1.PodClique{}
	if err := reader.Get(ctx, client.ObjectKeyFromObject(object), clique); err != nil {
		return err
	}
	scale := subresource.(*autoscalingv1.Scale)
	scale.ObjectMeta = clique.ObjectMeta
	scale.Spec.Replicas = clique.Spec.Replicas
	return nil
}

func (s *testScaleAPI) Update(ctx context.Context, writer client.Client, name string, object client.Object,
	options ...client.SubResourceUpdateOption) error {
	if name != "scale" {
		return writer.SubResource(name).Update(ctx, object, options...)
	}
	state := journalState{}
	if _, err := s.journal.Load(ctx, &state); err != nil {
		return err
	}
	s.acceptedRevisionAtWrite = state.AppliedRevision
	clique := &grovev1alpha1.PodClique{}
	if err := writer.Get(ctx, client.ObjectKeyFromObject(object), clique); err != nil {
		return err
	}
	s.updates++
	if s.writeError != nil {
		return s.writeError
	}
	parsed := &client.SubResourceUpdateOptions{}
	parsed.ApplyOptions(options)
	scale := parsed.SubResourceBody.(*autoscalingv1.Scale)
	if scale.ResourceVersion != clique.ResourceVersion || scale.UID != clique.UID {
		return apierrors.NewConflict(grovev1alpha1.Resource("podcliques"), clique.Name, errors.New("stale scale write"))
	}
	clique.Spec.Replicas = scale.Spec.Replicas
	return writer.Update(ctx, clique)
}
