//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestGroveRestartPreservesLogicalWorldAndReinitializesAuthority(t *testing.T) {
	const newRuntime = "new-runtime"
	const newClique = "new-clique"

	t.Log("initialize a three-allocation world and protect its exact Pod lifetimes")
	fixture := newGroveRestartTestWorld(t, 3)
	ctx := t.Context()
	controller := &Reconciler{Client: fixture.client, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: fixture.backend}}
	req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(fixture.group)}
	for step := 0; step < 5; step++ {
		_, err := controller.Reconcile(ctx, req)
		require.NoError(t, err)
	}
	require.NoError(t, fixture.client.Get(ctx, req.NamespacedName, fixture.group))
	original := engineGroupControllerTestCheckpoint(t, ctx, fixture.client, fixture.group)
	require.Equal(t, int32(3), original.State.Membership.Observed.CommittedTopology.ReplicaCount())
	spec := fixture.group.Spec.DeepCopy()
	logicalUID := fixture.group.UID

	t.Log("let Grove recreate the clique while old processes are still terminating")
	require.NoError(t, fixture.client.Delete(ctx, fixture.clique))
	for _, pod := range fixture.pods {
		require.NoError(t, fixture.client.Get(ctx, client.ObjectKeyFromObject(pod), pod))
		require.Contains(t, pod.Finalizers, engineGroupProcessFinalizer)
		require.NoError(t, fixture.client.Delete(ctx, pod))
	}
	candidate := fixture.clique.DeepCopy()
	candidate.UID, candidate.ResourceVersion = newClique, ""
	require.NoError(t, fixture.client.Create(ctx, candidate))
	_, err := controller.Reconcile(ctx, req)
	require.ErrorContains(t, err, "not proven stopped")
	require.NoError(t, fixture.client.Get(ctx, req.NamespacedName, fixture.group))
	assert.Equal(t, string(fixture.clique.UID), fixture.group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	assert.Equal(t, metav1.ConditionUnknown, meta.FindStatusCondition(fixture.group.Status.Conditions, engineGroupConditionAvailable).Status)
	assert.Equal(t, original.State, engineGroupControllerTestCheckpoint(t, ctx, fixture.client, fixture.group).State)

	t.Log("report actual kubelet termination and provide a fresh one-allocation engine lifetime")
	for _, pod := range fixture.pods {
		require.NoError(t, fixture.client.Get(ctx, client.ObjectKeyFromObject(pod), pod))
		pod.Status.Phase = corev1.PodFailed
		pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1}}}}
		require.NoError(t, fixture.client.Status().Update(ctx, pod))
	}
	newPod := fixture.pods[0].DeepCopy()
	newPod.Name, newPod.UID, newPod.ResourceVersion = "new-worker-0", "new-pod-uid", ""
	newPod.DeletionTimestamp, newPod.Finalizers = nil, nil
	newPod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(candidate, grove.SchemeGroupVersion.WithKind("PodClique"))}
	newPod.Status = corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}}
	require.NoError(t, fixture.client.Create(ctx, newPod))
	backend := newEngineGroupControllerTestBackend(1)
	backend.capacity.Allocations[0].Incarnation.CapacityRefs = []domain.CapacityRef{{Name: newPod.Name, UID: domain.PodUID(newPod.UID)}}
	backend.capacity.Allocations[0].Incarnation.Members[0].RuntimeIncarnation = newRuntime
	backend.topology.Replicas[0].Members[0].RuntimeIncarnation = newRuntime
	backend.traffic.Admitted[0].Members[0].RuntimeIncarnation = newRuntime
	controller.RuntimeProvider = engineGroupControllerTestRuntimeProvider{backend: backend}
	_, err = controller.Reconcile(ctx, req)
	require.NoError(t, err)
	require.NoError(t, fixture.client.Get(ctx, req.NamespacedName, fixture.group))
	assert.Equal(t, string(candidate.UID), fixture.group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	for _, pod := range fixture.pods {
		assert.True(t, apierrors.IsNotFound(fixture.client.Get(ctx, client.ObjectKeyFromObject(pod), &corev1.Pod{})))
	}

	t.Log("restart the controller after binding changes but before checkpoint replacement")
	controller = &Reconciler{Client: fixture.client, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
	for step := 0; step < 2; step++ {
		_, err := controller.Reconcile(ctx, req)
		require.NoError(t, err)
	}
	require.NoError(t, fixture.client.Get(ctx, req.NamespacedName, fixture.group))
	checkpoint := engineGroupControllerTestCheckpoint(t, ctx, fixture.client, fixture.group)
	assert.Equal(t, candidate.UID, checkpoint.BindingUID)
	assert.Equal(t, logicalUID, fixture.group.UID)
	assert.Equal(t, *spec, fixture.group.Spec)
	assert.Nil(t, checkpoint.State.Transition)
	assert.Nil(t, checkpoint.State.Capacity.Desired)
	assert.Nil(t, checkpoint.State.Traffic.Desired)
	assert.Nil(t, fixture.group.Status.Operation)
	assert.Empty(t, fixture.group.Status.ReleaseAuthorizations)
	assert.Equal(t, int32(1), fixture.group.Status.ActiveNativeMemberCount)
	assert.Equal(t, newRuntime, fixture.group.Status.Topology.Replicas[0].NativeMembers[0].RuntimeIncarnation)
	assert.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(fixture.group.Status.Conditions, engineGroupConditionAvailable).Status)

	t.Log("resume convergence from fresh formation toward the preserved three-allocation target")
	_, err = controller.Reconcile(ctx, req)
	require.NoError(t, err)
	require.NoError(t, fixture.client.Get(ctx, req.NamespacedName, fixture.group))
	checkpoint = engineGroupControllerTestCheckpoint(t, ctx, fixture.client, fixture.group)
	require.NotNil(t, checkpoint.State.Transition)
	assert.Equal(t, int32(3), fixture.group.Status.Operation.TargetReplicas)
	assert.Equal(t, domain.RuntimeIncarnationID(newRuntime), checkpoint.State.Topologies.Snapshots[0].Replicas[0].Members[0].RuntimeIncarnation)
	assert.Equal(t, 0, backend.membershipApplyCount(), "persist a new plan before submitting any effect")
}

func TestGroveRestartRequiresExactStopEvidence(t *testing.T) {
	for _, name := range []string{"API disappearance", "NotReady process", "terminal phase without container stop", "stop checkpoint failure"} {
		t.Run(name, func(t *testing.T) {
			t.Log("protect a running world before permitting membership effects")
			fixture := newGroveRestartTestWorld(t, 1)
			ctx := t.Context()
			kube := &checkpointTestClient{Client: fixture.client}
			controller := &Reconciler{Client: kube, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: fixture.backend}}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(fixture.group)}
			for step := 0; step < 4; step++ {
				_, err := controller.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			require.NoError(t, kube.Delete(ctx, fixture.clique))
			pod := fixture.pods[0]
			require.NoError(t, kube.Get(ctx, client.ObjectKeyFromObject(pod), pod))
			switch name {
			case "API disappearance":
				pod.Finalizers = nil
				require.NoError(t, kube.Update(ctx, pod))
				require.NoError(t, kube.Delete(ctx, pod))
			case "NotReady process":
				pod.Status.Conditions[0].Status = corev1.ConditionFalse
				require.NoError(t, kube.Status().Update(ctx, pod))
			case "terminal phase without container stop":
				pod.Status.Phase = corev1.PodFailed
				require.NoError(t, kube.Status().Update(ctx, pod))
			case "stop checkpoint failure":
				require.NoError(t, kube.Delete(ctx, pod))
				require.NoError(t, kube.Get(ctx, client.ObjectKeyFromObject(pod), pod))
				pod.Status.Phase = corev1.PodFailed
				pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1}}}}
				require.NoError(t, kube.Status().Update(ctx, pod))
				kube.failCheckpoint = true
			}

			t.Log("refuse rebinding or native effects even after a fresh clique appears")
			candidate := fixture.clique.DeepCopy()
			candidate.UID, candidate.ResourceVersion = "new-clique", ""
			require.NoError(t, kube.Create(ctx, candidate))
			_, err := controller.Reconcile(ctx, req)
			require.Error(t, err)
			require.NoError(t, kube.Get(ctx, req.NamespacedName, fixture.group))
			assert.Equal(t, string(fixture.clique.UID), fixture.group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
			assert.Equal(t, 0, fixture.backend.membershipApplyCount())
			if name == "stop checkpoint failure" {
				require.NoError(t, kube.Get(ctx, client.ObjectKeyFromObject(pod), pod))
				assert.Contains(t, pod.Finalizers, engineGroupProcessFinalizer)
			}
		})
	}
}

func TestGroveRestartVerificationAndPublicationAreRestartSafe(t *testing.T) {
	const newRuntime = "verification-runtime"
	const newClique = "verification-clique"

	t.Log("initialize and protect the old world's exact process")
	fixture := newGroveRestartTestWorld(t, 1)
	ctx := t.Context()
	kube := &checkpointTestClient{Client: fixture.client}
	controller := &Reconciler{Client: kube, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: fixture.backend}}
	req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(fixture.group)}
	for step := 0; step < 4; step++ {
		_, err := controller.Reconcile(ctx, req)
		require.NoError(t, err)
	}

	t.Log("stop the old process and recreate native resources with different physical identities")
	pod := fixture.pods[0]
	require.NoError(t, kube.Get(ctx, client.ObjectKeyFromObject(pod), pod))
	pod.Status.Phase = corev1.PodFailed
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1}}}}
	require.NoError(t, kube.Status().Update(ctx, pod))
	require.NoError(t, kube.Delete(ctx, pod))
	require.NoError(t, kube.Delete(ctx, fixture.clique))
	candidate := fixture.clique.DeepCopy()
	candidate.UID, candidate.ResourceVersion = newClique, ""
	require.NoError(t, kube.Create(ctx, candidate))
	newPod := pod.DeepCopy()
	newPod.Name, newPod.UID, newPod.ResourceVersion = "new-worker", "new-pod", ""
	newPod.DeletionTimestamp, newPod.Finalizers = nil, nil
	newPod.OwnerReferences[0].UID = candidate.UID
	newPod.Status = corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}}
	require.NoError(t, kube.Create(ctx, newPod))
	backend := newEngineGroupControllerTestBackend(1)
	backend.capacity.Allocations[0].Incarnation.CapacityRefs = []domain.CapacityRef{{Name: newPod.Name, UID: domain.PodUID(newPod.UID)}}
	backend.capacity.Allocations[0].Incarnation.Members[0].RuntimeIncarnation = newRuntime
	backend.topology.Replicas[0].Members[0].RuntimeIncarnation = newRuntime
	backend.traffic.Admitted[0].Members[0].RuntimeIncarnation = newRuntime
	backend.verificationFailure = &domain.Failure{Classification: domain.FailureClassificationRetryable, Reason: "ServingNotReadyYet"}
	controller.RuntimeProvider = engineGroupControllerTestRuntimeProvider{backend: backend}
	for step := 0; step < 2; step++ {
		_, err := controller.Reconcile(ctx, req)
		require.NoError(t, err)
	}

	t.Log("a fresh but unverified world cannot replace checkpoint authority or publish healthy status")
	_, err := controller.Reconcile(ctx, req)
	require.ErrorContains(t, err, "no matching serving proof")
	require.NoError(t, kube.Get(ctx, req.NamespacedName, fixture.group))
	var checkpoint engineGroupCheckpoint
	store := engineGroupCheckpointStore(controller, fixture.group)
	_, err = store.Load(ctx, &checkpoint)
	require.NoError(t, err)
	assert.Equal(t, fixture.clique.UID, checkpoint.BindingUID)
	assert.Equal(t, metav1.ConditionUnknown, meta.FindStatusCondition(fixture.group.Status.Conditions, engineGroupConditionAvailable).Status)

	t.Log("persist successful fresh formation even if its public projection fails")
	backend.verificationFailure = nil
	kube.failStatus = true
	_, err = controller.Reconcile(ctx, req)
	require.ErrorContains(t, err, "public status write failed")
	_, err = store.Load(ctx, &checkpoint)
	require.NoError(t, err)
	assert.Equal(t, candidate.UID, checkpoint.BindingUID)
	assert.Equal(t, domain.RuntimeIncarnationID(newRuntime), checkpoint.State.Membership.Observed.CommittedTopology.Replicas[0].Members[0].RuntimeIncarnation)

	t.Log("restart from the new checkpoint without replaying old targets or requiring another topology change")
	kube.failStatus = false
	controller = &Reconciler{Client: kube, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
	_, err = controller.Reconcile(ctx, req)
	require.NoError(t, err)
	require.NoError(t, kube.Get(ctx, req.NamespacedName, fixture.group))
	assert.Equal(t, newRuntime, fixture.group.Status.Topology.Replicas[0].NativeMembers[0].RuntimeIncarnation)
	assert.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(fixture.group.Status.Conditions, engineGroupConditionAvailable).Status)
	assert.Nil(t, fixture.group.Status.Operation)
	assert.Equal(t, 0, backend.membershipApplyCount())
}

func TestGroveProcessStopProofSurvivesProductionPodProjection(t *testing.T) {
	for _, test := range []struct {
		name    string
		phase   corev1.PodPhase
		stopped bool
		unknown bool
		want    bool
	}{
		{name: "running is not fenced", phase: corev1.PodRunning},
		{name: "terminal phase alone is not fenced", phase: corev1.PodFailed},
		{name: "kubelet confirms terminal process", phase: corev1.PodFailed, stopped: true, want: true},
		{name: "missing runtime status is not proof", phase: corev1.PodFailed, stopped: true, unknown: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("use the same compact Pod representation as the production informer")
			pod := &corev1.Pod{Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever, Containers: []corev1.Container{{Name: "main"}}}, Status: corev1.PodStatus{Phase: test.phase}}
			if test.stopped {
				pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1}}}}
			}
			if test.unknown {
				pod.Status.ContainerStatuses[0].State.Terminated.Reason = "ContainerStatusUnknown"
			}
			assert.Equal(t, test.want, podProcessesStopped(podcache.Project(pod)))
		})
	}
}

func TestRestartedGroveFormationRequiresServingEvidence(t *testing.T) {
	const newClique = "formation-clique"

	for _, test := range []struct {
		name       string
		terminal   bool
		incomplete bool
		wantError  string
	}{
		{name: "terminal verification survives controller restart", terminal: true, wantError: "failed serving verification"},
		{name: "incomplete formation cannot replace old authority", incomplete: true, wantError: "has not formed its initial serving membership"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("initialize an owned old world before testing fresh formation")
			fixture := newGroveRestartTestWorld(t, 1)
			ctx := t.Context()
			controller := &Reconciler{Client: fixture.client, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: fixture.backend}}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(fixture.group)}
			for step := 0; step < 4; step++ {
				_, err := controller.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			require.NoError(t, fixture.client.Get(ctx, req.NamespacedName, fixture.group))
			store := engineGroupCheckpointStore(controller, fixture.group)
			var original engineGroupCheckpoint
			snapshot, err := store.Load(ctx, &original)
			require.NoError(t, err)

			t.Log("provide new formation that cannot yet establish serving authority")
			fixture.group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = newClique
			require.NoError(t, fixture.client.Update(ctx, fixture.group))
			backend := newEngineGroupControllerTestBackend(1)
			runtime, err := (engineGroupControllerTestRuntimeProvider{backend: backend}).Resolve(ctx, fixture.group)
			require.NoError(t, err)
			if test.terminal {
				backend.verificationFailure = &domain.Failure{Classification: domain.FailureClassificationTerminal, Reason: "ServingWedged"}
			}
			if test.incomplete {
				runtime.Profile.MinSafeServingNativeMembers = 2
			}
			fenceStore := groveWorldFenceStore(fixture.client, fixture.group, newClique)
			var fence groveWorldFence
			fenceSnapshot, err := fenceStore.Load(ctx, &fence)
			require.NoError(t, err)
			_, err = fenceStore.Save(ctx, fenceSnapshot, groveWorldFence{CliqueUID: newClique, PreviousUID: fixture.clique.UID})
			require.NoError(t, err)
			_, err = controller.initializeRestartedEngineGroup(ctx, fixture.group, runtime, store, snapshot)
			require.ErrorContains(t, err, test.wantError)

			t.Log("retain the previous checkpoint and fail closed instead of accepting fresh health")
			var retained engineGroupCheckpoint
			_, err = store.Load(ctx, &retained)
			require.NoError(t, err)
			assert.Equal(t, original, retained)
			require.NoError(t, fixture.client.Get(ctx, req.NamespacedName, fixture.group))
			assert.Equal(t, metav1.ConditionUnknown, meta.FindStatusCondition(fixture.group.Status.Conditions, engineGroupConditionAvailable).Status)
			assert.Equal(t, 0, backend.membershipApplyCount())
			if test.terminal {
				t.Log("keep a definitive failure after restart even if a later probe would succeed")
				backend.verificationFailure = nil
				controller = &Reconciler{Client: fixture.client, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
				_, err = controller.initializeRestartedEngineGroup(ctx, fixture.group, runtime, store, snapshot)
				require.ErrorContains(t, err, "requires recreation after terminal verification failure")
				_, err = store.Load(ctx, &retained)
				require.NoError(t, err)
				assert.Equal(t, original, retained)
			}
		})
	}
}

func TestGroveRestartVerifiesFormationWithoutAnOldCoordinatorCheckpoint(t *testing.T) {
	const newClique = "interrupted-formation-clique"

	t.Log("retain stop proof when Grove interrupts startup before the first coordinator checkpoint")
	fixture := newGroveRestartTestWorld(t, 1)
	ctx := t.Context()
	controller := &Reconciler{Client: fixture.client, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: fixture.backend}}
	runtime, err := controller.RuntimeProvider.Resolve(ctx, fixture.group)
	require.NoError(t, err)
	fixture.group.Status.Profile = runtime.Profile.DeepCopy()
	require.NoError(t, fixture.client.Status().Update(ctx, fixture.group))
	for _, fence := range []groveWorldFence{
		{CliqueUID: fixture.clique.UID, Pods: []grovePodFence{{Ref: domain.CapacityRef{Name: fixture.pods[0].Name, UID: domain.PodUID(fixture.pods[0].UID)}, SlotID: "slot-0", Stopped: true}}},
		{CliqueUID: newClique, PreviousUID: fixture.clique.UID},
	} {
		store := groveWorldFenceStore(fixture.client, fixture.group, fence.CliqueUID)
		var saved groveWorldFence
		snapshot, err := store.Load(ctx, &saved)
		require.NoError(t, err)
		_, err = store.Save(ctx, snapshot, fence)
		require.NoError(t, err)
	}
	fixture.group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = newClique
	require.NoError(t, fixture.client.Update(ctx, fixture.group))

	t.Log("refuse first-checkpoint initialization without a fresh serving proof")
	fixture.backend.verificationFailure = &domain.Failure{Classification: domain.FailureClassificationTerminal, Reason: "ServingWedged"}
	store := engineGroupCheckpointStore(controller, fixture.group)
	var checkpoint engineGroupCheckpoint
	snapshot, err := store.Load(ctx, &checkpoint)
	require.NoError(t, err)
	require.False(t, snapshot.Exists())
	_, err = controller.initializeEngineGroupStatus(ctx, fixture.group, runtime, store, snapshot)
	require.ErrorContains(t, err, "failed serving verification")
	snapshot, err = store.Load(ctx, &checkpoint)
	require.NoError(t, err)
	assert.False(t, snapshot.Exists(), "unverified formation cannot publish effect authority")
	assert.Equal(t, 0, fixture.backend.membershipApplyCount())
}

func TestGroveCapacityEventsWakeTheLogicalWorld(t *testing.T) {
	for _, object := range []client.Object{
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "test"}},
		&grove.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "clique", Namespace: "test"}},
	} {
		t.Run(fmt.Sprintf("%T", object), func(t *testing.T) {
			t.Log("ignore resources outside the Engine Group binding")
			controller := &Reconciler{}
			assert.Empty(t, controller.engineGroupsForCapacity(t.Context(), object))

			t.Log("map native clique and Pod lifecycle changes to the same logical world")
			object.SetLabels(map[string]string{consts.KubeLabelDynamoEngineGroup: "group"})
			assert.Equal(t, []reconcile.Request{{NamespacedName: types.NamespacedName{Namespace: "test", Name: "group"}}},
				controller.engineGroupsForCapacity(t.Context(), object))
		})
	}
}

type groveRestartTestWorld struct {
	client  client.Client
	group   *api.DynamoGraphDeploymentEngineGroup
	clique  *grove.PodClique
	pods    []*corev1.Pod
	backend *engineGroupControllerTestBackend
}

// newGroveRestartTestWorld constructs an owned world; tests drive its lifecycle themselves.
func newGroveRestartTestWorld(t *testing.T, replicas int) groveRestartTestWorld {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, grove.AddToScheme(scheme))
	dgd := &api.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "test", UID: "dgd-uid"}}
	pcs := &grove.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components), Namespace: "test", UID: "pcs-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, api.GroupVersion.WithKind("DynamoGraphDeployment"))}}}
	world := &grove.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{Name: "world", Namespace: "test", UID: "pcsg-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grove.SchemeGroupVersion.WithKind("PodCliqueSet"))}}}
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name: "group", Namespace: "test", UID: "group-uid", Generation: 1,
			Labels:          map[string]string{consts.KubeLabelDynamoEngineGroupRuntime: consts.KubeLabelDynamoEngineGroupSGLang, consts.KubeLabelDynamoEngineGroupWorldIndex: "0"},
			Annotations:     map[string]string{consts.KubeAnnotationDynamoEngineGroupPodClique: "clique", consts.KubeAnnotationDynamoEngineGroupPodCliqueUID: "old-clique"},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, api.GroupVersion.WithKind("DynamoGraphDeployment"))},
		},
		Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: int32(replicas), Policy: &api.EngineGroupScalingPolicy{MinReplicas: ptr.To(int32(1)), MaxReplicas: ptr.To(int32(8))}},
	}
	clique := &grove.PodClique{ObjectMeta: metav1.ObjectMeta{
		Name: "clique", Namespace: "test", UID: "old-clique",
		Labels:          map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name, grovecommon.LabelPodCliqueScalingGroupReplicaIndex: "0"},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(world, grove.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
	}}
	objects := []client.Object{dgd, pcs, world, group, clique}
	pods := make([]*corev1.Pod, replicas)
	for i := range pods {
		pods[i] = &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("worker-%d", i), Namespace: "test", UID: types.UID(fmt.Sprintf("uid-%d", i)),
				Labels:          map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name, grovecommon.LabelPodClique: clique.Name, grovecommon.LabelPodCliquePodIndex: fmt.Sprint(i)},
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(clique, grove.SchemeGroupVersion.WithKind("PodClique"))},
			},
			Spec:   corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever, Containers: []corev1.Container{{Name: "main"}}},
			Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}},
		}
		objects = append(objects, pods[i])
	}
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).WithStatusSubresource(group, &corev1.Pod{}).Build()
	return groveRestartTestWorld{client: kube, group: group, clique: clique, pods: pods, backend: newEngineGroupControllerTestBackend(replicas)}
}
