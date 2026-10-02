//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestEngineGroupControllerPollsSettledWorldsWithoutPodUpdates(t *testing.T) {
	tests := []struct {
		name    string
		target  int32
		outcome nvidiacomv1beta1.EngineGroupTransitionOutcome
	}{
		{name: "initial healthy world", target: 1},
		{name: "completed growth", target: 2, outcome: nvidiacomv1beta1.EngineGroupTransitionOutcomeCompleted},
		{name: "blocked serving verification", target: 2, outcome: nvidiacomv1beta1.EngineGroupTransitionOutcomeBlocked},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("create a ready Pod and a controller whose runtime observations change independently of Kubernetes")
			ctx := context.Background()
			backend := newEngineGroupControllerTestBackend(1)
			backend.ambiguousApply = false
			if test.outcome == nvidiacomv1beta1.EngineGroupTransitionOutcomeBlocked {
				backend.verificationFailure = &enginegroup.Failure{
					Classification: enginegroup.FailureClassificationTerminal,
					Reason:         "ServingCheckFailed",
					Message:        "the committed data path cannot make progress",
				}
			}
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "polling", Namespace: "test", UID: "group-uid"},
				Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: test.target},
			}
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "worker-0", Namespace: group.Namespace, UID: "uid-0"},
				Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{
					Type: corev1.PodReady, Status: corev1.ConditionTrue,
				}}},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			require.NoError(t, corev1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(group, pod).WithObjects(group, pod).Build()
			reconciler := &DynamoGraphDeploymentEngineGroupReconciler{
				Client:          kubeClient,
				RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend},
			}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
			podBefore := &corev1.Pod{}
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(pod), podBefore))

			t.Log("reconcile until the initial world or growth transition reaches its settled state")
			settled := false
			for step := 0; step < 64; step++ {
				_, err := reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
				require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
				journal := group.Status.Reconciliation
				if journal != nil && ((test.outcome == "" && journal.Transition == nil) ||
					(journal.Transition != nil && journal.Transition.Outcome == test.outcome)) {
					settled = true
					break
				}
			}
			require.True(t, settled, "journal: %#v", group.Status.Reconciliation)
			result, err := reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Equal(t, 10*time.Second, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			initialStatus := group.Status.DeepCopy()
			membershipCalls := backend.membershipApplyCount()

			t.Log("let a runtime become unavailable without changing Pod readiness or submitting a Kubernetes event")
			backend.mu.Lock()
			backend.capacity.Allocations[0].Available = false
			backend.mu.Unlock()
			result, err = reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Positive(t, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			assert.Equal(t, initialStatus.AvailableReplicas-1, group.Status.AvailableReplicas)
			available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			assert.Equal(t, metav1.ConditionFalse, available.Status)
			assert.Equal(t, initialStatus.Reconciliation.Membership.Desired, group.Status.Reconciliation.Membership.Desired)
			assert.Equal(t, membershipCalls, backend.membershipApplyCount())

			t.Log("restore availability and keep observation scheduled without an allocation event")
			backend.mu.Lock()
			backend.capacity.Allocations[0].Available = true
			backend.mu.Unlock()
			result, err = reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Equal(t, 10*time.Second, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			assert.Equal(t, initialStatus.AvailableReplicas, group.Status.AvailableReplicas)

			if test.outcome != "" {
				t.Log("inject routing drift after progress stops and reassert the last accepted absolute target")
				status := engineGroupStatusFromAPI(group.Status.Reconciliation)
				require.NotNil(t, status.Traffic.Accepted)
				backend.mu.Lock()
				if test.outcome == nvidiacomv1beta1.EngineGroupTransitionOutcomeBlocked {
					backend.traffic.Admitted = cloneEngineGroupControllerTestMemberships(backend.topology.Replicas)
				} else {
					backend.traffic.Admitted = nil
				}
				backend.mu.Unlock()
				result, err = reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
				assert.Equal(t, engineGroupRequeueAfter, result.RequeueAfter)
				backend.mu.Lock()
				repaired := cloneEngineGroupControllerTestTraffic(backend.traffic)
				backend.mu.Unlock()
				assert.Equal(t, status.Traffic.Accepted.Admitted, repaired.Admitted)
				result, err = reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
				assert.Equal(t, 10*time.Second, result.RequeueAfter)
			}

			t.Log("observe a new process in the same slot without adopting its incarnation or admitting it")
			backend.mu.Lock()
			backend.capacity.Allocations[0].Incarnation.Members[0].RuntimeIncarnation = "runtime-restarted"
			backend.mu.Unlock()
			result, err = reconciler.Reconcile(ctx, req)
			if test.outcome != "" {
				require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
			} else {
				require.NoError(t, err)
			}
			assert.Positive(t, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			available = meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			assert.Equal(t, metav1.ConditionFalse, available.Status)
			assert.Equal(t, initialStatus.Reconciliation.Registry, group.Status.Reconciliation.Registry)
			assert.Equal(t, initialStatus.Reconciliation.Membership.Desired, group.Status.Reconciliation.Membership.Desired)
			assert.Equal(t, membershipCalls, backend.membershipApplyCount())

			t.Log("verify every runtime-only change left the Kubernetes Pod and desired size untouched")
			podAfter := &corev1.Pod{}
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(pod), podAfter))
			assert.Equal(t, podBefore, podAfter)
			assert.Equal(t, test.target, group.Spec.Replicas)
		})
	}
}

func TestEngineGroupRuntimeResolutionFailureInvalidatesHealth(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "allocation observation failure", err: errors.New("capacity unavailable")},
		{name: "runtime no longer selected", err: ErrEngineGroupRuntimeUnavailable},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("seed historical health evidence from the previous generation")
			ctx := t.Context()
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "group", Namespace: "test", Generation: 2},
				Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
				Status:     nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus{ObservedGeneration: 1},
			}
			for _, conditionType := range []string{engineGroupConditionAvailable, engineGroupConditionTopologyKnown,
				engineGroupConditionDegraded, engineGroupConditionTargetReached} {
				condition := metav1.Condition{Type: conditionType, Status: metav1.ConditionTrue, Reason: "PreviouslyObserved", ObservedGeneration: 1}
				if conditionType == engineGroupConditionDegraded {
					condition.Status = metav1.ConditionFalse
				}
				meta.SetStatusCondition(&group.Status.Conditions, condition)
			}
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(group).WithObjects(group).Build()
			reconciler := &DynamoGraphDeploymentEngineGroupReconciler{
				Client: kubeClient, RuntimeProvider: failingEngineGroupRuntimeProvider{err: test.err},
			}

			t.Log("fail runtime resolution before any coordinator observation can establish current health")
			_, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)})
			if !errors.Is(test.err, ErrEngineGroupRuntimeUnavailable) {
				require.ErrorIs(t, err, test.err)
			}
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(group), group))

			t.Log("retain historical generation while making every current health claim unknown")
			assert.Equal(t, int64(1), group.Status.ObservedGeneration)
			for _, conditionType := range []string{engineGroupConditionAvailable, engineGroupConditionTopologyKnown,
				engineGroupConditionDegraded, engineGroupConditionTargetReached} {
				condition := meta.FindStatusCondition(group.Status.Conditions, conditionType)
				require.NotNil(t, condition)
				assert.Equal(t, metav1.ConditionUnknown, condition.Status, conditionType)
				assert.Equal(t, group.Generation, condition.ObservedGeneration)
			}
		})
	}
}

type failingEngineGroupRuntimeProvider struct {
	err error
}

func (p failingEngineGroupRuntimeProvider) Resolve(context.Context, *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup) (EngineGroupRuntime, error) {
	return EngineGroupRuntime{}, p.err
}

func TestEngineGroupControllerPeriodicObservationFailsClosed(t *testing.T) {
	authorities := []enginegroup.ObservationAuthority{
		enginegroup.ObservationAuthorityCapacity,
		enginegroup.ObservationAuthorityTraffic,
		enginegroup.ObservationAuthorityMembership,
	}

	for _, authority := range authorities {
		t.Run(string(authority), func(t *testing.T) {
			t.Log("initialize a healthy world whose runtime can become unreachable without a Kubernetes update")
			ctx := context.Background()
			backend := newEngineGroupControllerTestBackend(1)
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "observation-failure", Namespace: "test", UID: "group-uid"},
				Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(group).WithObjects(group).Build()
			reconciler := &DynamoGraphDeploymentEngineGroupReconciler{
				Client:          kubeClient,
				RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend},
			}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
			for step := 0; step < 4; step++ {
				_, err := reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			require.Equal(t, metav1.ConditionTrue, available.Status)
			journal := group.Status.Reconciliation.DeepCopy()

			t.Log("fail one observation authority on the next scheduled reconcile")
			observationFailure := errors.New("runtime endpoint is unreachable")
			backend.mu.Lock()
			backend.observationErrors = map[enginegroup.ObservationAuthority]error{authority: observationFailure}
			backend.mu.Unlock()
			_, err := reconciler.Reconcile(ctx, req)
			require.ErrorIs(t, err, observationFailure)
			var typedFailure *enginegroup.ObservationError
			require.ErrorAs(t, err, &typedFailure)
			assert.Equal(t, authority, typedFailure.Authority)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))

			t.Log("preserve the recovery journal but never republish its old health as a fresh observation")
			for _, conditionType := range []string{
				engineGroupConditionAvailable, engineGroupConditionDegraded,
				engineGroupConditionTopologyKnown, engineGroupConditionTargetReached,
			} {
				condition := meta.FindStatusCondition(group.Status.Conditions, conditionType)
				require.NotNil(t, condition)
				assert.Equal(t, metav1.ConditionUnknown, condition.Status, conditionType)
			}
			assert.Equal(t, journal, group.Status.Reconciliation)
			assert.Equal(t, 0, backend.membershipApplyCount())
			assert.Equal(t, int32(1), group.Spec.Replicas)

			t.Log("recover observation authority and resume periodic healthy observation without an external event")
			backend.mu.Lock()
			delete(backend.observationErrors, authority)
			backend.mu.Unlock()
			result, err := reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Equal(t, 10*time.Second, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			available = meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			assert.Equal(t, metav1.ConditionTrue, available.Status)
		})
	}
}
