//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
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

func TestEngineGroupControllerPollsInitialObservationsWithoutBackoff(t *testing.T) {
	for _, authority := range []domain.ObservationAuthority{
		domain.ObservationAuthorityCapacity,
		domain.ObservationAuthorityTraffic,
		domain.ObservationAuthorityMembership,
	} {
		t.Run(string(authority), func(t *testing.T) {
			t.Log("start a world whose runtime is not yet observable")
			ctx := t.Context()
			backend := newEngineGroupControllerTestBackend(1)
			backend.observationErrors = map[domain.ObservationAuthority]error{
				authority: domain.ErrObservationPending,
			}
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				Name: "starting", Namespace: "test", UID: "group-uid",
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			require.NoError(t, corev1.AddToScheme(scheme))
			kube := fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(group).WithObjects(group).Build()
			reconciler := &Reconciler{Client: kube, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}

			t.Log("reach initial observation without returning an error that overrides the retry interval")
			for range 4 {
				_, err := reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			for range 3 {
				result, err := reconciler.Reconcile(ctx, req)
				require.NoError(t, err, "controller-runtime must use RequeueAfter rather than error backoff")
				assert.Equal(t, engineGroupRequeueAfter, result.RequeueAfter)
			}
			require.NoError(t, kube.Get(ctx, req.NamespacedName, group))

			t.Log("keep health unknown and leave membership effects and the recovery checkpoint untouched")
			for _, conditionType := range []string{
				engineGroupConditionAvailable, engineGroupConditionDegraded,
				engineGroupConditionTopologyKnown, engineGroupConditionTargetReached,
			} {
				condition := meta.FindStatusCondition(group.Status.Conditions, conditionType)
				require.NotNil(t, condition)
				assert.Equal(t, metav1.ConditionUnknown, condition.Status, conditionType)
			}
			_, snapshot, err := loadEngineGroupCheckpoint(ctx, engineGroupCheckpointStore(reconciler, group), group)
			require.NoError(t, err)
			assert.False(t, snapshot.Exists())
			assert.Zero(t, backend.membershipApplyCount())

			t.Log("resume from the scheduled retry when the engine becomes observable, with no spec or Pod update")
			delete(backend.observationErrors, authority)
			for range 4 {
				_, err := reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			require.NoError(t, kube.Get(ctx, req.NamespacedName, group))
			available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			assert.Equal(t, metav1.ConditionTrue, available.Status)
			assert.Equal(t, int32(1), group.Status.AvailableReplicas)
			assert.Equal(t, int32(1), group.Spec.Replicas)
			assert.Zero(t, backend.membershipApplyCount())
		})
	}
}

func TestEngineGroupControllerPollsSettledWorldsWithoutPodUpdates(t *testing.T) {
	tests := []struct {
		name    string
		target  int32
		outcome domain.TransitionOutcome
	}{
		{name: "initial healthy world", target: 1},
		{name: "completed growth", target: 2, outcome: domain.TransitionOutcomeCompleted},
		{name: "blocked serving verification", target: 2, outcome: domain.TransitionOutcomeBlocked},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("create a ready Pod and a controller whose runtime observations change independently of Kubernetes")
			ctx := context.Background()
			backend := newEngineGroupControllerTestBackend(1)
			backend.ambiguousApply = false
			if test.outcome == domain.TransitionOutcomeBlocked {
				backend.verificationFailure = &domain.Failure{
					Classification: domain.FailureClassificationTerminal,
					Reason:         "ServingCheckFailed",
					Message:        "the committed data path cannot make progress",
				}
			}
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				Name: "polling", Namespace: "test", UID: "group-uid",
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: test.target},
			}
			pod := &corev1.Pod{
				Name: "worker-0", Namespace: group.Namespace, UID: "uid-0",
				Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{
					Type: corev1.PodReady, Status: corev1.ConditionTrue,
				}}},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			require.NoError(t, corev1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(group, pod).WithObjects(group, pod).Build()
			reconciler := &Reconciler{
				Client:          kubeClient,
				RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend},
			}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
			podBefore := &corev1.Pod{}
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(pod), podBefore))

			t.Log("reconcile until the initial world or growth transition reaches its settled state")
			settled := false
			for range 64 {
				_, err := reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
				require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
				checkpoint, snapshot, loadErr := loadEngineGroupCheckpoint(ctx, engineGroupCheckpointStore(reconciler, group), group)
				require.NoError(t, loadErr)
				journal := &checkpoint.State
				if snapshot.Exists() && ((test.outcome == "" && journal.Transition == nil) ||
					(journal.Transition != nil && journal.Transition.Outcome == test.outcome)) {
					settled = true
					break
				}
			}
			require.True(t, settled, "operation: %#v", group.Status.Operation)
			result, err := reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Equal(t, 10*time.Second, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			initialStatus := group.Status.DeepCopy()
			initialCheckpoint := engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group)
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
			assert.Equal(t, initialCheckpoint.State.Membership.Desired, engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group).State.Membership.Desired)
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
				t.Log("observe lost registration without issuing another positive serving permission")
				backend.mu.Lock()
				backend.traffic.Admitted = nil
				backend.mu.Unlock()
				result, err = reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
				assert.Equal(t, 10*time.Second, result.RequeueAfter)
				require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
				available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
				require.NotNil(t, available)
				assert.Equal(t, metav1.ConditionFalse, available.Status)

				t.Log("the serving runtime restores registration independently")
				backend.mu.Lock()
				backend.traffic.Admitted = cloneEngineGroupControllerTestMemberships(backend.topology.Replicas)
				backend.mu.Unlock()
				result, err = reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
				assert.Equal(t, 10*time.Second, result.RequeueAfter)
			}

			t.Log("observe a new process in the same slot without adopting its incarnation or admitting it")
			backend.mu.Lock()
			backend.capacity.Allocations[0].Incarnation.Members[0].RuntimeIncarnation = "runtime-restarted"
			backend.mu.Unlock()
			result, err = reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Positive(t, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			available = meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			assert.Equal(t, metav1.ConditionUnknown, available.Status)
			recovery := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionRuntimeReady)
			require.NotNil(t, recovery)
			assert.Equal(t, "RecoveryRequired", recovery.Reason)
			assert.Equal(t, initialCheckpoint.State.Registry, engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group).State.Registry)
			assert.Equal(t, initialCheckpoint.State.Membership.Desired, engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group).State.Membership.Desired)
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
		{name: "runtime no longer selected", err: ErrRuntimeUnavailable},
		{name: "confirmed runtime recovery fault", err: fmt.Errorf("native instance changed: %w", domain.ErrRecoveryRequired)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("seed historical health evidence from the previous generation")
			ctx := t.Context()
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				Name: "group", Namespace: "test", Generation: 2,
				Spec:   nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
				Status: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus{ObservedGeneration: 1},
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
			require.NoError(t, corev1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(group).WithObjects(group).Build()
			reconciler := &Reconciler{
				Client: kubeClient, RuntimeProvider: failingEngineGroupRuntimeProvider{err: test.err},
			}

			t.Log("fail runtime resolution before any coordinator observation can establish current health")
			result, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)})
			if errors.Is(test.err, domain.ErrRecoveryRequired) || errors.Is(test.err, ErrRuntimeUnavailable) {
				require.NoError(t, err)
				assert.Equal(t, engineGroupObservationInterval, result.RequeueAfter)
			} else {
				require.ErrorIs(t, err, test.err)
				assert.Zero(t, result.RequeueAfter)
			}
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(group), group))
			if errors.Is(test.err, domain.ErrRecoveryRequired) {
				condition := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionRuntimeReady)
				require.NotNil(t, condition)
				assert.Equal(t, "RecoveryRequired", condition.Reason)
			}

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

func (p failingEngineGroupRuntimeProvider) Resolve(context.Context, *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup) (Runtime, error) {
	return Runtime{}, p.err
}

func TestEngineGroupControllerPeriodicObservationFailsClosed(t *testing.T) {
	authorities := []domain.ObservationAuthority{
		domain.ObservationAuthorityCapacity,
		domain.ObservationAuthorityTraffic,
		domain.ObservationAuthorityMembership,
	}

	for _, authority := range authorities {
		t.Run(string(authority), func(t *testing.T) {
			t.Log("initialize a healthy world whose runtime can become unreachable without a Kubernetes update")
			ctx := context.Background()
			backend := newEngineGroupControllerTestBackend(1)
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				Name: "observation-failure", Namespace: "test", UID: "group-uid",
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			require.NoError(t, corev1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(group).WithObjects(group).Build()
			reconciler := &Reconciler{
				Client:          kubeClient,
				RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend},
			}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
			for range 4 {
				_, err := reconciler.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			require.Equal(t, metav1.ConditionTrue, available.Status)
			journal := engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group).State

			t.Log("fail one observation authority on the next scheduled reconcile")
			observationFailure := errors.New("runtime endpoint is unreachable")
			backend.mu.Lock()
			backend.observationErrors = map[domain.ObservationAuthority]error{authority: observationFailure}
			backend.mu.Unlock()
			_, err := reconciler.Reconcile(ctx, req)
			require.ErrorIs(t, err, observationFailure)
			var typedFailure *domain.ObservationError
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
			assert.Equal(t, journal, engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group).State)
			assert.Equal(t, 0, backend.membershipApplyCount())
			assert.Equal(t, int32(1), group.Spec.Replicas)

			t.Log("a confirmed recovery fault has an actionable diagnosis and bounded observation, not transient backoff")
			backend.mu.Lock()
			backend.observationErrors[authority] = fmt.Errorf("native authority: %w", domain.ErrRecoveryRequired)
			backend.mu.Unlock()
			result, err := reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Equal(t, engineGroupObservationInterval, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			condition := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionRuntimeReady)
			require.NotNil(t, condition)
			assert.Equal(t, "RecoveryRequired", condition.Reason)
			assert.Equal(t, journal, engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group).State)

			t.Log("recover observation authority and resume periodic healthy observation without an external event")
			backend.mu.Lock()
			delete(backend.observationErrors, authority)
			backend.mu.Unlock()
			result, err = reconciler.Reconcile(ctx, req)
			require.NoError(t, err)
			assert.Equal(t, 10*time.Second, result.RequeueAfter)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			available = meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
			require.NotNil(t, available)
			assert.Equal(t, metav1.ConditionTrue, available.Status)
		})
	}
}
