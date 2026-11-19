//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"sync"
	"testing"
	"time"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestEngineGroupSlowSubmissionDoesNotStarveAnotherWorld(t *testing.T) {
	t.Log("start the real controller manager with independent slow and fast runtime authorities")
	env := sharedEnv.ForTest(t)
	blocked := &blockedMembership{
		entered: make(chan struct{}), release: make(chan struct{}),
	}
	slow := newEngineGroupControllerTestBackend(1)
	slow.ambiguousApply = false
	provider := isolationRuntimeProvider{slow: slow, fast: newEngineGroupControllerTestBackend(1), blocked: blocked}
	env.StartManager(func(mgr ctrl.Manager) error {
		return (&Reconciler{Client: mgr.GetClient(), RuntimeProvider: provider}).SetupWithManager(mgr, false)
	})
	// Unblock before the manager cleanup waits for outstanding reconciles.
	t.Cleanup(func() { close(blocked.release) })
	group := &api.DynamoGraphDeploymentEngineGroup{
		Name: "slow", Namespace: env.Namespace(), Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	t.Log("hold one world's native submission after the exact target is durable")
	require.NoError(t, env.Client().Create(t.Context(), group))
	select {
	case <-blocked.entered:
	case <-time.After(20 * time.Second):
		t.Fatal("slow world's submission was not reached")
	}
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(group), group))
	checkpoint := engineGroupControllerTestCheckpoint(t, t.Context(), env.Client(), group)
	require.NotNil(t, checkpoint.State.Membership.Desired)

	t.Log("observe an unrelated world while the first submission is still blocked")
	fast := &api.DynamoGraphDeploymentEngineGroup{
		Name: "fast", Namespace: env.Namespace(), Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	require.NoError(t, env.Client().Create(t.Context(), fast))
	require.Eventually(t, func() bool {
		current := &api.DynamoGraphDeploymentEngineGroup{}
		if err := env.Client().Get(t.Context(), client.ObjectKeyFromObject(fast), current); err != nil {
			return false
		}
		return meta.IsStatusConditionTrue(current.Status.Conditions, engineGroupConditionAvailable)
	}, 10*time.Second, 20*time.Millisecond)
}

type isolationRuntimeProvider struct {
	slow, fast *engineGroupControllerTestBackend
	blocked    *blockedMembership
}

func (p isolationRuntimeProvider) Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (Runtime, error) {
	backend := p.fast
	if group.Name == "slow" {
		backend = p.slow
	}
	resolved, err := (engineGroupControllerTestRuntimeProvider{backend: backend}).Resolve(ctx, group)
	if group.Name == "slow" {
		resolved.Membership = &gatedMembership{MembershipAdapter: resolved.Membership, gate: p.blocked}
	}
	return resolved, err
}

type blockedMembership struct {
	entered, release chan struct{}
	once             sync.Once
}

type gatedMembership struct {
	domain.MembershipAdapter
	gate *blockedMembership
}

func (m *gatedMembership) Apply(ctx context.Context, group domain.GroupID, target domain.MembershipTarget) error {
	m.gate.once.Do(func() { close(m.gate.entered) })
	select {
	case <-m.gate.release:
		return m.MembershipAdapter.Apply(ctx, group, target)
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestEngineGroupReconcileHasAnAggregateDeadline(t *testing.T) {
	t.Log("create an Engine Group with a caller deadline shorter than the reconcile budget")
	env := sharedEnv.ForTest(t)
	provider := deadlineRuntimeProvider{deadline: make(chan time.Duration, 1)}
	group := &api.DynamoGraphDeploymentEngineGroup{
		Name: "budget", Namespace: env.Namespace(), Spec: api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	require.NoError(t, env.Client().Create(t.Context(), group))
	reconciler := &Reconciler{Client: env.Client(), RuntimeProvider: provider}
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	// First pass may only install the finalizer; reach runtime resolution without
	// replacing the original caller deadline with a fresh per-subsystem budget.
	for range 4 {
		_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(group)})
		if err != nil {
			require.ErrorIs(t, err, context.DeadlineExceeded)
			break
		}
	}
	select {
	case remaining := <-provider.deadline:
		require.Positive(t, remaining)
		require.LessOrEqual(t, remaining, 100*time.Millisecond)
	case <-time.After(time.Second):
		t.Fatal("runtime did not receive the caller's bounded context")
	}
}

type deadlineRuntimeProvider struct{ deadline chan time.Duration }

func (p deadlineRuntimeProvider) Resolve(ctx context.Context, _ *api.DynamoGraphDeploymentEngineGroup) (Runtime, error) {
	deadline, ok := ctx.Deadline()
	if ok {
		p.deadline <- time.Until(deadline)
	}
	<-ctx.Done()
	return Runtime{}, ctx.Err()
}
