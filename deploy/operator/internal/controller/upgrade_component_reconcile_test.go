// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// upgradeComponentReconciles records candidate calls, never persisted predecessor status.
// Each manager owns one recorder. Gates are immutable after manager startup.
type upgradeComponentReconciles struct {
	next      reconcile.Reconciler
	gates     map[types.NamespacedName]chan struct{}
	mu        sync.Mutex
	completed map[types.NamespacedName]bool
}

func (r *upgradeComponentReconciles) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// The regression test can hold individual DCDs before their candidate renderer runs.
	if gate, blocked := r.gates[req.NamespacedName]; blocked {
		select {
		case <-gate:
		case <-ctx.Done():
			return ctrl.Result{}, ctx.Err()
		}
	}

	// A failed call must not count as candidate completion.
	result, err := r.next.Reconcile(ctx, req)
	if err == nil && result.IsZero() {
		r.mu.Lock()
		if r.completed == nil {
			r.completed = map[types.NamespacedName]bool{}
		}
		r.completed[req.NamespacedName] = true
		r.mu.Unlock()
	}
	return result, err
}

//nolint:unparam // Tagged cluster tests pass the actual component count; unit tests use one.
func (r *upgradeComponentReconciles) graphReady(objects []unstructured.Unstructured, components int) error {
	// Preserve the graph's independent inventory, ownership and generation checks.
	if err := upgradeComponentGraphReady(objects, components); err != nil {
		return err
	}

	// Every live child must have completed through this manager's candidate controller.
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, object := range objects {
		if object.GetKind() == "DynamoComponentDeployment" {
			key := types.NamespacedName{Namespace: object.GetNamespace(), Name: object.GetName()}
			if !r.completed[key] {
				return fmt.Errorf("DCD %s has no completed candidate reconciliation", key)
			}
		}
	}
	return nil
}

func TestUpgradeComponentReconcileRecorder(t *testing.T) {
	t.Log("Errors, requeues and repeated calls for another child cannot satisfy completion")
	recorder := &upgradeComponentReconciles{}
	object := unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoComponentDeployment",
		"metadata": map[string]any{"name": "worker", "namespace": "history", "uid": "dcd", "generation": int64(1)},
		"status":   map[string]any{"observedGeneration": int64(1)},
	}}
	workload := unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "apps/v1", "kind": "Deployment",
		"metadata": map[string]any{"annotations": map[string]any{"nvidia.com/last-applied-generation": "1"}, "ownerReferences": []any{map[string]any{"uid": "dcd", "controller": true}}},
	}}
	graph := []unstructured.Unstructured{object, workload}
	require.NoError(t, upgradeComponentGraphReady(graph, 1))
	require.ErrorContains(t, recorder.graphReady(graph, 1), "no completed candidate")

	for _, result := range []struct {
		name   string
		result ctrl.Result
		err    error
	}{
		{name: "error", err: fmt.Errorf("candidate failed")},
		{name: "requeue", result: ctrl.Result{RequeueAfter: 1}},
		{name: "success"},
	} {
		t.Run(result.name, func(t *testing.T) {
			recorder.next = reconcile.Func(func(context.Context, ctrl.Request) (ctrl.Result, error) { return result.result, result.err })
			req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "history", Name: "other"}}
			for range 2 {
				_, err := recorder.Reconcile(t.Context(), req)
				require.ErrorIs(t, err, result.err)
			}
			require.Error(t, recorder.graphReady(graph, 1))
			req.Name = "worker"
			_, err := recorder.Reconcile(t.Context(), req)
			require.ErrorIs(t, err, result.err)
			if result.err == nil && result.result.IsZero() {
				require.NoError(t, recorder.graphReady(graph, 1))
			} else {
				require.Error(t, recorder.graphReady(graph, 1))
			}
		})
	}
}
