// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"reflect"
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
	revision  uint64
}

func (r *upgradeComponentReconciles) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// Starting another call invalidates the earlier success, including while a gate is held.
	r.mu.Lock()
	delete(r.completed, req.NamespacedName)
	r.revision++
	r.mu.Unlock()

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
	r.mu.Lock()
	defer r.mu.Unlock()
	r.revision++
	if err == nil && result.IsZero() {
		if r.completed == nil {
			r.completed = map[types.NamespacedName]bool{}
		}
		r.completed[req.NamespacedName] = true
	}
	return result, err
}

//nolint:unparam // Tagged cluster tests pass the actual component count; unit tests use one.
func (r *upgradeComponentReconciles) graphReady(objects []unstructured.Unstructured, components int) error {
	// Preserve the graph's independent inventory, ownership and generation checks.
	if err := upgradeComponentGraphReady(objects, components); err != nil {
		return err
	}

	// The graph projection can lag the DCD status after a conflict and recovery.
	var graphStatus map[string]any
	for _, object := range objects {
		if object.GetKind() == "DynamoGraphDeployment" {
			var err error
			graphStatus, _, err = unstructured.NestedMap(object.Object, "status", "components")
			if err != nil {
				return err
			}
		}
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

			// Compare only projected facts, without requiring serving Pods or replica readiness.
			name, _, err := unstructured.NestedString(object.Object, "spec", "name")
			if err != nil {
				return err
			}
			component, found, err := unstructured.NestedMap(object.Object, "status", "component")
			if err != nil {
				return err
			}
			projected, ok := graphStatus[name].(map[string]any)
			if !found || !ok {
				return fmt.Errorf("DCD %s has no graph component projection", key)
			}
			for _, field := range []string{"runtimeNamespace", "servedModelName", "runtimeComponentName", "gpuPowerLimitWatts", "gpusPerEngine", "gpusPerReplica"} {
				if !reflect.DeepEqual(component[field], projected[field]) {
					return fmt.Errorf("DCD %s graph projection %s has not converged", key, field)
				}
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
		"spec":     map[string]any{"name": "worker"},
		"status":   map[string]any{"observedGeneration": int64(1), "component": map[string]any{"servedModelName": "model"}},
	}}
	workload := unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "apps/v1", "kind": "Deployment",
		"metadata": map[string]any{"annotations": map[string]any{"nvidia.com/last-applied-generation": "1"}, "ownerReferences": []any{map[string]any{"uid": "dcd", "controller": true}}},
	}}
	dgd := unstructured.Unstructured{Object: map[string]any{
		"kind":   "DynamoGraphDeployment",
		"status": map[string]any{"components": map[string]any{"worker": map[string]any{"servedModelName": "model"}}},
	}}
	graph := []unstructured.Unstructured{object, workload, dgd}
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
		{name: "error after success", err: fmt.Errorf("status update conflict")},
		{name: "recovered"},
		{name: "requeue after success", result: ctrl.Result{RequeueAfter: 1}},
		{name: "recovered again"},
	} {
		t.Run(result.name, func(t *testing.T) {
			recorder.next = reconcile.Func(func(_ context.Context, req ctrl.Request) (ctrl.Result, error) {
				if req.Name == "worker" {
					require.Error(t, recorder.graphReady(graph, 1), "an in-flight call must invalidate earlier success")
				}
				return result.result, result.err
			})
			readyBefore := recorder.graphReady(graph, 1) == nil
			req := ctrl.Request{Namespace: "history", Name: "other"}
			for range 2 {
				_, err := recorder.Reconcile(t.Context(), req)
				require.ErrorIs(t, err, result.err)
			}
			require.Equal(t, readyBefore, recorder.graphReady(graph, 1) == nil, "another child must not change this child's completion")
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

	t.Log("A recovered DCD must wait for its DGD projection to catch up")
	require.NoError(t, unstructured.SetNestedField(dgd.Object, "", "status", "components", "worker", "servedModelName"))
	require.ErrorContains(t, recorder.graphReady(graph, 1), "servedModelName has not converged")
}
