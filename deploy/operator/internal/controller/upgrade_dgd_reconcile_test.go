// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// upgradeDGDReconciles records successful candidate calls within one test manager.
// Requeues count as completed calls, matching the previous non-error metric barrier.
type upgradeDGDReconciles struct {
	next      reconcile.Reconciler
	mu        sync.Mutex
	successes map[types.NamespacedName]bool
}

func (r *upgradeDGDReconciles) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	result, err := r.next.Reconcile(ctx, req)
	if err == nil {
		r.mu.Lock()
		if r.successes == nil {
			r.successes = map[types.NamespacedName]bool{}
		}
		r.successes[req.NamespacedName] = true
		r.mu.Unlock()
	}
	return result, err
}

func (r *upgradeDGDReconciles) completed(key types.NamespacedName) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.successes[key]
}

func TestUpgradeDGDReconcileRecorder(t *testing.T) {
	t.Log("Only this manager's successful call for this DGD can release its barrier")
	key := types.NamespacedName{Namespace: "scenario", Name: "workers"}
	for _, tc := range []struct {
		name   string
		result ctrl.Result
		err    error
	}{
		{name: "success"},
		{name: "requeue", result: ctrl.Result{RequeueAfter: 1}},
		{name: "error", err: fmt.Errorf("candidate failed")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := &upgradeDGDReconciles{}
			otherManager := &upgradeDGDReconciles{}
			recorder.next = reconcile.Func(func(context.Context, ctrl.Request) (ctrl.Result, error) {
				require.False(t, recorder.completed(key), "an in-flight call has not completed")
				return tc.result, tc.err
			})
			result, err := recorder.Reconcile(t.Context(), ctrl.Request{NamespacedName: key})
			require.Equal(t, tc.result, result)
			require.ErrorIs(t, err, tc.err)
			require.Equal(t, tc.err == nil, recorder.completed(key))
			require.False(t, recorder.completed(types.NamespacedName{Namespace: "other", Name: key.Name}))
			require.False(t, recorder.completed(types.NamespacedName{Namespace: key.Namespace, Name: "other"}))
			require.False(t, otherManager.completed(key), "another manager cannot release this barrier")
		})
	}
}
