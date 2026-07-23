/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package paused

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestReconcilerPausesExactTrue(t *testing.T) {
	ctx := t.Context()
	object := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name:        "paused",
		Namespace:   "default",
		Annotations: map[string]string{Annotation: Value},
		Finalizers:  []string{"test-finalizer"},
	}}
	delegateCalled := false
	wrapped := Wrap(
		pausedTestClient(t, object),
		&corev1.ConfigMap{},
		reconcile.Func(func(context.Context, reconcile.Request) (reconcile.Result, error) {
			delegateCalled = true
			return reconcile.Result{}, nil
		}),
	)

	t.Log("Reconcile an object paused with the exact accepted value")
	if _, err := wrapped.Reconcile(ctx, reconcile.Request{NamespacedName: types.NamespacedName{
		Name: object.Name, Namespace: object.Namespace,
	}}); err != nil {
		t.Fatalf("reconcile paused object: %v", err)
	}

	t.Log("Verify the delegate did not run, leaving finalization untouched")
	if delegateCalled {
		t.Fatal("paused reconciler called its delegate")
	}
}

func TestReconcilerDelegatesOtherValues(t *testing.T) {
	values := []string{"", "false", "TRUE", "1"}
	for _, value := range values {
		object := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
			Name:        "not-paused",
			Namespace:   "default",
			Annotations: map[string]string{Annotation: value},
		}}
		delegateCalled := false
		wrapped := Wrap(
			pausedTestClient(t, object),
			&corev1.ConfigMap{},
			reconcile.Func(func(context.Context, reconcile.Request) (reconcile.Result, error) {
				delegateCalled = true
				return reconcile.Result{}, nil
			}),
		)

		t.Logf("Reconcile pause annotation value %q", value)
		if _, err := wrapped.Reconcile(t.Context(), reconcile.Request{NamespacedName: types.NamespacedName{
			Name: object.Name, Namespace: object.Namespace,
		}}); err != nil {
			t.Fatalf("reconcile pause value %q: %v", value, err)
		}
		if !delegateCalled {
			t.Fatalf("pause value %q did not call delegate", value)
		}
	}
}

func TestAnnotationChangedPredicate(t *testing.T) {
	pauseChanged := AnnotationChangedPredicate()
	oldObject := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Annotations: map[string]string{Annotation: Value, "other": "old"},
	}}
	unpausedObject := oldObject.DeepCopy()
	unpausedObject.Annotations[Annotation] = "false"
	otherChanged := oldObject.DeepCopy()
	otherChanged.Annotations["other"] = "new"

	t.Log("Wake reconciliation when the pause value changes")
	if !pauseChanged.Update(event.UpdateEvent{ObjectOld: oldObject, ObjectNew: unpausedObject}) {
		t.Fatal("pause annotation change was ignored")
	}

	t.Log("Ignore metadata changes unrelated to the pause value")
	if pauseChanged.Update(event.UpdateEvent{ObjectOld: oldObject, ObjectNew: otherChanged}) {
		t.Fatal("unrelated annotation change triggered pause predicate")
	}
}

func pausedTestClient(t *testing.T, objects ...client.Object) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core scheme: %v", err)
	}
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()
}
