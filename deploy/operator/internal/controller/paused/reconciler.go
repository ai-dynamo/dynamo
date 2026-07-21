/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package paused provides the opt-in pause boundary used by the DGD controller.
package paused

import (
	"context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

const (
	// Annotation marks an object for paused reconciliation.
	Annotation = "nvidia.com/paused"
	// Value is the only annotation value that pauses reconciliation.
	Value = "true"
)

type reconciler struct {
	reader    client.Reader
	prototype client.Object
	delegate  reconcile.Reconciler
}

// Wrap returns a reconciler that does not call delegate while the requested
// object has the pause annotation set to exactly "true".
func Wrap(reader client.Reader, prototype client.Object, delegate reconcile.Reconciler) reconcile.Reconciler {
	return &reconciler{reader: reader, prototype: prototype, delegate: delegate}
}

func (r *reconciler) Reconcile(ctx context.Context, request reconcile.Request) (reconcile.Result, error) {
	object := r.prototype.DeepCopyObject().(client.Object)
	if err := r.reader.Get(ctx, request.NamespacedName, object); err != nil {
		if apierrors.IsNotFound(err) {
			return r.delegate.Reconcile(ctx, request)
		}
		return reconcile.Result{}, err
	}
	if object.GetAnnotations()[Annotation] == Value {
		return reconcile.Result{}, nil
	}
	return r.delegate.Reconcile(ctx, request)
}

// AnnotationChangedPredicate triggers reconciliation when the pause value changes.
func AnnotationChangedPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(event.CreateEvent) bool { return false },
		DeleteFunc: func(event.DeleteEvent) bool { return false },
		UpdateFunc: func(update event.UpdateEvent) bool {
			return update.ObjectOld.GetAnnotations()[Annotation] != update.ObjectNew.GetAnnotations()[Annotation]
		},
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}
