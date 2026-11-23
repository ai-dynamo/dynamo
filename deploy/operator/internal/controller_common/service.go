// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller_common

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// SyncService applies desired without changing its inputs and returns the API response.
// parent and desired must be non-nil. Existing Services must be controlled by parent.
func SyncService(ctx context.Context, kubeClient client.Client, parent client.Object, desired *corev1.Service) (bool, *corev1.Service, error) {
	const fieldManager = "dynamo-operator-service"
	key := client.ObjectKeyFromObject(desired)
	var current corev1.Service
	createdNew := false
	if err := kubeClient.Get(ctx, key, &current); err != nil {
		if !apierrors.IsNotFound(err) {
			return false, nil, err
		}

		// Create first so a foreign Service cannot be adopted through a stale miss.
		created := desired.DeepCopy()
		if err := controllerutil.SetControllerReference(parent, created, kubeClient.Scheme()); err != nil {
			return false, nil, err
		}
		if err := kubeClient.Create(ctx, created, client.FieldOwner(fieldManager)); err != nil {
			return false, nil, fmt.Errorf("create Service %s: %w", key, err)
		}
		current = *created
		createdNew = true
	} else if err := CheckControllerOwnership(&current, parent, kubeClient.Scheme()); err != nil {
		return false, nil, err
	}
	if !current.DeletionTimestamp.IsZero() {
		return false, nil, fmt.Errorf("Service %s is being deleted", key)
	}

	// Apply authored fields only, preserving allocated addresses and other managers' fields.
	intent := desired.DeepCopy()
	if err := controllerutil.SetControllerReference(parent, intent, kubeClient.Scheme()); err != nil {
		return false, nil, err
	}
	intent.TypeMeta = metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Service"}
	intent.UID = current.UID
	intent.ResourceVersion = ""
	fields, err := runtime.DefaultUnstructuredConverter.ToUnstructured(intent)
	if err != nil {
		return false, nil, fmt.Errorf("build Service %s apply configuration: %w", key, err)
	}
	applyObject := &unstructured.Unstructured{Object: fields}
	if err := kubeClient.Apply(ctx, client.ApplyConfigurationFromUnstructured(applyObject), client.FieldOwner(fieldManager), client.ForceOwnership); err != nil {
		return false, nil, fmt.Errorf("apply Service %s: %w", key, err)
	}

	// Use the write response without reading the Service back through the cache.
	observed := &corev1.Service{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(applyObject.Object, observed); err != nil {
		return false, nil, fmt.Errorf("decode applied Service %s: %w", key, err)
	}
	return createdNew || observed.ResourceVersion != current.ResourceVersion, observed, nil
}
