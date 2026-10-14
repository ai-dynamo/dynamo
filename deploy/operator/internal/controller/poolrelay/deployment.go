/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"fmt"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	appsv1apply "k8s.io/client-go/applyconfigurations/apps/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// reconcileRelayDeployment applies the full PodTemplate with field ownership so
// deleting overrides removes managed fields while preserving API defaults.
// relay and desired must be non-nil. desired receives the persisted status.
func (r *relayReconciler) reconcileRelayDeployment(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, desired *appsv1.Deployment) error {
	// A create race must never turn into adoption of an unrelated workload.
	var current appsv1.Deployment
	err := r.Get(ctx, client.ObjectKeyFromObject(desired), &current)
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if err == nil && (!metav1.IsControlledBy(&current, relay) || !current.DeletionTimestamp.IsZero()) {
		return fmt.Errorf("Deployment %s is not owned by this DynamoPoolRelay or is being deleted", current.Name)
	}
	if err := controllerutil.SetControllerReference(relay, desired, r.Scheme()); err != nil {
		return err
	}
	if apierrors.IsNotFound(err) {
		created := desired.DeepCopy()
		if err := r.Create(ctx, created, client.FieldOwner("dynamo-relay-controller")); err != nil {
			return err
		}
		current = *created
	}

	// Fence replacement objects by UID and let the API track all managed Pod fields.
	desired.TypeMeta = metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"}
	desired.UID = current.UID
	// Skip matching owned fields, ignoring API defaults and admission-owned fields.
	owned, err := appsv1apply.ExtractDeployment(&current, "dynamo-relay-controller")
	if err != nil {
		return err
	}
	var wanted appsv1apply.DeploymentApplyConfiguration
	data, err := json.Marshal(desired)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &wanted); err != nil {
		return err
	}
	if equality.Semantic.DeepEqual(owned.Spec, wanted.Spec) && equality.Semantic.DeepEqual(owned.Labels, wanted.Labels) && equality.Semantic.DeepEqual(owned.Annotations, wanted.Annotations) && equality.Semantic.DeepEqual(owned.OwnerReferences, wanted.OwnerReferences) {
		*desired = current
		return nil
	}
	if err := r.Patch(ctx, desired, client.Apply, client.FieldOwner("dynamo-relay-controller"), client.ForceOwnership); err != nil {
		return fmt.Errorf("apply Relay Deployment %s: %w", desired.Name, err)
	}
	return nil
}

// validateRelayWorkload asks admission to validate copies before any dependency
// writes. API error messages may contain user-provided values, so do not publish them.
func (r *relayReconciler) validateRelayWorkload(ctx context.Context, resources []client.Object) error {
	for _, resource := range resources {
		candidate := resource.DeepCopyObject().(client.Object)
		switch candidate.(type) {
		case *appsv1.Deployment:
			candidate.GetObjectKind().SetGroupVersionKind(appsv1.SchemeGroupVersion.WithKind("Deployment"))
		case *corev1.Service:
			candidate.GetObjectKind().SetGroupVersionKind(corev1.SchemeGroupVersion.WithKind("Service"))
		default:
			continue
		}
		var err error
		if service, ok := candidate.(*corev1.Service); ok {
			current := &corev1.Service{}
			err = r.Get(ctx, client.ObjectKeyFromObject(service), current)
			if apierrors.IsNotFound(err) {
				err = r.Create(ctx, service, client.DryRunAll)
			} else if err == nil {
				current.Spec.Type, current.Spec.Selector, current.Spec.Ports = service.Spec.Type, service.Spec.Selector, service.Spec.Ports
				err = r.Update(ctx, current, client.DryRunAll)
			}
		} else {
			err = r.Patch(ctx, candidate, client.Apply, client.FieldOwner("dynamo-relay-controller"), client.ForceOwnership, client.DryRunAll)
		}
		if err != nil {
			return fmt.Errorf("%s %s failed API dry-run validation; check PodTemplate and Service configuration and operator permissions", candidate.GetObjectKind().GroupVersionKind().Kind, candidate.GetName())
		}
	}
	return nil
}
