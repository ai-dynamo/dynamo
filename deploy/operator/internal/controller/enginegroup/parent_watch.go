/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func (r *Reconciler) engineGroupsForParent(ctx context.Context, object client.Object) []reconcile.Request {
	// Resolve only the supported DGD -> PCS -> PCSG ownership chain; stale names never select replacement worlds.
	owner := metav1.GetControllerOf(object)
	switch parent := object.(type) {
	case *api.DynamoGraphDeployment:
		owner = metav1.NewControllerRef(parent, api.DynamoGraphDeploymentGVK)
	case *grove.PodCliqueSet:
	case *grove.PodCliqueScalingGroup:
		if owner == nil || owner.Kind != "PodCliqueSet" || owner.APIVersion != grove.SchemeGroupVersion.String() {
			return nil
		}
		pcs := &grove.PodCliqueSet{}
		if err := r.Get(ctx, client.ObjectKey{Namespace: object.GetNamespace(), Name: owner.Name}, pcs); err != nil {
			if !apierrors.IsNotFound(err) {
				log.FromContext(ctx).Error(err, "Unable to observe the Engine Group parent PodCliqueSet")
			}
			return nil // The PCS watch and bounded observation retry resolve informer lag.
		}
		if pcs.UID != owner.UID {
			return nil
		}
		owner = metav1.GetControllerOf(pcs)
	default:
		return nil
	}
	if owner == nil || owner.Kind != api.DynamoGraphDeploymentGVK.Kind || owner.APIVersion != api.GroupVersion.String() || owner.UID == "" {
		return nil
	}

	// Controller ownership is authoritative even when labels or the component list have changed.
	groups := &api.DynamoGraphDeploymentEngineGroupList{}
	if err := r.List(ctx, groups, client.InNamespace(object.GetNamespace())); err != nil {
		log.FromContext(ctx).Error(err, "Unable to enqueue Engine Groups for a parent change")
		return nil
	}
	var requests []reconcile.Request
	for i := range groups.Items {
		group := &groups.Items[i]
		groupOwner := metav1.GetControllerOf(group)
		if groupOwner != nil && groupOwner.Kind == owner.Kind && groupOwner.APIVersion == owner.APIVersion &&
			groupOwner.Name == owner.Name && groupOwner.UID == owner.UID {
			requests = append(requests, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)})
		}
	}
	return requests
}
