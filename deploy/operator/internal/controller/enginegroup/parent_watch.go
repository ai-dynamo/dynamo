/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func (r *Reconciler) engineGroupsForParent(ctx context.Context, object client.Object) []reconcile.Request {
	// A DGD event identifies its own exact lifetime, not whichever object currently has the same name.
	dgd, ok := object.(*api.DynamoGraphDeployment)
	if !ok {
		return nil
	}
	return engineGroupsForOwner(ctx, r.Client, dgd.Namespace, metav1.NewControllerRef(dgd, api.DynamoGraphDeploymentGVK))
}

// engineGroupsForOwner accepts nil when an observed dependency has no controller owner.
func engineGroupsForOwner(ctx context.Context, reader client.Reader, namespace string, owner *metav1.OwnerReference) []reconcile.Request {
	if owner == nil || owner.Kind != api.DynamoGraphDeploymentGVK.Kind || owner.APIVersion != api.GroupVersion.String() || owner.UID == "" {
		return nil
	}

	// Controller ownership is authoritative even when labels or the component list have changed.
	groups := &api.DynamoGraphDeploymentEngineGroupList{}
	if err := reader.List(ctx, groups, client.InNamespace(namespace)); err != nil {
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
