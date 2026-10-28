/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"

	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func (p *GroveWorkloadProvider) engineGroupsForParent(ctx context.Context, object client.Object) []reconcile.Request {
	// Resolve only the supported DGD -> PCS -> PCSG ownership chain; stale names never select replacement worlds.
	owner := metav1.GetControllerOf(object)
	switch object.(type) {
	case *grove.PodCliqueSet:
	case *grove.PodCliqueScalingGroup:
		if owner == nil || owner.Kind != "PodCliqueSet" || owner.APIVersion != grove.SchemeGroupVersion.String() {
			return nil
		}
		pcs := &grove.PodCliqueSet{}
		if err := p.Client.Get(ctx, client.ObjectKey{Namespace: object.GetNamespace(), Name: owner.Name}, pcs); err != nil {
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
	return engineGroupsForOwner(ctx, p.Client, object.GetNamespace(), owner)
}
