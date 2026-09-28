// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// dgdLPXHandoff owns only convergence of the generated child resource family.
// Build, Grove and scheduler lifecycles are exclusively the LPX controller's concern.
type dgdLPXHandoff struct {
	client client.Client
}

// Reconcile requires a DGD that selects LPX.
func (r *dgdLPXHandoff) Reconcile(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment) (*v1alpha1.LPXGraphDeployment, error) {
	child := &v1alpha1.LPXGraphDeployment{}
	err := r.client.Get(ctx, client.ObjectKeyFromObject(dgd), child)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	exists := err == nil
	if exists && !metav1.IsControlledBy(child, dgd) {
		return nil, fmt.Errorf("LPXGraphDeployment %q belongs to a different DGD; adoption is not supported", child.Name)
	}
	if exists && !child.DeletionTimestamp.IsZero() {
		return child, nil
	}

	restart := dynamo.LPXRestartToken(dgd, child.Annotations[dynamo.LPXRestartAnnotation])
	revision, err := dynamo.LPXInputRevision(dgd, restart)
	if err != nil {
		return nil, err
	}
	if exists && child.Spec.InputRevision == revision &&
		child.Annotations[dynamo.LPXRestartAnnotation] == restart {
		return child, nil
	}
	if !exists {
		child = &v1alpha1.LPXGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: dgd.Name, Namespace: dgd.Namespace},
		}
		if err := ctrl.SetControllerReference(dgd, child, r.client.Scheme()); err != nil {
			return nil, err
		}
	}
	child.Spec.InputRevision = revision
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, dynamo.LPXRestartAnnotation, restart)
	if exists {
		err = r.client.Update(ctx, child)
	} else {
		err = r.client.Create(ctx, child)
	}
	return child, err
}
