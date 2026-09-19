// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"strconv"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
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

func (r *dgdLPXHandoff) Reconcile(ctx context.Context, source *v1beta1.DynamoGraphDeployment) (*v1alpha1.LPXGraphDeployment, error) {
	child := &v1alpha1.LPXGraphDeployment{}
	err := r.client.Get(ctx, client.ObjectKeyFromObject(source), child)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	exists := err == nil
	if exists && !metav1.IsControlledBy(child, source) {
		return nil, fmt.Errorf("LPXGraphDeployment %q belongs to a different source; adoption is not supported", child.Name)
	}
	if !source.HasLPXComponent() {
		if !exists {
			return nil, nil
		}
		return child, r.deleteChild(ctx, child)
	}
	if exists && !child.DeletionTimestamp.IsZero() {
		return child, nil
	}

	restart := dynamo.LPXRestartToken(source, child.Annotations[dynamo.LPXRestartAnnotation])
	revision, err := dynamo.LPXInputRevision(source, restart)
	if err != nil {
		return nil, err
	}
	if exists && child.Spec.InputRevision == revision &&
		child.Annotations[dynamo.LPXRestartAnnotation] == restart {
		return child, nil
	}
	if !exists {
		child = &v1alpha1.LPXGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: source.Name, Namespace: source.Namespace},
		}
		if err := ctrl.SetControllerReference(source, child, r.client.Scheme()); err != nil {
			return nil, err
		}
	}
	child.Spec.InputRevision = revision
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, dynamo.LPXRestartAnnotation, restart)
	metav1.SetMetaDataAnnotation(&child.ObjectMeta, lpx.DGDGenerationAnnotation, strconv.FormatInt(source.Generation, 10))
	if exists {
		err = r.client.Update(ctx, child)
	} else {
		err = r.client.Create(ctx, child)
	}
	return child, err
}

func (r *dgdLPXHandoff) deleteChild(ctx context.Context, child *v1alpha1.LPXGraphDeployment) error {
	if !child.DeletionTimestamp.IsZero() {
		return nil
	}
	uid, version := child.UID, child.ResourceVersion
	return client.IgnoreNotFound(r.client.Delete(ctx, child, &client.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid, ResourceVersion: &version}}))
}

func (r *dgdLPXHandoff) Finalize(ctx context.Context, source *v1beta1.DynamoGraphDeployment) error {
	child := &v1alpha1.LPXGraphDeployment{}
	if err := r.client.Get(ctx, client.ObjectKeyFromObject(source), child); err != nil {
		return client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(child, source) {
		// It is not our dependent; do not delete it and do not wait for it.
		return nil
	}
	if err := r.deleteChild(ctx, child); err != nil {
		return err
	}
	return fmt.Errorf("waiting for LPXGraphDeployment %q to finish cleanup", child.Name)
}
