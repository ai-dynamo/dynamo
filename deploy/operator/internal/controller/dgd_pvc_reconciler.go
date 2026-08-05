/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdPVCReconciler owns compatibility PVCs preserved when an alpha DGD is
// converted to the beta API.
type dgdPVCReconciler struct {
	dgdResourceSyncer
}

func newDGDPVCReconciler(syncer dgdResourceSyncer) *dgdPVCReconciler {
	return &dgdPVCReconciler{dgdResourceSyncer: syncer}
}

func (r *dgdPVCReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)
	pvcs := dynamo.GetDGDPreservedAlphaPVCs(dgd)
	for _, pvcConfig := range pvcs {
		if pvcConfig.Name == nil || *pvcConfig.Name == "" {
			logger.Error(nil, "Legacy top-level PVC not reconcilable: name is required", "pvcConfig", pvcConfig)
			continue
		}

		pvcName := *pvcConfig.Name
		logger.Info("Reconciling legacy top-level PVC", "pvcName", pvcName, "namespace", dgd.Namespace)
		if err := r.reconcilePVC(ctx, dgd, pvcName, pvcConfig); err != nil {
			return err
		}
	}
	return nil
}

func (r *dgdPVCReconciler) reconcilePVC(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	pvcName string,
	pvcConfig nvidiacomv1alpha1.PVC,
) error {
	logger := log.FromContext(ctx)
	pvc := &corev1.PersistentVolumeClaim{}
	key := types.NamespacedName{Name: pvcName, Namespace: dgd.Namespace}
	if err := r.Get(ctx, key, pvc); err != nil {
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("unable to retrieve legacy top-level PVC %q: %w", pvcName, err)
		}
		if pvcConfig.Create == nil || !*pvcConfig.Create {
			return fmt.Errorf("legacy top-level PVC %q does not exist and create is not enabled: %w", pvcName, err)
		}

		pvc = constructPVC(dgd, pvcConfig)
		if err := controllerutil.SetControllerReference(dgd, pvc, r.Scheme()); err != nil {
			return fmt.Errorf("failed to set controller reference for legacy top-level PVC %q: %w", pvcName, err)
		}
		if err := r.Create(ctx, pvc); err != nil {
			if apierrors.IsAlreadyExists(err) {
				return nil
			}
			return fmt.Errorf("failed to create legacy top-level PVC %q: %w", pvcName, err)
		}
		logger.Info("Legacy top-level PVC created", "pvcName", pvcName, "namespace", dgd.Namespace)
	}

	return nil
}
