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
	"slices"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/multicluster"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdServiceExportReconciler exports the PoolRelay and frontend Services of a DGD
// marked as a Global Router pool through the configured Multi-Cluster Services
// implementation, and withdraws the exports when the mark is removed. It creates
// and deletes only ServiceExports that the DGD controls, at the
// multicluster.x-k8s.io version that the gates resolved.
type dgdServiceExportReconciler struct {
	dgdResourceSyncer
	config *configv1alpha1.OperatorConfiguration
	gates  features.Gates
}

func newDGDServiceExportReconciler(
	syncer dgdResourceSyncer,
	config *configv1alpha1.OperatorConfiguration,
	gates features.Gates,
) *dgdServiceExportReconciler {
	return &dgdServiceExportReconciler{dgdResourceSyncer: syncer, config: config, gates: gates}
}

// Reconcile converges the DGD's ServiceExports on its pool mark. Without a
// configured MCS implementation the operator exports nothing and leaves existing
// exports alone.
func (r *dgdServiceExportReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if !r.gates.Enabled(features.MultiClusterExport) {
		return nil
	}

	// Export each of the pool's Services, or withdraw the export.
	logger := log.FromContext(ctx)
	implementation := r.config.MultiCluster.MCSImplementation
	pool := multicluster.IsPool(dgd.Annotations)
	maxLength := multicluster.MaxServiceNameLength(implementation)
	for _, name := range multicluster.ServiceNames(dgd.Name) {
		// A Service the implementation cannot import with its cluster is not exported.
		exportable := len(name) <= maxLength
		if pool && !exportable {
			logger.Error(nil, "Service name is too long to export through this MCS implementation; shorten the DGD name",
				"service", name, "implementation", implementation, "maxLength", maxLength)
		}

		var err error
		if pool && exportable {
			err = r.export(ctx, dgd, name)
		} else {
			err = r.withdraw(ctx, dgd, name)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// export creates the ServiceExport for the Service named name unless one exists.
// An existing ServiceExport is left as it is, including one a user created.
func (r *dgdServiceExportReconciler) export(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	name string,
) error {
	// An existing export, the DGD's or a user's, needs nothing.
	err := r.Get(ctx, types.NamespacedName{Namespace: dgd.Namespace, Name: name},
		multicluster.ServiceExportMetadata(r.gates.ServiceExportVersion))
	if err == nil {
		return nil
	}
	if !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to get ServiceExport %s: %w", name, err)
	}

	// Owned by the DGD, the export is deleted with it.
	serviceExport := multicluster.RenderServiceExport(r.gates.ServiceExportVersion, dgd.Namespace, name, dgd.Name)
	if err := ctrl.SetControllerReference(dgd, serviceExport, r.Scheme()); err != nil {
		return fmt.Errorf("failed to set the owner of ServiceExport %s: %w", name, err)
	}
	if err := r.Create(ctx, serviceExport); err != nil {
		// The cache has not seen an export created moments ago. Its watch event
		// requeues the DGD.
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("failed to create ServiceExport %s: %w", name, err)
	}
	r.recorder.Eventf(dgd, nil, corev1.EventTypeNormal, "CreateServiceExport", "Create",
		"Exported Service %s through MCS implementation %s", name, r.config.MultiCluster.MCSImplementation)
	return nil
}

// withdraw deletes the ServiceExport for the Service named name if the DGD
// controls it and it is not already being deleted.
func (r *dgdServiceExportReconciler) withdraw(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	name string,
) error {
	// Only an export the DGD controls is withdrawn.
	existing := multicluster.ServiceExportMetadata(r.gates.ServiceExportVersion)
	err := r.Get(ctx, types.NamespacedName{Namespace: dgd.Namespace, Name: name}, existing)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get ServiceExport %s: %w", name, err)
	}
	if !metav1.IsControlledBy(existing, dgd) || existing.DeletionTimestamp != nil {
		return nil
	}

	// The UID precondition deletes exactly the export just read.
	if err := r.Delete(ctx, existing, client.Preconditions{UID: &existing.UID}); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to delete ServiceExport %s: %w", name, err)
	}
	r.recorder.Eventf(dgd, nil, corev1.EventTypeNormal, "DeleteServiceExport", "Delete",
		"Withdrew the export of Service %s", name)
	return nil
}

// mapServiceExportToDGDRequests requeues each DGD in the export's namespace that
// exports a Service of that name, whether or not it owns the export: the DGD
// withdraws an export it created moments before losing its pool mark, and takes
// over when a user's export of the same Service is deleted.
func (r *DynamoGraphDeploymentReconciler) mapServiceExportToDGDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	deployments := &nvidiacomv1beta1.DynamoGraphDeploymentList{}
	if err := r.List(ctx, deployments, client.InNamespace(obj.GetNamespace())); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoGraphDeployments for ServiceExport event")
		return nil
	}

	requests := make([]ctrl.Request, 0)
	for i := range deployments.Items {
		if slices.Contains(multicluster.ServiceNames(deployments.Items[i].Name), obj.GetName()) {
			requests = append(requests, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(&deployments.Items[i])})
		}
	}
	return requests
}
