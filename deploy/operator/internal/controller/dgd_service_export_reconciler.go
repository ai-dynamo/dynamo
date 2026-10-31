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
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/multicluster"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// globalRouterPoolExportedConditionType reports on a Global Router pool's DGD
// whether its Services are exported, as far as the operator and the MCS
// implementation can tell.
const globalRouterPoolExportedConditionType = "GlobalRouterPoolExported"

// Reasons of the GlobalRouterPoolExported condition.
const (
	reasonPoolExported           = "Exported"
	reasonPoolExportConflict     = "ExportConflict"
	reasonPoolExportPending      = "ExportPending"
	reasonPoolExportFailed       = "ExportFailed"
	reasonPoolInvalidServiceName = "InvalidServiceName"
	reasonPoolServiceNameTooLong = "ServiceNameTooLong"
	reasonPoolNoFrontend         = "NoFrontend"
	reasonPoolClusterIDMismatch  = "ClusterIDMismatch"
	reasonPoolExportRejected     = "ExportRejected"
)

// poolProblemReasons are the reasons of the GlobalRouterPoolExported condition
// that are recorded as a Warning event when they are newly raised. A failed
// reconcile records its own event.
var poolProblemReasons = []string{
	reasonPoolExportConflict,
	reasonPoolInvalidServiceName,
	reasonPoolServiceNameTooLong,
	reasonPoolNoFrontend,
	reasonPoolClusterIDMismatch,
	reasonPoolExportRejected,
}

// clusterSetIDTTL is how long the operator reuses the cluster's ID in its
// ClusterSet before reading it again.
const clusterSetIDTTL = time.Minute

// dgdServiceExportReconciler exports the PoolRelay and frontend Services of a DGD
// that is a Global Router pool through the configured Multi-Cluster Services
// implementation, and withdraws the exports when it stops being one. A DGD is a
// pool while a DynamoPoolRelay of the same name exists. It creates and deletes
// only ServiceExports that the DGD controls, at the multicluster.x-k8s.io version
// that the gates resolved.
type dgdServiceExportReconciler struct {
	dgdResourceSyncer
	config *configv1alpha1.OperatorConfiguration
	gates  features.Gates
	// exports reads ServiceExports, which are unstructured: the manager's client
	// would read them from the API server, so this is the manager's cache, filled
	// by the controller's ServiceExport watch.
	exports client.Reader
	// clusterSetID returns the cluster's ID in its ClusterSet, or "" when it is
	// unknown, which skips the check of the pool's cluster ID.
	clusterSetID func(context.Context) string
}

func newDGDServiceExportReconciler(
	syncer dgdResourceSyncer,
	config *configv1alpha1.OperatorConfiguration,
	gates features.Gates,
	exports client.Reader,
	clusterSetID func(context.Context) string,
) *dgdServiceExportReconciler {
	return &dgdServiceExportReconciler{
		dgdResourceSyncer: syncer, config: config, gates: gates, exports: exports, clusterSetID: clusterSetID,
	}
}

// Reconcile converges the DGD's ServiceExports on whether it is a pool, and
// returns the GlobalRouterPoolExported condition for the DGD, or nil when the DGD
// is not a pool or nothing is exported. A failure returns the error with an
// ExportFailed condition when the DGD is a pool, or with nil when that is
// unknown. Without a configured MCS implementation the operator exports nothing
// and leaves existing exports alone.
func (r *dgdServiceExportReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (*metav1.Condition, error) {
	if !r.gates.Enabled(features.MultiClusterExport) {
		return nil, nil
	}
	relay, err := r.poolRelay(ctx, dgd)
	if err != nil {
		return nil, err
	}
	condition, err := r.reconcileExports(ctx, dgd, relay)
	if err != nil && relay != nil {
		return newPoolCondition(dgd, metav1.ConditionFalse, reasonPoolExportFailed, "%v", err), err
	}
	return condition, err
}

// unexportedService is a pool Service that the implementation cannot export.
type unexportedService struct {
	name    string
	problem multicluster.NameProblem
	detail  string
}

// reconcileExports exports the Services of the pool that relay makes the DGD,
// withdraws the DGD's other exports, and returns the pool's condition. A nil
// relay withdraws every export the DGD controls.
func (r *dgdServiceExportReconciler) reconcileExports(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	relay *nvidiacomv1alpha1.DynamoPoolRelay,
) (*metav1.Condition, error) {
	implementation := r.config.MultiCluster.MCSImplementation
	existing, err := multicluster.ListExports(ctx, r.exports, r.gates.ServiceExportVersion, dgd.Namespace, implementation)
	if err != nil {
		return nil, err
	}

	// A pool exports each of its Services that the implementation can export. An
	// existing export, the DGD's or a user's, needs nothing.
	var exported []string
	var unexported []unexportedService
	if relay != nil {
		for _, name := range multicluster.ServiceNames(dgd) {
			if problem, detail := multicluster.CheckServiceName(implementation, name); problem != multicluster.NameExportable {
				unexported = append(unexported, unexportedService{name: name, problem: problem, detail: detail})
				continue
			}
			if _, found := existing[name]; !found {
				if err := r.export(ctx, dgd, name); err != nil {
					return nil, err
				}
			}
			exported = append(exported, name)
		}
	}

	// Withdraw the DGD's exports of Services it no longer exports: all of them once
	// it is no longer a pool, and a renamed or removed frontend's.
	for _, name := range slices.Sorted(maps.Keys(existing)) {
		if !slices.Contains(exported, name) {
			if err := r.withdraw(ctx, dgd, existing[name]); err != nil {
				return nil, err
			}
		}
	}

	if relay == nil {
		return nil, nil
	}
	return r.condition(ctx, dgd, relay, exported, unexported, existing), nil
}

// poolRelay returns the DynamoPoolRelay that makes the DGD a Global Router pool,
// or nil when there is none or it is being deleted.
func (r *dgdServiceExportReconciler) poolRelay(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (*nvidiacomv1alpha1.DynamoPoolRelay, error) {
	relay := &nvidiacomv1alpha1.DynamoPoolRelay{}
	err := r.Get(ctx, client.ObjectKeyFromObject(dgd), relay)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get DynamoPoolRelay %s: %w", dgd.Name, err)
	}
	if !relay.DeletionTimestamp.IsZero() {
		return nil, nil
	}
	return relay, nil
}

// export creates the DGD's ServiceExport for the Service named name.
func (r *dgdServiceExportReconciler) export(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	name string,
) error {
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

// withdraw deletes export if the DGD controls it and it is not already being
// deleted.
func (r *dgdServiceExportReconciler) withdraw(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	export multicluster.Export,
) error {
	if export.ControllerUID != dgd.UID || export.Deleting {
		return nil
	}

	// The UID precondition deletes exactly the export just read.
	serviceExport := multicluster.NewServiceExport(r.gates.ServiceExportVersion)
	serviceExport.SetNamespace(dgd.Namespace)
	serviceExport.SetName(export.Name)
	if err := r.Delete(ctx, serviceExport, client.Preconditions{UID: &export.UID}); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to delete ServiceExport %s: %w", export.Name, err)
	}
	r.recorder.Eventf(dgd, nil, corev1.EventTypeNormal, "DeleteServiceExport", "Delete",
		"Withdrew the export of Service %s", export.Name)
	return nil
}

// condition returns the pool's GlobalRouterPoolExported condition: the first
// problem that the operator or the MCS implementation finds, else whether every
// export is imported. existing holds the exports read before this reconcile
// created any.
func (r *dgdServiceExportReconciler) condition(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	relay *nvidiacomv1alpha1.DynamoPoolRelay,
	exported []string,
	unexported []unexportedService,
	existing map[string]multicluster.Export,
) *metav1.Condition {
	implementation := r.config.MultiCluster.MCSImplementation
	if names, detail := unexportedNames(unexported, multicluster.NameInvalid); len(names) > 0 {
		return newPoolCondition(dgd, metav1.ConditionFalse, reasonPoolInvalidServiceName,
			"No Service can be named %s, so it is not exported: %s", strings.Join(names, ", "), detail)
	}
	if names, detail := unexportedNames(unexported, multicluster.NameTooLong); len(names) > 0 {
		return newPoolCondition(dgd, metav1.ConditionFalse, reasonPoolServiceNameTooLong,
			"%s, so %s is not exported; shorten the DGD or component name", detail, strings.Join(names, ", "))
	}
	if len(multicluster.FrontendServiceNames(dgd)) == 0 {
		return newPoolCondition(dgd, metav1.ConditionFalse, reasonPoolNoFrontend,
			"The DGD has no frontend component, so the Global Router has nowhere to send the pool's requests")
	}

	// A standard implementation labels the pool's imports with the cluster's ID
	// in its ClusterSet, and the Global Discovery Service joins the pool's
	// registration to them only under the same cluster ID.
	if multicluster.UsesClusterSetID(implementation) && r.clusterSetID != nil {
		if clusterID := r.clusterSetID(ctx); clusterID != "" && clusterID != relay.Spec.ClusterID {
			return newPoolCondition(dgd, metav1.ConditionFalse, reasonPoolClusterIDMismatch,
				"DynamoPoolRelay %s has clusterId %q, but ClusterProperty %s gives this cluster the ID %q in its ClusterSet; "+
					"the Global Discovery Service lists the pool only when they match",
				relay.Name, relay.Spec.ClusterID, multicluster.ClusterIDProperty, clusterID)
		}
	}

	var pending, conflicting []string
	for _, name := range exported {
		export, found := existing[name]
		switch {
		case !found || export.Deleting:
			// Created by this reconcile, or waiting to be replaced. Its next watch
			// event requeues the DGD.
			pending = append(pending, name)
		case export.State == multicluster.ExportRejected:
			return newPoolCondition(dgd, metav1.ConditionFalse, reasonPoolExportRejected,
				"%s reports on the export of Service %s: %s", implementation, name, export.Detail)
		case export.State == multicluster.ExportPending:
			pending = append(pending, name)
		case export.State == multicluster.ExportConflicting:
			conflicting = append(conflicting, fmt.Sprintf("%s (%s)", name, export.Detail))
		}
	}
	if len(pending) > 0 {
		return newPoolCondition(dgd, metav1.ConditionUnknown, reasonPoolExportPending,
			"Waiting for %s to import %s", implementation, strings.Join(pending, ", "))
	}
	if len(conflicting) > 0 {
		return newPoolCondition(dgd, metav1.ConditionTrue, reasonPoolExportConflict,
			"Exported %s through %s, which resolves a conflict with another cluster's export of %s",
			strings.Join(exported, ", "), implementation, strings.Join(conflicting, ", "))
	}
	return newPoolCondition(dgd, metav1.ConditionTrue, reasonPoolExported,
		"Exported %s through %s", strings.Join(exported, ", "), implementation)
}

// unexportedNames returns the names of the services with problem, and the first
// one's detail.
func unexportedNames(services []unexportedService, problem multicluster.NameProblem) ([]string, string) {
	var names []string
	detail := ""
	for _, service := range services {
		if service.problem != problem {
			continue
		}
		if len(names) == 0 {
			detail = service.detail
		}
		names = append(names, service.name)
	}
	return names, detail
}

// newPoolCondition returns the DGD's GlobalRouterPoolExported condition.
func newPoolCondition(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status metav1.ConditionStatus,
	reason, format string,
	args ...any,
) *metav1.Condition {
	return &metav1.Condition{
		Type:               globalRouterPoolExportedConditionType,
		Status:             status,
		ObservedGeneration: dgd.Generation,
		Reason:             reason,
		Message:            fmt.Sprintf(format, args...),
	}
}

// clusterSetIDs keeps the cluster's ID in its ClusterSet, read through the API
// server at most once per clusterSetIDTTL: the cache does not hold it, and it
// rarely changes.
type clusterSetIDs struct {
	mu     sync.Mutex
	id     string
	readAt time.Time
}

// get returns the cluster's ID in its ClusterSet, or "" when it is unknown. A
// failed read is not kept, so the next call reads again.
func (c *clusterSetIDs) get(ctx context.Context, reader client.Reader, now time.Time) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.readAt.IsZero() && now.Sub(c.readAt) < clusterSetIDTTL {
		return c.id
	}
	id, err := multicluster.ReadClusterID(ctx, reader)
	if err != nil {
		log.FromContext(ctx).V(1).Info("Failed to read the cluster's ID in its ClusterSet; skipping the check", "error", err)
		return ""
	}
	c.id, c.readAt = id, now
	return id
}

// mapServiceExportToDGDRequests requeues each DGD in the export's namespace that
// controls the export or exports a Service of its name: the DGD withdraws an
// export it no longer needs, takes over when a user's export of the same Service
// is deleted, and follows what the MCS implementation reports.
func (r *DynamoGraphDeploymentReconciler) mapServiceExportToDGDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	deployments := &nvidiacomv1beta1.DynamoGraphDeploymentList{}
	if err := r.List(ctx, deployments, client.InNamespace(obj.GetNamespace())); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoGraphDeployments for ServiceExport event")
		return nil
	}

	owner := metav1.GetControllerOf(obj)
	requests := make([]ctrl.Request, 0)
	for i := range deployments.Items {
		dgd := &deployments.Items[i]
		controls := owner != nil && owner.UID == dgd.UID
		if controls || slices.Contains(multicluster.ServiceNames(dgd), obj.GetName()) {
			requests = append(requests, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
		}
	}
	return requests
}

// mapPoolRelayToDGDRequests requeues the DGD that a DynamoPoolRelay makes a
// Global Router pool: the one with the same name and namespace.
func mapPoolRelayToDGDRequests(_ context.Context, obj client.Object) []ctrl.Request {
	return []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(obj)}}
}

// serviceExportStateChanged passes the ServiceExport updates that can change a
// pool's condition: a new status from the MCS implementation, or the start of a
// deletion.
func serviceExportStateChanged(update event.UpdateEvent) bool {
	oldExport, oldOK := update.ObjectOld.(*unstructured.Unstructured)
	newExport, newOK := update.ObjectNew.(*unstructured.Unstructured)
	if !oldOK || !newOK {
		return false
	}
	return !equality.Semantic.DeepEqual(oldExport.Object["status"], newExport.Object["status"]) ||
		oldExport.GetDeletionTimestamp().IsZero() != newExport.GetDeletionTimestamp().IsZero()
}

// poolRelayChanged passes the DynamoPoolRelay updates that can change its pool:
// a new spec, such as its cluster ID, or the start of a deletion.
func poolRelayChanged(update event.UpdateEvent) bool {
	return update.ObjectOld.GetGeneration() != update.ObjectNew.GetGeneration() ||
		update.ObjectOld.GetDeletionTimestamp().IsZero() != update.ObjectNew.GetDeletionTimestamp().IsZero()
}
