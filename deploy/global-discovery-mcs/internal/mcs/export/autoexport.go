// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package export

import (
	"context"
	"fmt"
	"slices"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
)

// AutoExportControllerName names the label-driven export controller.
const AutoExportControllerName = "autoexport"

// alreadyExistsRequeue is how long a Service waits before its ServiceExport
// is created again after an AlreadyExists error.
const alreadyExistsRequeue = time.Second

// Managed reports whether the controller created the ServiceExport.
func Managed(se *mcsv1beta1.ServiceExport) bool {
	return se.Labels[api.LabelManagedBy] == api.ManagedByController
}

// managedLabels are the labels of every ServiceExport the controller
// creates.
func managedLabels(clusterset string) map[string]string {
	return map[string]string{api.LabelManagedBy: api.ManagedByController, api.LabelClusterSet: clusterset}
}

// ownerRef returns the owner reference of a ServiceExport to its Service.
func ownerRef(svc *corev1.Service) metav1.OwnerReference {
	return metav1.OwnerReference{
		APIVersion: "v1", Kind: "Service", Name: svc.Name, UID: svc.UID, Controller: new(true),
	}
}

func hasOwner(se *mcsv1beta1.ServiceExport, uid types.UID) bool {
	return slices.ContainsFunc(se.OwnerReferences, func(o metav1.OwnerReference) bool { return o.UID == uid })
}

// AutoExporterOptions configure an AutoExporter.
type AutoExporterOptions struct {
	Client                  ctrlclient.Client
	Identity                identity.Identity
	Metrics                 *metrics.Metrics
	MaxConcurrentReconciles int
	WatchesNamespace        func(string) bool
}

// AutoExporter is label-driven auto-export: a Service labelled
// dynamo.nvidia.com/mcs-export=true gets a ServiceExport of the same name,
// owned by the Service and labelled dynamo.nvidia.com/mcs-managed-by, which is
// deleted when the label goes. ServiceExports the controller did not create are
// never touched.
type AutoExporter struct {
	opts AutoExporterOptions
}

// NewAutoExporter returns an AutoExporter.
func NewAutoExporter(opts AutoExporterOptions) *AutoExporter {
	if opts.Metrics == nil {
		opts.Metrics = metrics.New()
	}
	if opts.MaxConcurrentReconciles < 1 {
		opts.MaxConcurrentReconciles = 1
	}
	if opts.WatchesNamespace == nil {
		opts.WatchesNamespace = func(string) bool { return true }
	}
	return &AutoExporter{opts: opts}
}

// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceexports,verbs=create;delete

// SetupWithManager registers the controller on Services and on the
// ServiceExports it manages.
func (a *AutoExporter) SetupWithManager(mgr ctrl.Manager) error {
	managed := predicate.NewPredicateFuncs(func(o ctrlclient.Object) bool {
		return o.GetLabels()[api.LabelManagedBy] == api.ManagedByController
	})
	return ctrl.NewControllerManagedBy(mgr).
		Named(AutoExportControllerName).
		For(&corev1.Service{}).
		Watches(&mcsv1beta1.ServiceExport{}, handler.EnqueueRequestsFromMapFunc(func(_ context.Context, o ctrlclient.Object) []reconcile.Request {
			return []reconcile.Request{{Namespace: o.GetNamespace(), Name: o.GetName()}}
		}), builder.WithPredicates(managed)).
		WithOptions(controller.Options{MaxConcurrentReconciles: a.opts.MaxConcurrentReconciles}).
		Complete(a)
}

// Reconcile implements reconcile.Reconciler for the Service req.
func (a *AutoExporter) Reconcile(ctx context.Context, req ctrl.Request) (res ctrl.Result, err error) {
	start := time.Now()
	defer func() { a.opts.Metrics.ObserveReconcile(metrics.ControllerAutoExport, start, err) }()
	if !a.opts.WatchesNamespace(req.Namespace) {
		return ctrl.Result{}, nil
	}
	logger := log.FromContext(ctx)
	var svc corev1.Service
	svcFound := true
	if err := a.opts.Client.Get(ctx, req.NamespacedName, &svc); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		svcFound = false
	}
	want := svcFound && svc.DeletionTimestamp.IsZero() && api.IsExported(svc.Labels)

	var se mcsv1beta1.ServiceExport
	seFound := true
	if err := a.opts.Client.Get(ctx, req.NamespacedName, &se); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		seFound = false
	}

	switch {
	case want && !seFound:
		se = mcsv1beta1.ServiceExport{
			Namespace: req.Namespace, Name: req.Name,
			Labels:          managedLabels(a.opts.Identity.ClusterSet),
			OwnerReferences: []metav1.OwnerReference{ownerRef(&svc)}}
		if err := a.opts.Client.Create(ctx, &se); err != nil {
			if apierrors.IsAlreadyExists(err) {
				// The cache has not seen the ServiceExport yet.
				return ctrl.Result{RequeueAfter: alreadyExistsRequeue}, nil
			}
			return ctrl.Result{}, fmt.Errorf("create ServiceExport: %w", err)
		}
		logger.Info("Created ServiceExport for a labelled Service")
	case want && Managed(&se) && !hasOwner(&se, svc.UID):
		// The Service was re-created: own the ServiceExport again.
		patch := ctrlclient.MergeFrom(se.DeepCopy())
		se.OwnerReferences = append(slices.DeleteFunc(se.OwnerReferences, func(o metav1.OwnerReference) bool {
			return o.Kind == "Service" && o.APIVersion == "v1"
		}), ownerRef(&svc))
		if err := a.opts.Client.Patch(ctx, &se, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("set ServiceExport owner: %w", err)
		}
	case !want && seFound && Managed(&se) && se.DeletionTimestamp.IsZero():
		if err := a.opts.Client.Delete(ctx, &se); ctrlclient.IgnoreNotFound(err) != nil {
			return ctrl.Result{}, fmt.Errorf("delete ServiceExport: %w", err)
		}
		logger.Info("Deleted the ServiceExport of a Service no longer labelled for export")
	}
	return ctrl.Result{}, nil
}
