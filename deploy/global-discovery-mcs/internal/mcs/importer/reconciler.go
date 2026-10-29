// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package importer

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/tracing"
)

// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceimports,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceimports/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=discovery.k8s.io,resources=endpointslices,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=namespaces,verbs=get;list;watch;create

// SetupWithManager registers the import controller: the importer's channel,
// Namespace creations (pending imports and recreated namespaces), and
// deletions of managed objects, which are repaired.
func (i *Importer) SetupWithManager(mgr ctrl.Manager) error {
	created := predicate.Funcs{
		CreateFunc:  func(event.CreateEvent) bool { return true },
		UpdateFunc:  func(event.UpdateEvent) bool { return false },
		DeleteFunc:  func(event.DeleteEvent) bool { return false },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
	deletedManaged := predicate.Funcs{
		CreateFunc:  func(event.CreateEvent) bool { return false },
		UpdateFunc:  func(event.UpdateEvent) bool { return false },
		DeleteFunc:  func(e event.DeleteEvent) bool { return isManaged(e.Object.GetLabels()) },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
	// A deleted managed object must be re-applied even though its import
	// did not change: forget what was applied.
	byServiceName := handler.EnqueueRequestsFromMapFunc(func(_ context.Context, o ctrlclient.Object) []reconcile.Request {
		name := o.GetLabels()[api.LabelMCSServiceName]
		if name == "" {
			return nil
		}
		key := types.NamespacedName{Namespace: o.GetNamespace(), Name: name}
		i.forgetApplied(key)
		return []reconcile.Request{{NamespacedName: key}}
	})
	byName := handler.EnqueueRequestsFromMapFunc(func(_ context.Context, o ctrlclient.Object) []reconcile.Request {
		key := types.NamespacedName{Namespace: o.GetNamespace(), Name: o.GetName()}
		i.forgetApplied(key)
		return []reconcile.Request{{NamespacedName: key}}
	})
	return ctrl.NewControllerManagedBy(mgr).
		Named(ControllerName).
		WatchesRawSource(source.Channel(i.events, &handler.TypedEnqueueRequestForObject[*mcsv1beta1.ServiceImport]{})).
		Watches(&corev1.Namespace{}, handler.EnqueueRequestsFromMapFunc(func(_ context.Context, o ctrlclient.Object) []reconcile.Request {
			keys := i.keysIn(o.GetName())
			reqs := make([]reconcile.Request, 0, len(keys))
			for _, key := range keys {
				i.forgetApplied(key)
				reqs = append(reqs, reconcile.Request{NamespacedName: key})
			}
			return reqs
		}), builder.WithPredicates(created)).
		Watches(&mcsv1beta1.ServiceImport{}, byName, builder.WithPredicates(deletedManaged)).
		Watches(&corev1.Service{}, byServiceName, builder.WithPredicates(deletedManaged)).
		Watches(&discoveryv1.EndpointSlice{}, byServiceName, builder.WithPredicates(deletedManaged)).
		WithOptions(controller.Options{MaxConcurrentReconciles: i.opts.MaxConcurrentReconciles}).
		Complete(i)
}

// Reconcile applies the import req, or deletes its objects when the registry no
// longer has it.
func (i *Importer) Reconcile(ctx context.Context, req ctrl.Request) (res ctrl.Result, err error) {
	start := time.Now()
	defer func() { i.opts.Metrics.ObserveReconcile(metrics.ControllerServiceImport, start, err) }()
	key := req.NamespacedName
	if !i.handles(key.Namespace) {
		return ctrl.Result{}, nil
	}
	imp, ok, synced := i.lookup(key)
	if !ok {
		i.setPending(key, false)
		if !synced {
			// The view is being rebuilt; SNAPSHOT_END collects what is gone.
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, i.remove(ctx, key)
	}
	ready, err := i.ensureNamespace(ctx, key.Namespace)
	if err != nil {
		return ctrl.Result{}, err
	}
	if !ready {
		i.setPending(key, true)
		log.FromContext(ctx).V(1).Info("Import is pending: its namespace does not exist")
		return ctrl.Result{}, nil
	}
	i.setPending(key, false)
	return ctrl.Result{}, i.apply(ctx, imp)
}

// ensureNamespace reports whether the namespace exists and is not
// terminating, creating it when CreateNamespaces is set.
func (i *Importer) ensureNamespace(ctx context.Context, name string) (bool, error) {
	var ns corev1.Namespace
	err := i.opts.Client.Get(ctx, types.NamespacedName{Name: name}, &ns)
	switch {
	case err == nil:
		return ns.DeletionTimestamp.IsZero(), nil
	case !apierrors.IsNotFound(err):
		return false, err
	case !i.opts.CreateNamespaces:
		return false, nil
	}
	ns = corev1.Namespace{Name: name, Labels: managedLabels(i.opts.Identity.ClusterSet)}
	if err := i.opts.Client.Create(ctx, &ns); err != nil && !apierrors.IsAlreadyExists(err) {
		return false, fmt.Errorf("create namespace %s: %w", name, err)
	}
	log.FromContext(ctx).Info("Created the namespace of an import", "namespace", name)
	return true, nil
}

// keptDerivedService returns the derived Service at derivedKey when it
// exists and keeps its type. A Service of the other type is deleted, since
// its clusterIP is immutable, and reported as not found. A Service of that
// name that the controller does not manage is an error.
func (i *Importer) keptDerivedService(ctx context.Context, derivedKey types.NamespacedName, headless bool) (svc corev1.Service, found bool, err error) {
	if err := i.opts.Client.Get(ctx, derivedKey, &svc); err != nil {
		if apierrors.IsNotFound(err) {
			return svc, false, nil
		}
		return svc, false, err
	}
	if !isManaged(svc.Labels) {
		return svc, false, fmt.Errorf("service %s exists and is not managed by %s; it blocks the derived Service of the import", derivedKey, api.ManagedByController)
	}
	if (svc.Spec.ClusterIP == corev1.ClusterIPNone) != headless {
		if err := i.opts.Client.Delete(ctx, &svc); ctrlclient.IgnoreNotFound(err) != nil {
			return svc, false, fmt.Errorf("delete derived Service to change its type: %w", err)
		}
		return svc, false, nil
	}
	return svc, true, nil
}

// apply writes the ServiceImport, the derived Service, the EndpointSlices
// and the ServiceImport status of imp.
func (i *Importer) apply(ctx context.Context, imp *mcsv1.ServiceImport) (err error) {
	id := i.opts.Identity
	key := keyOf(imp)
	hash := importHash(imp)
	if i.appliedHash(key) == hash {
		// Applied before and unchanged, for example a snapshot after a
		// reconnect: nothing to write.
		return nil
	}
	ctx, span := tracing.Start(ctx, "apply_service_import", id.ClusterSet, id.Cluster, key.Namespace, key.Name)
	defer func() { tracing.End(span, err) }()
	defer func() {
		if err == nil {
			i.setApplied(key, hash)
		}
	}()
	logger := log.FromContext(ctx)

	var existing mcsv1beta1.ServiceImport
	found := true
	if err := i.opts.Client.Get(ctx, key, &existing); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		found = false
	}
	if found && !isManaged(existing.Labels) && !i.opts.AdoptUnmanaged {
		logger.V(1).Info("Skipping an import whose ServiceImport another controller manages")
		hash = ""
		return nil
	}

	derivedKey := types.NamespacedName{Namespace: key.Namespace, Name: api.DerivedServiceName(key.Namespace, key.Name)}
	headless := imp.GetType() == mcsv1.ServiceType_SERVICE_TYPE_HEADLESS
	svc, svcFound, err := i.keptDerivedService(ctx, derivedKey, headless)
	if err != nil {
		return err
	}
	var ips []string
	if svcFound {
		ips = clusterIPs(&svc)
	}

	si, err := BuildServiceImport(imp, id.ClusterSet, ips)
	if err != nil {
		return err
	}
	if err := i.opts.Client.Apply(ctx, ctrlclient.ApplyConfigurationFromUnstructured(si), ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
		return fmt.Errorf("apply ServiceImport: %w", err)
	}
	uid := si.GetUID()

	derived := BuildDerivedService(imp, id.ClusterSet, uid)
	if err := i.opts.Client.Apply(ctx, derived, ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
		return fmt.Errorf("apply derived Service %s: %w", derivedKey.Name, err)
	}
	got := ips
	if !svcFound && !headless {
		// Created now: the API server allocated the ClusterIPs, which the
		// cache may not hold yet.
		var fresh corev1.Service
		if err := i.opts.APIReader.Get(ctx, derivedKey, &fresh); err != nil {
			return fmt.Errorf("read derived Service %s: %w", derivedKey.Name, err)
		}
		got = clusterIPs(&fresh)
	}
	if !slices.Equal(got, ips) {
		si, err = BuildServiceImport(imp, id.ClusterSet, got)
		if err != nil {
			return err
		}
		if err := i.opts.Client.Apply(ctx, ctrlclient.ApplyConfigurationFromUnstructured(si), ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
			return fmt.Errorf("apply ServiceImport ips: %w", err)
		}
	}

	if err := i.applySlices(ctx, imp, uid); err != nil {
		return err
	}

	// The current conditions keep their transition times when the status
	// does not change.
	if !found {
		existing = mcsv1beta1.ServiceImport{}
	}
	st := BuildServiceImportStatus(imp, existing.Status.Conditions, si.GetGeneration())
	if found && equality.Semantic.DeepEqual(st.Clusters, existing.Status.Clusters) && equalConditions(st.Conditions, existing.Status.Conditions) {
		return nil
	}
	// The status is written whole with a merge patch: the controller owns
	// it, and the merge does not depend on the cached copy being current.
	patch, err := json.Marshal(map[string]any{"status": map[string]any{"clusters": st.Clusters, "conditions": st.Conditions}})
	if err != nil {
		return err
	}
	target := &mcsv1beta1.ServiceImport{Namespace: key.Namespace, Name: key.Name}
	if err := i.opts.Client.Status().Patch(ctx, target, ctrlclient.RawPatch(types.MergePatchType, patch)); err != nil {
		return fmt.Errorf("patch ServiceImport status: %w", err)
	}
	logger.V(1).Info("Import applied", "clusters", len(st.Clusters))
	return nil
}

// applySlices applies the EndpointSlices of every source cluster and deletes
// managed slices of the import that are no longer wanted.
func (i *Importer) applySlices(ctx context.Context, imp *mcsv1.ServiceImport, owner types.UID) error {
	id := i.opts.Identity
	want := map[string]bool{}
	for _, s := range BuildEndpointSlices(imp, id.ClusterSet, id.Cluster, owner, i.opts.EndpointsPerSlice) {
		if err := i.opts.Client.Apply(ctx, s, ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
			return fmt.Errorf("apply EndpointSlice %s: %w", *s.Name, err)
		}
		want[*s.Name] = true
	}
	var list discoveryv1.EndpointSliceList
	if err := i.opts.Client.List(ctx, &list, ctrlclient.InNamespace(imp.GetNamespace()), ctrlclient.MatchingLabels{
		api.LabelEndpointSliceManagedBy: api.EndpointSliceManagedBy,
		api.LabelMCSServiceName:         imp.GetName(),
	}); err != nil {
		return err
	}
	for idx := range list.Items {
		s := &list.Items[idx]
		if want[s.Name] {
			continue
		}
		if err := i.opts.Client.Delete(ctx, s); ctrlclient.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete stale EndpointSlice %s: %w", s.Name, err)
		}
		i.opts.Metrics.GCDeletions.WithLabelValues(metrics.GCEndpointSlice).Inc()
	}
	return nil
}

// remove deletes the managed ServiceImport, derived Service and EndpointSlices
// of an import the registry no longer has. Owner references would cascade, but
// the controller does not rely on the garbage collector.
func (i *Importer) remove(ctx context.Context, key types.NamespacedName) error {
	i.forgetApplied(key)
	logger := log.FromContext(ctx)
	var si mcsv1beta1.ServiceImport
	err := i.opts.Client.Get(ctx, key, &si)
	switch {
	case err == nil && isManaged(si.Labels):
		if err := i.opts.Client.Delete(ctx, &si); ctrlclient.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete ServiceImport: %w", err)
		}
		i.opts.Metrics.GCDeletions.WithLabelValues(metrics.GCServiceImport).Inc()
		logger.Info("Deleted a ServiceImport that the registry no longer has")
	case err != nil && !apierrors.IsNotFound(err):
		return err
	}
	var svc corev1.Service
	err = i.opts.Client.Get(ctx, types.NamespacedName{Namespace: key.Namespace, Name: api.DerivedServiceName(key.Namespace, key.Name)}, &svc)
	switch {
	case err == nil && isManaged(svc.Labels):
		if err := i.opts.Client.Delete(ctx, &svc); ctrlclient.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete derived Service: %w", err)
		}
		i.opts.Metrics.GCDeletions.WithLabelValues(metrics.GCService).Inc()
	case err != nil && !apierrors.IsNotFound(err):
		return err
	}
	var list discoveryv1.EndpointSliceList
	if err := i.opts.Client.List(ctx, &list, ctrlclient.InNamespace(key.Namespace), ctrlclient.MatchingLabels{
		api.LabelEndpointSliceManagedBy: api.EndpointSliceManagedBy,
		api.LabelMCSServiceName:         key.Name,
	}); err != nil {
		return err
	}
	for idx := range list.Items {
		if err := i.opts.Client.Delete(ctx, &list.Items[idx]); ctrlclient.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete EndpointSlice %s: %w", list.Items[idx].Name, err)
		}
		i.opts.Metrics.GCDeletions.WithLabelValues(metrics.GCEndpointSlice).Inc()
	}
	return nil
}

// clusterIPs returns the ClusterIPs of a derived Service for spec.ips: at
// most two, none for a headless Service.
func clusterIPs(svc *corev1.Service) []string {
	var ips []string
	for _, ip := range svc.Spec.ClusterIPs {
		if ip != "" && ip != corev1.ClusterIPNone {
			ips = append(ips, ip)
		}
	}
	if len(ips) == 0 && svc.Spec.ClusterIP != "" && svc.Spec.ClusterIP != corev1.ClusterIPNone {
		ips = []string{svc.Spec.ClusterIP}
	}
	if len(ips) > 2 {
		ips = ips[:2]
	}
	return ips
}

// equalConditions compares condition lists ignoring order and heartbeat
// fields.
func equalConditions(a, b []metav1.Condition) bool {
	if len(a) != len(b) {
		return false
	}
	byType := map[string]metav1.Condition{}
	for _, c := range a {
		byType[c.Type] = c
	}
	for _, c := range b {
		o, ok := byType[c.Type]
		if !ok || o.Status != c.Status || o.Reason != c.Reason || o.Message != c.Message || o.ObservedGeneration != c.ObservedGeneration {
			return false
		}
	}
	return true
}
