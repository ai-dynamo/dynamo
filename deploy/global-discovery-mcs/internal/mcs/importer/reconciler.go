// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package importer

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
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

// Requeue delays of imports that cannot be applied yet.
const (
	// TerminatingNamespaceRequeue is how long an import waits while its
	// namespace is terminating before it is checked again.
	TerminatingNamespaceRequeue = 5 * time.Second
	// ForeignImportRequeue is how long an import blocked by a ServiceImport of
	// another implementation waits before it is checked again, in case the
	// watch missed that object going away.
	ForeignImportRequeue = 5 * time.Minute
)

// EventApplyFailed is the reason of the Warning Event emitted on a
// ServiceImport whose objects could not be applied.
const EventApplyFailed = "ApplyFailed"

// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceimports,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceimports/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=discovery.k8s.io,resources=endpointslices,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=namespaces,verbs=get;list;watch;create

// SetupWithManager registers the import controller: the importer's channel,
// Namespace creations and deletions (pending imports and recreated
// namespaces), and the changes to managed objects that the controller did not
// make itself (deletions and foreign updates), which are repaired. Deletions
// and ownership label changes of ServiceImports the controller does not
// manage let a blocked import through.
func (i *Importer) SetupWithManager(mgr ctrl.Manager) error {
	namespaces := predicate.Funcs{
		CreateFunc:  func(event.CreateEvent) bool { return true },
		UpdateFunc:  func(event.UpdateEvent) bool { return false },
		DeleteFunc:  func(event.DeleteEvent) bool { return true },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
	byLabel := predicate.Funcs{
		CreateFunc:  func(event.CreateEvent) bool { return false },
		UpdateFunc:  i.driftByLabel,
		DeleteFunc:  func(e event.DeleteEvent) bool { return isManaged(e.Object.GetLabels()) },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
	serviceImports := predicate.Funcs{
		CreateFunc:  func(event.CreateEvent) bool { return false },
		UpdateFunc:  i.serviceImportChanged,
		DeleteFunc:  func(event.DeleteEvent) bool { return true },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
	// A changed or deleted managed object must be re-applied even though its
	// import did not change: forget what was applied.
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
		}), builder.WithPredicates(namespaces)).
		Watches(&mcsv1beta1.ServiceImport{}, byName, builder.WithPredicates(serviceImports)).
		Watches(&corev1.Service{}, byServiceName, builder.WithPredicates(byLabel)).
		Watches(&discoveryv1.EndpointSlice{}, byServiceName, builder.WithPredicates(byLabel)).
		WithOptions(controller.Options{MaxConcurrentReconciles: i.opts.MaxConcurrentReconciles}).
		Complete(i)
}

// driftByLabel reports whether an update of a derived Service or an imported
// EndpointSlice is drift: the object is managed, the controller did not write
// the new version itself, and something it renders changed.
func (i *Importer) driftByLabel(e event.UpdateEvent) bool {
	if !isManaged(e.ObjectNew.GetLabels()) && !isManaged(e.ObjectOld.GetLabels()) {
		return false
	}
	name := e.ObjectNew.GetLabels()[api.LabelMCSServiceName]
	if name == "" {
		name = e.ObjectOld.GetLabels()[api.LabelMCSServiceName]
	}
	key := types.NamespacedName{Namespace: e.ObjectNew.GetNamespace(), Name: name}
	switch n := e.ObjectNew.(type) {
	case *corev1.Service:
		o, _ := e.ObjectOld.(*corev1.Service)
		return o != nil && !i.wrote(key, writtenService, n.ResourceVersion) &&
			(!equality.Semantic.DeepEqual(o.Spec, n.Spec) || metaChanged(o, n))
	case *discoveryv1.EndpointSlice:
		o, _ := e.ObjectOld.(*discoveryv1.EndpointSlice)
		return o != nil && !i.wrote(key, writtenSlice+n.Name, n.ResourceVersion) &&
			(o.AddressType != n.AddressType || !equality.Semantic.DeepEqual(o.Endpoints, n.Endpoints) ||
				!equality.Semantic.DeepEqual(o.Ports, n.Ports) || metaChanged(o, n))
	}
	return false
}

// serviceImportChanged reports whether an update of a ServiceImport needs a
// reconcile: a managed one changed by someone else, or an unmanaged one whose
// labels changed (it may have been released to this controller).
func (i *Importer) serviceImportChanged(e event.UpdateEvent) bool {
	n, ok := e.ObjectNew.(*mcsv1beta1.ServiceImport)
	o, ok2 := e.ObjectOld.(*mcsv1beta1.ServiceImport)
	if !ok || !ok2 {
		return false
	}
	if !isManaged(n.Labels) && !isManaged(o.Labels) {
		return !maps.Equal(o.Labels, n.Labels)
	}
	key := types.NamespacedName{Namespace: n.Namespace, Name: n.Name}
	return !i.wrote(key, writtenImport, n.ResourceVersion) &&
		(!equality.Semantic.DeepEqual(o.Spec, n.Spec) || metaChanged(o, n) || !maps.Equal(o.Annotations, n.Annotations))
}

// metaChanged reports whether the labels or owner references differ.
func metaChanged(o, n ctrlclient.Object) bool {
	return !maps.Equal(o.GetLabels(), n.GetLabels()) || !equality.Semantic.DeepEqual(o.GetOwnerReferences(), n.GetOwnerReferences())
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
	state, err := i.ensureNamespace(ctx, key.Namespace)
	if err != nil {
		return ctrl.Result{}, err
	}
	switch state {
	case namespaceTerminating:
		i.setPending(key, true)
		log.FromContext(ctx).V(1).Info("Import is pending: its namespace is terminating")
		return ctrl.Result{RequeueAfter: TerminatingNamespaceRequeue}, nil
	case namespaceMissing:
		i.setPending(key, true)
		log.FromContext(ctx).V(1).Info("Import is pending: its namespace does not exist")
		return ctrl.Result{}, nil
	}
	i.setPending(key, false)
	return i.apply(ctx, imp)
}

// namespaceState is what ensureNamespace found.
type namespaceState int

const (
	namespaceReady namespaceState = iota
	namespaceMissing
	namespaceTerminating
)

// ensureNamespace reports whether the namespace exists and is not
// terminating, creating it when CreateNamespaces is set.
func (i *Importer) ensureNamespace(ctx context.Context, name string) (namespaceState, error) {
	var ns corev1.Namespace
	err := i.opts.Client.Get(ctx, types.NamespacedName{Name: name}, &ns)
	switch {
	case err == nil:
		if !ns.DeletionTimestamp.IsZero() {
			return namespaceTerminating, nil
		}
		return namespaceReady, nil
	case !apierrors.IsNotFound(err):
		return namespaceMissing, err
	case !i.opts.CreateNamespaces:
		return namespaceMissing, nil
	}
	ns = corev1.Namespace{Name: name, Labels: managedLabels(i.opts.Identity.ClusterSet)}
	if err := i.opts.Client.Create(ctx, &ns); err != nil && !apierrors.IsAlreadyExists(err) {
		return namespaceMissing, fmt.Errorf("create namespace %s: %w", name, err)
	}
	log.FromContext(ctx).Info("Created the namespace of an import", "namespace", name)
	return namespaceReady, nil
}

// keptDerivedService returns the derived Service at derivedKey when it
// exists and keeps its type and primary IP family. A Service of the other
// type, or whose primary family differs from the first of families, is
// deleted, since clusterIP and the primary family are immutable, and reported
// as not found. A Service of that name that the controller does not manage
// is an error.
func (i *Importer) keptDerivedService(ctx context.Context, derivedKey types.NamespacedName, headless bool, families []corev1.IPFamily) (svc corev1.Service, found bool, err error) {
	if err := i.opts.Client.Get(ctx, derivedKey, &svc); err != nil {
		if apierrors.IsNotFound(err) {
			return svc, false, nil
		}
		return svc, false, err
	}
	if !isManaged(svc.Labels) {
		return svc, false, fmt.Errorf("service %s exists and is not managed by %s; it blocks the derived Service of the import", derivedKey, api.ManagedByController)
	}
	typeChanges := (svc.Spec.ClusterIP == corev1.ClusterIPNone) != headless
	familyChanges := len(families) > 0 && len(svc.Spec.IPFamilies) > 0 && svc.Spec.IPFamilies[0] != families[0]
	if typeChanges || familyChanges {
		if err := i.opts.Client.Delete(ctx, &svc); ctrlclient.IgnoreNotFound(err) != nil {
			return svc, false, fmt.Errorf("delete derived Service to change its type or IP family: %w", err)
		}
		log.FromContext(ctx).Info("Deleted the derived Service to re-create it", "service", derivedKey.Name, "typeChanged", typeChanges, "ipFamilyChanged", familyChanges)
		return svc, false, nil
	}
	return svc, true, nil
}

// apply writes the ServiceImport, the derived Service, the EndpointSlices
// and the ServiceImport status of imp. A failure after the ServiceImport
// exists is reported in its Ready condition and as a Warning Event.
func (i *Importer) apply(ctx context.Context, imp *mcsv1.ServiceImport) (res ctrl.Result, err error) {
	id := i.opts.Identity
	key := keyOf(imp)
	hash := importHash(imp)
	applied, epoch := i.appliedHash(key)
	if applied == hash {
		// Applied before and unchanged, for example a snapshot after a
		// reconnect: nothing to write. Drift and deletions forget the hash.
		return ctrl.Result{}, nil
	}
	ctx, span := tracing.Start(ctx, "apply_service_import", id.ClusterSet, id.Cluster, key.Namespace, key.Name)
	defer func() { tracing.End(span, err) }()
	logger := log.FromContext(ctx)

	var existing mcsv1beta1.ServiceImport
	found := true
	if err := i.opts.Client.Get(ctx, key, &existing); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		found = false
	}
	if found && !isManaged(existing.Labels) && !i.opts.AdoptUnmanaged {
		logger.V(1).Info("Skipping an import whose ServiceImport another controller manages")
		return ctrl.Result{RequeueAfter: ForeignImportRequeue}, nil
	}

	step, err := i.applyObjects(ctx, imp, &existing, found)
	if err != nil {
		// Some objects may have been written: the previous hash no longer
		// describes the cluster.
		i.forgetApplied(key)
		if found || step != stepServiceImport {
			i.reportFailure(ctx, key, &existing, step, err)
		}
		return ctrl.Result{}, err
	}
	i.setApplied(key, hash, epoch)
	return ctrl.Result{}, nil
}

// Steps of applyObjects, named in the Ready condition on failure.
const (
	stepServiceImport = "ServiceImport"
	stepService       = "derived Service"
	stepSlices        = "EndpointSlices"
	stepStatus        = "ServiceImport status"
)

// applyObjects applies the objects of imp in order and returns the step that
// failed. existing is the current ServiceImport when found; its conditions
// keep their transition times.
func (i *Importer) applyObjects(ctx context.Context, imp *mcsv1.ServiceImport, existing *mcsv1beta1.ServiceImport, found bool) (string, error) {
	id := i.opts.Identity
	key := keyOf(imp)
	logger := log.FromContext(ctx)

	derivedKey := types.NamespacedName{Namespace: key.Namespace, Name: api.DerivedServiceName(key.Namespace, key.Name)}
	headless := imp.GetType() == mcsv1.ServiceType_SERVICE_TYPE_HEADLESS
	_, families := DerivedIPFamilies(imp, i.opts.ClusterIPFamilies)
	svc, svcFound, err := i.keptDerivedService(ctx, derivedKey, headless, families)
	if err != nil {
		return stepService, err
	}
	var ips []string
	if svcFound {
		ips = clusterIPs(svc.Spec.ClusterIPs, svc.Spec.ClusterIP)
	}

	si, err := BuildServiceImport(imp, id.ClusterSet, ips)
	if err != nil {
		return stepServiceImport, err
	}
	if err := i.opts.Client.Apply(ctx, ctrlclient.ApplyConfigurationFromUnstructured(si), ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
		return stepServiceImport, fmt.Errorf("apply ServiceImport: %w", err)
	}
	i.recordWritten(key, writtenImport, si.GetResourceVersion())
	uid := si.GetUID()
	if !found {
		// From here on a failure is reported on the ServiceImport.
		*existing = mcsv1beta1.ServiceImport{}
		existing.Namespace, existing.Name, existing.UID, existing.Generation = key.Namespace, key.Name, uid, si.GetGeneration()
	}
	if err := InvalidPorts(imp); err != nil {
		// The ServiceImport shows the merged ports; a Service cannot.
		return stepService, fmt.Errorf("the merged ports are not valid Service ports: %w", err)
	}

	if svcFound && portsDiffer(svc.Spec.Ports, imp) {
		// spec.ports is a map list keyed by port and protocol: a port someone
		// else changed would survive the apply as an extra entry, or make
		// the apply invalid when its name is taken. Replace the list first.
		if err := i.replaceServicePorts(ctx, key, derivedKey, imp); err != nil {
			return stepService, err
		}
	}
	derived := BuildDerivedService(imp, id.ClusterSet, uid, i.opts.ClusterIPFamilies)
	if err := i.opts.Client.Apply(ctx, derived, ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
		return stepService, fmt.Errorf("apply derived Service %s: %w", derivedKey.Name, err)
	}
	if derived.ResourceVersion != nil {
		i.recordWritten(key, writtenService, *derived.ResourceVersion)
	}
	if len(derived.Spec.Ports) > len(imp.GetPorts()) {
		// The cached copy lagged: the answer carries ports the controller
		// did not set.
		if err := i.replaceServicePorts(ctx, key, derivedKey, imp); err != nil {
			return stepService, err
		}
	}
	// The client decodes the API server's answer into the apply
	// configuration, so the ClusterIPs it allocated, now or for a
	// re-created Service, are at hand without another read.
	got := clusterIPs(derived.Spec.ClusterIPs, deref(derived.Spec.ClusterIP))
	if !slices.Equal(got, ips) {
		si, err = BuildServiceImport(imp, id.ClusterSet, got)
		if err != nil {
			return stepServiceImport, err
		}
		if err := i.opts.Client.Apply(ctx, ctrlclient.ApplyConfigurationFromUnstructured(si), ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
			return stepServiceImport, fmt.Errorf("apply ServiceImport ips: %w", err)
		}
		i.recordWritten(key, writtenImport, si.GetResourceVersion())
	}

	if err := i.applySlices(ctx, imp, uid); err != nil {
		return stepSlices, err
	}

	// The current conditions keep their transition times when the status
	// does not change.
	st := BuildServiceImportStatus(imp, existing.Status.Conditions, si.GetGeneration())
	if found && equality.Semantic.DeepEqual(st.Clusters, existing.Status.Clusters) && equalConditions(st.Conditions, existing.Status.Conditions) {
		return "", nil
	}
	// The status is written whole with a merge patch: the controller owns
	// it, and the merge does not depend on the cached copy being current.
	patch, err := json.Marshal(map[string]any{"status": map[string]any{"clusters": st.Clusters, "conditions": st.Conditions}})
	if err != nil {
		return stepStatus, err
	}
	target := &mcsv1beta1.ServiceImport{Namespace: key.Namespace, Name: key.Name}
	if err := i.opts.Client.Status().Patch(ctx, target, ctrlclient.RawPatch(types.MergePatchType, patch)); err != nil {
		return stepStatus, fmt.Errorf("patch ServiceImport status: %w", err)
	}
	i.recordWritten(key, writtenImport, target.ResourceVersion)
	logger.V(1).Info("Import applied", "clusters", len(st.Clusters))
	return "", nil
}

// portsDiffer reports whether the ports of a derived Service are not the
// merged ports of imp, by name, port, protocol and app protocol.
func portsDiffer(existing []corev1.ServicePort, imp *mcsv1.ServiceImport) bool {
	desired := importPorts(imp)
	if len(existing) != len(desired) {
		return true
	}
	for _, e := range existing {
		if !slices.ContainsFunc(desired, func(d mcsv1beta1.ServicePort) bool {
			return d.Name == e.Name && d.Port == e.Port && d.Protocol == e.Protocol && deref(d.AppProtocol) == deref(e.AppProtocol)
		}) {
			return true
		}
	}
	return false
}

// replaceServicePorts sets spec.ports of the derived Service to the merged
// ports with a merge patch, which replaces the list whole, and takes
// ownership of every entry for the following applies.
func (i *Importer) replaceServicePorts(ctx context.Context, key, derivedKey types.NamespacedName, imp *mcsv1.ServiceImport) error {
	ports := make([]corev1.ServicePort, 0, len(imp.GetPorts()))
	for _, p := range importPorts(imp) {
		ports = append(ports, corev1.ServicePort{Name: p.Name, Protocol: p.Protocol, Port: p.Port, AppProtocol: p.AppProtocol})
	}
	patch, err := json.Marshal(map[string]any{"spec": map[string]any{"ports": ports}})
	if err != nil {
		return err
	}
	svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Namespace: derivedKey.Namespace, Name: derivedKey.Name}}
	if err := i.opts.Client.Patch(ctx, svc, ctrlclient.RawPatch(types.MergePatchType, patch), ctrlclient.FieldOwner(api.FieldManager)); err != nil {
		return fmt.Errorf("replace the ports of derived Service %s: %w", derivedKey.Name, err)
	}
	i.recordWritten(key, writtenService, svc.ResourceVersion)
	log.FromContext(ctx).Info("Replaced the ports of the derived Service, which carried ports the controller did not set", "service", derivedKey.Name)
	return nil
}

// reportFailure sets Ready=False, reason ApplyFailed, on the ServiceImport
// of key and emits a Warning Event, so that an import that stops converging
// is visible. The next successful apply replaces the condition. Failures of
// the report itself are logged; the apply error is what the caller returns.
func (i *Importer) reportFailure(ctx context.Context, key types.NamespacedName, existing *mcsv1beta1.ServiceImport, step string, cause error) {
	msg := fmt.Sprintf("Applying the %s failed: %v", step, cause)
	if len(msg) > 1024 {
		msg = msg[:1024]
	}
	conds := slices.Clone(existing.Status.Conditions)
	setCondition(&conds, metav1.Condition{
		Type: string(mcsv1beta1.ServiceImportConditionReady), Status: metav1.ConditionFalse,
		Reason: api.ConditionReasonApplyFailed, Message: msg, ObservedGeneration: existing.Generation,
	})
	patch, err := json.Marshal(map[string]any{"status": map[string]any{"conditions": conds}})
	if err == nil {
		target := &mcsv1beta1.ServiceImport{Namespace: key.Namespace, Name: key.Name}
		err = i.opts.Client.Status().Patch(ctx, target, ctrlclient.RawPatch(types.MergePatchType, patch))
		if err == nil {
			i.recordWritten(key, writtenImport, target.ResourceVersion)
		}
	}
	if err != nil && !apierrors.IsNotFound(err) {
		log.FromContext(ctx).V(1).Info("Reporting the apply failure on the ServiceImport failed", "error", err.Error())
	}
	if i.opts.Recorder != nil {
		regarding := &mcsv1beta1.ServiceImport{
			TypeMeta:   metav1.TypeMeta{APIVersion: importAPIVersion, Kind: importKind},
			ObjectMeta: metav1.ObjectMeta{Namespace: key.Namespace, Name: key.Name, UID: existing.UID},
		}
		i.opts.Recorder.Eventf(regarding, nil, corev1.EventTypeWarning, EventApplyFailed, "Apply", "%s", msg)
	}
}

// applySlices applies the EndpointSlices of every source cluster and deletes
// managed slices of the import that are no longer wanted: the slices the
// previous apply wrote and this one does not, by name and whatever the
// cache says, then the stale slices the cache knows. When the slice count
// shrinks the API server is listed directly, since the cache may lag behind
// the slices just written.
func (i *Importer) applySlices(ctx context.Context, imp *mcsv1.ServiceImport, owner types.UID) error {
	id := i.opts.Identity
	key := keyOf(imp)
	previous := i.writtenSlices(key)
	want := map[string]bool{}
	for _, s := range BuildEndpointSlices(imp, id.ClusterSet, id.Cluster, owner, i.opts.EndpointsPerSlice, i.opts.OmitTargetRefs) {
		if err := i.opts.Client.Apply(ctx, s, ctrlclient.FieldOwner(api.FieldManager), ctrlclient.ForceOwnership); err != nil {
			return fmt.Errorf("apply EndpointSlice %s: %w", *s.Name, err)
		}
		i.recordWritten(key, writtenSlice+*s.Name, deref(s.ResourceVersion))
		want[*s.Name] = true
	}
	for _, name := range previous {
		if want[name] {
			continue
		}
		stale := &discoveryv1.EndpointSlice{ObjectMeta: metav1.ObjectMeta{Namespace: imp.GetNamespace(), Name: name}}
		if err := i.deleteSlice(ctx, stale); err != nil {
			return err
		}
		i.forgetWritten(key, writtenSlice+name)
	}
	reader := ctrlclient.Reader(i.opts.Client)
	if len(previous) > 0 && len(want) < len(previous) {
		reader = i.opts.APIReader
	}
	var list discoveryv1.EndpointSliceList
	if err := reader.List(ctx, &list, ctrlclient.InNamespace(imp.GetNamespace()), ctrlclient.MatchingLabels{
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
		if err := i.deleteSlice(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// deleteSlice deletes a stale imported EndpointSlice; a missing one is fine.
func (i *Importer) deleteSlice(ctx context.Context, s *discoveryv1.EndpointSlice) error {
	err := i.opts.Client.Delete(ctx, s)
	switch {
	case apierrors.IsNotFound(err):
		return nil
	case err != nil:
		return fmt.Errorf("delete stale EndpointSlice %s: %w", s.Name, err)
	}
	i.opts.Metrics.GCDeletions.WithLabelValues(metrics.GCEndpointSlice).Inc()
	return nil
}

// remove deletes the managed ServiceImport, derived Service and EndpointSlices
// of an import the registry no longer has. Owner references would cascade, but
// the controller does not rely on the garbage collector.
func (i *Importer) remove(ctx context.Context, key types.NamespacedName) error {
	i.forgetApplied(key)
	i.forgetWritten(key, "")
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
func clusterIPs(clusterIPs []string, clusterIP string) []string {
	var ips []string
	for _, ip := range clusterIPs {
		if ip != "" && ip != corev1.ClusterIPNone {
			ips = append(ips, ip)
		}
	}
	if len(ips) == 0 && clusterIP != "" && clusterIP != corev1.ClusterIPNone {
		ips = []string{clusterIP}
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
