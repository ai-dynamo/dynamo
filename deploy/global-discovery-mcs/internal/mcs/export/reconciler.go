// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package export writes this cluster's exports to the registry.
//
// Reconciler handles every ServiceExport, whether a chart created it (KEP-1645
// behaviour) or the label-driven AutoExporter did: it adds the finalizer
// dynamo.nvidia.com/mcs-service-export, validates the Service of the same name,
// puts the export spec when its hash changes, replaces the endpoint set when
// the hash of the Service's EndpointSlices changes, and writes the Valid
// condition locally and the Ready and Conflict conditions the registry reports
// for this cluster. Deleting a ServiceExport deletes the registry record before
// the finalizer is removed. Resyncer garbage-collects registry records without
// a local ServiceExport and resends everything after startup, client reconnects
// and re-registration.
package export

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/tracing"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

// ControllerName names the ServiceExport controller.
const ControllerName = "serviceexport"

// DefaultNotRegisteredRequeue is how long an export waits for the cluster
// registration before it is retried.
const DefaultNotRegisteredRequeue = 2 * time.Second

// Event reasons emitted on ServiceExports.
const (
	EventInvalid            = "InvalidExport"
	EventEndpointsTruncated = "EndpointsTruncated"
	EventExportFailed       = "ExportFailed"
)

// Registrar is the part of the cluster lease the reconciler needs.
type Registrar interface {
	Registered() bool
	Kick()
}

// Options configure a Reconciler.
type Options struct {
	// Client is the manager's client.
	Client ctrlclient.Client
	// APIReader reads from the API server, bypassing the cache, for the
	// sweep of namespaces outside the watch set; nil uses Client.
	APIReader ctrlclient.Reader
	Registry  *client.Client
	Identity  identity.Identity
	Lease     Registrar
	Metrics   *metrics.Metrics
	// Recorder emits Events on ServiceExports; nil emits none.
	Recorder events.EventRecorder
	// MaxConcurrentReconciles bounds the workers; default 1.
	MaxConcurrentReconciles int
	// WatchesNamespace reports whether a namespace is handled; nil means
	// every namespace.
	WatchesNamespace func(string) bool
	// NotRegisteredRequeue is the retry delay while the cluster is not
	// registered; default DefaultNotRegisteredRequeue.
	NotRegisteredRequeue time.Duration
	// Restricted is set when the controller watches a subset of the
	// namespaces: SweepUnwatched then releases ServiceExports outside the
	// watch set that still carry this controller's finalizer.
	Restricted bool
}

// exportState is what the reconciler last wrote to the registry for one export.
// The generations are safe for concurrent use; the other fields are
// guarded by Reconciler.mu.
type exportState struct {
	specGen      client.Generation
	endpointsGen client.Generation

	specHash      string
	endpointsHash string
	// withdrawn is set after the registry record was deleted for an invalid
	// export, so that it is not deleted on every reconcile.
	withdrawn bool
	phase     string
	endpoints int
	// epoch counts the times the sent hashes of this export were cleared
	// because the registry no longer held what was written; a write that
	// started before such a clear does not record its hash.
	epoch uint64
}

// sentView is a copy of the write bookkeeping of one export, taken with the
// reset epoch and the key's epoch it belongs to.
type sentView struct {
	specHash      string
	endpointsHash string
	withdrawn     bool
	epoch         uint64
	keyEpoch      uint64
}

// Reconciler is the ServiceExport reconciler.
type Reconciler struct {
	opts   Options
	events chan event.TypedGenericEvent[*mcsv1beta1.ServiceExport]

	mu     sync.Mutex
	states map[types.NamespacedName]*exportState
	own    map[types.NamespacedName]*mcsv1.ClusterStatus
	// resetEpoch counts ResetSent calls; a write that started before a
	// reset does not record its hash, so the reset always resends.
	resetEpoch uint64
}

// NewReconciler returns a Reconciler.
func NewReconciler(opts Options) *Reconciler {
	if opts.Metrics == nil {
		opts.Metrics = metrics.New()
	}
	if opts.MaxConcurrentReconciles < 1 {
		opts.MaxConcurrentReconciles = 1
	}
	if opts.NotRegisteredRequeue <= 0 {
		opts.NotRegisteredRequeue = DefaultNotRegisteredRequeue
	}
	if opts.WatchesNamespace == nil {
		opts.WatchesNamespace = func(string) bool { return true }
	}
	if opts.APIReader == nil {
		opts.APIReader = opts.Client
	}
	return &Reconciler{
		opts:   opts,
		events: make(chan event.TypedGenericEvent[*mcsv1beta1.ServiceExport], 4096),
		states: map[types.NamespacedName]*exportState{},
		own:    map[types.NamespacedName]*mcsv1.ClusterStatus{},
	}
}

// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceexports,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceexports/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceexports/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch
// +kubebuilder:rbac:groups=discovery.k8s.io,resources=endpointslices,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=create;patch

// SetupWithManager registers the controller: ServiceExports, the Services
// of the same name, their EndpointSlices except imported ones, and the
// channel the importer feeds with this cluster's conditions.
func (r *Reconciler) SetupWithManager(mgr ctrl.Manager) error {
	notImported := predicate.NewPredicateFuncs(func(o ctrlclient.Object) bool {
		return o.GetLabels()[api.LabelEndpointSliceManagedBy] != api.EndpointSliceManagedBy
	})
	return ctrl.NewControllerManagedBy(mgr).
		Named(ControllerName).
		For(&mcsv1beta1.ServiceExport{}).
		Watches(&corev1.Service{}, handler.EnqueueRequestsFromMapFunc(r.exportOfService)).
		Watches(&discoveryv1.EndpointSlice{}, handler.EnqueueRequestsFromMapFunc(r.exportOfSlice), builder.WithPredicates(notImported)).
		WatchesRawSource(source.Channel(r.events, &handler.TypedEnqueueRequestForObject[*mcsv1beta1.ServiceExport]{})).
		WithOptions(controller.Options{MaxConcurrentReconciles: r.opts.MaxConcurrentReconciles}).
		Complete(r)
}

// exportOfService maps a Service to the ServiceExport of the same name when
// one exists.
func (r *Reconciler) exportOfService(ctx context.Context, o ctrlclient.Object) []reconcile.Request {
	return r.existingExport(ctx, types.NamespacedName{Namespace: o.GetNamespace(), Name: o.GetName()})
}

// exportOfSlice maps an EndpointSlice to the ServiceExport of its Service.
func (r *Reconciler) exportOfSlice(ctx context.Context, o ctrlclient.Object) []reconcile.Request {
	name := o.GetLabels()[discoveryv1.LabelServiceName]
	if name == "" {
		return nil
	}
	return r.existingExport(ctx, types.NamespacedName{Namespace: o.GetNamespace(), Name: name})
}

func (r *Reconciler) existingExport(ctx context.Context, key types.NamespacedName) []reconcile.Request {
	var se mcsv1beta1.ServiceExport
	if err := r.opts.Client.Get(ctx, key, &se); err != nil {
		return nil
	}
	return []reconcile.Request{{NamespacedName: key}}
}

// SetOwnStatus records this cluster's status of the import namespace/name as
// the registry reported it, or nil when the registry reports none, together
// with the generation of this cluster's endpoint set in the import, and
// reconciles the ServiceExport of that name. The importer calls it.
//
// The report also tells the reconciler what the registry holds: a nil
// status means the export record is gone, and a participating cluster
// without an endpoint generation means the endpoint set is gone. In both
// cases the sent hashes are cleared so that the next reconcile writes
// again, whatever the heartbeats say.
func (r *Reconciler) SetOwnStatus(namespace, name string, st *mcsv1.ClusterStatus, endpointsGeneration uint64) {
	key := types.NamespacedName{Namespace: namespace, Name: name}
	r.mu.Lock()
	if st == nil {
		delete(r.own, key)
	} else {
		r.own[key] = st
	}
	if x := r.states[key]; x != nil && !x.withdrawn {
		switch {
		case st == nil && (x.specHash != "" || x.endpointsHash != ""):
			x.specHash, x.endpointsHash = "", ""
			x.epoch++
		case st != nil && st.GetParticipating() && endpointsGeneration == 0 && x.endpointsHash != "":
			x.endpointsHash = ""
			x.epoch++
		}
	}
	r.mu.Unlock()
	r.enqueue(key)
}

func (r *Reconciler) ownStatus(key types.NamespacedName) *mcsv1.ClusterStatus {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.own[key]
}

func (r *Reconciler) enqueue(key types.NamespacedName) {
	r.events <- genericEvent(key)
}

func genericEvent(key types.NamespacedName) event.TypedGenericEvent[*mcsv1beta1.ServiceExport] {
	return event.TypedGenericEvent[*mcsv1beta1.ServiceExport]{
		Object: &mcsv1beta1.ServiceExport{Namespace: key.Namespace, Name: key.Name},
	}
}

// state returns the state of key, creating it, and a copy of its write
// bookkeeping.
func (r *Reconciler) state(key types.NamespacedName) (*exportState, sentView) {
	r.mu.Lock()
	defer r.mu.Unlock()
	st := r.states[key]
	if st == nil {
		st = &exportState{phase: metrics.ExportPending}
		r.states[key] = st
		r.updateGaugesLocked()
	}
	return st, sentView{specHash: st.specHash, endpointsHash: st.endpointsHash, withdrawn: st.withdrawn, epoch: r.resetEpoch, keyEpoch: st.epoch}
}

// recordSent applies a successful write to the bookkeeping of key unless
// ResetSent ran or the key's hashes were cleared since view was taken.
func (r *Reconciler) recordSent(key types.NamespacedName, view sentView, apply func(st *exportState)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if st := r.states[key]; st != nil && view.epoch == r.resetEpoch && view.keyEpoch == st.epoch {
		apply(st)
	}
}

// setPhase records the export state and refreshes the export gauges.
func (r *Reconciler) setPhase(key types.NamespacedName, phase string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if st := r.states[key]; st != nil {
		st.phase = phase
	}
	r.updateGaugesLocked()
}

func (r *Reconciler) forget(key types.NamespacedName) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.states, key)
	r.updateGaugesLocked()
}

func (r *Reconciler) updateGaugesLocked() {
	counts := map[string]int{}
	endpoints := 0
	for _, st := range r.states {
		counts[st.phase]++
		if st.phase == metrics.ExportExported {
			endpoints += st.endpoints
		}
	}
	for _, p := range metrics.ExportStates {
		r.opts.Metrics.Exports.WithLabelValues(p).Set(float64(counts[p]))
	}
	r.opts.Metrics.ExportedEndpoints.Set(float64(endpoints))
}

// ResetSent forgets what was written to the registry, keeping the generations,
// so that the next reconcile of every export writes again.
func (r *Reconciler) ResetSent() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.resetEpoch++
	for _, st := range r.states {
		st.specHash, st.endpointsHash, st.withdrawn = "", "", false
	}
}

// Reconcile implements reconcile.Reconciler.
func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (res ctrl.Result, err error) {
	start := time.Now()
	defer func() { r.opts.Metrics.ObserveReconcile(metrics.ControllerServiceExport, start, err) }()
	if !r.opts.WatchesNamespace(req.Namespace) {
		return ctrl.Result{}, nil
	}
	var se mcsv1beta1.ServiceExport
	if err := r.opts.Client.Get(ctx, req.NamespacedName, &se); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, r.goneWithoutFinalizer(ctx, req.NamespacedName)
	}
	if !se.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, r.finalize(ctx, &se)
	}
	if !controllerutil.ContainsFinalizer(&se, api.FinalizerServiceExport) {
		patch := ctrlclient.MergeFromWithOptions(se.DeepCopy(), ctrlclient.MergeFromWithOptimisticLock{})
		controllerutil.AddFinalizer(&se, api.FinalizerServiceExport)
		if err := r.opts.Client.Patch(ctx, &se, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("add finalizer: %w", err)
		}
	}
	key := req.NamespacedName
	st, sent := r.state(key)

	var svc corev1.Service
	if err := r.opts.Client.Get(ctx, key, &svc); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, r.invalid(ctx, &se, st, sent, api.ConditionReasonNoService,
			fmt.Sprintf("Service %s/%s does not exist", key.Namespace, key.Name))
	}
	if svc.Spec.Type == corev1.ServiceTypeExternalName {
		return ctrl.Result{}, r.invalid(ctx, &se, st, sent, api.ConditionReasonInvalidServiceType,
			"Service of type ExternalName cannot be exported")
	}

	result := r.export(ctx, &se, &svc, st, sent)
	if err := r.writeStatus(ctx, &se, nil, result); err != nil {
		return ctrl.Result{}, err
	}
	switch {
	case result.notRegistered:
		return ctrl.Result{RequeueAfter: r.opts.NotRegisteredRequeue}, nil
	case result.transient != nil:
		return ctrl.Result{}, result.transient
	}
	return ctrl.Result{}, nil
}

// exportResult is the outcome of writing one export.
type exportResult struct {
	notRegistered bool
	// failed is a permanent failure; it sets Ready=False/Failed.
	failed error
	// transient is retried with backoff.
	transient error
}

// export writes the spec and the endpoint set when they changed.
func (r *Reconciler) export(ctx context.Context, se *mcsv1beta1.ServiceExport, svc *corev1.Service, st *exportState, sent sentView) exportResult {
	key := types.NamespacedName{Namespace: se.Namespace, Name: se.Name}
	logger := log.FromContext(ctx)
	if r.opts.Lease != nil && !r.opts.Lease.Registered() {
		r.setPhase(key, metrics.ExportPending)
		return exportResult{notRegistered: true}
	}
	spec := BuildSpec(svc, se)
	if hash := SpecHash(spec); hash != sent.specHash {
		if res := r.classify(key, r.putSpec(ctx, key, spec, st), metrics.OpPutExport); res != nil {
			return *res
		}
		r.recordSent(key, sent, func(st *exportState) { st.specHash, st.withdrawn = hash, false })
		logger.V(1).Info("Export spec written to the registry", "generation", st.specGen.Last())
	}

	var list discoveryv1.EndpointSliceList
	if err := r.opts.Client.List(ctx, &list, ctrlclient.InNamespace(key.Namespace),
		ctrlclient.MatchingLabels{discoveryv1.LabelServiceName: key.Name}); err != nil {
		return exportResult{transient: fmt.Errorf("list EndpointSlices: %w", err)}
	}
	own := list.Items[:0]
	for _, s := range list.Items {
		if s.Labels[api.LabelEndpointSliceManagedBy] != api.EndpointSliceManagedBy {
			own = append(own, s)
		}
	}
	eps, dropped := BuildEndpoints(own, api.MaxEndpointsPerExport)
	if hash := EndpointsHash(eps); hash != sent.endpointsHash {
		if dropped > 0 {
			msg := fmt.Sprintf("Service has %d endpoints; exporting %d, ready ones first", len(eps)+dropped, len(eps))
			logger.Info("Endpoint set truncated", "dropped", dropped, "limit", api.MaxEndpointsPerExport)
			r.event(se, corev1.EventTypeWarning, EventEndpointsTruncated, msg)
		}
		if res := r.classifyEndpoints(key, r.replaceEndpoints(ctx, key, eps, st)); res != nil {
			return *res
		}
		r.recordSent(key, sent, func(st *exportState) { st.endpointsHash, st.endpoints = hash, len(eps) })
		logger.V(1).Info("Endpoint set written to the registry", "endpoints", len(eps), "generation", st.endpointsGen.Last())
	}
	r.setPhase(key, metrics.ExportExported)
	return exportResult{}
}

// putSpec writes the spec with the next generation, retrying once above the
// stored generation on STALE_GENERATION.
func (r *Reconciler) putSpec(ctx context.Context, key types.NamespacedName, spec *mcsv1.ServiceExportSpec, st *exportState) (err error) {
	id := r.opts.Identity
	ctx, span := tracing.Start(ctx, "put_service_export", id.ClusterSet, id.Cluster, key.Namespace, key.Name)
	defer func() { tracing.End(span, err) }()
	req := &mcsv1.PutServiceExportRequest{
		Clusterset: id.ClusterSet, Cluster: id.Cluster, Namespace: key.Namespace, Name: key.Name,
		Spec: spec, Generation: st.specGen.Next(),
	}
	_, err = r.opts.Registry.ServiceExports().Put(ctx, req)
	if cur, ok := client.StaleGeneration(err); ok {
		r.opts.Metrics.ExportWrites.WithLabelValues(metrics.OpPutExport, metrics.WriteStaleGeneration).Inc()
		st.specGen.Observe(cur)
		req.Generation = st.specGen.Next()
		_, err = r.opts.Registry.ServiceExports().Put(ctx, req)
	}
	return err
}

// replaceEndpoints writes the endpoint set like putSpec writes the spec.
func (r *Reconciler) replaceEndpoints(ctx context.Context, key types.NamespacedName, eps []*mcsv1.Endpoint, st *exportState) (err error) {
	id := r.opts.Identity
	ctx, span := tracing.Start(ctx, "replace_endpoints", id.ClusterSet, id.Cluster, key.Namespace, key.Name)
	defer func() { tracing.End(span, err) }()
	req := &mcsv1.ReplaceEndpointsRequest{
		Clusterset: id.ClusterSet, Cluster: id.Cluster, Namespace: key.Namespace, Name: key.Name,
		Endpoints: eps, Generation: st.endpointsGen.Next(),
	}
	_, err = r.opts.Registry.ServiceExports().ReplaceEndpoints(ctx, req)
	if cur, ok := client.StaleGeneration(err); ok {
		r.opts.Metrics.EndpointReplacements.WithLabelValues(metrics.WriteStaleGeneration).Inc()
		st.endpointsGen.Observe(cur)
		req.Generation = st.endpointsGen.Next()
		_, err = r.opts.Registry.ServiceExports().ReplaceEndpoints(ctx, req)
	}
	return err
}

// classify records the result of an export write and turns an error into an
// exportResult, or returns nil on success.
func (r *Reconciler) classify(key types.NamespacedName, err error, op string) *exportResult {
	res := r.classifyErr(key, err)
	result := metrics.WriteSuccess
	switch {
	case res == nil:
	case res.notRegistered:
		result = metrics.WriteNotRegistered
	default:
		result = metrics.WriteError
	}
	r.opts.Metrics.ExportWrites.WithLabelValues(op, result).Inc()
	return res
}

func (r *Reconciler) classifyEndpoints(key types.NamespacedName, err error) *exportResult {
	res := r.classifyErr(key, err)
	result := metrics.WriteSuccess
	switch {
	case res == nil:
	case res.notRegistered:
		result = metrics.WriteNotRegistered
	default:
		result = metrics.WriteError
	}
	r.opts.Metrics.EndpointReplacements.WithLabelValues(result).Inc()
	return res
}

func (r *Reconciler) classifyErr(key types.NamespacedName, err error) *exportResult {
	if err == nil {
		return nil
	}
	if client.IsClusterNotRegistered(err) {
		if r.opts.Lease != nil {
			r.opts.Lease.Kick()
		}
		r.setPhase(key, metrics.ExportPending)
		return &exportResult{notRegistered: true}
	}
	r.setPhase(key, metrics.ExportFailed)
	switch status.Code(err) {
	case codes.InvalidArgument, codes.FailedPrecondition, codes.OutOfRange, codes.PermissionDenied:
		return &exportResult{failed: err}
	}
	return &exportResult{transient: err}
}

// invalid marks an export Valid=False, writes the status and withdraws the
// registry record.
func (r *Reconciler) invalid(ctx context.Context, se *mcsv1beta1.ServiceExport, _ *exportState, sent sentView, reason, msg string) error {
	key := types.NamespacedName{Namespace: se.Namespace, Name: se.Name}
	r.setPhase(key, metrics.ExportInvalid)
	if err := r.writeStatus(ctx, se, &metav1.Condition{
		Type: api.ConditionValid, Status: metav1.ConditionFalse, Reason: reason, Message: msg,
	}, exportResult{}); err != nil {
		return err
	}
	if sent.withdrawn {
		return nil
	}
	r.event(se, corev1.EventTypeWarning, EventInvalid, msg)
	if err := r.deleteRecord(ctx, key); err != nil {
		return err
	}
	r.recordSent(key, sent, func(st *exportState) { st.withdrawn, st.specHash, st.endpointsHash = true, "", "" })
	return nil
}

// writeStatus sets Valid (invalid, or True when nil) and Ready and Conflict
// from the registry and patches the status when it changed.
func (r *Reconciler) writeStatus(ctx context.Context, se *mcsv1beta1.ServiceExport, invalid *metav1.Condition, res exportResult) error {
	key := types.NamespacedName{Namespace: se.Namespace, Name: se.Name}
	orig := se.DeepCopy()
	conds := &se.Status.Conditions
	gen := se.Generation
	set := func(c metav1.Condition) {
		c.ObservedGeneration = gen
		meta.SetStatusCondition(conds, c)
	}
	if invalid != nil {
		set(*invalid)
		set(metav1.Condition{Type: api.ConditionReady, Status: metav1.ConditionFalse, Reason: api.ConditionReasonFailed,
			Message: "ServiceExport is not valid: " + invalid.Message})
		meta.RemoveStatusCondition(conds, api.ConditionConflict)
	} else {
		set(metav1.Condition{Type: api.ConditionValid, Status: metav1.ConditionTrue, Reason: api.ConditionReasonValid,
			Message: "Service is exportable"})
		own := r.ownStatus(key)
		ready, conflict := registryCondition(own, api.ConditionReady), registryCondition(own, api.ConditionConflict)
		switch {
		case res.failed != nil:
			set(metav1.Condition{Type: api.ConditionReady, Status: metav1.ConditionFalse, Reason: api.ConditionReasonFailed,
				Message: "The registry rejected the export: " + status.Convert(res.failed).Message()})
			r.event(se, corev1.EventTypeWarning, EventExportFailed, status.Convert(res.failed).Message())
		case res.notRegistered:
			set(metav1.Condition{Type: api.ConditionReady, Status: metav1.ConditionUnknown, Reason: api.ConditionReasonPending,
				Message: "Waiting for the cluster to register with the registry"})
		case ready != nil:
			set(*ready)
		default:
			set(metav1.Condition{Type: api.ConditionReady, Status: metav1.ConditionUnknown, Reason: api.ConditionReasonPending,
				Message: "Waiting for the registry to report the export"})
		}
		if conflict != nil {
			set(*conflict)
		} else {
			meta.RemoveStatusCondition(conds, api.ConditionConflict)
		}
	}
	if equalConditions(orig.Status.Conditions, se.Status.Conditions) {
		return nil
	}
	if err := r.opts.Client.Status().Patch(ctx, se, ctrlclient.MergeFrom(orig)); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("patch ServiceExport status: %w", err)
	}
	return nil
}

// finalize deletes the registry record of a ServiceExport being deleted, then
// removes the finalizer.
func (r *Reconciler) finalize(ctx context.Context, se *mcsv1beta1.ServiceExport) error {
	if !controllerutil.ContainsFinalizer(se, api.FinalizerServiceExport) {
		return nil
	}
	key := types.NamespacedName{Namespace: se.Namespace, Name: se.Name}
	if err := r.deleteRecord(ctx, key); err != nil {
		return err
	}
	patch := ctrlclient.MergeFromWithOptions(se.DeepCopy(), ctrlclient.MergeFromWithOptimisticLock{})
	controllerutil.RemoveFinalizer(se, api.FinalizerServiceExport)
	if err := r.opts.Client.Patch(ctx, se, patch); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("remove finalizer: %w", err)
	}
	r.forget(key)
	log.FromContext(ctx).Info("Export withdrawn from the registry", "clusterset", r.opts.Identity.ClusterSet, "cluster", r.opts.Identity.Cluster)
	return nil
}

// goneWithoutFinalizer withdraws an export whose ServiceExport disappeared
// without passing through finalize, for example after its finalizer was
// removed by hand.
func (r *Reconciler) goneWithoutFinalizer(ctx context.Context, key types.NamespacedName) error {
	r.mu.Lock()
	_, known := r.states[key]
	r.mu.Unlock()
	if !known {
		return nil
	}
	if err := r.deleteRecord(ctx, key); err != nil {
		return err
	}
	r.forget(key)
	return nil
}

// SweepUnwatched releases the ServiceExports in namespaces outside the watch
// set that still carry this controller's finalizer, which a narrowed
// --watch-namespaces leaves behind: the caches no longer cover them, so no
// reconcile would. Their registry records go with the next resync, which
// deletes every record without a ServiceExport in a watched namespace. It
// reads the API server directly and does nothing when every namespace is
// watched.
func (r *Reconciler) SweepUnwatched(ctx context.Context) error {
	if !r.opts.Restricted {
		return nil
	}
	var list mcsv1beta1.ServiceExportList
	if err := r.opts.APIReader.List(ctx, &list); err != nil {
		return fmt.Errorf("list ServiceExports of every namespace: %w", err)
	}
	logger := log.FromContext(ctx)
	for i := range list.Items {
		se := &list.Items[i]
		if r.opts.WatchesNamespace(se.Namespace) || !controllerutil.ContainsFinalizer(se, api.FinalizerServiceExport) {
			continue
		}
		patch := ctrlclient.MergeFromWithOptions(se.DeepCopy(), ctrlclient.MergeFromWithOptimisticLock{})
		controllerutil.RemoveFinalizer(se, api.FinalizerServiceExport)
		if err := r.opts.Client.Patch(ctx, se, patch); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("remove the finalizer of %s/%s outside the watched namespaces: %w", se.Namespace, se.Name, err)
		}
		logger.Info("Released a ServiceExport in a namespace no longer watched", "namespace", se.Namespace, "name", se.Name)
	}
	return nil
}

// deleteRecord deletes the registry export record of key; deleting a missing
// record succeeds.
func (r *Reconciler) deleteRecord(ctx context.Context, key types.NamespacedName) (err error) {
	id := r.opts.Identity
	ctx, span := tracing.Start(ctx, "delete_service_export", id.ClusterSet, id.Cluster, key.Namespace, key.Name)
	defer func() { tracing.End(span, err) }()
	_, err = r.opts.Registry.ServiceExports().Delete(ctx, &mcsv1.DeleteServiceExportRequest{
		Clusterset: id.ClusterSet, Cluster: id.Cluster, Namespace: key.Namespace, Name: key.Name,
	})
	if status.Code(err) == codes.NotFound {
		err = nil
	}
	result := metrics.WriteSuccess
	if err != nil {
		result = metrics.WriteError
		err = fmt.Errorf("delete export %s from the registry: %w", key, err)
	}
	r.opts.Metrics.ExportWrites.WithLabelValues(metrics.OpDeleteExport, result).Inc()
	return err
}

// eventAction is the action of every Event the reconciler emits.
const eventAction = "Export"

func (r *Reconciler) event(se *mcsv1beta1.ServiceExport, typ, reason, msg string) {
	if r.opts.Recorder != nil {
		r.opts.Recorder.Eventf(se, nil, typ, reason, eventAction, "%s", msg)
	}
}
