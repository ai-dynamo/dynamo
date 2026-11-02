// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package importer materialises the clusterset's ServiceImports from the
// registry. Importer is a leader-only runnable holding one
// WatchServiceImports stream with endpoints: SNAPSHOT_START clears its map of
// imports, UPSERT and DELETE update it, and SNAPSHOT_END garbage-collects the
// managed objects of imports that are gone. Every change enqueues the import
// into a controller whose Reconcile applies, with server-side apply and field
// manager mcs-controller, the ServiceImport, the derived Service and the
// EndpointSlices of every source cluster, or deletes them. An import whose
// namespace does not exist is held pending until a Namespace event, or the
// namespace is created when CreateNamespaces is set.
//
// Reconciliation is level-based: the hash of the last applied import is a
// short-circuit for an unchanged snapshot only. Any change to a managed
// object that the controller did not write itself, a deleted managed
// object, a namespace event and the periodic resync (ResyncPeriod) forget
// the hash, so the next reconcile applies the import again.
package importer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"net"
	"slices"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

// ControllerName names the import controller.
const ControllerName = "serviceimport"

// Options configure an Importer.
type Options struct {
	// Client is the manager's client.
	Client ctrlclient.Client
	// APIReader reads from the API server, bypassing the cache, where the
	// cache may lag behind the controller's own writes; nil uses Client.
	APIReader ctrlclient.Reader
	Registry  *client.Client
	Identity  identity.Identity
	Metrics   *metrics.Metrics
	Logger    logr.Logger
	// Recorder emits Events on ServiceImports; nil emits none.
	Recorder events.EventRecorder
	// EndpointsPerSlice bounds imported EndpointSlices; default 100.
	EndpointsPerSlice int
	// CreateNamespaces creates a missing namespace instead of holding the
	// import pending.
	CreateNamespaces bool
	// AdoptUnmanaged takes over ServiceImports another implementation
	// created.
	AdoptUnmanaged bool
	// OmitTargetRefs leaves targetRef off imported endpoints, for
	// installations whose service mesh resolves target references locally.
	OmitTargetRefs bool
	// ClusterIPFamilies are the IP families of the importing cluster's
	// Service network (DetectClusterIPFamilies); derived Services list only
	// families the cluster has. Nil lists none and lets the API server
	// choose.
	ClusterIPFamilies []corev1.IPFamily
	// WatchNamespaces restricts the imports applied; empty means all.
	WatchNamespaces []string
	// MaxConcurrentReconciles bounds the workers; default 1.
	MaxConcurrentReconciles int
	// OwnStatus receives this cluster's status of every import it exports,
	// with the generation of this cluster's endpoint set in the import, and
	// nil once the registry no longer reports it. The ServiceExport
	// reconciler copies the conditions onto the ServiceExport.
	OwnStatus func(namespace, name string, st *mcsv1.ClusterStatus, endpointsGeneration uint64)
	// IdleTimeout reopens a watch that delivered nothing for this long;
	// default client.DefaultWatchIdleTimeout.
	IdleTimeout time.Duration
	// ResyncPeriod is the interval at which every import is applied again;
	// 0 disables the periodic resync.
	ResyncPeriod time.Duration
}

// Importer keeps the local view of the clusterset's imports and applies it.
type Importer struct {
	opts   Options
	events chan event.TypedGenericEvent[*mcsv1beta1.ServiceImport]

	mu          sync.Mutex
	ctx         context.Context
	imports     map[types.NamespacedName]*mcsv1.ServiceImport
	synced      bool
	pending     map[types.NamespacedName]bool
	ownReported map[types.NamespacedName]bool
	ownSeen     map[types.NamespacedName]bool
	// applied is the hash of the import last applied per key, so that an
	// unchanged import costs no API calls; epochs counts the times each key
	// was forgotten, so that an apply that started before a forget does not
	// record its hash.
	applied map[types.NamespacedName]string
	epochs  map[types.NamespacedName]uint64
	// written is, per key, the resourceVersion the controller last wrote to
	// each of its objects (writtenID): update events carrying one of them
	// are the controller's own writes, not drift. The slice entries also
	// remember which slices the previous apply wrote.
	written map[types.NamespacedName]map[string]string
}

// New returns an Importer. Add it to the manager and call SetupWithManager.
func New(opts Options) *Importer {
	if opts.Metrics == nil {
		opts.Metrics = metrics.New()
	}
	if opts.EndpointsPerSlice < 1 {
		opts.EndpointsPerSlice = 100
	}
	if opts.MaxConcurrentReconciles < 1 {
		opts.MaxConcurrentReconciles = 1
	}
	if opts.IdleTimeout == 0 {
		opts.IdleTimeout = client.DefaultWatchIdleTimeout
	}
	if opts.APIReader == nil {
		opts.APIReader = opts.Client
	}
	return &Importer{
		opts:        opts,
		events:      make(chan event.TypedGenericEvent[*mcsv1beta1.ServiceImport], 4096),
		ctx:         context.Background(),
		imports:     map[types.NamespacedName]*mcsv1.ServiceImport{},
		pending:     map[types.NamespacedName]bool{},
		ownReported: map[types.NamespacedName]bool{},
		applied:     map[types.NamespacedName]string{},
		epochs:      map[types.NamespacedName]uint64{},
		written:     map[types.NamespacedName]map[string]string{},
	}
}

// appliedHash returns the hash last applied for key, or empty, and the
// key's epoch, which setApplied needs.
func (i *Importer) appliedHash(key types.NamespacedName) (string, uint64) {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.applied[key], i.epochs[key]
}

// setApplied records the hash of an applied import unless the key was
// forgotten since epoch was read.
func (i *Importer) setApplied(key types.NamespacedName, hash string, epoch uint64) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.epochs[key] == epoch {
		i.applied[key] = hash
	}
}

// forgetApplied makes the next reconcile of key apply its import again,
// and an apply of key already running not record its hash.
func (i *Importer) forgetApplied(key types.NamespacedName) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.forgetLocked(key)
}

func (i *Importer) forgetLocked(key types.NamespacedName) {
	delete(i.applied, key)
	i.epochs[key]++
}

// Identifiers of the objects of one import in written.
const (
	writtenImport  = "import"
	writtenService = "service"
	writtenSlice   = "slice/"
)

// recordWritten remembers that the controller wrote resourceVersion rv to the
// object id of key.
func (i *Importer) recordWritten(key types.NamespacedName, id, rv string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	m := i.written[key]
	if m == nil {
		m = map[string]string{}
		i.written[key] = m
	}
	m[id] = rv
}

// forgetWritten drops the record of object id of key, or of every object
// when id is empty.
func (i *Importer) forgetWritten(key types.NamespacedName, id string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if id == "" {
		delete(i.written, key)
		return
	}
	delete(i.written[key], id)
}

// wrote reports whether rv is the resourceVersion the controller last wrote to
// object id of key.
func (i *Importer) wrote(key types.NamespacedName, id, rv string) bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return rv != "" && i.written[key][id] == rv
}

// writtenSlices returns the names of the EndpointSlices the previous apply
// of key wrote.
func (i *Importer) writtenSlices(key types.NamespacedName) []string {
	i.mu.Lock()
	defer i.mu.Unlock()
	var names []string
	for id := range i.written[key] {
		if name, ok := cutPrefix(id, writtenSlice); ok {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	return names
}

func cutPrefix(s, prefix string) (string, bool) {
	if len(s) >= len(prefix) && s[:len(prefix)] == prefix {
		return s[len(prefix):], true
	}
	return "", false
}

// importHash is a stable hash of the parts of an import the controller
// renders: the merged spec, labels, annotations, conflict, the participating
// clusters and the endpoints of each source cluster. Timestamps and
// generations are left out: a resync in one exporting cluster bumps them
// without changing anything the importer writes.
func importHash(imp *mcsv1.ServiceImport) string {
	b, err := proto.MarshalOptions{Deterministic: true}.Marshal(renderedImport(imp))
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// renderedImport returns a copy of imp reduced to the fields the controller
// renders into Kubernetes objects.
func renderedImport(imp *mcsv1.ServiceImport) *mcsv1.ServiceImport {
	out := &mcsv1.ServiceImport{
		Clusterset:            imp.GetClusterset(),
		Namespace:             imp.GetNamespace(),
		Name:                  imp.GetName(),
		Type:                  imp.GetType(),
		Ports:                 imp.GetPorts(),
		IpFamilies:            imp.GetIpFamilies(),
		SessionAffinity:       imp.GetSessionAffinity(),
		SessionAffinityConfig: imp.GetSessionAffinityConfig(),
		Annotations:           imp.GetAnnotations(),
		Labels:                imp.GetLabels(),
		Conflict:              imp.GetConflict(),
	}
	for _, c := range imp.GetClusters() {
		out.Clusters = append(out.Clusters, &mcsv1.ClusterStatus{Cluster: c.GetCluster(), Participating: c.GetParticipating()})
	}
	for _, ce := range imp.GetEndpoints() {
		out.Endpoints = append(out.Endpoints, &mcsv1.ClusterEndpoints{Cluster: ce.GetCluster(), Region: ce.GetRegion(), Endpoints: ce.GetEndpoints()})
	}
	return out
}

// NeedLeaderElection makes the importer leader-only.
func (i *Importer) NeedLeaderElection() bool { return true }

// handles reports whether imports of namespace ns are applied.
func (i *Importer) handles(ns string) bool {
	return len(i.opts.WatchNamespaces) == 0 || slices.Contains(i.opts.WatchNamespaces, ns)
}

// Start sweeps the namespaces no longer watched, starts the periodic resync
// and runs the watch until ctx is done.
func (i *Importer) Start(ctx context.Context) error {
	i.mu.Lock()
	i.ctx = ctx
	i.mu.Unlock()
	if err := i.SweepUnwatched(ctx); err != nil {
		// The managed objects outside the watched namespaces stay until the
		// next leader start; the imports themselves are not affected.
		i.opts.Logger.Info("Sweep of namespaces no longer watched failed", "error", err.Error())
	}
	if i.opts.ResyncPeriod > 0 {
		go i.periodicResync(ctx)
	}
	req := &mcsv1.WatchServiceImportsRequest{Clusterset: i.opts.Identity.ClusterSet, IncludeEndpoints: true}
	if len(i.opts.WatchNamespaces) == 1 {
		req.Namespace = i.opts.WatchNamespaces[0]
	}
	loop := client.NewWatchLoop(i.opts.Registry, client.WatchLoopConfig[*mcsv1.ServiceImport]{
		Name:  client.WatchServiceImports,
		Watch: i.opts.Registry.ServiceImports().Watcher(req),
		Handler: client.WatchHandler[*mcsv1.ServiceImport]{
			SnapshotStart: i.SnapshotStart,
			Upsert:        i.Upsert,
			Delete:        i.Delete,
			Synced:        i.Synced,
			Bookmark:      func(string) { i.opts.Metrics.ImportEvents.WithLabelValues("bookmark").Inc() },
		},
		OnResync: func() {
			i.opts.Metrics.ResyncAttempts.WithLabelValues(metrics.ResyncSnapshot).Inc()
			i.opts.Metrics.Resyncs.WithLabelValues(metrics.ResyncSnapshot).Inc()
			i.opts.Logger.Info("Registry import watch restarted with a new snapshot")
		},
		IdleTimeout: i.opts.IdleTimeout,
	})
	i.opts.Logger.Info("Watching registry service imports", "clusterset", i.opts.Identity.ClusterSet, "cluster", i.opts.Identity.Cluster)
	return loop.Run(ctx)
}

// periodicResync runs Resync every ResyncPeriod until ctx is done.
func (i *Importer) periodicResync(ctx context.Context) {
	ticker := time.NewTicker(i.opts.ResyncPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			i.Resync()
		}
	}
}

// Resync forgets every applied hash and reconciles every import, so that
// drift the watches did not report is repaired within one period.
func (i *Importer) Resync() {
	i.opts.Metrics.ResyncAttempts.WithLabelValues(metrics.ResyncImportPeriodic).Inc()
	i.mu.Lock()
	keys := slices.Collect(maps.Keys(i.imports))
	for _, key := range keys {
		i.forgetLocked(key)
	}
	i.mu.Unlock()
	for _, key := range keys {
		i.enqueue(key)
	}
	i.opts.Metrics.Resyncs.WithLabelValues(metrics.ResyncImportPeriodic).Inc()
	i.opts.Logger.V(1).Info("Imports enqueued for the periodic resync", "imports", len(keys))
}

// SweepUnwatched deletes the managed ServiceImports, derived Services and
// EndpointSlices in namespaces the controller no longer watches, which a
// narrowed --watch-namespaces leaves behind: the caches no longer cover
// them, so no reconcile would. It reads the API server directly and does
// nothing when every namespace is watched.
func (i *Importer) SweepUnwatched(ctx context.Context) error {
	if len(i.opts.WatchNamespaces) == 0 {
		return nil
	}
	managed := ctrlclient.MatchingLabels{api.LabelManagedBy: api.ManagedByController}
	var imports mcsv1beta1.ServiceImportList
	if err := i.opts.APIReader.List(ctx, &imports, managed); err != nil {
		return fmt.Errorf("list managed ServiceImports of every namespace: %w", err)
	}
	var svcs corev1.ServiceList
	if err := i.opts.APIReader.List(ctx, &svcs, managed, ctrlclient.HasLabels{api.LabelMCSServiceName}); err != nil {
		return fmt.Errorf("list derived Services of every namespace: %w", err)
	}
	var eps discoveryv1.EndpointSliceList
	if err := i.opts.APIReader.List(ctx, &eps, ctrlclient.MatchingLabels{api.LabelEndpointSliceManagedBy: api.EndpointSliceManagedBy}); err != nil {
		return fmt.Errorf("list imported EndpointSlices of every namespace: %w", err)
	}
	var objs []ctrlclient.Object
	for idx := range imports.Items {
		objs = append(objs, &imports.Items[idx])
	}
	for idx := range svcs.Items {
		objs = append(objs, &svcs.Items[idx])
	}
	for idx := range eps.Items {
		objs = append(objs, &eps.Items[idx])
	}
	for _, o := range objs {
		if i.handles(o.GetNamespace()) {
			continue
		}
		if err := i.opts.Client.Delete(ctx, o); ctrlclient.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete %s/%s outside the watched namespaces: %w", o.GetNamespace(), o.GetName(), err)
		}
		i.opts.Logger.Info("Deleted a managed object in a namespace no longer watched", "namespace", o.GetNamespace(), "name", o.GetName())
	}
	return nil
}

// DetectClusterIPFamilies returns the IP families of the cluster's Service
// network, read from the kubernetes Service of the default namespace,
// which the API server creates with its own ClusterIPs.
func DetectClusterIPFamilies(ctx context.Context, reader ctrlclient.Reader) ([]corev1.IPFamily, error) {
	var svc corev1.Service
	if err := reader.Get(ctx, types.NamespacedName{Namespace: metav1NamespaceDefault, Name: "kubernetes"}, &svc); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read the kubernetes Service: %w", err)
	}
	if len(svc.Spec.IPFamilies) > 0 {
		return slices.Clone(svc.Spec.IPFamilies), nil
	}
	var fams []corev1.IPFamily
	for _, ip := range svc.Spec.ClusterIPs {
		switch parsed := net.ParseIP(ip); {
		case parsed == nil:
		case parsed.To4() != nil:
			fams = append(fams, corev1.IPv4Protocol)
		default:
			fams = append(fams, corev1.IPv6Protocol)
		}
	}
	return fams, nil
}

const metav1NamespaceDefault = "default"

func keyOf(imp *mcsv1.ServiceImport) types.NamespacedName {
	return types.NamespacedName{Namespace: imp.GetNamespace(), Name: imp.GetName()}
}

// SnapshotStart handles SNAPSHOT_START: the following upserts until
// SNAPSHOT_END are the complete set of imports.
func (i *Importer) SnapshotStart() {
	i.opts.Metrics.ImportEvents.WithLabelValues("snapshot_start").Inc()
	i.mu.Lock()
	defer i.mu.Unlock()
	i.imports = map[types.NamespacedName]*mcsv1.ServiceImport{}
	i.synced = false
	i.ownSeen = map[types.NamespacedName]bool{}
	i.opts.Metrics.Imports.Set(0)
}

// Upsert handles a new or changed import.
func (i *Importer) Upsert(imp *mcsv1.ServiceImport) {
	i.opts.Metrics.ImportEvents.WithLabelValues("upsert").Inc()
	key := keyOf(imp)
	own := ownStatus(imp, i.opts.Identity.Cluster)
	i.mu.Lock()
	report := own != nil || i.ownReported[key]
	if own != nil {
		i.ownReported[key] = true
		if i.ownSeen != nil {
			i.ownSeen[key] = true
		}
	} else {
		delete(i.ownReported, key)
	}
	stored := i.handles(key.Namespace)
	if stored {
		i.imports[key] = imp
	}
	i.opts.Metrics.Imports.Set(float64(len(i.imports)))
	i.mu.Unlock()
	if report && i.opts.OwnStatus != nil {
		i.opts.OwnStatus(key.Namespace, key.Name, own, ownEndpointsGeneration(imp, i.opts.Identity.Cluster))
	}
	if stored {
		i.enqueue(key)
	}
}

// Delete handles a removed import.
func (i *Importer) Delete(imp *mcsv1.ServiceImport) {
	i.opts.Metrics.ImportEvents.WithLabelValues("delete").Inc()
	key := keyOf(imp)
	i.mu.Lock()
	report := i.ownReported[key]
	delete(i.ownReported, key)
	delete(i.imports, key)
	i.opts.Metrics.Imports.Set(float64(len(i.imports)))
	i.mu.Unlock()
	if report && i.opts.OwnStatus != nil {
		i.opts.OwnStatus(key.Namespace, key.Name, nil, 0)
	}
	if i.handles(key.Namespace) {
		i.enqueue(key)
	}
}

// Synced handles SNAPSHOT_END: own statuses missing from the snapshot are
// cleared, pending imports missing from it are dropped, and the managed
// objects of imports missing from it are deleted.
func (i *Importer) Synced(string) {
	i.opts.Metrics.ImportEvents.WithLabelValues("snapshot_end").Inc()
	i.mu.Lock()
	i.synced = true
	var cleared []types.NamespacedName
	if i.ownSeen != nil {
		for key := range i.ownReported {
			if !i.ownSeen[key] {
				cleared = append(cleared, key)
				delete(i.ownReported, key)
			}
		}
		i.ownSeen = nil
	}
	// A pending import deleted from the registry while the watch was down
	// has no managed objects for collectGarbage to find.
	for key := range i.pending {
		if _, ok := i.imports[key]; !ok {
			delete(i.pending, key)
		}
	}
	i.opts.Metrics.ImportsPendingNamespace.Set(float64(len(i.pending)))
	ctx := i.ctx
	i.mu.Unlock()
	if i.opts.OwnStatus != nil {
		for _, key := range cleared {
			i.opts.OwnStatus(key.Namespace, key.Name, nil, 0)
		}
	}
	if err := i.collectGarbage(ctx); err != nil {
		i.opts.Logger.Info("Garbage collection of imports failed; retrying after the next snapshot", "error", err.Error())
	}
}

// collectGarbage enqueues every import with managed objects in the cluster
// but none in the registry; Reconcile deletes them.
func (i *Importer) collectGarbage(ctx context.Context) error {
	keys := map[types.NamespacedName]bool{}
	var imports mcsv1beta1.ServiceImportList
	if err := i.opts.Client.List(ctx, &imports, ctrlclient.MatchingLabels{api.LabelManagedBy: api.ManagedByController}); err != nil {
		return err
	}
	for _, si := range imports.Items {
		keys[types.NamespacedName{Namespace: si.Namespace, Name: si.Name}] = true
	}
	var svcs corev1.ServiceList
	if err := i.opts.Client.List(ctx, &svcs, ctrlclient.MatchingLabels{api.LabelManagedBy: api.ManagedByController},
		ctrlclient.HasLabels{api.LabelMCSServiceName}); err != nil {
		return err
	}
	for _, s := range svcs.Items {
		keys[types.NamespacedName{Namespace: s.Namespace, Name: s.Labels[api.LabelMCSServiceName]}] = true
	}
	var eps discoveryv1.EndpointSliceList
	if err := i.opts.Client.List(ctx, &eps, ctrlclient.MatchingLabels{api.LabelEndpointSliceManagedBy: api.EndpointSliceManagedBy},
		ctrlclient.HasLabels{api.LabelMCSServiceName}); err != nil {
		return err
	}
	for _, s := range eps.Items {
		keys[types.NamespacedName{Namespace: s.Namespace, Name: s.Labels[api.LabelMCSServiceName]}] = true
	}
	i.mu.Lock()
	var gone []types.NamespacedName
	for key := range keys {
		if _, ok := i.imports[key]; !ok && i.handles(key.Namespace) {
			gone = append(gone, key)
		}
	}
	i.mu.Unlock()
	for _, key := range gone {
		i.enqueue(key)
	}
	return nil
}

// enqueue reconciles key, waiting while the controller's queue is full.
func (i *Importer) enqueue(key types.NamespacedName) {
	ev := event.TypedGenericEvent[*mcsv1beta1.ServiceImport]{
		Object: &mcsv1beta1.ServiceImport{Namespace: key.Namespace, Name: key.Name},
	}
	i.mu.Lock()
	ctx := i.ctx
	i.mu.Unlock()
	select {
	case i.events <- ev:
	case <-ctx.Done():
	}
}

// lookup returns the import of key and whether the view is complete.
func (i *Importer) lookup(key types.NamespacedName) (imp *mcsv1.ServiceImport, ok, synced bool) {
	i.mu.Lock()
	defer i.mu.Unlock()
	imp, ok = i.imports[key]
	return imp, ok, i.synced
}

// keysIn returns the imports of namespace ns.
func (i *Importer) keysIn(ns string) []types.NamespacedName {
	i.mu.Lock()
	defer i.mu.Unlock()
	var keys []types.NamespacedName
	for key := range i.imports {
		if key.Namespace == ns {
			keys = append(keys, key)
		}
	}
	return keys
}

// setPending records whether key waits for its namespace.
func (i *Importer) setPending(key types.NamespacedName, pending bool) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if pending {
		i.pending[key] = true
	} else {
		delete(i.pending, key)
	}
	i.opts.Metrics.ImportsPendingNamespace.Set(float64(len(i.pending)))
}

// Pending returns the imports waiting for their namespace.
func (i *Importer) Pending() []types.NamespacedName {
	i.mu.Lock()
	defer i.mu.Unlock()
	return slices.Collect(maps.Keys(i.pending))
}

// ownStatus returns the status of cluster in imp, or nil.
func ownStatus(imp *mcsv1.ServiceImport, cluster string) *mcsv1.ClusterStatus {
	for _, c := range imp.GetClusters() {
		if c.GetCluster() == cluster {
			return c
		}
	}
	return nil
}

// ownEndpointsGeneration returns the generation of the endpoint set cluster
// contributes to imp, or 0 when it contributes none.
func ownEndpointsGeneration(imp *mcsv1.ServiceImport, cluster string) uint64 {
	for _, ce := range imp.GetEndpoints() {
		if ce.GetCluster() == cluster {
			return ce.GetGeneration()
		}
	}
	return 0
}
