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
package importer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
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
	// APIReader reads from the API server, bypassing the cache, for the
	// ClusterIPs of a derived Service just created; nil uses Client.
	APIReader ctrlclient.Reader
	Registry  *client.Client
	Identity  identity.Identity
	Metrics   *metrics.Metrics
	Logger    logr.Logger
	// EndpointsPerSlice bounds imported EndpointSlices; default 100.
	EndpointsPerSlice int
	// CreateNamespaces creates a missing namespace instead of holding the
	// import pending.
	CreateNamespaces bool
	// AdoptUnmanaged takes over ServiceImports another implementation
	// created.
	AdoptUnmanaged bool
	// WatchNamespaces restricts the imports applied; empty means all.
	WatchNamespaces []string
	// MaxConcurrentReconciles bounds the workers; default 1.
	MaxConcurrentReconciles int
	// OwnStatus receives this cluster's status of every import it exports, and
	// nil once the registry no longer reports it. The ServiceExport reconciler
	// copies the conditions onto the ServiceExport.
	OwnStatus func(namespace, name string, st *mcsv1.ClusterStatus)
	// IdleTimeout reopens a watch that delivered nothing for this long;
	// default client.DefaultWatchIdleTimeout.
	IdleTimeout time.Duration
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
	// unchanged import costs no API calls.
	applied map[types.NamespacedName]string
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
	}
}

func (i *Importer) appliedHash(key types.NamespacedName) string {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.applied[key]
}

// setApplied records the hash of an applied import; an empty hash forgets
// it.
func (i *Importer) setApplied(key types.NamespacedName, hash string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if hash == "" {
		delete(i.applied, key)
		return
	}
	i.applied[key] = hash
}

func (i *Importer) forgetApplied(key types.NamespacedName) { i.setApplied(key, "") }

// importHash is a stable hash of an import as the registry sent it.
func importHash(imp *mcsv1.ServiceImport) string {
	b, err := proto.MarshalOptions{Deterministic: true}.Marshal(imp)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// NeedLeaderElection makes the importer leader-only.
func (i *Importer) NeedLeaderElection() bool { return true }

// handles reports whether imports of namespace ns are applied.
func (i *Importer) handles(ns string) bool {
	return len(i.opts.WatchNamespaces) == 0 || slices.Contains(i.opts.WatchNamespaces, ns)
}

// Start runs the watch until ctx is done.
func (i *Importer) Start(ctx context.Context) error {
	i.mu.Lock()
	i.ctx = ctx
	i.mu.Unlock()
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
			i.opts.Metrics.Resyncs.WithLabelValues(metrics.ResyncSnapshot).Inc()
			i.opts.Logger.Info("Registry import watch restarted with a new snapshot")
		},
		IdleTimeout: i.opts.IdleTimeout,
	})
	i.opts.Logger.Info("Watching registry service imports", "clusterset", i.opts.Identity.ClusterSet, "cluster", i.opts.Identity.Cluster)
	return loop.Run(ctx)
}

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
		i.opts.OwnStatus(key.Namespace, key.Name, own)
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
		i.opts.OwnStatus(key.Namespace, key.Name, nil)
	}
	if i.handles(key.Namespace) {
		i.enqueue(key)
	}
}

// Synced handles SNAPSHOT_END: own statuses missing from the snapshot are
// cleared and the managed objects of imports missing from it are deleted.
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
	ctx := i.ctx
	i.mu.Unlock()
	if i.opts.OwnStatus != nil {
		for _, key := range cleared {
			i.opts.OwnStatus(key.Namespace, key.Name, nil)
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
