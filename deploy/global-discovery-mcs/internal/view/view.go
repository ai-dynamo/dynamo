// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package view is the in-memory materialised view of every clusterset a
// mcs-registry replica serves, and the watch hub on top of it.
//
// The view holds the raw store records of each clusterset and derives from
// them the objects the API returns: clusters with their lease state, merged
// ServiceImports (internal/registry/merge). Every change of a derived object
// is numbered with a per-clusterset sequence and recorded in a ring buffer,
// so that watchers can resume (revision "view_epoch:seq").
//
// Records enter the view three ways:
//
//   - Refresher reloads each clusterset from the store every refresh
//     interval (Replace).
//   - API handlers apply their own successful writes immediately
//     (ApplyCluster, ApplyExport, ... : the write-through, or ApplyLocal,
//     family), so a writer reads its write at once on this replica.
//   - EvaluateLeases re-evaluates cluster leases every lease evaluation
//     interval without a store read.
//
// A refresh that started before a local write cannot undo it: writes applied
// locally win over a snapshot for LocalWriteWindow unless the snapshot holds
// a newer version.
package view

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// Kind is the kind of a derived object.
type Kind int

const (
	KindCluster Kind = iota
	KindServiceImport
	numKinds
)

var kindNames = [numKinds]string{metrics.KindCluster, metrics.KindServiceImport}

// String returns the metrics label value of the kind.
func (k Kind) String() string { return kindNames[k] }

// Object is an immutable derived object. Callers must not modify Full or
// Lite; clone them first.
type Object struct {
	Key string
	// Full is the complete message.
	Full proto.Message
	// Lite is Full without endpoint lists (counts are kept). It is Full for
	// kinds without endpoints.
	Lite proto.Message
}

// Message returns Full or Lite.
func (o *Object) Message(includeEndpoints bool) proto.Message {
	if includeEndpoints {
		return o.Full
	}
	return o.Lite
}

// Event is one change of a derived object. Old is nil for a creation, New
// is nil for a removal.
type Event struct {
	Seq  uint64
	Kind Kind
	Key  string
	Old  *Object
	New  *Object
}

// Options configures a View.
type Options struct {
	Clock clock.Clock
	// LeaseSkew is the clock skew allowance before a lease expires.
	LeaseSkew time.Duration
	// WatchHistory is the number of events kept per clusterset for resume.
	WatchHistory int
	// LocalWriteWindow is how long a write applied locally wins over a
	// refresh snapshot that does not hold it; one refresh interval.
	LocalWriteWindow time.Duration
	// Epoch overrides the view epoch; a unique one is minted when empty.
	Epoch   string
	Metrics *metrics.Metrics
	Logger  *zap.Logger
}

// View is the materialised view of every clusterset of one replica. It is
// safe for concurrent use.
type View struct {
	opts  Options
	clock clock.Clock
	log   *zap.Logger
	epoch string

	mu         sync.RWMutex
	sets       map[string]*clusterSet
	discovered bool
	kick       chan struct{}
}

// New returns an empty view.
func New(opts Options) *View {
	if opts.Clock == nil {
		opts.Clock = clock.Real()
	}
	if opts.Logger == nil {
		opts.Logger = zap.NewNop()
	}
	if opts.WatchHistory <= 0 {
		opts.WatchHistory = 10000
	}
	epoch := opts.Epoch
	if epoch == "" {
		epoch = NewEpoch(opts.Clock.Now())
	}
	return &View{
		opts:  opts,
		clock: opts.Clock,
		log:   opts.Logger,
		epoch: epoch,
		sets:  map[string]*clusterSet{},
		kick:  make(chan struct{}, 1),
	}
}

// NewEpoch mints a view epoch: the start time in unix milliseconds and 32
// random bits, unique per replica start.
func NewEpoch(now time.Time) string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return strconv.FormatInt(now.UnixMilli(), 10) + "-" + hex.EncodeToString(b[:])
}

// Epoch returns the view epoch of this replica.
func (v *View) Epoch() string { return v.epoch }

// Revision formats a revision token.
func Revision(epoch string, seq uint64) string {
	return epoch + ":" + strconv.FormatUint(seq, 10)
}

// ParseRevision splits a revision token.
func ParseRevision(rev string) (epoch string, seq uint64, err error) {
	i := strings.LastIndexByte(rev, ':')
	if i <= 0 {
		return "", 0, fmt.Errorf("revision %q: want view_epoch:seq", rev)
	}
	seq, err = strconv.ParseUint(rev[i+1:], 10, 64)
	if err != nil {
		return "", 0, fmt.Errorf("revision %q: %w", rev, err)
	}
	return rev[:i], seq, nil
}

// Kicked is signalled when a clusterset is created by a request, so the
// refresher can load it before its next tick.
func (v *View) Kicked() <-chan struct{} { return v.kick }

// get returns a clusterset or nil.
func (v *View) get(name string) *clusterSet {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.sets[name]
}

// ensure returns a clusterset, creating it when absent.
func (v *View) ensure(name string) *clusterSet {
	if cs := v.get(name); cs != nil {
		return cs
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if cs := v.sets[name]; cs != nil {
		return cs
	}
	cs := newClusterSet(name, v.opts.WatchHistory)
	v.sets[name] = cs
	select {
	case v.kick <- struct{}{}:
	default:
	}
	return cs
}

// Ensure makes the view track a clusterset, for example one a watcher asks
// for before any record exists.
func (v *View) Ensure(name string) { v.ensure(name) }

// ClusterSetNames returns the names of every tracked clusterset, sorted.
func (v *View) ClusterSetNames() []string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	names := make([]string, 0, len(v.sets))
	for n := range v.sets {
		names = append(names, n)
	}
	slices.Sort(names)
	return names
}

// SetDiscovered records the clustersets the store knows. They count for
// readiness. Tracked clustersets the store does not know are dropped when
// they hold no record, no watcher and no recent local write.
func (v *View) SetDiscovered(names []string) {
	known := map[string]bool{}
	for _, n := range names {
		known[n] = true
		v.ensure(n).setDiscovered(true)
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	v.discovered = true
	for name, cs := range v.sets {
		if known[name] {
			continue
		}
		cs.mu.Lock()
		cs.discovered = false
		idle := cs.idle(v.clock.Now(), v.opts.LocalWriteWindow)
		cs.mu.Unlock()
		if idle {
			delete(v.sets, name)
			if v.opts.Metrics != nil {
				v.opts.Metrics.DeleteClusterSet(name)
			}
		}
	}
}

// Discovered reports whether SetDiscovered ran at least once.
func (v *View) Discovered() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.discovered
}

// Ready reports whether the view is fresh enough to serve: clustersets were
// discovered at least once, and every clusterset the store knows was
// refreshed within maxAge. The string explains a false result.
func (v *View) Ready(now time.Time, maxAge time.Duration) (bool, string) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if !v.discovered {
		return false, "clustersets not discovered yet"
	}
	for _, name := range slices.Sorted(maps.Keys(v.sets)) {
		cs := v.sets[name]
		cs.mu.Lock()
		discovered, refreshed, last := cs.discovered, cs.refreshed, cs.lastRefresh
		cs.mu.Unlock()
		if !discovered {
			continue
		}
		if !refreshed {
			return false, fmt.Sprintf("clusterset %q not loaded yet", name)
		}
		if age := now.Sub(last); age > maxAge {
			return false, fmt.Sprintf("clusterset %q view is %s old", name, age.Round(time.Millisecond))
		}
	}
	return true, ""
}

// Age returns the age of the oldest view of a discovered clusterset, or 0.
func (v *View) Age(now time.Time) time.Duration {
	v.mu.RLock()
	defer v.mu.RUnlock()
	var oldest time.Duration
	for _, cs := range v.sets {
		cs.mu.Lock()
		if cs.discovered && cs.refreshed {
			oldest = max(oldest, now.Sub(cs.lastRefresh))
		}
		cs.mu.Unlock()
	}
	return oldest
}

// Replace replaces the records of a clusterset with a snapshot loaded from
// the store. loadStart is the time the load began; local writes newer than
// loadStart minus LocalWriteWindow are kept unless the snapshot holds a
// newer version. A snapshot whose load began before the one applied last is
// ignored, so concurrent loads cannot move the view backwards.
func (v *View) Replace(snap store.ClusterSetSnapshot, loadStart time.Time) {
	cs := v.ensure(snap.ClusterSet)
	now := v.clock.Now()
	cs.mu.Lock()
	defer cs.mu.Unlock()
	if loadStart.Before(cs.lastLoadStart) {
		return
	}
	cs.lastLoadStart = loadStart
	cs.replace(snap, loadStart.Add(-v.opts.LocalWriteWindow))
	cs.refreshed = true
	cs.lastRefresh = now
	v.recompute(cs, now)
}

// EvaluateLeases re-evaluates every lease without a store read and
// publishes the resulting changes.
func (v *View) EvaluateLeases() {
	now := v.clock.Now()
	for _, name := range v.ClusterSetNames() {
		cs := v.get(name)
		if cs == nil {
			continue
		}
		cs.mu.Lock()
		if cs.leasesChanged(now, v.opts.LeaseSkew) {
			v.recompute(cs, now)
		}
		cs.mu.Unlock()
	}
	if v.opts.Metrics != nil {
		v.opts.Metrics.ViewAge.Set(v.Age(now).Seconds())
	}
}

// Records returns a copy of the raw records of a clusterset.
func (v *View) Records(clusterSet string) store.ClusterSetSnapshot {
	snap := store.ClusterSetSnapshot{ClusterSet: clusterSet}
	cs := v.get(clusterSet)
	if cs == nil {
		return snap
	}
	cs.mu.Lock()
	defer cs.mu.Unlock()
	for _, r := range cs.clusters {
		snap.Clusters = append(snap.Clusters, r.Clone())
	}
	for _, r := range cs.exports {
		snap.Exports = append(snap.Exports, r.Clone())
	}
	for _, r := range cs.endpoints {
		snap.EndpointSets = append(snap.EndpointSets, r.Clone())
	}
	snap.Sort()
	return snap
}

// LeaseExpired evaluates a lease at now with the view's skew allowance.
func (v *View) LeaseExpired(l store.Lease, now time.Time) bool {
	return registry.LeaseExpired(l.RenewedAt, l.TTL, now, v.opts.LeaseSkew)
}

// Now returns the view's clock time.
func (v *View) Now() time.Time { return v.clock.Now() }
