// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
)

var (
	// ErrTooManyWatchers is returned by Watch when the replica serves
	// MaxWatchers streams already.
	ErrTooManyWatchers = errors.New("view: too many watchers")
	// ErrShuttingDown is returned by Watch when the hub is closed.
	ErrShuttingDown = errors.New("view: shutting down")
)

// ring is a fixed-size buffer of the latest events of a clusterset.
type ring struct {
	buf   []Event
	start int
	n     int
}

func newRing(size int) *ring { return &ring{buf: make([]Event, size)} }

func (r *ring) push(ev Event) {
	if len(r.buf) == 0 {
		return
	}
	if r.n < len(r.buf) {
		r.buf[(r.start+r.n)%len(r.buf)] = ev
		r.n++
		return
	}
	r.buf[r.start] = ev
	r.start = (r.start + 1) % len(r.buf)
}

// covers reports whether every event after seq up to current is buffered.
func (r *ring) covers(seq, current uint64) bool {
	if seq >= current {
		return seq == current
	}
	return r.n > 0 && r.buf[r.start].Seq <= seq+1
}

// since returns the buffered events after seq, oldest first.
func (r *ring) since(seq uint64) []Event {
	var out []Event
	for i := range r.n {
		ev := r.buf[(r.start+i)%len(r.buf)]
		if ev.Seq > seq {
			out = append(out, ev)
		}
	}
	return out
}

// subscriber is one watch stream of a clusterset. queue, lagged and pos are
// guarded by the clusterset's mu.
type subscriber struct {
	filter Filter
	limit  int
	queue  []Event
	lagged bool
	// pos is the sequence of the last event taken from the queue or of
	// the last snapshot.
	pos    uint64
	notify chan struct{}
}

func (s *subscriber) signal() {
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

// relevant reports whether an event concerns the subscriber's filter: the
// object matches before or after the change.
func (s *subscriber) relevant(ev Event) bool {
	if ev.Kind != s.filter.Kind() {
		return false
	}
	return (ev.New != nil && s.filter.Match(ev.New.Full)) || (ev.Old != nil && s.filter.Match(ev.Old.Full))
}

// publish numbers an event, records it and queues it for every subscriber it
// concerns. A subscriber whose queue is full is marked lagged and will be
// re-snapshotted. cs.mu must be held.
func (v *View) publish(cs *clusterSet, kind Kind, key string, old, new *Object) {
	cs.seq++
	ev := Event{Seq: cs.seq, Kind: kind, Key: key, Old: old, New: new}
	cs.ring.push(ev)
	for s := range cs.subs {
		if s.lagged || !s.relevant(ev) {
			continue
		}
		if len(s.queue) >= s.limit {
			s.lagged = true
			s.queue = nil
		} else {
			s.queue = append(s.queue, ev)
		}
		s.signal()
	}
}

// WatchEvent is one event sent to a watcher.
type WatchEvent struct {
	Type mcsv1.EventType
	// Object is set for UPSERT and DELETE.
	Object proto.Message
	// Revision is set for SNAPSHOT_END, BOOKMARK and deltas.
	Revision string
}

// HubOptions configures a Hub.
type HubOptions struct {
	Clock clock.Clock
	// BookmarkInterval is the interval of BOOKMARK events.
	BookmarkInterval time.Duration
	// QueueSize is the per-watcher event queue; an overflowing watcher is
	// re-snapshotted.
	QueueSize int
	// MaxWatchers caps concurrent watch streams.
	MaxWatchers int
	Metrics     *metrics.Metrics
	Logger      *zap.Logger
}

// Hub serves watches over a View.
type Hub struct {
	view *View
	opts HubOptions

	mu       sync.Mutex
	watchers int
	closed   bool
	closing  chan struct{}
}

// NewHub returns a hub over v.
func NewHub(v *View, opts HubOptions) *Hub {
	if opts.Clock == nil {
		opts.Clock = v.clock
	}
	if opts.Logger == nil {
		opts.Logger = zap.NewNop()
	}
	if opts.BookmarkInterval <= 0 {
		opts.BookmarkInterval = 5 * time.Second
	}
	if opts.QueueSize <= 0 {
		opts.QueueSize = 1024
	}
	if opts.MaxWatchers <= 0 {
		opts.MaxWatchers = 5000
	}
	return &Hub{view: v, opts: opts, closing: make(chan struct{})}
}

// Watchers returns the number of open watch streams.
func (h *Hub) Watchers() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.watchers
}

// Close ends every watch with ErrShuttingDown and refuses new ones. It is
// idempotent.
func (h *Hub) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.closed {
		h.closed = true
		close(h.closing)
	}
}

func (h *Hub) acquire(kind Kind) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	m := h.opts.Metrics
	switch {
	case h.closed:
		if m != nil {
			m.WatchRejected.WithLabelValues(metrics.RejectShuttingDown).Inc()
		}
		return ErrShuttingDown
	case h.watchers >= h.opts.MaxWatchers:
		if m != nil {
			m.WatchRejected.WithLabelValues(metrics.RejectTooManyWatchers).Inc()
		}
		return ErrTooManyWatchers
	}
	h.watchers++
	if m != nil {
		m.Watchers.WithLabelValues(kind.String()).Inc()
	}
	return nil
}

func (h *Hub) release(kind Kind) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.watchers--
	if h.opts.Metrics != nil {
		h.opts.Metrics.Watchers.WithLabelValues(kind.String()).Dec()
	}
}

// Watch streams the objects of clusterSet that match f to send until ctx is
// done, send fails or the hub closes: a snapshot (SNAPSHOT_START, one UPSERT
// per object, SNAPSHOT_END) unless resume names a revision this replica can
// replay from, then deltas and a BOOKMARK every bookmark interval. A watcher
// that falls QueueSize events behind gets a new snapshot. Watch returns
// ErrTooManyWatchers, ErrShuttingDown, ctx.Err() or the error of send.
func (h *Hub) Watch(ctx context.Context, clusterSet string, f Filter, resume string, send func(WatchEvent) error) error {
	kind := f.Kind()
	if err := h.acquire(kind); err != nil {
		return err
	}
	defer h.release(kind)

	cs := h.view.ensure(clusterSet)
	sub := &subscriber{filter: f, limit: h.opts.QueueSize, notify: make(chan struct{}, 1)}
	cs.mu.Lock()
	initial, reason := h.start(cs, sub, resume)
	cs.subs[sub] = struct{}{}
	cs.mu.Unlock()
	defer func() {
		cs.mu.Lock()
		delete(cs.subs, sub)
		cs.mu.Unlock()
	}()
	if reason != "" {
		h.resync(kind, reason)
	}
	if err := h.sendAll(kind, send, initial); err != nil {
		return err
	}

	ticker := h.opts.Clock.NewTicker(h.opts.BookmarkInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-h.closing:
			return ErrShuttingDown
		case <-sub.notify:
			cs.mu.Lock()
			lagged := sub.lagged
			var evs []WatchEvent
			if lagged {
				evs = h.snapshotLocked(cs, sub)
			} else {
				evs = h.deltasLocked(sub, sub.queue)
				if n := len(sub.queue); n > 0 {
					sub.pos = sub.queue[n-1].Seq
				}
				sub.queue = nil
			}
			cs.mu.Unlock()
			if lagged {
				h.resync(kind, metrics.ResyncLagged)
			}
			if err := h.sendAll(kind, send, evs); err != nil {
				return err
			}
		case <-ticker.C():
			cs.mu.Lock()
			pos := sub.pos
			if len(sub.queue) == 0 && !sub.lagged {
				pos = cs.seq
			}
			cs.mu.Unlock()
			if err := h.sendAll(kind, send, []WatchEvent{{
				Type: mcsv1.EventType_EVENT_TYPE_BOOKMARK, Revision: Revision(h.view.epoch, pos),
			}}); err != nil {
				return err
			}
		}
	}
}

// start returns the first events of a new subscriber: a replay of the
// deltas after a usable resume revision, else a snapshot, with the resync
// reason when a resume revision could not be used. cs.mu must be held.
func (h *Hub) start(cs *clusterSet, sub *subscriber, resume string) ([]WatchEvent, string) {
	if resume == "" {
		return h.snapshotLocked(cs, sub), ""
	}
	epoch, seq, err := ParseRevision(resume)
	switch {
	case err != nil || epoch != h.view.epoch || seq > cs.seq:
		return h.snapshotLocked(cs, sub), metrics.ResyncRevisionUnknown
	case !cs.ring.covers(seq, cs.seq):
		return h.snapshotLocked(cs, sub), metrics.ResyncRevisionExpired
	}
	sub.pos = cs.seq
	return h.deltasLocked(sub, cs.ring.since(seq)), ""
}

// snapshotLocked returns a snapshot of the subscriber's objects and resets
// its queue. cs.mu must be held.
func (h *Hub) snapshotLocked(cs *clusterSet, sub *subscriber) []WatchEvent {
	objs := cs.matching(sub.filter)
	evs := make([]WatchEvent, 0, len(objs)+2)
	evs = append(evs, WatchEvent{Type: mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START})
	for _, o := range objs {
		evs = append(evs, WatchEvent{Type: mcsv1.EventType_EVENT_TYPE_UPSERT, Object: o.Message(sub.filter.IncludeEndpoints())})
	}
	evs = append(evs, WatchEvent{Type: mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END, Revision: Revision(h.view.epoch, cs.seq)})
	sub.pos = cs.seq
	sub.queue = nil
	sub.lagged = false
	return evs
}

// deltasLocked converts events to the subscriber's view: an object that
// matches after the change is an UPSERT; one that matched only before is a
// DELETE carrying its latest state.
func (h *Hub) deltasLocked(sub *subscriber, events []Event) []WatchEvent {
	var out []WatchEvent
	f := sub.filter
	for _, ev := range events {
		if ev.Kind != f.Kind() {
			continue
		}
		rev := Revision(h.view.epoch, ev.Seq)
		switch {
		case ev.New != nil && f.Match(ev.New.Full):
			out = append(out, WatchEvent{Type: mcsv1.EventType_EVENT_TYPE_UPSERT, Object: ev.New.Message(f.IncludeEndpoints()), Revision: rev})
		case ev.Old != nil && f.Match(ev.Old.Full):
			last := ev.New
			if last == nil {
				last = ev.Old
			}
			out = append(out, WatchEvent{Type: mcsv1.EventType_EVENT_TYPE_DELETE, Object: last.Message(f.IncludeEndpoints()), Revision: rev})
		}
	}
	return out
}

func (h *Hub) resync(kind Kind, reason string) {
	if h.opts.Metrics != nil {
		h.opts.Metrics.WatchResyncs.WithLabelValues(kind.String(), reason).Inc()
	}
}

func eventTypeLabel(t mcsv1.EventType) string {
	return strings.ToLower(strings.TrimPrefix(t.String(), "EVENT_TYPE_"))
}

func (h *Hub) sendAll(kind Kind, send func(WatchEvent) error, evs []WatchEvent) error {
	for _, ev := range evs {
		if err := send(ev); err != nil {
			return err
		}
		if h.opts.Metrics != nil {
			h.opts.Metrics.WatchEventsSent.WithLabelValues(kind.String(), eventTypeLabel(ev.Type)).Inc()
		}
	}
	return nil
}
