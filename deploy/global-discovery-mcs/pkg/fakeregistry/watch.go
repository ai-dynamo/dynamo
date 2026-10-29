// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"context"
	"maps"
	"slices"
	"time"

	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// subscriber is one open watch stream. Events are delivered to ch under the
// server lock; when ch is full the watcher is flagged on lag and
// re-snapshotted.
type subscriber struct {
	cs   string
	kind kind
	ch   chan event
	lag  chan struct{}
	kill chan error
}

// watchSpec describes one watch: what it selects and how events are sent.
type watchSpec struct {
	kind   kind
	cs     string
	resume string
	// match reports whether an object passes the watcher's filters.
	match func(proto.Message) bool
	// view returns the object as this watcher sees it, for example without
	// endpoint lists. It must not modify its argument.
	view func(proto.Message) proto.Message
	// send sends one event. obj is nil and rev empty where the event type
	// carries none.
	send func(typ mcsv1.EventType, obj proto.Message, rev string) error
}

// snapshotLocked returns the keys and objects of the view that match, in
// key order, and the current revision sequence.
func (s *Server) snapshotLocked(cs *clusterSet, w *watchSpec) ([]string, []proto.Message, uint64) {
	var keys []string
	var objs []proto.Message
	for _, key := range slices.Sorted(maps.Keys(cs.views[w.kind])) {
		obj := cs.views[w.kind][key]
		if w.match(obj) {
			keys = append(keys, key)
			objs = append(objs, obj)
		}
	}
	return keys, objs, s.seq
}

// subscribe registers a watcher. It returns either a snapshot (keys, objects
// and revision sequence) or, when resume is a usable revision, the events
// to replay and the keys the client may hold.
func (s *Server) subscribe(w *watchSpec) (sub *subscriber, resumed bool, keys []string, objs []proto.Message, seq uint64, replay []event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.refreshedLocked(w.cs)
	sub = &subscriber{
		cs:   w.cs,
		kind: w.kind,
		ch:   make(chan event, s.queueSize),
		lag:  make(chan struct{}, 1),
		kill: make(chan error, 1),
	}
	s.subs[sub] = struct{}{}
	if from, ok := s.parseRevision(w.resume); ok && from <= s.seq && from >= cs.history[w.kind].truncated {
		for _, ev := range cs.history[w.kind].events {
			if ev.seq > from {
				replay = append(replay, ev)
			}
		}
		// The client may hold any object that matches now or that a
		// replayed event touches; a DELETE for a key it lacks is harmless.
		keys, _, _ = s.snapshotLocked(cs, w)
		for _, ev := range replay {
			keys = append(keys, ev.key)
		}
		return sub, true, keys, nil, from, replay
	}
	keys, objs, seq = s.snapshotLocked(cs, w)
	return sub, false, keys, objs, seq, nil
}

func (s *Server) unsubscribe(sub *subscriber) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.subs, sub)
}

// resnapshot drains a lagging watcher and returns a new snapshot.
func (s *Server) resnapshot(sub *subscriber, w *watchSpec) ([]string, []proto.Message, uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(sub.ch) > 0 {
		<-sub.ch
	}
	select {
	case <-sub.lag:
	default:
	}
	return s.snapshotLocked(s.setLocked(w.cs), w)
}

// bookmarkSeq returns the revision sequence a bookmark may carry: the
// current one when every event delivered to the watcher has been processed,
// else the last processed one.
func (s *Server) bookmarkSeq(sub *subscriber, processed uint64) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(sub.ch) == 0 {
		return s.seq
	}
	return processed
}

// serveWatch runs one watch stream until the client goes away, the
// listener stops or CloseWatches ends it.
func (s *Server) serveWatch(ctx context.Context, w *watchSpec) error {
	if err := require("clusterset", w.cs); err != nil {
		return err
	}
	sub, resumed, keys, objs, seq, replay := s.subscribe(w)
	defer s.unsubscribe(sub)

	sent := map[string]bool{}
	sendSnapshot := func(keys []string, objs []proto.Message, seq uint64) error {
		clear(sent)
		if err := w.send(mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START, nil, ""); err != nil {
			return err
		}
		for i, obj := range objs {
			if err := w.send(mcsv1.EventType_EVENT_TYPE_UPSERT, w.view(obj), ""); err != nil {
				return err
			}
			sent[keys[i]] = true
		}
		return w.send(mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END, nil, s.revision(seq))
	}
	processed := seq
	process := func(ev event) error {
		processed = ev.seq
		rev := s.revision(ev.seq)
		switch ev.typ {
		case mcsv1.EventType_EVENT_TYPE_UPSERT:
			if w.match(ev.obj) {
				sent[ev.key] = true
				return w.send(mcsv1.EventType_EVENT_TYPE_UPSERT, w.view(ev.obj), rev)
			}
			if sent[ev.key] {
				delete(sent, ev.key)
				return w.send(mcsv1.EventType_EVENT_TYPE_DELETE, w.view(ev.obj), rev)
			}
		case mcsv1.EventType_EVENT_TYPE_DELETE:
			if sent[ev.key] {
				delete(sent, ev.key)
				return w.send(mcsv1.EventType_EVENT_TYPE_DELETE, w.view(ev.obj), rev)
			}
		}
		return nil
	}

	if resumed {
		for _, k := range keys {
			sent[k] = true
		}
		for _, ev := range replay {
			if err := process(ev); err != nil {
				return err
			}
		}
	} else if err := sendSnapshot(keys, objs, seq); err != nil {
		return err
	}

	var tick <-chan time.Time
	if s.bookmarkInterval > 0 {
		t := time.NewTicker(s.bookmarkInterval)
		defer t.Stop()
		tick = t.C
	}
	for {
		select {
		case <-ctx.Done():
			return status.FromContextError(ctx.Err()).Err()
		case err := <-sub.kill:
			return err
		case <-sub.lag:
			keys, objs, seq := s.resnapshot(sub, w)
			processed = seq
			if err := sendSnapshot(keys, objs, seq); err != nil {
				return err
			}
		case ev := <-sub.ch:
			if ev.seq <= processed {
				continue
			}
			if err := process(ev); err != nil {
				return err
			}
		case <-tick:
			seq := s.bookmarkSeq(sub, processed)
			if err := w.send(mcsv1.EventType_EVENT_TYPE_BOOKMARK, nil, s.revision(seq)); err != nil {
				return err
			}
		}
	}
}
