// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"maps"
	"slices"
	"time"

	"google.golang.org/protobuf/proto"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// kind is a watchable object kind. Each kind has its own broadcast and
// history per clusterset.
type kind int

const (
	kindClusters kind = iota
	kindImports
	numKinds
)

func (k kind) String() string {
	return [...]string{"clusters", "service_imports"}[k]
}

type clusterRec struct {
	// c is the stored cluster; Lease.State and Lease.ExpiresAt are filled
	// when it is read.
	c   *mcsv1.Cluster
	ttl time.Duration
}

type exportKey struct {
	cluster, namespace, name string
}

type endpointRec struct {
	endpoints  []*mcsv1.Endpoint
	generation uint64
}

// event is one change of one object, in the order the server made it.
type event struct {
	typ mcsv1.EventType
	key string
	obj proto.Message
	seq uint64
}

// history keeps the latest events of one kind of one clusterset for resume.
type history struct {
	events []event
	// truncated is the highest sequence number dropped from events; every
	// event after it is still present.
	truncated uint64
}

// clusterSet holds the records of one clusterset and its last computed
// views, from which watch events are diffed.
type clusterSet struct {
	name      string
	clusters  map[string]*clusterRec
	exports   map[exportKey]*mcsv1.ServiceExport
	endpoints map[exportKey]*endpointRec
	views     [numKinds]map[string]proto.Message
	history   [numKinds]history
}

func (s *Server) setLocked(name string) *clusterSet {
	cs := s.sets[name]
	if cs == nil {
		cs = &clusterSet{
			name:      name,
			clusters:  map[string]*clusterRec{},
			exports:   map[exportKey]*mcsv1.ServiceExport{},
			endpoints: map[exportKey]*endpointRec{},
		}
		for k := range cs.views {
			cs.views[k] = map[string]proto.Message{}
		}
		s.sets[name] = cs
	}
	return cs
}

// commitLocked recomputes the views of cs at the server clock and
// broadcasts every difference to the previous views: UPSERT for new and
// changed objects, DELETE (with the last object) for removed ones.
func (s *Server) commitLocked(cs *clusterSet) {
	next := s.computeLocked(cs)
	for k := range numKinds {
		prev := cs.views[k]
		for _, key := range slices.Sorted(maps.Keys(next[k])) {
			obj := next[k][key]
			if old, ok := prev[key]; !ok || !proto.Equal(old, obj) {
				s.emitLocked(cs, k, mcsv1.EventType_EVENT_TYPE_UPSERT, key, obj)
			}
		}
		for _, key := range slices.Sorted(maps.Keys(prev)) {
			if _, ok := next[k][key]; !ok {
				s.emitLocked(cs, k, mcsv1.EventType_EVENT_TYPE_DELETE, key, prev[key])
			}
		}
		cs.views[k] = next[k]
	}
}

func (s *Server) emitLocked(cs *clusterSet, k kind, typ mcsv1.EventType, key string, obj proto.Message) {
	s.seq++
	ev := event{typ: typ, key: key, obj: obj, seq: s.seq}
	h := &cs.history[k]
	h.events = append(h.events, ev)
	if over := len(h.events) - s.historySize; over > 0 {
		h.truncated = h.events[over-1].seq
		h.events = slices.Clone(h.events[over:])
	}
	for sub := range s.subs {
		if sub.cs != cs.name || sub.kind != k {
			continue
		}
		select {
		case sub.ch <- ev:
		default:
			select {
			case sub.lag <- struct{}{}:
			default:
			}
		}
	}
}

// refreshedLocked returns cs with its views re-evaluated at the server
// clock, so reads see current lease states.
func (s *Server) refreshedLocked(name string) *clusterSet {
	cs := s.setLocked(name)
	s.commitLocked(cs)
	return cs
}

// sortedViews returns the objects of one view that match, in key order,
// passed through view.
func sortedViews[T proto.Message](cs *clusterSet, k kind, match func(T) bool) []T {
	var out []T
	for _, key := range slices.Sorted(maps.Keys(cs.views[k])) {
		obj := cs.views[k][key].(T)
		if match == nil || match(obj) {
			out = append(out, obj)
		}
	}
	return out
}
