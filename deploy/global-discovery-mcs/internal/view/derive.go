// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"cmp"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/convert"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry/merge"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// memoEntry caches a derived import by a fingerprint of its inputs.
type memoEntry struct {
	fp  string
	obj *Object
}

// endpointCache holds the proto form of one endpoint set. The messages are
// shared by every import that includes the set and never modified.
type endpointCache struct {
	gen     uint64
	updated time.Time
	eps     []*mcsv1.Endpoint
	total   int32
	ready   int32
}

// recompute evaluates leases, derives every object of the clusterset and
// publishes the differences to watchers. cs.mu must be held.
func (v *View) recompute(cs *clusterSet, now time.Time) {
	leases := cs.leaseStates(now, v.opts.LeaseSkew)
	v.countExpirations(cs, leases)
	cs.clusterLease = leases

	var next [numKinds]map[string]*Object
	next[KindCluster] = cs.deriveClusters()
	next[KindServiceImport] = cs.deriveImports(now)
	for k := range cs.epCache {
		if _, ok := cs.endpoints[k]; !ok {
			delete(cs.epCache, k)
		}
	}

	changed := false
	for k := range numKinds {
		if v.diff(cs, k, next[k]) {
			changed = true
		}
	}
	if changed {
		cs.index = buildIndexes(cs.objects)
	}
	v.setGauges(cs)
}

// diff publishes the changes from the current objects of a kind to next and
// installs next. Unchanged objects keep their previous pointer.
func (v *View) diff(cs *clusterSet, kind Kind, next map[string]*Object) bool {
	old := cs.objects[kind]
	keys := slices.Sorted(maps.Keys(old))
	for k := range next {
		if _, ok := old[k]; !ok {
			keys = append(keys, k)
		}
	}
	slices.Sort(keys)
	changed := false
	for _, key := range keys {
		o, n := old[key], next[key]
		switch {
		case n == nil:
			v.publish(cs, kind, key, o, nil)
			changed = true
		case o == nil:
			v.publish(cs, kind, key, nil, n)
			changed = true
		case o == n:
		case proto.Equal(o.Full, n.Full):
			next[key] = o
		default:
			v.publish(cs, kind, key, o, n)
			changed = true
		}
	}
	cs.objects[kind] = next
	return changed
}

func (v *View) countExpirations(cs *clusterSet, leases map[string]registry.LeaseState) {
	for name, s := range leases {
		if s == registry.LeaseStateExpired && cs.clusterLease[name] == registry.LeaseStateActive {
			v.log.Info("cluster lease expired", zap.String("clusterset", cs.name), zap.String("cluster", name))
			if v.opts.Metrics != nil {
				v.opts.Metrics.LeaseExpirations.WithLabelValues(metrics.LeaseCluster).Inc()
			}
		}
	}
}

func (v *View) setGauges(cs *clusterSet) {
	m := v.opts.Metrics
	if m == nil {
		return
	}
	g := metrics.ClusterSetGauges{Exports: len(cs.exports)}
	for _, s := range cs.clusterLease {
		if s == registry.LeaseStateActive {
			g.ActiveClusters++
		} else {
			g.ExpiredClusters++
		}
	}
	for key := range cs.endpoints {
		ep, _ := cs.endpointsOf(key)
		g.ReadyEndpoints += int(ep.ready)
		g.NotReadyEnds += int(ep.total - ep.ready)
	}
	for _, o := range cs.objects[KindServiceImport] {
		g.Imports++
		if o.Full.(*mcsv1.ServiceImport).GetConflict() != nil {
			g.ConflictedImports++
		}
	}
	m.SetClusterSet(cs.name, g)
}

// endpointsOf returns the proto form of an endpoint set.
func (cs *clusterSet) endpointsOf(key store.ExportKey) (endpointCache, bool) {
	rec, ok := cs.endpoints[key]
	if !ok {
		return endpointCache{}, false
	}
	if c, ok := cs.epCache[key]; ok && c.gen == rec.Generation && c.updated.Equal(rec.UpdatedAt) {
		return c, true
	}
	c := endpointCache{gen: rec.Generation, updated: rec.UpdatedAt}
	for _, e := range rec.Endpoints {
		c.eps = append(c.eps, convert.EndpointToProto(e))
		c.total++
		if e.Conditions.Ready {
			c.ready++
		}
	}
	cs.epCache[key] = c
	return c, true
}

func (cs *clusterSet) leaseActive(cluster string) bool {
	return cs.clusterLease[cluster] == registry.LeaseStateActive
}

func (cs *clusterSet) leaseStateProto(cluster string) mcsv1.LeaseState {
	if _, ok := cs.clusters[cluster]; !ok {
		return mcsv1.LeaseState_LEASE_STATE_UNSPECIFIED
	}
	return convert.LeaseStateToProto(cs.clusterLease[cluster])
}

func (cs *clusterSet) deriveClusters() map[string]*Object {
	out := make(map[string]*Object, len(cs.clusters))
	for name, r := range cs.clusters {
		c := convert.ClusterToProto(r, cs.clusterLease[name])
		out[name] = &Object{Key: name, Full: c, Lite: c}
	}
	return out
}

// ImportKey is the object key of a ServiceImport.
func ImportKey(namespace, name string) string { return namespace + "/" + name }

func (cs *clusterSet) deriveImports(now time.Time) map[string]*Object {
	groups := map[string][]store.ExportRecord{}
	for _, r := range cs.exports {
		k := ImportKey(r.Key.Namespace, r.Key.Name)
		groups[k] = append(groups[k], r)
	}
	out := make(map[string]*Object, len(groups))
	for key, recs := range groups {
		slices.SortFunc(recs, func(a, b store.ExportRecord) int { return cmp.Compare(a.Key.Cluster, b.Key.Cluster) })
		fp := cs.importFingerprint(recs)
		if m, ok := cs.importMemo[key]; ok && m.fp == fp {
			out[key] = m.obj
			continue
		}
		obj := cs.buildImport(key, recs, cs.objects[KindServiceImport][key], now)
		cs.importMemo[key] = memoEntry{fp: fp, obj: obj}
		out[key] = obj
	}
	for key := range cs.importMemo {
		if _, ok := groups[key]; !ok {
			delete(cs.importMemo, key)
		}
	}
	return out
}

// importFingerprint covers every input of a merged import: the export and
// endpoint set versions and the lease state, generation and region of each
// exporting cluster.
func (cs *clusterSet) importFingerprint(recs []store.ExportRecord) string {
	var b strings.Builder
	for _, r := range recs {
		cl := cs.clusters[r.Key.Cluster]
		ep := cs.endpoints[r.Key]
		fmt.Fprintf(&b, "%s|%d|%d|%d|%d|%s|%d|%s;", r.Key.Cluster, r.Generation, r.UpdatedAt.UnixNano(),
			ep.Generation, ep.UpdatedAt.UnixNano(), cs.leaseStateProto(r.Key.Cluster), cl.Generation, cl.Region)
	}
	return b.String()
}

func conflictToProto(c *merge.Conflict) *mcsv1.Conflict {
	if c == nil {
		return nil
	}
	return &mcsv1.Conflict{Reason: c.Reason, Message: c.Message, WinningCluster: c.WinningCluster}
}

var conditionStatuses = map[merge.ConditionStatus]mcsv1.ConditionStatus{
	merge.ConditionTrue:    mcsv1.ConditionStatus_CONDITION_STATUS_TRUE,
	merge.ConditionFalse:   mcsv1.ConditionStatus_CONDITION_STATUS_FALSE,
	merge.ConditionUnknown: mcsv1.ConditionStatus_CONDITION_STATUS_UNKNOWN,
}

// conditionsToProto converts conditions, keeping the transition time of a
// previous condition with the same type, status and reason.
func conditionsToProto(conds []merge.Condition, prev []*mcsv1.Condition, now time.Time) []*mcsv1.Condition {
	out := make([]*mcsv1.Condition, 0, len(conds))
	for _, c := range conds {
		status := conditionStatuses[c.Status]
		at := convert.Timestamp(now)
		for _, p := range prev {
			if p.GetType() == c.Type && p.GetStatus() == status && p.GetReason() == c.Reason {
				at = p.GetLastTransitionAt()
			}
		}
		out = append(out, &mcsv1.Condition{Type: c.Type, Status: status, Reason: c.Reason, Message: c.Message, LastTransitionAt: at})
	}
	return out
}

func maxTime(a, b time.Time) time.Time {
	if b.After(a) {
		return b
	}
	return a
}

func (cs *clusterSet) buildImport(key string, recs []store.ExportRecord, prev *Object, now time.Time) *Object {
	byCluster := make(map[string]store.ExportRecord, len(recs))
	inputs := make([]merge.Export, 0, len(recs))
	for _, r := range recs {
		byCluster[r.Key.Cluster] = r
		inputs = append(inputs, merge.Export{Cluster: r.Key.Cluster, LeaseActive: cs.leaseActive(r.Key.Cluster), Spec: r.Spec})
	}
	res := merge.Merge(inputs)
	si := &mcsv1.ServiceImport{
		Clusterset:            cs.name,
		Namespace:             recs[0].Key.Namespace,
		Name:                  recs[0].Key.Name,
		Type:                  convert.ServiceTypeToProto(res.Spec.Type),
		Ports:                 convert.ServicePortsToProto(res.Spec.Ports),
		IpFamilies:            convert.IPFamiliesToProto(res.Spec.IPFamilies),
		SessionAffinity:       convert.SessionAffinityToProto(res.Spec.SessionAffinity),
		SessionAffinityConfig: convert.SessionAffinityConfigToProto(res.Spec.SessionAffinityConfig),
		InternalTrafficPolicy: convert.InternalTrafficPolicyToProto(res.Spec.InternalTrafficPolicy),
		TrafficDistribution:   res.Spec.TrafficDistribution,
		Annotations:           maps.Clone(res.Spec.Annotations),
		Labels:                maps.Clone(res.Spec.Labels),
		Conflict:              conflictToProto(res.Conflict),
	}
	prevConds := map[string][]*mcsv1.Condition{}
	if prev != nil {
		for _, c := range prev.Full.(*mcsv1.ServiceImport).GetClusters() {
			prevConds[c.GetCluster()] = c.GetConditions()
		}
	}
	var updated time.Time
	for _, cr := range res.Clusters {
		rec := byCluster[cr.Cluster]
		region := cs.clusters[cr.Cluster].Region
		ep, _ := cs.endpointsOf(rec.Key)
		si.Clusters = append(si.Clusters, &mcsv1.ClusterStatus{
			Cluster:            cr.Cluster,
			Region:             region,
			LeaseState:         cs.leaseStateProto(cr.Cluster),
			Participating:      cr.Participating,
			Conflict:           conflictToProto(cr.Conflict),
			EndpointCount:      ep.total,
			ReadyEndpointCount: ep.ready,
			ExportGeneration:   rec.Generation,
			Conditions:         conditionsToProto(cr.Conditions, prevConds[cr.Cluster], now),
		})
		if cr.Participating {
			si.Endpoints = append(si.Endpoints, &mcsv1.ClusterEndpoints{
				Cluster:            cr.Cluster,
				Region:             region,
				Endpoints:          ep.eps,
				Generation:         ep.gen,
				EndpointCount:      ep.total,
				ReadyEndpointCount: ep.ready,
			})
		}
		updated = maxTime(updated, maxTime(rec.UpdatedAt, cs.endpoints[rec.Key].UpdatedAt))
	}
	si.UpdatedAt = convert.Timestamp(updated)
	return &Object{Key: key, Full: si, Lite: liteImport(si)}
}

// liteClusterEndpoints copies endpoint groups without their endpoint lists.
func liteClusterEndpoints(in []*mcsv1.ClusterEndpoints) []*mcsv1.ClusterEndpoints {
	if in == nil {
		return nil
	}
	out := make([]*mcsv1.ClusterEndpoints, 0, len(in))
	for _, ce := range in {
		out = append(out, &mcsv1.ClusterEndpoints{
			Cluster:            ce.GetCluster(),
			Region:             ce.GetRegion(),
			Generation:         ce.GetGeneration(),
			EndpointCount:      ce.GetEndpointCount(),
			ReadyEndpointCount: ce.GetReadyEndpointCount(),
		})
	}
	return out
}

// liteImport returns a ServiceImport sharing every field of si except the
// endpoint lists.
func liteImport(si *mcsv1.ServiceImport) *mcsv1.ServiceImport {
	return &mcsv1.ServiceImport{
		Clusterset:            si.Clusterset,
		Namespace:             si.Namespace,
		Name:                  si.Name,
		Type:                  si.Type,
		Ports:                 si.Ports,
		IpFamilies:            si.IpFamilies,
		SessionAffinity:       si.SessionAffinity,
		SessionAffinityConfig: si.SessionAffinityConfig,
		InternalTrafficPolicy: si.InternalTrafficPolicy,
		TrafficDistribution:   si.TrafficDistribution,
		Annotations:           si.Annotations,
		Labels:                si.Labels,
		Conflict:              si.Conflict,
		Clusters:              si.Clusters,
		Endpoints:             liteClusterEndpoints(si.Endpoints),
		UpdatedAt:             si.UpdatedAt,
	}
}
