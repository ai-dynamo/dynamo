// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"cmp"
	"maps"
	"slices"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
)

// computeLocked derives every view of cs at the server clock. The returned
// objects are fresh and never mutated afterwards.
func (s *Server) computeLocked(cs *clusterSet) [numKinds]map[string]proto.Message {
	now := s.now()
	var v [numKinds]map[string]proto.Message
	for k := range v {
		v[k] = map[string]proto.Message{}
	}
	active := map[string]bool{}
	region := map[string]string{}
	for name, rec := range cs.clusters {
		c := clusterView(rec, now)
		v[kindClusters][name] = c
		active[name] = c.GetLease().GetState() == mcsv1.LeaseState_LEASE_STATE_ACTIVE
		region[name] = c.GetRegion()
	}
	for key, imp := range importsView(cs, active, region) {
		v[kindImports][key] = imp
	}
	return v
}

func leaseState(renewedAt time.Time, ttl time.Duration, now time.Time) mcsv1.LeaseState {
	if registry.EvaluateLease(renewedAt, ttl, now, registry.DefaultLeaseSkew) == registry.LeaseStateActive {
		return mcsv1.LeaseState_LEASE_STATE_ACTIVE
	}
	return mcsv1.LeaseState_LEASE_STATE_EXPIRED
}

// leaseView fills the read-time fields of a stored lease.
func leaseView(l *mcsv1.Lease, ttl time.Duration, now time.Time) *mcsv1.Lease {
	if l == nil {
		return nil
	}
	l = proto.CloneOf(l)
	renewed := l.GetRenewedAt().AsTime()
	l.ExpiresAt = timestamppb.New(registry.LeaseExpiresAt(renewed, ttl))
	l.State = leaseState(renewed, ttl, now)
	return l
}

func clusterView(rec *clusterRec, now time.Time) *mcsv1.Cluster {
	c := proto.CloneOf(rec.c)
	c.Lease = leaseView(rec.c.GetLease(), rec.ttl, now)
	return c
}

func countEndpoints(eps []*mcsv1.Endpoint) (total, ready int32) {
	for _, ep := range eps {
		total++
		if ep.GetConditions().GetReady() {
			ready++
		}
	}
	return total, ready
}

func clusterEndpoints(cluster, region string, rec *endpointRec) *mcsv1.ClusterEndpoints {
	ce := &mcsv1.ClusterEndpoints{Cluster: cluster, Region: region}
	if rec != nil {
		ce.Generation = rec.generation
		ce.EndpointCount, ce.ReadyEndpointCount = countEndpoints(rec.endpoints)
		for _, ep := range rec.endpoints {
			ce.Endpoints = append(ce.Endpoints, proto.CloneOf(ep))
		}
	}
	return ce
}

// exportOrder sorts exports oldest first: export creation time, then
// cluster name.
func exportOrder(a, b *mcsv1.ServiceExport) int {
	ta, tb := a.GetSpec().GetExportCreationTimestamp().AsTime(), b.GetSpec().GetExportCreationTimestamp().AsTime()
	if c := ta.Compare(tb); c != 0 {
		return c
	}
	return cmp.Compare(a.GetCluster(), b.GetCluster())
}

func importsView(cs *clusterSet, active map[string]bool, region map[string]string) map[string]*mcsv1.ServiceImport {
	groups := map[string][]*mcsv1.ServiceExport{}
	for key, e := range cs.exports {
		groups[key.namespace+"/"+key.name] = append(groups[key.namespace+"/"+key.name], e)
	}
	out := map[string]*mcsv1.ServiceImport{}
	for key, exports := range groups {
		slices.SortFunc(exports, exportOrder)
		var participants []*mcsv1.ServiceExport
		for _, e := range exports {
			if active[e.GetCluster()] {
				participants = append(participants, e)
			}
		}
		winner := exports[0]
		if len(participants) > 0 {
			winner = participants[0]
		}
		spec := winner.GetSpec()
		imp := &mcsv1.ServiceImport{
			Clusterset:            cs.name,
			Namespace:             winner.GetNamespace(),
			Name:                  winner.GetName(),
			Type:                  spec.GetType(),
			IpFamilies:            slices.Clone(spec.GetIpFamilies()),
			SessionAffinity:       spec.GetSessionAffinity(),
			SessionAffinityConfig: proto.CloneOf(spec.GetSessionAffinityConfig()),
			Annotations:           maps.Clone(spec.GetAnnotations()),
			Labels:                maps.Clone(spec.GetLabels()),
		}
		portSources := participants
		if len(portSources) == 0 {
			portSources = []*mcsv1.ServiceExport{winner}
		}
		type portKey struct {
			port     int32
			protocol mcsv1.Protocol
		}
		seen := map[portKey]bool{}
		for _, e := range portSources {
			for _, p := range e.GetSpec().GetPorts() {
				k := portKey{p.GetPort(), p.GetProtocol()}
				if !seen[k] {
					seen[k] = true
					imp.Ports = append(imp.Ports, proto.CloneOf(p))
				}
			}
		}
		var updated time.Time
		byCluster := slices.Clone(exports)
		slices.SortFunc(byCluster, func(a, b *mcsv1.ServiceExport) int { return cmp.Compare(a.GetCluster(), b.GetCluster()) })
		for _, e := range byCluster {
			ek := exportKey{e.GetCluster(), e.GetNamespace(), e.GetName()}
			epRec := cs.endpoints[ek]
			participating := active[e.GetCluster()]
			st := &mcsv1.ClusterStatus{
				Cluster:          e.GetCluster(),
				Region:           region[e.GetCluster()],
				LeaseState:       mcsv1.LeaseState_LEASE_STATE_EXPIRED,
				Participating:    participating,
				ExportGeneration: e.GetGeneration(),
				Conditions: []*mcsv1.Condition{{
					Type:   api.ConditionConflict,
					Status: mcsv1.ConditionStatus_CONDITION_STATUS_FALSE,
					Reason: api.ConditionReasonNoConflicts,
				}},
			}
			ready := &mcsv1.Condition{Type: api.ConditionReady, Status: mcsv1.ConditionStatus_CONDITION_STATUS_TRUE, Reason: "Exported"}
			if participating {
				st.LeaseState = mcsv1.LeaseState_LEASE_STATE_ACTIVE
				if epRec != nil {
					st.EndpointCount, st.ReadyEndpointCount = countEndpoints(epRec.endpoints)
					imp.Endpoints = append(imp.Endpoints, clusterEndpoints(e.GetCluster(), region[e.GetCluster()], epRec))
				}
			} else {
				ready.Status = mcsv1.ConditionStatus_CONDITION_STATUS_FALSE
				ready.Reason = api.ConditionReasonClusterLeaseExpired
			}
			st.Conditions = append([]*mcsv1.Condition{ready}, st.Conditions...)
			imp.Clusters = append(imp.Clusters, st)
			if t := e.GetUpdatedAt().AsTime(); t.After(updated) {
				updated = t
			}
		}
		imp.UpdatedAt = timestamppb.New(updated)
		out[key] = imp
	}
	return out
}
