// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"net/netip"

	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
)

// The seeding helpers write state the way a controller would, without the
// generation bookkeeping, and broadcast to watchers like the RPCs. They
// suit tests of consumers of imports; tests of writers should call the
// RPCs.

// EnsureCluster registers a cluster with the default 30 s lease when it is
// not registered, and renews its lease otherwise. An empty region becomes
// the server region. It returns the lease epoch.
func (s *Server) EnsureCluster(clusterset, cluster, region string) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ensureClusterLocked(clusterset, cluster, region)
}

func (s *Server) ensureClusterLocked(clusterset, cluster, region string) uint64 {
	cs := s.setLocked(clusterset)
	if rec := cs.clusters[cluster]; rec != nil {
		rec.c.Lease.RenewedAt = timestamppb.New(s.now())
		s.commitLocked(cs)
		return rec.c.GetLease().GetEpoch()
	}
	if region == "" {
		region = s.region
	}
	rec := s.registerClusterLocked(&mcsv1.RegisterClusterRequest{Clusterset: clusterset, Cluster: cluster, Region: region})
	return rec.c.GetLease().GetEpoch()
}

// RemoveCluster deregisters a cluster and removes everything it owns, as
// DeregisterCluster does.
func (s *Server) RemoveCluster(clusterset, cluster string) {
	s.removeCluster(clusterset, cluster)
}

// PutExport stores the export of namespace/name from cluster with spec and
// replaces its endpoint set, registering the cluster first when needed.
// Generations are bumped automatically. A nil endpoints slice leaves the
// endpoint set unchanged; an empty one clears it.
func (s *Server) PutExport(clusterset, cluster, namespace, name string, spec *mcsv1.ServiceExportSpec, endpoints []*mcsv1.Endpoint) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.setLocked(clusterset)
	if cs.clusters[cluster] == nil {
		s.ensureClusterLocked(clusterset, cluster, "")
	}
	key := exportKey{cluster, namespace, name}
	gen := registry.NextGeneration(cs.exports[key].GetGeneration(), s.now())
	_ = s.putExportLocked(cs, key, spec, gen)
	if endpoints != nil {
		s.setEndpointsLocked(cs, key, endpoints)
	}
}

// SetEndpoints replaces the endpoint set of an export, bumping its
// generation.
func (s *Server) SetEndpoints(clusterset, cluster, namespace, name string, endpoints []*mcsv1.Endpoint) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.setEndpointsLocked(s.setLocked(clusterset), exportKey{cluster, namespace, name}, endpoints)
}

func (s *Server) setEndpointsLocked(cs *clusterSet, key exportKey, endpoints []*mcsv1.Endpoint) {
	var prev uint64
	if old := cs.endpoints[key]; old != nil {
		prev = old.generation
	}
	_ = s.putEndpointsLocked(cs, key, endpoints, registry.NextGeneration(prev, s.now()))
}

// DeleteExport removes an export and its endpoint set. It is idempotent.
func (s *Server) DeleteExport(clusterset, cluster, namespace, name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cs := s.setLocked(clusterset)
	key := exportKey{cluster, namespace, name}
	delete(cs.exports, key)
	delete(cs.endpoints, key)
	s.commitLocked(cs)
}

// ServiceSpec returns the export spec of a ClusterSetIP Service labelled
// for a label-driven export (api.LabelExport), with ports. Build ports with
// TCPPort.
func ServiceSpec(ports ...*mcsv1.ServicePort) *mcsv1.ServiceExportSpec {
	return &mcsv1.ServiceExportSpec{
		Type:   mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
		Labels: map[string]string{api.LabelExport: api.ExportEnabled},
		Ports:  ports,
	}
}

// TCPPort returns a named TCP Service port.
func TCPPort(name string, port int32) *mcsv1.ServicePort {
	return &mcsv1.ServicePort{Name: name, Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: port}
}

// ReadyEndpoints returns one ready, serving endpoint per address, with the
// address type inferred from the address.
func ReadyEndpoints(addresses ...string) []*mcsv1.Endpoint {
	eps := make([]*mcsv1.Endpoint, 0, len(addresses))
	for _, a := range addresses {
		eps = append(eps, &mcsv1.Endpoint{
			Addresses:   []string{a},
			AddressType: addressType(a),
			Conditions:  &mcsv1.EndpointConditions{Ready: true, Serving: true},
		})
	}
	return eps
}

// WithHostnames sets the EndpointSlice hostname of eps[i] to hostnames[i],
// as a StatefulSet behind a headless Service has them. Extra names are
// ignored. It modifies and returns eps.
func WithHostnames(eps []*mcsv1.Endpoint, hostnames ...string) []*mcsv1.Endpoint {
	for i := 0; i < len(eps) && i < len(hostnames); i++ {
		eps[i].Hostname = hostnames[i]
	}
	return eps
}

// WithPodNames gives eps[i] the target reference attributes the
// mcs-controller exports for a pod named podNames[i]: targetRef.kind Pod
// and targetRef.name (api.EndpointAttrTargetName). An empty name leaves its
// endpoint unchanged and extra names are ignored. It modifies and returns
// eps.
func WithPodNames(eps []*mcsv1.Endpoint, podNames ...string) []*mcsv1.Endpoint {
	for i := 0; i < len(eps) && i < len(podNames); i++ {
		if podNames[i] == "" {
			continue
		}
		if eps[i].Attributes == nil {
			eps[i].Attributes = map[string]string{}
		}
		eps[i].Attributes["targetRef.kind"] = "Pod"
		eps[i].Attributes[api.EndpointAttrTargetName] = podNames[i]
	}
	return eps
}

func addressType(a string) mcsv1.AddressType {
	ip, err := netip.ParseAddr(a)
	switch {
	case err != nil:
		return mcsv1.AddressType_ADDRESS_TYPE_FQDN
	case ip.Is4():
		return mcsv1.AddressType_ADDRESS_TYPE_IPV4
	}
	return mcsv1.AddressType_ADDRESS_TYPE_IPV6
}
