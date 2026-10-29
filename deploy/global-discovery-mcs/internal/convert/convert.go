// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package convert maps between the store record types, which use Kubernetes
// spellings, and the dynamo.mcs.v1 messages. Conversions are total: an
// unknown or unspecified enum maps to the zero value on the other side.
// Defaults (TCP, ClusterSetIP, None) are applied by the API layer, not here.
package convert

import (
	"maps"
	"slices"
	"time"

	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// Timestamp returns t as a Timestamp, or nil for the zero time.
func Timestamp(t time.Time) *timestamppb.Timestamp {
	if t.IsZero() {
		return nil
	}
	return timestamppb.New(t)
}

// Time returns ts as a time in UTC, or the zero time for nil.
func Time(ts *timestamppb.Timestamp) time.Time {
	if ts == nil {
		return time.Time{}
	}
	return ts.AsTime().UTC()
}

// Duration returns d as a Duration, or nil for zero.
func Duration(d time.Duration) *durationpb.Duration {
	if d == 0 {
		return nil
	}
	return durationpb.New(d)
}

var serviceTypes = map[store.ServiceType]mcsv1.ServiceType{
	store.ServiceTypeClusterSetIP: mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
	store.ServiceTypeHeadless:     mcsv1.ServiceType_SERVICE_TYPE_HEADLESS,
}

var protocols = map[store.Protocol]mcsv1.Protocol{
	store.ProtocolTCP:  mcsv1.Protocol_PROTOCOL_TCP,
	store.ProtocolUDP:  mcsv1.Protocol_PROTOCOL_UDP,
	store.ProtocolSCTP: mcsv1.Protocol_PROTOCOL_SCTP,
}

var sessionAffinities = map[store.SessionAffinity]mcsv1.SessionAffinity{
	store.SessionAffinityNone:     mcsv1.SessionAffinity_SESSION_AFFINITY_NONE,
	store.SessionAffinityClientIP: mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP,
}

var ipFamilies = map[store.IPFamily]mcsv1.IPFamily{
	store.IPFamilyIPv4: mcsv1.IPFamily_IP_FAMILY_IPV4,
	store.IPFamilyIPv6: mcsv1.IPFamily_IP_FAMILY_IPV6,
}

var trafficPolicies = map[store.InternalTrafficPolicy]mcsv1.InternalTrafficPolicy{
	store.InternalTrafficPolicyCluster: mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_CLUSTER,
	store.InternalTrafficPolicyLocal:   mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_LOCAL,
}

var addressTypes = map[store.AddressType]mcsv1.AddressType{
	store.AddressTypeIPv4: mcsv1.AddressType_ADDRESS_TYPE_IPV4,
	store.AddressTypeIPv6: mcsv1.AddressType_ADDRESS_TYPE_IPV6,
	store.AddressTypeFQDN: mcsv1.AddressType_ADDRESS_TYPE_FQDN,
}

var leaseStates = map[registry.LeaseState]mcsv1.LeaseState{
	registry.LeaseStateActive:  mcsv1.LeaseState_LEASE_STATE_ACTIVE,
	registry.LeaseStateExpired: mcsv1.LeaseState_LEASE_STATE_EXPIRED,
}

func reverse[K, V comparable](m map[K]V) map[V]K {
	out := make(map[V]K, len(m))
	for k, v := range m {
		out[v] = k
	}
	return out
}

var (
	serviceTypesRev      = reverse(serviceTypes)
	protocolsRev         = reverse(protocols)
	sessionAffinitiesRev = reverse(sessionAffinities)
	ipFamiliesRev        = reverse(ipFamilies)
	trafficPoliciesRev   = reverse(trafficPolicies)
	addressTypesRev      = reverse(addressTypes)
)

// ServiceTypeToProto maps a store service type.
func ServiceTypeToProto(t store.ServiceType) mcsv1.ServiceType { return serviceTypes[t] }

// ServiceTypeFromProto maps a proto service type.
func ServiceTypeFromProto(t mcsv1.ServiceType) store.ServiceType { return serviceTypesRev[t] }

// ProtocolToProto maps a store protocol.
func ProtocolToProto(p store.Protocol) mcsv1.Protocol { return protocols[p] }

// ProtocolFromProto maps a proto protocol.
func ProtocolFromProto(p mcsv1.Protocol) store.Protocol { return protocolsRev[p] }

// SessionAffinityToProto maps a store session affinity.
func SessionAffinityToProto(a store.SessionAffinity) mcsv1.SessionAffinity {
	return sessionAffinities[a]
}

// SessionAffinityFromProto maps a proto session affinity.
func SessionAffinityFromProto(a mcsv1.SessionAffinity) store.SessionAffinity {
	return sessionAffinitiesRev[a]
}

// IPFamilyToProto maps a store IP family.
func IPFamilyToProto(f store.IPFamily) mcsv1.IPFamily { return ipFamilies[f] }

// IPFamilyFromProto maps a proto IP family.
func IPFamilyFromProto(f mcsv1.IPFamily) store.IPFamily { return ipFamiliesRev[f] }

// InternalTrafficPolicyToProto maps a store internal traffic policy.
func InternalTrafficPolicyToProto(p store.InternalTrafficPolicy) mcsv1.InternalTrafficPolicy {
	return trafficPolicies[p]
}

// InternalTrafficPolicyFromProto maps a proto internal traffic policy.
func InternalTrafficPolicyFromProto(p mcsv1.InternalTrafficPolicy) store.InternalTrafficPolicy {
	return trafficPoliciesRev[p]
}

// AddressTypeToProto maps a store address type.
func AddressTypeToProto(t store.AddressType) mcsv1.AddressType { return addressTypes[t] }

// AddressTypeFromProto maps a proto address type.
func AddressTypeFromProto(t mcsv1.AddressType) store.AddressType { return addressTypesRev[t] }

// LeaseStateToProto maps a lease state.
func LeaseStateToProto(s registry.LeaseState) mcsv1.LeaseState { return leaseStates[s] }

// ExportSpecToProto converts an export spec.
func ExportSpecToProto(s store.ExportSpec) *mcsv1.ServiceExportSpec {
	out := &mcsv1.ServiceExportSpec{
		Annotations:             maps.Clone(s.Annotations),
		Labels:                  maps.Clone(s.Labels),
		ExportCreationTimestamp: Timestamp(s.ExportCreationTimestamp),
		Ports:                   ServicePortsToProto(s.Ports),
		Type:                    ServiceTypeToProto(s.Type),
		SessionAffinity:         SessionAffinityToProto(s.SessionAffinity),
		SessionAffinityConfig:   SessionAffinityConfigToProto(s.SessionAffinityConfig),
		IpFamilies:              IPFamiliesToProto(s.IPFamilies),
		InternalTrafficPolicy:   InternalTrafficPolicyToProto(s.InternalTrafficPolicy),
		TrafficDistribution:     s.TrafficDistribution,
	}
	return out
}

// ExportSpecFromProto converts an export spec; nil yields the zero spec.
func ExportSpecFromProto(s *mcsv1.ServiceExportSpec) store.ExportSpec {
	if s == nil {
		return store.ExportSpec{}
	}
	out := store.ExportSpec{
		Annotations:             maps.Clone(s.GetAnnotations()),
		Labels:                  maps.Clone(s.GetLabels()),
		ExportCreationTimestamp: Time(s.GetExportCreationTimestamp()),
		Type:                    ServiceTypeFromProto(s.GetType()),
		SessionAffinity:         SessionAffinityFromProto(s.GetSessionAffinity()),
		InternalTrafficPolicy:   InternalTrafficPolicyFromProto(s.GetInternalTrafficPolicy()),
		TrafficDistribution:     s.GetTrafficDistribution(),
	}
	for _, p := range s.GetPorts() {
		out.Ports = append(out.Ports, store.ServicePort{
			Name: p.GetName(), Protocol: ProtocolFromProto(p.GetProtocol()), AppProtocol: p.GetAppProtocol(), Port: p.GetPort(),
		})
	}
	if c := s.GetSessionAffinityConfig(); c != nil {
		out.SessionAffinityConfig = &store.SessionAffinityConfig{}
		if ip := c.GetClientIp(); ip != nil {
			out.SessionAffinityConfig.ClientIP = &store.ClientIPConfig{TimeoutSeconds: ip.GetTimeoutSeconds()}
		}
	}
	for _, f := range s.GetIpFamilies() {
		out.IPFamilies = append(out.IPFamilies, IPFamilyFromProto(f))
	}
	return out
}

// ServicePortsToProto converts Service ports.
func ServicePortsToProto(ports []store.ServicePort) []*mcsv1.ServicePort {
	if ports == nil {
		return nil
	}
	out := make([]*mcsv1.ServicePort, 0, len(ports))
	for _, p := range ports {
		out = append(out, &mcsv1.ServicePort{
			Name: p.Name, Protocol: ProtocolToProto(p.Protocol), AppProtocol: p.AppProtocol, Port: p.Port,
		})
	}
	return out
}

// SessionAffinityConfigToProto converts a session affinity config.
func SessionAffinityConfigToProto(c *store.SessionAffinityConfig) *mcsv1.SessionAffinityConfig {
	if c == nil {
		return nil
	}
	out := &mcsv1.SessionAffinityConfig{}
	if c.ClientIP != nil {
		out.ClientIp = &mcsv1.ClientIPConfig{TimeoutSeconds: c.ClientIP.TimeoutSeconds}
	}
	return out
}

// IPFamiliesToProto converts IP families.
func IPFamiliesToProto(fs []store.IPFamily) []mcsv1.IPFamily {
	if fs == nil {
		return nil
	}
	out := make([]mcsv1.IPFamily, 0, len(fs))
	for _, f := range fs {
		out = append(out, IPFamilyToProto(f))
	}
	return out
}

// EndpointToProto converts an endpoint.
func EndpointToProto(e store.Endpoint) *mcsv1.Endpoint {
	out := &mcsv1.Endpoint{
		Addresses:   slices.Clone(e.Addresses),
		AddressType: AddressTypeToProto(e.AddressType),
		Conditions: &mcsv1.EndpointConditions{
			Ready: e.Conditions.Ready, Serving: e.Conditions.Serving, Terminating: e.Conditions.Terminating,
		},
		Hostname:   e.Hostname,
		NodeName:   e.NodeName,
		Zone:       e.Zone,
		Attributes: maps.Clone(e.Attributes),
	}
	for _, p := range e.Ports {
		out.Ports = append(out.Ports, &mcsv1.EndpointPort{
			Name: p.Name, Protocol: ProtocolToProto(p.Protocol), AppProtocol: p.AppProtocol, Port: p.Port,
		})
	}
	return out
}

// EndpointFromProto converts an endpoint.
func EndpointFromProto(e *mcsv1.Endpoint) store.Endpoint {
	out := store.Endpoint{
		Addresses:   slices.Clone(e.GetAddresses()),
		AddressType: AddressTypeFromProto(e.GetAddressType()),
		Conditions: store.EndpointConditions{
			Ready:       e.GetConditions().GetReady(),
			Serving:     e.GetConditions().GetServing(),
			Terminating: e.GetConditions().GetTerminating(),
		},
		Hostname:   e.GetHostname(),
		NodeName:   e.GetNodeName(),
		Zone:       e.GetZone(),
		Attributes: maps.Clone(e.GetAttributes()),
	}
	for _, p := range e.GetPorts() {
		out.Ports = append(out.Ports, store.EndpointPort{
			Name: p.GetName(), Protocol: ProtocolFromProto(p.GetProtocol()), AppProtocol: p.GetAppProtocol(), Port: p.GetPort(),
		})
	}
	return out
}

// LeaseToProto converts a lease evaluated to state. A lease with epoch 0
// (none) yields nil.
func LeaseToProto(l store.Lease, state registry.LeaseState) *mcsv1.Lease {
	if l.Epoch == 0 {
		return nil
	}
	return &mcsv1.Lease{
		Epoch:     l.Epoch,
		Ttl:       Duration(l.TTL),
		RenewedAt: Timestamp(l.RenewedAt),
		ExpiresAt: Timestamp(l.ExpiresAt()),
		State:     LeaseStateToProto(state),
	}
}

// ClusterToProto converts a cluster registration whose lease evaluated to
// state.
func ClusterToProto(r store.ClusterRecord, state registry.LeaseState) *mcsv1.Cluster {
	return &mcsv1.Cluster{
		Clusterset:   r.Key.ClusterSet,
		Cluster:      r.Key.Cluster,
		Region:       r.Region,
		Zones:        slices.Clone(r.Zones),
		Provider:     r.Provider,
		Labels:       maps.Clone(r.Labels),
		AgentId:      r.AgentID,
		AgentVersion: r.AgentVersion,
		Lease:        LeaseToProto(r.Lease, state),
		Generation:   r.Generation,
		RegisteredAt: Timestamp(r.RegisteredAt),
		UpdatedAt:    Timestamp(r.UpdatedAt),
	}
}
