// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package store

import (
	"maps"
	"slices"
	"time"
)

// The spec types below mirror the dynamo.mcs.v1 messages field by field.
// Enumerations use the Kubernetes and KEP-1645 spellings, so that the
// controller maps Kubernetes objects without translation and the API layer
// maps them to and from the proto enums.

// ServiceType is the KEP-1645 ServiceImport type.
type ServiceType string

const (
	ServiceTypeClusterSetIP ServiceType = "ClusterSetIP"
	ServiceTypeHeadless     ServiceType = "Headless"
)

// Protocol is a Kubernetes port protocol.
type Protocol string

const (
	ProtocolTCP  Protocol = "TCP"
	ProtocolUDP  Protocol = "UDP"
	ProtocolSCTP Protocol = "SCTP"
)

// SessionAffinity is the Kubernetes Service session affinity.
type SessionAffinity string

const (
	SessionAffinityNone     SessionAffinity = "None"
	SessionAffinityClientIP SessionAffinity = "ClientIP"
)

// IPFamily is a Kubernetes IP family.
type IPFamily string

const (
	IPFamilyIPv4 IPFamily = "IPv4"
	IPFamilyIPv6 IPFamily = "IPv6"
)

// InternalTrafficPolicy is the Kubernetes internal traffic policy; empty
// means unset.
type InternalTrafficPolicy string

const (
	InternalTrafficPolicyCluster InternalTrafficPolicy = "Cluster"
	InternalTrafficPolicyLocal   InternalTrafficPolicy = "Local"
)

// AddressType is the EndpointSlice address type.
type AddressType string

const (
	AddressTypeIPv4 AddressType = "IPv4"
	AddressTypeIPv6 AddressType = "IPv6"
	AddressTypeFQDN AddressType = "FQDN"
)

// ServicePort mirrors dynamo.mcs.v1.ServicePort.
type ServicePort struct {
	Name        string
	Protocol    Protocol
	AppProtocol string
	Port        int32
}

// SessionAffinityConfig mirrors dynamo.mcs.v1.SessionAffinityConfig.
type SessionAffinityConfig struct {
	ClientIP *ClientIPConfig
}

// ClientIPConfig mirrors dynamo.mcs.v1.ClientIPConfig.
type ClientIPConfig struct {
	TimeoutSeconds int32
}

// ExportSpec mirrors dynamo.mcs.v1.ServiceExportSpec.
type ExportSpec struct {
	Annotations             map[string]string
	Labels                  map[string]string
	ExportCreationTimestamp time.Time
	Ports                   []ServicePort
	Type                    ServiceType
	SessionAffinity         SessionAffinity
	SessionAffinityConfig   *SessionAffinityConfig
	IPFamilies              []IPFamily
	InternalTrafficPolicy   InternalTrafficPolicy
	TrafficDistribution     string
}

// EndpointConditions mirrors dynamo.mcs.v1.EndpointConditions.
type EndpointConditions struct {
	Ready       bool
	Serving     bool
	Terminating bool
}

// EndpointPort mirrors dynamo.mcs.v1.EndpointPort.
type EndpointPort struct {
	Name        string
	Protocol    Protocol
	AppProtocol string
	Port        int32
}

// Endpoint mirrors dynamo.mcs.v1.Endpoint.
type Endpoint struct {
	Addresses   []string
	AddressType AddressType
	Conditions  EndpointConditions
	Hostname    string
	NodeName    string
	Zone        string
	Ports       []EndpointPort
	Attributes  map[string]string
}

// Clone returns a deep copy.
func (r ClusterRecord) Clone() ClusterRecord {
	r.Zones = slices.Clone(r.Zones)
	r.Labels = maps.Clone(r.Labels)
	return r
}

// Clone returns a deep copy.
func (r ExportRecord) Clone() ExportRecord {
	r.Spec = r.Spec.Clone()
	return r
}

// Clone returns a deep copy.
func (r EndpointSetRecord) Clone() EndpointSetRecord {
	r.Endpoints = cloneSlice(r.Endpoints, Endpoint.Clone)
	return r
}

// Clone returns a deep copy.
func (s ExportSpec) Clone() ExportSpec {
	s.Annotations = maps.Clone(s.Annotations)
	s.Labels = maps.Clone(s.Labels)
	s.Ports = slices.Clone(s.Ports)
	if s.SessionAffinityConfig != nil {
		c := *s.SessionAffinityConfig
		if c.ClientIP != nil {
			ip := *c.ClientIP
			c.ClientIP = &ip
		}
		s.SessionAffinityConfig = &c
	}
	s.IPFamilies = slices.Clone(s.IPFamilies)
	return s
}

// Clone returns a deep copy.
func (e Endpoint) Clone() Endpoint {
	e.Addresses = slices.Clone(e.Addresses)
	e.Ports = slices.Clone(e.Ports)
	e.Attributes = maps.Clone(e.Attributes)
	return e
}

func cloneSlice[T any](in []T, clone func(T) T) []T {
	if in == nil {
		return nil
	}
	out := make([]T, len(in))
	for i, v := range in {
		out[i] = clone(v)
	}
	return out
}
