// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package convert maps between Kubernetes Service and EndpointSlice spellings
// and the dynamo.mcs.v1 messages, in both directions. Values the target side
// does not know map to the unspecified value (towards the registry) or to the
// Kubernetes default (towards Kubernetes).
package convert

import (
	"net/netip"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// ProtocolToRegistry maps a Kubernetes protocol; empty means TCP.
func ProtocolToRegistry(p corev1.Protocol) mcsv1.Protocol {
	switch p {
	case corev1.ProtocolTCP, "":
		return mcsv1.Protocol_PROTOCOL_TCP
	case corev1.ProtocolUDP:
		return mcsv1.Protocol_PROTOCOL_UDP
	case corev1.ProtocolSCTP:
		return mcsv1.Protocol_PROTOCOL_SCTP
	}
	return mcsv1.Protocol_PROTOCOL_UNSPECIFIED
}

// ProtocolFromRegistry maps a registry protocol; unspecified means TCP.
func ProtocolFromRegistry(p mcsv1.Protocol) corev1.Protocol {
	switch p {
	case mcsv1.Protocol_PROTOCOL_UDP:
		return corev1.ProtocolUDP
	case mcsv1.Protocol_PROTOCOL_SCTP:
		return corev1.ProtocolSCTP
	}
	return corev1.ProtocolTCP
}

// ServiceTypeOf returns HEADLESS for a Service with clusterIP None and
// CLUSTER_SET_IP otherwise (ClusterIP, NodePort and LoadBalancer Services
// export their pod endpoints). ExternalName Services cannot be exported;
// the caller rejects them first.
func ServiceTypeOf(svc *corev1.Service) mcsv1.ServiceType {
	if svc.Spec.ClusterIP == corev1.ClusterIPNone {
		return mcsv1.ServiceType_SERVICE_TYPE_HEADLESS
	}
	return mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP
}

// ImportTypeFromRegistry maps an import type; unspecified means ClusterSetIP.
func ImportTypeFromRegistry(t mcsv1.ServiceType) mcsv1beta1.ServiceImportType {
	if t == mcsv1.ServiceType_SERVICE_TYPE_HEADLESS {
		return mcsv1beta1.Headless
	}
	return mcsv1beta1.ClusterSetIP
}

// SessionAffinityToRegistry maps a Service session affinity; empty means None.
func SessionAffinityToRegistry(a corev1.ServiceAffinity) mcsv1.SessionAffinity {
	if a == corev1.ServiceAffinityClientIP {
		return mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP
	}
	return mcsv1.SessionAffinity_SESSION_AFFINITY_NONE
}

// SessionAffinityFromRegistry maps a registry session affinity; unspecified
// means None.
func SessionAffinityFromRegistry(a mcsv1.SessionAffinity) corev1.ServiceAffinity {
	if a == mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP {
		return corev1.ServiceAffinityClientIP
	}
	return corev1.ServiceAffinityNone
}

// SessionAffinityConfigToRegistry maps the ClientIP timeout, or returns nil.
func SessionAffinityConfigToRegistry(c *corev1.SessionAffinityConfig) *mcsv1.SessionAffinityConfig {
	if c == nil || c.ClientIP == nil || c.ClientIP.TimeoutSeconds == nil {
		return nil
	}
	return &mcsv1.SessionAffinityConfig{ClientIp: &mcsv1.ClientIPConfig{TimeoutSeconds: *c.ClientIP.TimeoutSeconds}}
}

// SessionAffinityConfigFromRegistry maps the ClientIP timeout, or returns nil.
func SessionAffinityConfigFromRegistry(c *mcsv1.SessionAffinityConfig) *corev1.SessionAffinityConfig {
	if c.GetClientIp() == nil {
		return nil
	}
	t := c.GetClientIp().GetTimeoutSeconds()
	return &corev1.SessionAffinityConfig{ClientIP: &corev1.ClientIPConfig{TimeoutSeconds: &t}}
}

// IPFamiliesToRegistry maps IP families, dropping unknown ones.
func IPFamiliesToRegistry(fams []corev1.IPFamily) []mcsv1.IPFamily {
	var out []mcsv1.IPFamily
	for _, f := range fams {
		switch f {
		case corev1.IPv4Protocol:
			out = append(out, mcsv1.IPFamily_IP_FAMILY_IPV4)
		case corev1.IPv6Protocol:
			out = append(out, mcsv1.IPFamily_IP_FAMILY_IPV6)
		}
	}
	return out
}

// IPFamiliesFromRegistry maps IP families, dropping unspecified ones.
func IPFamiliesFromRegistry(fams []mcsv1.IPFamily) []corev1.IPFamily {
	var out []corev1.IPFamily
	for _, f := range fams {
		switch f {
		case mcsv1.IPFamily_IP_FAMILY_IPV4:
			out = append(out, corev1.IPv4Protocol)
		case mcsv1.IPFamily_IP_FAMILY_IPV6:
			out = append(out, corev1.IPv6Protocol)
		}
	}
	return out
}

// InternalTrafficPolicyToRegistry maps a Service internal traffic policy.
func InternalTrafficPolicyToRegistry(p *corev1.ServiceInternalTrafficPolicy) mcsv1.InternalTrafficPolicy {
	if p == nil {
		return mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_UNSPECIFIED
	}
	switch *p {
	case corev1.ServiceInternalTrafficPolicyCluster:
		return mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_CLUSTER
	case corev1.ServiceInternalTrafficPolicyLocal:
		return mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_LOCAL
	}
	return mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_UNSPECIFIED
}

// AddressTypeToRegistry maps an EndpointSlice address type.
func AddressTypeToRegistry(t discoveryv1.AddressType) mcsv1.AddressType {
	switch t {
	case discoveryv1.AddressTypeIPv4:
		return mcsv1.AddressType_ADDRESS_TYPE_IPV4
	case discoveryv1.AddressTypeIPv6:
		return mcsv1.AddressType_ADDRESS_TYPE_IPV6
	case discoveryv1.AddressTypeFQDN:
		return mcsv1.AddressType_ADDRESS_TYPE_FQDN
	}
	return mcsv1.AddressType_ADDRESS_TYPE_UNSPECIFIED
}

// AddressTypeFromRegistry maps a registry address type; unspecified is inferred
// from the first address.
func AddressTypeFromRegistry(t mcsv1.AddressType, addresses []string) discoveryv1.AddressType {
	switch t {
	case mcsv1.AddressType_ADDRESS_TYPE_IPV4:
		return discoveryv1.AddressTypeIPv4
	case mcsv1.AddressType_ADDRESS_TYPE_IPV6:
		return discoveryv1.AddressTypeIPv6
	case mcsv1.AddressType_ADDRESS_TYPE_FQDN:
		return discoveryv1.AddressTypeFQDN
	}
	if len(addresses) > 0 {
		if ip, err := netip.ParseAddr(addresses[0]); err == nil {
			if ip.Is4() {
				return discoveryv1.AddressTypeIPv4
			}
			return discoveryv1.AddressTypeIPv6
		}
	}
	return discoveryv1.AddressTypeFQDN
}
