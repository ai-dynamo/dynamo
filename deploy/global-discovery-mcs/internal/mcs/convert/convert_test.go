// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package convert

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

func TestProtocols(t *testing.T) {
	for k8s, fakeReg := range map[corev1.Protocol]mcsv1.Protocol{
		"":                  mcsv1.Protocol_PROTOCOL_TCP,
		corev1.ProtocolTCP:  mcsv1.Protocol_PROTOCOL_TCP,
		corev1.ProtocolUDP:  mcsv1.Protocol_PROTOCOL_UDP,
		corev1.ProtocolSCTP: mcsv1.Protocol_PROTOCOL_SCTP,
		"QUIC":              mcsv1.Protocol_PROTOCOL_UNSPECIFIED,
	} {
		assert.Equal(t, fakeReg, ProtocolToRegistry(k8s), k8s)
	}
	for fakeReg, k8s := range map[mcsv1.Protocol]corev1.Protocol{
		mcsv1.Protocol_PROTOCOL_UNSPECIFIED: corev1.ProtocolTCP,
		mcsv1.Protocol_PROTOCOL_TCP:         corev1.ProtocolTCP,
		mcsv1.Protocol_PROTOCOL_UDP:         corev1.ProtocolUDP,
		mcsv1.Protocol_PROTOCOL_SCTP:        corev1.ProtocolSCTP,
	} {
		assert.Equal(t, k8s, ProtocolFromRegistry(fakeReg), fakeReg.String())
	}
}

func TestServiceTypes(t *testing.T) {
	tests := []struct {
		name string
		spec corev1.ServiceSpec
		want mcsv1.ServiceType
	}{
		{"cluster ip", corev1.ServiceSpec{Type: corev1.ServiceTypeClusterIP, ClusterIP: "10.0.0.1"}, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP},
		{"headless", corev1.ServiceSpec{Type: corev1.ServiceTypeClusterIP, ClusterIP: corev1.ClusterIPNone}, mcsv1.ServiceType_SERVICE_TYPE_HEADLESS},
		{"node port", corev1.ServiceSpec{Type: corev1.ServiceTypeNodePort, ClusterIP: "10.0.0.2"}, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP},
		{"load balancer", corev1.ServiceSpec{Type: corev1.ServiceTypeLoadBalancer, ClusterIP: "10.0.0.3"}, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP},
		{"unallocated", corev1.ServiceSpec{}, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, ServiceTypeOf(&corev1.Service{Spec: tt.spec}))
		})
	}
	assert.Equal(t, mcsv1beta1.Headless, ImportTypeFromRegistry(mcsv1.ServiceType_SERVICE_TYPE_HEADLESS))
	assert.Equal(t, mcsv1beta1.ClusterSetIP, ImportTypeFromRegistry(mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP))
	assert.Equal(t, mcsv1beta1.ClusterSetIP, ImportTypeFromRegistry(mcsv1.ServiceType_SERVICE_TYPE_UNSPECIFIED))
}

func TestSessionAffinity(t *testing.T) {
	assert.Equal(t, mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP, SessionAffinityToRegistry(corev1.ServiceAffinityClientIP))
	assert.Equal(t, mcsv1.SessionAffinity_SESSION_AFFINITY_NONE, SessionAffinityToRegistry(corev1.ServiceAffinityNone))
	assert.Equal(t, mcsv1.SessionAffinity_SESSION_AFFINITY_NONE, SessionAffinityToRegistry(""))
	assert.Equal(t, corev1.ServiceAffinityClientIP, SessionAffinityFromRegistry(mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP))
	assert.Equal(t, corev1.ServiceAffinityNone, SessionAffinityFromRegistry(mcsv1.SessionAffinity_SESSION_AFFINITY_UNSPECIFIED))

	timeout := int32(600)
	cfg := SessionAffinityConfigToRegistry(&corev1.SessionAffinityConfig{ClientIP: &corev1.ClientIPConfig{TimeoutSeconds: &timeout}})
	assert.Equal(t, int32(600), cfg.GetClientIp().GetTimeoutSeconds())
	assert.Nil(t, SessionAffinityConfigToRegistry(nil))
	assert.Nil(t, SessionAffinityConfigToRegistry(&corev1.SessionAffinityConfig{}))
	back := SessionAffinityConfigFromRegistry(cfg)
	assert.Equal(t, int32(600), *back.ClientIP.TimeoutSeconds)
	assert.Nil(t, SessionAffinityConfigFromRegistry(nil))
}

func TestIPFamiliesAndPolicies(t *testing.T) {
	fams := IPFamiliesToRegistry([]corev1.IPFamily{corev1.IPv6Protocol, corev1.IPv4Protocol, "IPv5"})
	assert.Equal(t, []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV6, mcsv1.IPFamily_IP_FAMILY_IPV4}, fams)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv6Protocol, corev1.IPv4Protocol},
		IPFamiliesFromRegistry(append(fams, mcsv1.IPFamily_IP_FAMILY_UNSPECIFIED)))

	local, cluster, other := corev1.ServiceInternalTrafficPolicyLocal, corev1.ServiceInternalTrafficPolicyCluster, corev1.ServiceInternalTrafficPolicy("x")
	assert.Equal(t, mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_LOCAL, InternalTrafficPolicyToRegistry(&local))
	assert.Equal(t, mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_CLUSTER, InternalTrafficPolicyToRegistry(&cluster))
	assert.Equal(t, mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_UNSPECIFIED, InternalTrafficPolicyToRegistry(&other))
	assert.Equal(t, mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_UNSPECIFIED, InternalTrafficPolicyToRegistry(nil))
}

func TestAddressTypes(t *testing.T) {
	for k8s, fakeReg := range map[discoveryv1.AddressType]mcsv1.AddressType{
		discoveryv1.AddressTypeIPv4: mcsv1.AddressType_ADDRESS_TYPE_IPV4,
		discoveryv1.AddressTypeIPv6: mcsv1.AddressType_ADDRESS_TYPE_IPV6,
		discoveryv1.AddressTypeFQDN: mcsv1.AddressType_ADDRESS_TYPE_FQDN,
		"other":                     mcsv1.AddressType_ADDRESS_TYPE_UNSPECIFIED,
	} {
		assert.Equal(t, fakeReg, AddressTypeToRegistry(k8s), k8s)
		if fakeReg != mcsv1.AddressType_ADDRESS_TYPE_UNSPECIFIED {
			assert.Equal(t, k8s, AddressTypeFromRegistry(fakeReg, nil))
		}
	}
	unspecified := mcsv1.AddressType_ADDRESS_TYPE_UNSPECIFIED
	assert.Equal(t, discoveryv1.AddressTypeIPv4, AddressTypeFromRegistry(unspecified, []string{"10.0.0.1"}))
	assert.Equal(t, discoveryv1.AddressTypeIPv6, AddressTypeFromRegistry(unspecified, []string{"fd00::1"}))
	assert.Equal(t, discoveryv1.AddressTypeFQDN, AddressTypeFromRegistry(unspecified, []string{"pod.example"}))
	assert.Equal(t, discoveryv1.AddressTypeFQDN, AddressTypeFromRegistry(unspecified, nil))
}
