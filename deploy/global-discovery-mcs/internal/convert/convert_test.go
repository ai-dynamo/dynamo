// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package convert

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

var t0 = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

func TestEnumsRoundTrip(t *testing.T) {
	for k, v := range serviceTypes {
		assert.Equal(t, k, ServiceTypeFromProto(ServiceTypeToProto(k)))
		assert.Equal(t, v, ServiceTypeToProto(k))
	}
	for k := range protocols {
		assert.Equal(t, k, ProtocolFromProto(ProtocolToProto(k)))
	}
	for k := range sessionAffinities {
		assert.Equal(t, k, SessionAffinityFromProto(SessionAffinityToProto(k)))
	}
	for k := range ipFamilies {
		assert.Equal(t, k, IPFamilyFromProto(IPFamilyToProto(k)))
	}
	for k := range trafficPolicies {
		assert.Equal(t, k, InternalTrafficPolicyFromProto(InternalTrafficPolicyToProto(k)))
	}
	for k := range addressTypes {
		assert.Equal(t, k, AddressTypeFromProto(AddressTypeToProto(k)))
	}
	assert.Equal(t, store.ServiceType(""), ServiceTypeFromProto(mcsv1.ServiceType_SERVICE_TYPE_UNSPECIFIED))
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_EXPIRED, LeaseStateToProto(registry.LeaseStateExpired))
}

func TestTimes(t *testing.T) {
	assert.Nil(t, Timestamp(time.Time{}))
	assert.True(t, Time(nil).IsZero())
	assert.Equal(t, t0, Time(Timestamp(t0)))
	assert.Nil(t, Duration(0))
	assert.Equal(t, time.Second, Duration(time.Second).AsDuration())
}

func TestExportSpecRoundTrip(t *testing.T) {
	spec := store.ExportSpec{
		Annotations:             map[string]string{"a": "1"},
		Labels:                  map[string]string{"l": "2"},
		ExportCreationTimestamp: t0,
		Ports:                   []store.ServicePort{{Name: "http", Protocol: store.ProtocolTCP, AppProtocol: "h2c", Port: 80}},
		Type:                    store.ServiceTypeHeadless,
		SessionAffinity:         store.SessionAffinityClientIP,
		SessionAffinityConfig:   &store.SessionAffinityConfig{ClientIP: &store.ClientIPConfig{TimeoutSeconds: 30}},
		IPFamilies:              []store.IPFamily{store.IPFamilyIPv6, store.IPFamilyIPv4},
		InternalTrafficPolicy:   store.InternalTrafficPolicyLocal,
		TrafficDistribution:     "PreferClose",
	}
	assert.Equal(t, spec, ExportSpecFromProto(ExportSpecToProto(spec)))
	assert.Equal(t, store.ExportSpec{}, ExportSpecFromProto(nil))

	empty := ExportSpecFromProto(&mcsv1.ServiceExportSpec{SessionAffinityConfig: &mcsv1.SessionAffinityConfig{}})
	require.NotNil(t, empty.SessionAffinityConfig)
	assert.Nil(t, empty.SessionAffinityConfig.ClientIP)
	assert.Nil(t, SessionAffinityConfigToProto(nil))
	assert.Nil(t, ServicePortsToProto(nil))
	assert.Nil(t, IPFamiliesToProto(nil))
}

func TestEndpointRoundTrip(t *testing.T) {
	ep := store.Endpoint{
		Addresses:   []string{"10.0.0.1"},
		AddressType: store.AddressTypeIPv4,
		Conditions:  store.EndpointConditions{Ready: true, Serving: true},
		Hostname:    "pod-0",
		NodeName:    "node-a",
		Zone:        "us-east-1a",
		Ports:       []store.EndpointPort{{Name: "grpc", Protocol: store.ProtocolTCP, Port: 50071}},
		Attributes:  map[string]string{"k": "v"},
	}
	assert.Equal(t, ep, EndpointFromProto(EndpointToProto(ep)))
}

func TestCluster(t *testing.T) {
	rec := store.ClusterRecord{
		Key:          store.ClusterKey{ClusterSet: "cs", Cluster: "a"},
		Region:       "us-east-1",
		Zones:        []string{"z"},
		Provider:     "eks",
		Labels:       map[string]string{"env": "test"},
		AgentID:      "agent",
		AgentVersion: "v1",
		Lease:        store.Lease{Epoch: 7, TTL: 30 * time.Second, RenewedAt: t0},
		Generation:   3,
		RegisteredAt: t0,
		UpdatedAt:    t0,
	}
	c := ClusterToProto(rec, registry.LeaseStateActive)
	assert.Equal(t, "a", c.GetCluster())
	assert.Equal(t, uint64(7), c.GetLease().GetEpoch())
	assert.Equal(t, t0.Add(30*time.Second), c.GetLease().GetExpiresAt().AsTime())
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_ACTIVE, c.GetLease().GetState())
	assert.Nil(t, LeaseToProto(store.Lease{}, registry.LeaseStateActive))
}
