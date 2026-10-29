// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// Attribute names. Conditions and updates use the top-level attributes;
// everything else of a record is in body, a serialized dynamo.mcs.v1
// message. A field kept at top level is never also set in body, so each
// value has one source of truth.
const (
	attrPK             = "pk"
	attrSK             = "sk"
	attrBody           = "body"
	attrGen            = "gen"
	attrLeaseEpoch     = "lease_epoch"
	attrLeaseRenewedAt = "lease_renewed_at"
	attrUpdatedAt      = "updated_at"
	attrExpiresAt      = "expires_at"
)

// item is the attribute layout shared by every item kind. Times are Unix
// nanoseconds, except expires_at, which DynamoDB TTL reads as Unix seconds.
// Absent pointers mean a zero time.Time; gen is a pointer so that a stored
// generation 0 is still present for the gen < :gen condition.
type item struct {
	PK             string  `dynamodbav:"pk"`
	SK             string  `dynamodbav:"sk"`
	Body           []byte  `dynamodbav:"body,omitempty"`
	Gen            *uint64 `dynamodbav:"gen,omitempty"`
	LeaseEpoch     uint64  `dynamodbav:"lease_epoch,omitempty"`
	LeaseRenewedAt *int64  `dynamodbav:"lease_renewed_at,omitempty"`
	UpdatedAt      *int64  `dynamodbav:"updated_at,omitempty"`
	ExpiresAt      int64   `dynamodbav:"expires_at,omitempty"`
}

func (it item) attributes() (map[string]ddbtypes.AttributeValue, error) {
	av, err := attributevalue.MarshalMap(it)
	if err != nil {
		return nil, fmt.Errorf("dynamodb: marshal item %s/%s: %w", it.PK, it.SK, err)
	}
	return av, nil
}

func parseItem(av map[string]ddbtypes.AttributeValue) (item, error) {
	var it item
	if err := attributevalue.UnmarshalMap(av, &it); err != nil {
		return item{}, fmt.Errorf("dynamodb: unmarshal item: %w", err)
	}
	return it, nil
}

func (it item) gen() uint64 {
	if it.Gen == nil {
		return 0
	}
	return *it.Gen
}

var marshalOptions = proto.MarshalOptions{Deterministic: true}

func marshalBody(m proto.Message) ([]byte, error) {
	b, err := marshalOptions.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("dynamodb: marshal body: %w", err)
	}
	return b, nil
}

func unmarshalBody(it item, m proto.Message) error {
	if err := proto.Unmarshal(it.Body, m); err != nil {
		return fmt.Errorf("dynamodb: unmarshal body of %s/%s: %w", it.PK, it.SK, err)
	}
	return nil
}

// Times and durations.

func unixNanos(t time.Time) *int64 {
	if t.IsZero() {
		return nil
	}
	n := t.UnixNano()
	return &n
}

func fromUnixNanos(n *int64) time.Time {
	if n == nil {
		return time.Time{}
	}
	return time.Unix(0, *n).UTC()
}

func timestamp(t time.Time) *timestamppb.Timestamp {
	if t.IsZero() {
		return nil
	}
	return timestamppb.New(t)
}

func fromTimestamp(ts *timestamppb.Timestamp) time.Time {
	if ts == nil {
		return time.Time{}
	}
	return ts.AsTime()
}

func leaseTTL(ttl time.Duration) *mcsv1.Lease {
	if ttl == 0 {
		return nil
	}
	return &mcsv1.Lease{Ttl: durationpb.New(ttl)}
}

func fromLeaseTTL(l *mcsv1.Lease) time.Duration {
	if l == nil || l.Ttl == nil {
		return 0
	}
	return l.Ttl.AsDuration()
}

// Clusters.

func encodeCluster(rec store.ClusterRecord, expiresAt int64) (item, error) {
	body, err := marshalBody(&mcsv1.Cluster{
		Region:       rec.Region,
		Zones:        rec.Zones,
		Provider:     rec.Provider,
		Labels:       rec.Labels,
		AgentId:      rec.AgentID,
		AgentVersion: rec.AgentVersion,
		Lease:        leaseTTL(rec.Lease.TTL),
		RegisteredAt: timestamp(rec.RegisteredAt),
	})
	if err != nil {
		return item{}, err
	}
	k := clusterKey(rec.Key)
	return item{
		PK:             k.pk,
		SK:             k.sk,
		Body:           body,
		Gen:            new(rec.Generation),
		LeaseEpoch:     rec.Lease.Epoch,
		LeaseRenewedAt: unixNanos(rec.Lease.RenewedAt),
		UpdatedAt:      unixNanos(rec.UpdatedAt),
		ExpiresAt:      expiresAt,
	}, nil
}

func decodeCluster(it item) (store.ClusterRecord, error) {
	cs, ok := parseClusterSetPK(it.PK)
	if !ok {
		return store.ClusterRecord{}, fmt.Errorf("dynamodb: %q is not a clusterset partition", it.PK)
	}
	key, ok := parseClusterSK(cs, it.SK)
	if !ok {
		return store.ClusterRecord{}, fmt.Errorf("dynamodb: %q is not a cluster key", it.SK)
	}
	var body mcsv1.Cluster
	if err := unmarshalBody(it, &body); err != nil {
		return store.ClusterRecord{}, err
	}
	return store.ClusterRecord{
		Key:          key,
		Region:       body.Region,
		Zones:        body.Zones,
		Provider:     body.Provider,
		Labels:       body.Labels,
		AgentID:      body.AgentId,
		AgentVersion: body.AgentVersion,
		Lease: store.Lease{
			Epoch:     it.LeaseEpoch,
			TTL:       fromLeaseTTL(body.Lease),
			RenewedAt: fromUnixNanos(it.LeaseRenewedAt),
		},
		Generation:   it.gen(),
		RegisteredAt: fromTimestamp(body.RegisteredAt),
		UpdatedAt:    fromUnixNanos(it.UpdatedAt),
	}, nil
}

// Exports.

func encodeExport(rec store.ExportRecord) (item, error) {
	var e encoder
	spec := e.exportSpec(rec.Spec)
	if e.err != nil {
		return item{}, fmt.Errorf("dynamodb: export %s/%s/%s: %w", rec.Key.Cluster, rec.Key.Namespace, rec.Key.Name, e.err)
	}
	body, err := marshalBody(spec)
	if err != nil {
		return item{}, err
	}
	k := exportKey(rec.Key)
	return item{PK: k.pk, SK: k.sk, Body: body, Gen: new(rec.Generation), UpdatedAt: unixNanos(rec.UpdatedAt)}, nil
}

func decodeExport(it item) (store.ExportRecord, error) {
	cs, ok := parseClusterSetPK(it.PK)
	if !ok {
		return store.ExportRecord{}, fmt.Errorf("dynamodb: %q is not a clusterset partition", it.PK)
	}
	key, ok := parseExportSK(cs, it.SK)
	if !ok {
		return store.ExportRecord{}, fmt.Errorf("dynamodb: %q is not an export key", it.SK)
	}
	var body mcsv1.ServiceExportSpec
	if err := unmarshalBody(it, &body); err != nil {
		return store.ExportRecord{}, err
	}
	return store.ExportRecord{
		Key:        key,
		Spec:       decodeExportSpec(&body),
		Generation: it.gen(),
		UpdatedAt:  fromUnixNanos(it.UpdatedAt),
	}, nil
}

// Endpoint sets.

func encodeEndpointSet(rec store.EndpointSetRecord) (item, error) {
	var e encoder
	msg := &mcsv1.ClusterEndpoints{Endpoints: make([]*mcsv1.Endpoint, 0, len(rec.Endpoints))}
	for _, ep := range rec.Endpoints {
		msg.Endpoints = append(msg.Endpoints, e.endpoint(ep))
	}
	if e.err != nil {
		return item{}, fmt.Errorf("dynamodb: endpoint set %s/%s/%s: %w", rec.Key.Cluster, rec.Key.Namespace, rec.Key.Name, e.err)
	}
	body, err := marshalBody(msg)
	if err != nil {
		return item{}, err
	}
	k := endpointSetKey(rec.Key)
	return item{PK: k.pk, SK: k.sk, Body: body, Gen: new(rec.Generation), UpdatedAt: unixNanos(rec.UpdatedAt)}, nil
}

func decodeEndpointSet(it item) (store.EndpointSetRecord, error) {
	cs, cluster, ok := parseEndpointPK(it.PK)
	if !ok {
		return store.EndpointSetRecord{}, fmt.Errorf("dynamodb: %q is not an endpoint partition", it.PK)
	}
	key, ok := parseEndpointSetSK(cs, cluster, it.SK)
	if !ok {
		return store.EndpointSetRecord{}, fmt.Errorf("dynamodb: %q is not an endpoint set key", it.SK)
	}
	var body mcsv1.ClusterEndpoints
	if err := unmarshalBody(it, &body); err != nil {
		return store.EndpointSetRecord{}, err
	}
	rec := store.EndpointSetRecord{Key: key, Generation: it.gen(), UpdatedAt: fromUnixNanos(it.UpdatedAt)}
	if len(body.Endpoints) > 0 {
		rec.Endpoints = make([]store.Endpoint, 0, len(body.Endpoints))
		for _, ep := range body.Endpoints {
			rec.Endpoints = append(rec.Endpoints, decodeEndpoint(ep))
		}
	}
	return rec, nil
}

type enumMap[S ~string, P ~int32] struct {
	name      string
	toProto   map[S]P
	fromProto map[P]S
}

func newEnumMap[S ~string, P ~int32](name string, pairs map[S]P) enumMap[S, P] {
	m := enumMap[S, P]{name: name, toProto: pairs, fromProto: make(map[P]S, len(pairs))}
	for s, p := range pairs {
		m.fromProto[p] = s
	}
	return m
}

func (m enumMap[S, P]) encode(s S) (P, error) {
	if s == "" {
		return 0, nil
	}
	p, ok := m.toProto[s]
	if !ok {
		return 0, fmt.Errorf("unknown %s %q", m.name, s)
	}
	return p, nil
}

func (m enumMap[S, P]) decode(p P) S {
	return m.fromProto[p]
}

var (
	serviceTypes = newEnumMap("service type", map[store.ServiceType]mcsv1.ServiceType{
		store.ServiceTypeClusterSetIP: mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
		store.ServiceTypeHeadless:     mcsv1.ServiceType_SERVICE_TYPE_HEADLESS,
	})
	protocols = newEnumMap("protocol", map[store.Protocol]mcsv1.Protocol{
		store.ProtocolTCP:  mcsv1.Protocol_PROTOCOL_TCP,
		store.ProtocolUDP:  mcsv1.Protocol_PROTOCOL_UDP,
		store.ProtocolSCTP: mcsv1.Protocol_PROTOCOL_SCTP,
	})
	sessionAffinities = newEnumMap("session affinity", map[store.SessionAffinity]mcsv1.SessionAffinity{
		store.SessionAffinityNone:     mcsv1.SessionAffinity_SESSION_AFFINITY_NONE,
		store.SessionAffinityClientIP: mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP,
	})
	ipFamilies = newEnumMap("IP family", map[store.IPFamily]mcsv1.IPFamily{
		store.IPFamilyIPv4: mcsv1.IPFamily_IP_FAMILY_IPV4,
		store.IPFamilyIPv6: mcsv1.IPFamily_IP_FAMILY_IPV6,
	})
	trafficPolicies = newEnumMap("internal traffic policy", map[store.InternalTrafficPolicy]mcsv1.InternalTrafficPolicy{
		store.InternalTrafficPolicyCluster: mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_CLUSTER,
		store.InternalTrafficPolicyLocal:   mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_LOCAL,
	})
	addressTypes = newEnumMap("address type", map[store.AddressType]mcsv1.AddressType{
		store.AddressTypeIPv4: mcsv1.AddressType_ADDRESS_TYPE_IPV4,
		store.AddressTypeIPv6: mcsv1.AddressType_ADDRESS_TYPE_IPV6,
		store.AddressTypeFQDN: mcsv1.AddressType_ADDRESS_TYPE_FQDN,
	})
)

// encoder converts store values to proto messages and keeps the first
// enumeration error.
type encoder struct {
	err error
}

func encodeEnum[S ~string, P ~int32](e *encoder, m enumMap[S, P], s S) P {
	p, err := m.encode(s)
	if err != nil && e.err == nil {
		e.err = err
	}
	return p
}

func (e *encoder) exportSpec(s store.ExportSpec) *mcsv1.ServiceExportSpec {
	msg := &mcsv1.ServiceExportSpec{
		Annotations:             s.Annotations,
		Labels:                  s.Labels,
		ExportCreationTimestamp: timestamp(s.ExportCreationTimestamp),
		Type:                    encodeEnum(e, serviceTypes, s.Type),
		SessionAffinity:         encodeEnum(e, sessionAffinities, s.SessionAffinity),
		InternalTrafficPolicy:   encodeEnum(e, trafficPolicies, s.InternalTrafficPolicy),
		TrafficDistribution:     s.TrafficDistribution,
	}
	for _, p := range s.Ports {
		msg.Ports = append(msg.Ports, &mcsv1.ServicePort{
			Name:        p.Name,
			Protocol:    encodeEnum(e, protocols, p.Protocol),
			AppProtocol: p.AppProtocol,
			Port:        p.Port,
		})
	}
	if c := s.SessionAffinityConfig; c != nil {
		msg.SessionAffinityConfig = &mcsv1.SessionAffinityConfig{}
		if c.ClientIP != nil {
			msg.SessionAffinityConfig.ClientIp = &mcsv1.ClientIPConfig{TimeoutSeconds: c.ClientIP.TimeoutSeconds}
		}
	}
	for _, f := range s.IPFamilies {
		msg.IpFamilies = append(msg.IpFamilies, encodeEnum(e, ipFamilies, f))
	}
	return msg
}

func decodeExportSpec(msg *mcsv1.ServiceExportSpec) store.ExportSpec {
	s := store.ExportSpec{
		Annotations:             msg.Annotations,
		Labels:                  msg.Labels,
		ExportCreationTimestamp: fromTimestamp(msg.ExportCreationTimestamp),
		Type:                    serviceTypes.decode(msg.Type),
		SessionAffinity:         sessionAffinities.decode(msg.SessionAffinity),
		InternalTrafficPolicy:   trafficPolicies.decode(msg.InternalTrafficPolicy),
		TrafficDistribution:     msg.TrafficDistribution,
	}
	for _, p := range msg.Ports {
		s.Ports = append(s.Ports, store.ServicePort{
			Name:        p.Name,
			Protocol:    protocols.decode(p.Protocol),
			AppProtocol: p.AppProtocol,
			Port:        p.Port,
		})
	}
	if c := msg.SessionAffinityConfig; c != nil {
		s.SessionAffinityConfig = &store.SessionAffinityConfig{}
		if c.ClientIp != nil {
			s.SessionAffinityConfig.ClientIP = &store.ClientIPConfig{TimeoutSeconds: c.ClientIp.TimeoutSeconds}
		}
	}
	for _, f := range msg.IpFamilies {
		s.IPFamilies = append(s.IPFamilies, ipFamilies.decode(f))
	}
	return s
}

func (e *encoder) endpoint(ep store.Endpoint) *mcsv1.Endpoint {
	msg := &mcsv1.Endpoint{
		Addresses:   ep.Addresses,
		AddressType: encodeEnum(e, addressTypes, ep.AddressType),
		Conditions: &mcsv1.EndpointConditions{
			Ready:       ep.Conditions.Ready,
			Serving:     ep.Conditions.Serving,
			Terminating: ep.Conditions.Terminating,
		},
		Hostname:   ep.Hostname,
		NodeName:   ep.NodeName,
		Zone:       ep.Zone,
		Attributes: ep.Attributes,
	}
	for _, p := range ep.Ports {
		msg.Ports = append(msg.Ports, &mcsv1.EndpointPort{
			Name:        p.Name,
			Protocol:    encodeEnum(e, protocols, p.Protocol),
			AppProtocol: p.AppProtocol,
			Port:        p.Port,
		})
	}
	return msg
}

func decodeEndpoint(msg *mcsv1.Endpoint) store.Endpoint {
	ep := store.Endpoint{
		Addresses:   msg.Addresses,
		AddressType: addressTypes.decode(msg.AddressType),
		Hostname:    msg.Hostname,
		NodeName:    msg.NodeName,
		Zone:        msg.Zone,
		Attributes:  msg.Attributes,
	}
	if c := msg.Conditions; c != nil {
		ep.Conditions = store.EndpointConditions{Ready: c.Ready, Serving: c.Serving, Terminating: c.Terminating}
	}
	for _, p := range msg.Ports {
		ep.Ports = append(ep.Ports, store.EndpointPort{
			Name:        p.Name,
			Protocol:    protocols.decode(p.Protocol),
			AppProtocol: p.AppProtocol,
			Port:        p.Port,
		})
	}
	return ep
}
