// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package export

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// Prefixed keys a Service may carry; the controller forwards them.
const (
	tierLabel        = api.Prefix + "tier"
	modelsAnnotation = api.Prefix + "models"
)

func TestBuildSpec(t *testing.T) {
	created := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	timeout := int32(300)
	local := corev1.ServiceInternalTrafficPolicyLocal
	svc := &corev1.Service{
		Namespace: "llm", Name: "frontend",
		Labels: map[string]string{
			api.LabelExport: api.ExportEnabled, tierLabel: "gold",
			api.LabelManagedBy: "someone", api.LabelClusterSet: "spoofed", "app": "frontend",
		},
		Annotations: map[string]string{modelsAnnotation: "m1,m2", "kubectl.kubernetes.io/last-applied-configuration": "{}"},
		Spec: corev1.ServiceSpec{
			Type:                  corev1.ServiceTypeLoadBalancer,
			ClusterIP:             "10.0.0.1",
			SessionAffinity:       corev1.ServiceAffinityClientIP,
			SessionAffinityConfig: &corev1.SessionAffinityConfig{ClientIP: &corev1.ClientIPConfig{TimeoutSeconds: &timeout}},
			IPFamilies:            []corev1.IPFamily{corev1.IPv4Protocol},
			InternalTrafficPolicy: &local,
			TrafficDistribution:   new("PreferClose"),
			Ports: []corev1.ServicePort{
				{Name: "http", Protocol: corev1.ProtocolTCP, Port: 8000, AppProtocol: new("http")},
				{Name: "metrics", Port: 9090},
			},
		},
	}
	se := &mcsv1beta1.ServiceExport{
		CreationTimestamp: metav1.NewTime(created),
		Spec: mcsv1beta1.ServiceExportSpec{
			ExportedLabels:      map[string]string{"team": "inference", tierLabel: "explicit"},
			ExportedAnnotations: map[string]string{"note": "x"},
		},
	}
	spec := BuildSpec(svc, se)
	want := &mcsv1.ServiceExportSpec{
		Labels:                map[string]string{tierLabel: "explicit", "team": "inference"},
		Annotations:           map[string]string{modelsAnnotation: "m1,m2", "note": "x"},
		Type:                  mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
		SessionAffinity:       mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP,
		SessionAffinityConfig: &mcsv1.SessionAffinityConfig{ClientIp: &mcsv1.ClientIPConfig{TimeoutSeconds: 300}},
		IpFamilies:            []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV4},
		InternalTrafficPolicy: mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_LOCAL,
		TrafficDistribution:   "PreferClose",
		Ports: []*mcsv1.ServicePort{
			{Name: "http", Protocol: mcsv1.Protocol_PROTOCOL_TCP, AppProtocol: "http", Port: 8000},
			{Name: "metrics", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 9090},
		},
	}
	want.ExportCreationTimestamp = spec.GetExportCreationTimestamp()
	assert.True(t, proto.Equal(want, spec), "got %v", spec)
	assert.Equal(t, created, spec.GetExportCreationTimestamp().AsTime())

	// Headless, no affinity: the affinity config is dropped; no labels.
	svc.Spec.ClusterIP = corev1.ClusterIPNone
	svc.Spec.SessionAffinity = corev1.ServiceAffinityNone
	svc.Labels, svc.Annotations = nil, nil
	spec = BuildSpec(svc, &mcsv1beta1.ServiceExport{})
	assert.Equal(t, mcsv1.ServiceType_SERVICE_TYPE_HEADLESS, spec.GetType())
	assert.Nil(t, spec.GetSessionAffinityConfig())
	assert.Nil(t, spec.GetLabels())
	assert.Nil(t, spec.GetExportCreationTimestamp())
}

// Prefixed annotations are forwarded as they are, and the target reference
// attribute is the one consumers of the registry read.
func TestPrefixedAnnotationsAreForwarded(t *testing.T) {
	assert.Equal(t, api.EndpointAttrTargetName, AttrTargetName)
	assert.Equal(t, api.EndpointAttrTargetNamespace, AttrTargetNamespace)
	svc := &corev1.Service{
		Namespace: "llm", Name: "router",
		Labels:      map[string]string{api.LabelExport: api.ExportEnabled},
		Annotations: map[string]string{modelsAnnotation: "m1", api.Prefix + "port-name": "grpc"},
		Spec:        corev1.ServiceSpec{ClusterIP: "10.0.0.1", Ports: []corev1.ServicePort{{Name: "grpc", Port: 50071}}},
	}
	spec := BuildSpec(svc, &mcsv1beta1.ServiceExport{})
	assert.Equal(t, svc.Annotations, spec.GetAnnotations())
	assert.Nil(t, spec.GetLabels(), "the export trigger label is not forwarded")
}

func TestSpecHashIsStable(t *testing.T) {
	a := &mcsv1.ServiceExportSpec{Labels: map[string]string{}, Annotations: map[string]string{}}
	b := &mcsv1.ServiceExportSpec{Labels: map[string]string{}, Annotations: map[string]string{}}
	for i := range 50 {
		a.Labels[fmt.Sprintf("k%d", i)] = "v"
		b.Labels[fmt.Sprintf("k%d", 49-i)] = "v"
	}
	assert.Equal(t, SpecHash(a), SpecHash(b))
	b.Labels["k0"] = "changed"
	assert.NotEqual(t, SpecHash(a), SpecHash(b))
}

func slice(name string, addressType discoveryv1.AddressType, ports []discoveryv1.EndpointPort, eps ...discoveryv1.Endpoint) discoveryv1.EndpointSlice {
	return discoveryv1.EndpointSlice{Name: name, AddressType: addressType, Ports: ports, Endpoints: eps}
}

func TestBuildEndpoints(t *testing.T) {
	ports := []discoveryv1.EndpointPort{
		{Name: new("metrics"), Protocol: new(corev1.ProtocolTCP), Port: new(int32(9090))},
		{Name: new("http"), Port: new(int32(8000)), AppProtocol: new("http")},
	}
	pod := &corev1.ObjectReference{Kind: "Pod", Namespace: "llm", Name: "frontend-0", UID: types.UID("u0")}
	slices := []discoveryv1.EndpointSlice{
		slice("s1", discoveryv1.AddressTypeIPv4, ports,
			discoveryv1.Endpoint{Addresses: []string{"10.0.0.2"}, Conditions: discoveryv1.EndpointConditions{Ready: new(false), Serving: new(true), Terminating: new(true)},
				Hostname: new("frontend-1"), NodeName: new("node-1"), Zone: new("us-east-1a")},
			discoveryv1.Endpoint{Addresses: []string{"10.0.0.1"}, TargetRef: pod},
			discoveryv1.Endpoint{Addresses: nil},
		),
		// The EndpointSlice controller may list an endpoint twice while it
		// moves between slices.
		slice("s2", discoveryv1.AddressTypeIPv4, ports, discoveryv1.Endpoint{Addresses: []string{"10.0.0.1"}, TargetRef: pod}),
		slice("s3", discoveryv1.AddressTypeIPv6, nil, discoveryv1.Endpoint{Addresses: []string{"fd00::1"}, Conditions: discoveryv1.EndpointConditions{Ready: new(true)}},
			// Not ready with serving unset: serving, as the API defines a nil serving.
			discoveryv1.Endpoint{Addresses: []string{"fd00::2"}, Conditions: discoveryv1.EndpointConditions{Ready: new(false)}}),
	}
	deleting := slice("s4", discoveryv1.AddressTypeIPv4, ports, discoveryv1.Endpoint{Addresses: []string{"10.9.9.9"}})
	deleting.DeletionTimestamp = new(metav1.Now())
	slices = append(slices, deleting)

	eps, dropped := BuildEndpoints(slices, 0)
	require.Zero(t, dropped)
	wantPorts := []*mcsv1.EndpointPort{
		{Name: "http", Protocol: mcsv1.Protocol_PROTOCOL_TCP, AppProtocol: "http", Port: 8000},
		{Name: "metrics", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 9090},
	}
	want := []*mcsv1.Endpoint{
		{Addresses: []string{"10.0.0.1"}, AddressType: mcsv1.AddressType_ADDRESS_TYPE_IPV4,
			Conditions: &mcsv1.EndpointConditions{Ready: true, Serving: true}, Ports: wantPorts,
			Attributes: map[string]string{AttrTargetKind: "Pod", AttrTargetNamespace: "llm", AttrTargetName: "frontend-0", AttrTargetUID: "u0"}},
		{Addresses: []string{"10.0.0.2"}, AddressType: mcsv1.AddressType_ADDRESS_TYPE_IPV4,
			Conditions: &mcsv1.EndpointConditions{Ready: false, Serving: true, Terminating: true}, Ports: wantPorts,
			Hostname: "frontend-1", NodeName: "node-1", Zone: "us-east-1a"},
		{Addresses: []string{"fd00::1"}, AddressType: mcsv1.AddressType_ADDRESS_TYPE_IPV6,
			Conditions: &mcsv1.EndpointConditions{Ready: true, Serving: true}},
		{Addresses: []string{"fd00::2"}, AddressType: mcsv1.AddressType_ADDRESS_TYPE_IPV6,
			Conditions: &mcsv1.EndpointConditions{Ready: false, Serving: true}},
	}
	require.Len(t, eps, len(want))
	for i := range want {
		assert.True(t, proto.Equal(want[i], eps[i]), "endpoint %d: got %v", i, eps[i])
	}

	// The hash does not depend on the order of slices or endpoints.
	reversed := []discoveryv1.EndpointSlice{slices[3], slices[2], slices[1], slices[0]}
	reversed[3].Endpoints = []discoveryv1.Endpoint{slices[0].Endpoints[2], slices[0].Endpoints[1], slices[0].Endpoints[0]}
	reversed[1].Endpoints = []discoveryv1.Endpoint{slices[2].Endpoints[1], slices[2].Endpoints[0]}
	again, _ := BuildEndpoints(reversed, 0)
	assert.Equal(t, EndpointsHash(eps), EndpointsHash(again))

	// A ready flip changes it.
	slices[2].Endpoints[0].Conditions.Ready = new(false)
	changed, _ := BuildEndpoints(slices, 0)
	assert.NotEqual(t, EndpointsHash(eps), EndpointsHash(changed))
}

func TestBuildEndpointsTruncatesReadyFirst(t *testing.T) {
	var eps []discoveryv1.Endpoint
	for i := range 10 {
		eps = append(eps, discoveryv1.Endpoint{
			Addresses:  []string{fmt.Sprintf("10.0.0.%d", i)},
			Conditions: discoveryv1.EndpointConditions{Ready: new(i%2 == 1)},
		})
	}
	got, dropped := BuildEndpoints([]discoveryv1.EndpointSlice{slice("s", discoveryv1.AddressTypeIPv4, nil, eps...)}, 4)
	assert.Equal(t, 6, dropped)
	require.Len(t, got, 4)
	for i, ep := range got {
		assert.True(t, ep.GetConditions().GetReady(), ep.GetAddresses())
		if i > 0 {
			assert.Less(t, got[i-1].GetAddresses()[0], ep.GetAddresses()[0], "sorted")
		}
	}
}

func TestConditionFromRegistry(t *testing.T) {
	at := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	c := ConditionFromRegistry(&mcsv1.Condition{Type: api.ConditionReady, Status: mcsv1.ConditionStatus_CONDITION_STATUS_TRUE, Reason: "Ready", Message: "ok"})
	assert.Equal(t, metav1.Condition{Type: api.ConditionReady, Status: metav1.ConditionTrue, Reason: "Ready", Message: "ok"}, c)
	c = ConditionFromRegistry(&mcsv1.Condition{Type: api.ConditionConflict, Status: mcsv1.ConditionStatus_CONDITION_STATUS_FALSE, Reason: "PortConflict,TypeConflict"})
	assert.Equal(t, metav1.ConditionFalse, c.Status)
	assert.Equal(t, "PortConflict,TypeConflict", c.Reason)
	c = ConditionFromRegistry(&mcsv1.Condition{Type: "X", Reason: "not valid!", Message: string(make([]byte, 40000))})
	assert.Equal(t, metav1.ConditionUnknown, c.Status)
	assert.Equal(t, "Unknown", c.Reason)
	assert.Len(t, c.Message, 32768)
	c = ConditionFromRegistry(&mcsv1.Condition{Type: "X", Reason: "R", LastTransitionAt: timestampOf(at)})
	assert.True(t, c.LastTransitionTime.Time.Equal(at))
}

func timestampOf(t time.Time) *timestamppb.Timestamp { return timestamppb.New(t) }
