// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package importer

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	discoveryv1ac "k8s.io/client-go/applyconfigurations/discovery/v1"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

const (
	testNS = "llm"
	uid    = types.UID("si-uid")
)

func endpoint(addr string, ready bool, ports ...*mcsv1.EndpointPort) *mcsv1.Endpoint {
	return &mcsv1.Endpoint{
		Addresses: []string{addr}, Conditions: &mcsv1.EndpointConditions{Ready: ready, Serving: ready},
		Ports: ports, Hostname: "h-" + addr, NodeName: "node-" + addr, Zone: "z1",
	}
}

var (
	httpPort    = &mcsv1.EndpointPort{Name: "http", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 8080}
	metricsPort = &mcsv1.EndpointPort{Name: "metrics", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 9090}
)

func testImport() *mcsv1.ServiceImport {
	return &mcsv1.ServiceImport{
		Clusterset: "cs", Namespace: testNS, Name: "frontend",
		Type:                  mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
		Ports:                 []*mcsv1.ServicePort{{Name: "http", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 80, AppProtocol: "http"}},
		SessionAffinity:       mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP,
		SessionAffinityConfig: &mcsv1.SessionAffinityConfig{ClientIp: &mcsv1.ClientIPConfig{TimeoutSeconds: 60}},
		IpFamilies:            []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV4},
		Labels:                map[string]string{api.Prefix + "tier": "gold", "bad key!": "x", api.LabelManagedBy: "spoofed"},
		Annotations:           map[string]string{api.Prefix + "models": "m1", "Bad Key": "x"},
		Clusters: []*mcsv1.ClusterStatus{
			{Cluster: "a", Participating: true}, {Cluster: "b", Participating: true}, {Cluster: "c", Participating: false},
		},
		Endpoints: []*mcsv1.ClusterEndpoints{
			{Cluster: "a", Region: "us-east-1", Generation: 7, Endpoints: []*mcsv1.Endpoint{endpoint("10.0.0.2", true, httpPort, metricsPort), endpoint("10.0.0.1", false, httpPort)}},
			{Cluster: "b", Endpoints: []*mcsv1.Endpoint{endpoint("10.1.0.1", true, httpPort)}},
		},
	}
}

func TestBuildServiceImport(t *testing.T) {
	u, err := BuildServiceImport(testImport(), "cs", []string{"10.96.0.10"})
	require.NoError(t, err)
	var si mcsv1beta1.ServiceImport
	require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(u.Object, &si))
	assert.Equal(t, "multicluster.x-k8s.io/v1beta1", u.GetAPIVersion())
	assert.Equal(t, "ServiceImport", u.GetKind())
	assert.Equal(t, map[string]string{api.Prefix + "tier": "gold", api.LabelManagedBy: api.ManagedByController, api.LabelClusterSet: "cs"}, si.Labels)
	assert.Equal(t, map[string]string{api.Prefix + "models": "m1", api.AnnotationMCSDerivedService: api.DerivedServiceName(testNS, "frontend")}, si.Annotations)
	assert.Equal(t, mcsv1beta1.ClusterSetIP, si.Spec.Type)
	assert.Equal(t, []string{"10.96.0.10"}, si.Spec.IPs)
	assert.Equal(t, corev1.ServiceAffinityClientIP, si.Spec.SessionAffinity)
	assert.Equal(t, int32(60), *si.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol}, si.Spec.IPFamilies)
	require.Len(t, si.Spec.Ports, 1)
	assert.Equal(t, mcsv1beta1.ServicePort{Name: "http", Protocol: corev1.ProtocolTCP, Port: 80, AppProtocol: new("http")}, si.Spec.Ports[0])
	_, hasStatus := u.Object["status"]
	assert.False(t, hasStatus)

	empty, err := BuildServiceImport(&mcsv1.ServiceImport{Namespace: testNS, Name: "x", Type: mcsv1.ServiceType_SERVICE_TYPE_HEADLESS}, "cs", nil)
	require.NoError(t, err)
	ports, found, _ := unstructuredSlice(empty.Object, "spec", "ports")
	assert.True(t, found, "ports is required, even empty")
	assert.Empty(t, ports)
	typ, _, _ := unstructuredString(empty.Object, "spec", "type")
	assert.Equal(t, "Headless", typ)
}

func TestBuildServiceImportStatus(t *testing.T) {
	imp := testImport()
	st := BuildServiceImportStatus(imp, nil, 3)
	assert.Equal(t, []mcsv1beta1.ClusterStatus{{Cluster: "a"}, {Cluster: "b"}}, st.Clusters)
	require.Len(t, st.Conditions, 2)
	ready := st.Conditions[0]
	assert.Equal(t, "Ready", ready.Type)
	assert.Equal(t, metav1.ConditionTrue, ready.Status)
	assert.Equal(t, int64(3), ready.ObservedGeneration)
	assert.Contains(t, ready.Message, "2 of 3")
	assert.Equal(t, metav1.ConditionFalse, st.Conditions[1].Status)

	// Unchanged status keeps the transition time; a change moves it.
	old := metav1.NewTime(ready.LastTransitionTime.Add(-3600e9))
	st.Conditions[0].LastTransitionTime = old
	st.Conditions[1].LastTransitionTime = old
	imp.Conflict = &mcsv1.Conflict{Reason: api.ConditionReasonPortConflict, Message: "ports differ"}
	st2 := BuildServiceImportStatus(imp, st.Conditions, 3)
	assert.Equal(t, old, st2.Conditions[0].LastTransitionTime)
	assert.Equal(t, metav1.ConditionTrue, st2.Conditions[1].Status)
	assert.Equal(t, api.ConditionReasonPortConflict, st2.Conditions[1].Reason)
	assert.NotEqual(t, old, st2.Conditions[1].LastTransitionTime)

	for _, c := range imp.Clusters {
		c.Participating = false
	}
	st3 := BuildServiceImportStatus(imp, nil, 3)
	assert.Empty(t, st3.Clusters)
	assert.NotNil(t, st3.Clusters, "an empty list, so that the merge patch clears it")
	assert.Equal(t, metav1.ConditionFalse, st3.Conditions[0].Status)
	assert.Equal(t, "Pending", st3.Conditions[0].Reason)
}

func TestBuildDerivedService(t *testing.T) {
	imp := testImport()
	ac := BuildDerivedService(imp, "cs", uid, []corev1.IPFamily{corev1.IPv4Protocol})
	assert.Equal(t, api.DerivedServiceName(testNS, "frontend"), *ac.Name)
	assert.Equal(t, testNS, *ac.Namespace)
	assert.Equal(t, map[string]string{api.LabelManagedBy: api.ManagedByController, api.LabelClusterSet: "cs", api.LabelMCSServiceName: "frontend"}, ac.Labels)
	require.Len(t, ac.OwnerReferences, 1)
	assert.Equal(t, uid, *ac.OwnerReferences[0].UID)
	assert.Equal(t, "ServiceImport", *ac.OwnerReferences[0].Kind)
	assert.True(t, *ac.OwnerReferences[0].Controller)
	assert.Equal(t, corev1.ServiceTypeClusterIP, *ac.Spec.Type)
	assert.Nil(t, ac.Spec.ClusterIP, "allocated by the API server")
	assert.Nil(t, ac.Spec.Selector)
	assert.Equal(t, corev1.ServiceAffinityClientIP, *ac.Spec.SessionAffinity)
	assert.Equal(t, int32(60), *ac.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol}, ac.Spec.IPFamilies)
	assert.Equal(t, corev1.IPFamilyPolicySingleStack, *ac.Spec.IPFamilyPolicy)
	require.Len(t, ac.Spec.Ports, 1)
	assert.Equal(t, "http", *ac.Spec.Ports[0].Name)
	assert.Equal(t, int32(80), *ac.Spec.Ports[0].Port)
	assert.Equal(t, "http", *ac.Spec.Ports[0].AppProtocol)

	imp.Type = mcsv1.ServiceType_SERVICE_TYPE_HEADLESS
	imp.SessionAffinity = mcsv1.SessionAffinity_SESSION_AFFINITY_NONE
	imp.IpFamilies = []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV4, mcsv1.IPFamily_IP_FAMILY_IPV6}
	ac = BuildDerivedService(imp, "cs", uid, []corev1.IPFamily{corev1.IPv4Protocol})
	assert.Equal(t, corev1.ClusterIPNone, *ac.Spec.ClusterIP)
	assert.Nil(t, ac.Spec.SessionAffinityConfig)
	assert.Equal(t, corev1.IPFamilyPolicyPreferDualStack, *ac.Spec.IPFamilyPolicy)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol}, ac.Spec.IPFamilies, "only the families the importing cluster has")

	imp.IpFamilies = nil
	ac = BuildDerivedService(imp, "cs", uid, nil)
	assert.Nil(t, ac.Spec.IPFamilyPolicy)
	assert.Nil(t, ac.Spec.IPFamilies)
}

// The import's IP families describe the exporters; the derived Service lists
// only those the importing cluster has and the API server picks otherwise.
func TestDerivedIPFamilies(t *testing.T) {
	v4, v6 := corev1.IPv4Protocol, corev1.IPv6Protocol
	single, dual := corev1.IPFamilyPolicySingleStack, corev1.IPFamilyPolicyPreferDualStack
	for _, tt := range []struct {
		name     string
		imported []mcsv1.IPFamily
		cluster  []corev1.IPFamily
		policy   *corev1.IPFamilyPolicy
		families []corev1.IPFamily
	}{
		{"unknown", nil, []corev1.IPFamily{v4}, nil, nil},
		{"IPv4 on IPv4", []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV4}, []corev1.IPFamily{v4}, &single, []corev1.IPFamily{v4}},
		{"IPv6 on IPv4 only: the API server picks", []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV6}, []corev1.IPFamily{v4}, &single, nil},
		{"dual on IPv4 only", []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV4, mcsv1.IPFamily_IP_FAMILY_IPV6}, []corev1.IPFamily{v4}, &dual, []corev1.IPFamily{v4}},
		{"dual IPv6 first on dual", []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV6, mcsv1.IPFamily_IP_FAMILY_IPV4}, []corev1.IPFamily{v4, v6}, &dual, []corev1.IPFamily{v6, v4}},
		{"cluster families unknown", []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV4}, nil, &single, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			policy, fams := DerivedIPFamilies(&mcsv1.ServiceImport{IpFamilies: tt.imported}, tt.cluster)
			assert.Equal(t, tt.policy, policy)
			assert.Equal(t, tt.families, fams)
		})
	}
}

func TestInvalidPorts(t *testing.T) {
	tcp := mcsv1.Protocol_PROTOCOL_TCP
	assert.NoError(t, InvalidPorts(&mcsv1.ServiceImport{}))
	assert.NoError(t, InvalidPorts(&mcsv1.ServiceImport{Ports: []*mcsv1.ServicePort{{Port: 80, Protocol: tcp}}}), "one unnamed port is a valid Service")
	assert.NoError(t, InvalidPorts(testImport()))
	err := InvalidPorts(&mcsv1.ServiceImport{Ports: []*mcsv1.ServicePort{{Name: "http", Port: 80, Protocol: tcp}, {Port: 81, Protocol: tcp}}})
	assert.ErrorContains(t, err, "81/TCP has no name")
	err = InvalidPorts(&mcsv1.ServiceImport{Ports: []*mcsv1.ServicePort{{Name: "http", Port: 80, Protocol: tcp}, {Name: "http", Port: 81, Protocol: tcp}}})
	assert.ErrorContains(t, err, `"http" is used twice`)
}

func names(slices []*discoveryv1ac.EndpointSliceApplyConfiguration) []string {
	var out []string
	for _, s := range slices {
		out = append(out, *s.Name)
	}
	return out
}

func TestBuildEndpointSlicesPacking(t *testing.T) {
	imp := testImport()
	derived := api.DerivedServiceName(testNS, "frontend")
	// Five more IPv4 endpoints and one IPv6 endpoint in cluster a.
	for i := 3; i < 8; i++ {
		imp.Endpoints[0].Endpoints = append(imp.Endpoints[0].Endpoints, endpoint(fmt.Sprintf("10.0.0.%d", i), true, httpPort))
	}
	imp.Endpoints[0].Endpoints = append(imp.Endpoints[0].Endpoints,
		&mcsv1.Endpoint{Addresses: []string{"fd00::1"}, Ports: []*mcsv1.EndpointPort{httpPort}},
		&mcsv1.Endpoint{}, // no address: skipped
	)
	slices := BuildEndpointSlices(imp, "cs", "a", uid, 3, false)
	assert.Equal(t, []string{
		derived + "-a-v4-0", derived + "-a-v4-1", derived + "-a-v4-2", derived + "-a-v6-0", derived + "-b-v4-0",
	}, names(slices), "indexed per address type, so that IPv4 growth never renames the IPv6 slice")

	a0 := slices[0]
	assert.Equal(t, discoveryv1.AddressTypeIPv4, *a0.AddressType)
	assert.Equal(t, map[string]string{
		discoveryv1.LabelServiceName:    derived,
		api.LabelMCSServiceName:         "frontend",
		api.LabelMCSSourceCluster:       "a",
		api.LabelTopologyRegion:         "us-east-1",
		api.LabelEndpointSliceManagedBy: api.EndpointSliceManagedBy,
		api.LabelManagedBy:              api.ManagedByController,
		api.LabelClusterSet:             "cs",
	}, a0.Labels)
	require.Len(t, a0.OwnerReferences, 1)
	assert.Equal(t, uid, *a0.OwnerReferences[0].UID)
	// Ports are filtered to the merged names (metrics is not exported).
	require.Len(t, a0.Ports, 1)
	assert.Equal(t, "http", *a0.Ports[0].Name)
	assert.Equal(t, int32(8080), *a0.Ports[0].Port)
	require.Len(t, a0.Endpoints, 3)
	assert.Equal(t, []string{"10.0.0.1"}, a0.Endpoints[0].Addresses)
	assert.False(t, *a0.Endpoints[0].Conditions.Ready)
	assert.Equal(t, "node-10.0.0.1", *a0.Endpoints[0].NodeName, "local node names are kept")
	assert.Equal(t, "h-10.0.0.1", *a0.Endpoints[0].Hostname)
	assert.Equal(t, "z1", *a0.Endpoints[0].Zone)
	assert.Len(t, slices[1].Endpoints, 3)
	assert.Len(t, slices[2].Endpoints, 1)
	assert.Equal(t, discoveryv1.AddressTypeIPv6, *slices[3].AddressType)

	b0 := slices[4]
	assert.Equal(t, "b", b0.Labels[api.LabelMCSSourceCluster])
	assert.NotContains(t, b0.Labels, api.LabelTopologyRegion, "no region label for a cluster without a region")
	assert.Nil(t, b0.Endpoints[0].NodeName, "remote node names are dropped")

	// Deterministic whatever the order the registry sends.
	eps := imp.Endpoints[0].Endpoints
	for i, j := 0, len(eps)-1; i < j; i, j = i+1, j-1 {
		eps[i], eps[j] = eps[j], eps[i]
	}
	again := BuildEndpointSlices(imp, "cs", "a", uid, 3, false)
	require.Equal(t, names(slices), names(again))
	for i := range slices {
		assert.Equal(t, slices[i].Endpoints, again[i].Endpoints, *slices[i].Name)
	}

	// Default size.
	assert.Len(t, BuildEndpointSlices(imp, "cs", "a", uid, 0, false), 3)
	assert.Equal(t, derived+"-a-v4-12", SliceName(derived, "a", discoveryv1.AddressTypeIPv4, 12))
	assert.Equal(t, derived+"-a-v6-0", SliceName(derived, "a", discoveryv1.AddressTypeIPv6, 0))
	assert.Equal(t, derived+"-a-fqdn-1", SliceName(derived, "a", discoveryv1.AddressTypeFQDN, 1))

	// A region that is not a label value is left off instead of failing the apply.
	imp.Endpoints[0].Region = "us east/1"
	a0 = BuildEndpointSlices(imp, "cs", "a", uid, 3, false)[0]
	assert.NotContains(t, a0.Labels, api.LabelTopologyRegion)
}

func TestBuildEndpointSlicesTargetRef(t *testing.T) {
	t.Log("Endpoints exported with and without target reference attributes")
	imp := testImport()
	named := endpoint("10.1.0.2", true, httpPort)
	named.Attributes = map[string]string{api.EndpointAttrTargetName: "frontend-0", api.EndpointAttrTargetUID: "remote-uid"}
	namespaced := endpoint("10.1.0.3", true, httpPort)
	namespaced.Attributes = map[string]string{
		api.EndpointAttrTargetName: "frontend-1", api.EndpointAttrTargetKind: "Pod", api.EndpointAttrTargetNamespace: "llm",
	}
	kindOnly := endpoint("10.1.0.4", true, httpPort)
	kindOnly.Attributes = map[string]string{api.EndpointAttrTargetKind: "Pod"}
	imp.Endpoints[1].Endpoints = append(imp.Endpoints[1].Endpoints, named, namespaced, kindOnly)

	t.Log("Each imported endpoint names its pod when the export named one")
	b0 := BuildEndpointSlices(imp, "cs", "a", uid, 10, false)[1]
	require.Equal(t, "b", b0.Labels[api.LabelMCSSourceCluster])
	require.Len(t, b0.Endpoints, 4)
	assert.Nil(t, b0.Endpoints[0].TargetRef, "no attributes")
	ref := b0.Endpoints[1].TargetRef
	require.NotNil(t, ref)
	assert.Equal(t, "Pod", *ref.Kind, "Pod when the kind is not exported")
	assert.Equal(t, "frontend-0", *ref.Name)
	assert.Nil(t, ref.Namespace)
	assert.Nil(t, ref.UID, "the uid of another cluster's pod is not copied")
	ref = b0.Endpoints[2].TargetRef
	require.NotNil(t, ref)
	assert.Equal(t, "frontend-1", *ref.Name)
	assert.Equal(t, "llm", *ref.Namespace)
	assert.Nil(t, b0.Endpoints[3].TargetRef, "a kind without a name is no reference")

	t.Log("With target references omitted no endpoint carries one")
	for _, e := range BuildEndpointSlices(imp, "cs", "a", uid, 10, true)[1].Endpoints {
		assert.Nil(t, e.TargetRef)
	}
}

func unstructuredSlice(obj map[string]any, fields ...string) ([]any, bool, error) {
	return unstructured.NestedSlice(obj, fields...)
}

func unstructuredString(obj map[string]any, fields ...string) (string, bool, error) {
	return unstructured.NestedString(obj, fields...)
}
