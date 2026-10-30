// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package integration

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

func TestImportTrafficPoliciesAndEndpointSliceStatus(t *testing.T) {
	requireEnv(t)
	ctx := mcstest.Ctx(t)
	ns := createNamespace(t)
	url, reg := startServer(t)
	generation := &client.Generation{}

	t.Log("Publish non-default policies through the real registry and import them through the API server")
	_, err := reg.Clusters().Register(ctx, &mcsv1.RegisterClusterRequest{Clusterset: mcstest.ClusterSet, Cluster: remote})
	require.NoError(t, err)
	spec := remoteSpec()
	spec.InternalTrafficPolicy = mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_LOCAL
	spec.TrafficDistribution = "PreferClose"
	put := &mcsv1.PutServiceExportRequest{
		Clusterset: mcstest.ClusterSet, Cluster: remote, Namespace: ns, Name: "frontend",
		Spec: spec, Generation: generation.Next(),
	}
	_, err = reg.ServiceExports().Put(ctx, put)
	require.NoError(t, err)
	replace := &mcsv1.ReplaceEndpointsRequest{
		Clusterset: mcstest.ClusterSet, Cluster: remote, Namespace: ns, Name: "frontend",
		Endpoints: remoteEndpoints("10.9.0.1"), Generation: generation.Next(),
	}
	_, err = reg.ServiceExports().ReplaceEndpoints(ctx, replace)
	require.NoError(t, err)
	a := startApp(t, url, ns)

	var si mcsv1beta1.ServiceImport
	var svc corev1.Service
	derived := api.DerivedServiceName(ns, "frontend")
	eventually(t, func() bool {
		return get(t, ns, "frontend", &si) && get(t, ns, derived, &svc) &&
			si.Status.EndpointSliceObjects == mcsv1beta1.EndpointSliceObjectsPresent
	}, "import has EndpointSlice objects")
	assert.Equal(t, new(corev1.ServiceInternalTrafficPolicyLocal), si.Spec.InternalTrafficPolicy)
	assert.Equal(t, new(corev1.ServiceInternalTrafficPolicyLocal), svc.Spec.InternalTrafficPolicy)
	assert.Equal(t, new("PreferClose"), si.Spec.TrafficDistribution)
	assert.Equal(t, new("PreferClose"), svc.Spec.TrafficDistribution)
	require.Len(t, importedSlices(t, ns, "frontend"), 1)

	t.Log("Changing policies clears the previously owned trafficDistribution field under server-side apply")
	spec.InternalTrafficPolicy = mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_CLUSTER
	spec.TrafficDistribution = ""
	put.Generation = generation.Next()
	_, err = reg.ServiceExports().Put(ctx, put)
	require.NoError(t, err)
	eventually(t, func() bool {
		return get(t, ns, "frontend", &si) && get(t, ns, derived, &svc) &&
			si.Spec.InternalTrafficPolicy != nil && *si.Spec.InternalTrafficPolicy == corev1.ServiceInternalTrafficPolicyCluster &&
			svc.Spec.InternalTrafficPolicy != nil && *svc.Spec.InternalTrafficPolicy == corev1.ServiceInternalTrafficPolicyCluster &&
			si.Spec.TrafficDistribution == nil && svc.Spec.TrafficDistribution == nil
	}, "Cluster policy and cleared distribution on both imported resources")

	t.Log("An omitted policy remains omitted on ServiceImport while the derived Service receives Kubernetes defaulting")
	spec.InternalTrafficPolicy = mcsv1.InternalTrafficPolicy_INTERNAL_TRAFFIC_POLICY_UNSPECIFIED
	put.Generation = generation.Next()
	_, err = reg.ServiceExports().Put(ctx, put)
	require.NoError(t, err)
	eventually(t, func() bool {
		if !get(t, ns, "frontend", &si) || !get(t, ns, derived, &svc) {
			return false
		}
		ready := meta.FindStatusCondition(si.Status.Conditions, api.ConditionReady)
		return si.Spec.InternalTrafficPolicy == nil && svc.Spec.InternalTrafficPolicy != nil &&
			*svc.Spec.InternalTrafficPolicy == corev1.ServiceInternalTrafficPolicyCluster &&
			ready != nil && ready.ObservedGeneration == si.Generation &&
			si.Status.EndpointSliceObjects == mcsv1beta1.EndpointSliceObjectsPresent
	}, "omission is preserved and Service policy defaults to Cluster")

	t.Log("Repair a legacy endpointSliceObjects status when the other status fields still match")
	before := si.DeepCopy()
	si.Status.EndpointSliceObjects = mcsv1beta1.EndpointSliceObjectsAbsent
	require.NoError(t, k8sClient.Status().Update(ctx, &si))
	var cached mcsv1beta1.ServiceImport
	eventually(t, func() bool {
		return a.Manager.GetClient().Get(ctx, ctrlclient.ObjectKeyFromObject(&si), &cached) == nil &&
			cached.Status.EndpointSliceObjects == mcsv1beta1.EndpointSliceObjectsAbsent
	}, "the controller cache observes the legacy status before the new registry event")

	// An annotation-only registry update schedules application without changing spec generation or conditions.
	spec.Annotations = map[string]string{api.Prefix + "status-repair": "true"}
	put.Generation = generation.Next()
	_, err = reg.ServiceExports().Put(ctx, put)
	require.NoError(t, err)
	eventually(t, func() bool {
		return get(t, ns, "frontend", &si) && si.Annotations[api.Prefix+"status-repair"] == "true" &&
			si.Status.EndpointSliceObjects == mcsv1beta1.EndpointSliceObjectsPresent
	}, "endpointSliceObjects repaired despite unchanged clusters and conditions")
	assert.Equal(t, before.Status.Clusters, si.Status.Clusters)
	assert.Equal(t, before.Status.Conditions, si.Status.Conditions)
	assert.Equal(t, before.Generation, si.Generation)

	t.Log("Removing all endpoints removes imported EndpointSlice objects and reports Absent")
	replace.Endpoints = nil
	replace.Generation = generation.Next()
	_, err = reg.ServiceExports().ReplaceEndpoints(ctx, replace)
	require.NoError(t, err)
	eventually(t, func() bool {
		return get(t, ns, "frontend", &si) && si.Status.EndpointSliceObjects == mcsv1beta1.EndpointSliceObjectsAbsent &&
			len(importedSlices(t, ns, "frontend")) == 0
	}, "no imported EndpointSlice objects remain")

	t.Log("Non-ready addresses still create imported EndpointSlice objects and report Present")
	replace.Endpoints = remoteEndpoints("10.9.0.1")
	replace.Endpoints[0].Conditions.Ready = false
	replace.Endpoints[0].Conditions.Serving = false
	replace.Generation = generation.Next()
	_, err = reg.ServiceExports().ReplaceEndpoints(ctx, replace)
	require.NoError(t, err)
	eventually(t, func() bool {
		return get(t, ns, "frontend", &si) && si.Status.EndpointSliceObjects == mcsv1beta1.EndpointSliceObjectsPresent &&
			len(importedSlices(t, ns, "frontend")) == 1
	}, "EndpointSlice presence is independent of endpoint readiness")
	slices := importedSlices(t, ns, "frontend")
	require.Len(t, slices[0].Endpoints, 1)
	assert.Equal(t, new(false), slices[0].Endpoints[0].Conditions.Ready)
	assert.Equal(t, []mcsv1beta1.ClusterStatus{{Cluster: remote}}, si.Status.Clusters)
}
