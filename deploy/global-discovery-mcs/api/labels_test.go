// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// The values are a wire contract with the Rust crate, Helm charts and other
// clusters; a change here is a breaking change.
func TestWellKnownKeys(t *testing.T) {
	want := map[string]string{
		LabelExport:                 "dynamo.nvidia.com/mcs-export",
		LabelManagedBy:              "dynamo.nvidia.com/mcs-managed-by",
		LabelClusterSet:             "dynamo.nvidia.com/mcs-clusterset",
		FinalizerServiceExport:      "dynamo.nvidia.com/mcs-service-export",
		LabelEndpointSliceManagedBy: "endpointslice.kubernetes.io/managed-by",
		LabelKubernetesServiceName:  "kubernetes.io/service-name",
		LabelMCSServiceName:         "multicluster.kubernetes.io/service-name",
		LabelMCSSourceCluster:       "multicluster.kubernetes.io/source-cluster",
		AnnotationMCSDerivedService: "multicluster.kubernetes.io/derived-service",
	}
	for got, expected := range want {
		assert.Equal(t, expected, got)
	}

	assert.Equal(t, "true", ExportEnabled)
	assert.Equal(t, "targetRef.name", EndpointAttrTargetName)
	assert.Equal(t, "mcs-controller", ManagedByController)
	assert.Equal(t, "mcs-controller.dynamo.nvidia.com", EndpointSliceManagedBy)
	assert.Equal(t, "clusterset.k8s.io", ClusterPropertyClusterSet)
	assert.Equal(t, "cluster.clusterset.k8s.io", ClusterPropertyClusterID)
}

// Every key must be a valid Kubernetes qualified name: an optional DNS
// subdomain prefix and a name of at most 63 characters.
func TestKeysAreQualifiedNames(t *testing.T) {
	prefix := regexp.MustCompile(`^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$`)
	name := regexp.MustCompile(`^[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?$`)
	keys := []string{
		LabelExport, LabelManagedBy, LabelClusterSet, FinalizerServiceExport,
		LabelEndpointSliceManagedBy, LabelKubernetesServiceName, LabelMCSServiceName,
		LabelMCSSourceCluster, AnnotationMCSDerivedService,
	}
	for _, k := range keys {
		p, n, ok := cut(k)
		if assert.True(t, ok, k) {
			assert.Regexp(t, prefix, p, k)
			assert.LessOrEqual(t, len(p), 253, k)
			assert.Regexp(t, name, n, k)
			assert.LessOrEqual(t, len(n), 63, k)
		}
	}
	// Label values must be at most 63 characters as well.
	for _, v := range []string{ManagedByController, EndpointSliceManagedBy} {
		assert.Regexp(t, name, v)
		assert.LessOrEqual(t, len(v), 63, v)
	}
}

func cut(key string) (prefix, name string, ok bool) {
	before, after, ok := strings.CutLast(key, "/")
	if !ok {
		return "", key, false
	}
	return before, after, true
}

func TestIsExported(t *testing.T) {
	assert.True(t, IsExported(map[string]string{LabelExport: "true"}))
	assert.False(t, IsExported(map[string]string{LabelExport: "True"}))
	assert.False(t, IsExported(map[string]string{LabelExport: "false"}))
	assert.False(t, IsExported(map[string]string{}))
	assert.False(t, IsExported(nil))
}

func TestIsMCSKey(t *testing.T) {
	assert.True(t, IsMCSKey(LabelExport))
	assert.True(t, IsMCSKey("dynamo.nvidia.com/mcs-anything"))
	assert.False(t, IsMCSKey("dynamo.nvidia.com/mcs-"))
	assert.False(t, IsMCSKey("app.kubernetes.io/name"))
	assert.False(t, IsMCSKey("x.dynamo.nvidia.com/mcs-role"))
}
