/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
)

func TestGenerateHTTPRouteBindsPoolToGateway(t *testing.T) {
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "ns"},
	}
	// cross-namespace Gateway (gatewayNamespace != pool namespace)
	route := GenerateHTTPRoute(dgd, nil, "my-gw", "gw-ns")

	if route.Name != "graph-route" || route.Namespace != "ns" {
		t.Fatalf("route meta = %s/%s, want ns/graph-route", route.Namespace, route.Name)
	}

	// parentRef -> the named Gateway in the core gateway API group, in gw-ns
	if n := len(route.Spec.ParentRefs); n != 1 {
		t.Fatalf("parentRefs = %d, want 1", n)
	}
	parent := route.Spec.ParentRefs[0]
	if string(parent.Name) != "my-gw" {
		t.Fatalf("parentRef name = %q, want my-gw", parent.Name)
	}
	if parent.Group == nil || string(*parent.Group) != gatewayAPIGroup {
		t.Fatalf("parentRef group = %v, want %s", parent.Group, gatewayAPIGroup)
	}
	if parent.Kind == nil || string(*parent.Kind) != "Gateway" {
		t.Fatalf("parentRef kind = %v, want Gateway", parent.Kind)
	}
	if parent.Namespace == nil || string(*parent.Namespace) != "gw-ns" {
		t.Fatalf("parentRef namespace = %v, want gw-ns (cross-namespace)", parent.Namespace)
	}

	// empty gatewayNamespace defaults to the pool's namespace (same-ns)
	if sameNs := GenerateHTTPRoute(dgd, nil, "my-gw", ""); sameNs.Spec.ParentRefs[0].Namespace == nil ||
		string(*sameNs.Spec.ParentRefs[0].Namespace) != "ns" {
		t.Fatalf("empty gatewayNamespace should default to pool ns 'ns', got %v", sameNs.Spec.ParentRefs[0].Namespace)
	}

	// single rule, PathPrefix("/") match, backendRef -> the InferencePool
	if n := len(route.Spec.Rules); n != 1 {
		t.Fatalf("rules = %d, want 1", n)
	}
	rule := route.Spec.Rules[0]
	if n := len(rule.Matches); n != 1 || rule.Matches[0].Path == nil ||
		*rule.Matches[0].Path.Type != gatewayv1.PathMatchPathPrefix ||
		*rule.Matches[0].Path.Value != defaultHTTPRoutePathPrefix {
		t.Fatalf("match = %#v, want PathPrefix %q", rule.Matches, defaultHTTPRoutePathPrefix)
	}
	if n := len(rule.BackendRefs); n != 1 {
		t.Fatalf("backendRefs = %d, want 1", n)
	}
	be := rule.BackendRefs[0].BackendRef.BackendObjectReference
	if be.Group == nil || string(*be.Group) != InferencePoolGroup {
		t.Fatalf("backendRef group = %v, want %s (InferencePool)", be.Group, InferencePoolGroup)
	}
	if be.Kind == nil || string(*be.Kind) != "InferencePool" {
		t.Fatalf("backendRef kind = %v, want InferencePool", be.Kind)
	}
	if want := GetPoolName(dgd.Name, nil); string(be.Name) != want {
		t.Fatalf("backendRef name = %q, want pool %q", be.Name, want)
	}
	if be.Port == nil || int32(*be.Port) != int32(consts.DynamoServicePort) {
		t.Fatalf("backendRef port = %v, want %d", be.Port, consts.DynamoServicePort)
	}
}
