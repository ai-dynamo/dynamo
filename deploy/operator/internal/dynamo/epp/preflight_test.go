/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
)

func gw(ns string, from *gatewayv1.FromNamespaces, programmed bool) *gatewayv1.Gateway {
	g := &gatewayv1.Gateway{
		ObjectMeta: metav1.ObjectMeta{Name: "inference-gateway", Namespace: ns},
		Spec: gatewayv1.GatewaySpec{
			Listeners: []gatewayv1.Listener{{Name: "http", Port: 80, Protocol: gatewayv1.HTTPProtocolType}},
		},
	}
	if from != nil {
		g.Spec.Listeners[0].AllowedRoutes = &gatewayv1.AllowedRoutes{
			Namespaces: &gatewayv1.RouteNamespaces{From: from},
		}
	}
	if programmed {
		g.Status.Conditions = []metav1.Condition{{Type: "Programmed", Status: metav1.ConditionTrue}}
	}
	return g
}

func TestPreflight_NotFound(t *testing.T) {
	r := PreflightGateway(nil, "igw-demo")
	if r.Ready || r.Reason != reasonNotFound {
		t.Fatalf("nil gateway: got ready=%v reason=%q, want not-ready GatewayNotFound", r.Ready, r.Reason)
	}
}

func TestPreflight_CrossNamespaceAllowedAll(t *testing.T) {
	// shared gateway in its own ns, allowedRoutes=All, programmed → ready cross-ns.
	r := PreflightGateway(gw("inference-gateway", ptr.To(gatewayv1.NamespacesFromAll), true), "igw-demo")
	if !r.Ready || r.Reason != reasonReady {
		t.Fatalf("All+programmed cross-ns: got ready=%v reason=%q, want Ready", r.Ready, r.Reason)
	}
}

func TestPreflight_SameNamespaceDeniesCrossNs(t *testing.T) {
	// the silent-404 we hit: shared gateway, allowedRoutes=Same, route in a
	// different namespace → not ready, with the actionable reason.
	r := PreflightGateway(gw("inference-gateway", ptr.To(gatewayv1.NamespacesFromSame), true), "igw-demo")
	if r.Ready || r.Reason != reasonNamespaceDenied {
		t.Fatalf("Same cross-ns: got ready=%v reason=%q, want GatewayNamespaceNotAllowed", r.Ready, r.Reason)
	}
}

func TestPreflight_DefaultAllowedRoutesIsSame(t *testing.T) {
	// nil allowedRoutes defaults to Same → cross-ns denied.
	r := PreflightGateway(gw("inference-gateway", nil, true), "igw-demo")
	if r.Ready || r.Reason != reasonNamespaceDenied {
		t.Fatalf("nil allowedRoutes cross-ns: got ready=%v reason=%q, want GatewayNamespaceNotAllowed", r.Ready, r.Reason)
	}
}

func TestPreflight_SameNamespaceAllowedWhenColocated(t *testing.T) {
	// allowedRoutes=Same and the route is in the gateway's own ns → ready.
	r := PreflightGateway(gw("igw-demo", ptr.To(gatewayv1.NamespacesFromSame), true), "igw-demo")
	if !r.Ready {
		t.Fatalf("Same same-ns: got ready=%v reason=%q, want Ready", r.Ready, r.Reason)
	}
}

func TestPreflight_NotProgrammed(t *testing.T) {
	r := PreflightGateway(gw("inference-gateway", ptr.To(gatewayv1.NamespacesFromAll), false), "igw-demo")
	if r.Ready || r.Reason != reasonNotProgrammed {
		t.Fatalf("not programmed: got ready=%v reason=%q, want GatewayNotProgrammed", r.Ready, r.Reason)
	}
}

func TestPreflight_SelectorTreatedAdmissible(t *testing.T) {
	// Selector can't be evaluated here; treat as admissible to avoid false denial.
	r := PreflightGateway(gw("inference-gateway", ptr.To(gatewayv1.NamespacesFromSelector), true), "igw-demo")
	if !r.Ready {
		t.Fatalf("Selector: got ready=%v reason=%q, want Ready (admissible-unverified)", r.Ready, r.Reason)
	}
}
