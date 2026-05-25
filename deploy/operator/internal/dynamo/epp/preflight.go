/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"fmt"

	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
)

// GatewayPreflightResult reports whether a generated HTTPRoute in routeNamespace
// can actually attach to gw, with an actionable reason/message when it can't.
// It converts the silent failure modes (the route is created but never attaches,
// so requests 404 at the gateway with no signal) into a status condition.
type GatewayPreflightResult struct {
	Ready   bool
	Reason  string // one of consts.ReasonInferenceGateway*
	Message string
}

// reason/message strings are kept here as plain literals so this package has no
// dependency on internal/consts; the controller maps Reason onto the consts.
const (
	reasonReady           = "Ready"
	reasonNotFound        = "GatewayNotFound"
	reasonNamespaceDenied = "GatewayNamespaceNotAllowed"
	reasonNotProgrammed   = "GatewayNotProgrammed"
	gatewayProgrammedType = "Programmed"
)

// PreflightGateway checks, in order: the gateway exists, at least one listener
// admits routeNamespace (the allowedRoutes check — the cross-namespace silent
// 404 we hit in practice), and the gateway is Programmed. gw==nil means the
// controller's Get returned NotFound.
//
// allowedRoutes semantics (Gateway API): a listener's
// allowedRoutes.namespaces.from of "All" admits any namespace; "Same" admits
// only the gateway's own namespace; nil defaults to "Same". "Selector" can't be
// fully evaluated here (needs a namespace-label lookup), so it is treated as
// admissible-but-unverified to avoid false negatives.
func PreflightGateway(gw *gatewayv1.Gateway, routeNamespace string) GatewayPreflightResult {
	if gw == nil {
		return GatewayPreflightResult{
			Ready:   false,
			Reason:  reasonNotFound,
			Message: "referenced Gateway not found; create it (or fix inferenceGateway.gatewayName/gatewayNamespace) — the EPP InferencePool is created but no traffic will route until the Gateway exists",
		}
	}

	if !listenerAdmitsNamespace(gw, routeNamespace) {
		return GatewayPreflightResult{
			Ready:  false,
			Reason: reasonNamespaceDenied,
			Message: fmt.Sprintf(
				"Gateway %q (ns %q) has no listener admitting routes from namespace %q; "+
					"the HTTPRoute will not attach and requests will 404. Fix: set the "+
					"Gateway listener's spec.listeners[].allowedRoutes.namespaces.from to \"All\" "+
					"(or a selector matching %q).",
				gw.Name, gw.Namespace, routeNamespace, routeNamespace),
		}
	}

	if !gatewayProgrammed(gw) {
		return GatewayPreflightResult{
			Ready:   false,
			Reason:  reasonNotProgrammed,
			Message: fmt.Sprintf("Gateway %q is not Programmed yet; its data plane is still coming up", gw.Name),
		}
	}

	return GatewayPreflightResult{
		Ready:   true,
		Reason:  reasonReady,
		Message: fmt.Sprintf("HTTPRoute attaches to Gateway %q in namespace %q", gw.Name, gw.Namespace),
	}
}

// listenerAdmitsNamespace returns true if any listener admits routeNamespace.
func listenerAdmitsNamespace(gw *gatewayv1.Gateway, routeNamespace string) bool {
	for _, l := range gw.Spec.Listeners {
		from := gatewayv1.NamespacesFromSame // Gateway API default when unset
		if l.AllowedRoutes != nil && l.AllowedRoutes.Namespaces != nil && l.AllowedRoutes.Namespaces.From != nil {
			from = *l.AllowedRoutes.Namespaces.From
		}
		switch from {
		case gatewayv1.NamespacesFromAll:
			return true
		case gatewayv1.NamespacesFromSame:
			if routeNamespace == gw.Namespace {
				return true
			}
		case gatewayv1.NamespacesFromSelector:
			// Can't evaluate the label selector here without a Namespace lookup;
			// treat as admissible to avoid a false "denied" (the gateway
			// controller is the source of truth for selector matching).
			return true
		}
	}
	return false
}

// gatewayProgrammed returns true if the Gateway's Programmed condition is True.
func gatewayProgrammed(gw *gatewayv1.Gateway) bool {
	for _, c := range gw.Status.Conditions {
		if c.Type == gatewayProgrammedType {
			return c.Status == "True"
		}
	}
	return false
}
