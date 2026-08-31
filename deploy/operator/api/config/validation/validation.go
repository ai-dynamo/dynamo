/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package validation

import (
	"net/url"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateOperatorConfiguration validates an OperatorConfiguration object.
func ValidateOperatorConfiguration(config *configv1alpha1.OperatorConfiguration) field.ErrorList {
	if config == nil {
		return field.ErrorList{field.Required(field.NewPath(""), "operator configuration is required")}
	}

	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validateServer(&config.Server, field.NewPath("server"))...)
	allErrs = append(allErrs, validateLeaderElection(&config.LeaderElection, field.NewPath("leaderElection"))...)
	allErrs = append(allErrs, validateNamespace(&config.Namespace, field.NewPath("namespace"))...)
	allErrs = append(allErrs, validateMPI(&config.MPI, field.NewPath("mpi"))...)
	allErrs = append(allErrs, validateInfrastructure(&config.Infrastructure, field.NewPath("infrastructure"))...)
	allErrs = append(allErrs, validateDiscovery(&config.Discovery, field.NewPath("discovery"))...)
	allErrs = append(allErrs, validateRBAC(config)...)
	allErrs = append(allErrs, validateOrchestrators(&config.Orchestrators, field.NewPath("orchestrators"))...)
	allErrs = append(allErrs, validateIngress(&config.Ingress, field.NewPath("ingress"))...)
	allErrs = append(allErrs, validateServiceMesh(&config.ServiceMesh, field.NewPath("serviceMesh"))...)

	return allErrs
}

func validateServer(server *configv1alpha1.ServerConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if server.Metrics.Port < 0 || server.Metrics.Port > 65535 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("metrics", "port"), server.Metrics.Port, "must be between 0 and 65535"))
	}
	if server.HealthProbe.Port < 0 || server.HealthProbe.Port > 65535 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("healthProbe", "port"), server.HealthProbe.Port, "must be between 0 and 65535"))
	}
	if server.Webhook.Port < 0 || server.Webhook.Port > 65535 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("webhook", "port"), server.Webhook.Port, "must be between 0 and 65535"))
	}

	return allErrs
}

func validateLeaderElection(le *configv1alpha1.LeaderElectionConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if le.Enabled && le.ID == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("id"), "leader election ID is required when leader election is enabled"))
	}

	return allErrs
}

func validateNamespace(ns *configv1alpha1.NamespaceConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	// Namespace-restricted mode validations
	if ns.Restricted != "" {
		scopePath := fldPath.Child("scope")
		if ns.Scope.LeaseDuration.Duration <= 0 {
			allErrs = append(allErrs, field.Invalid(scopePath.Child("leaseDuration"), ns.Scope.LeaseDuration.Duration, "must be greater than 0 in namespace-restricted mode"))
		}
		if ns.Scope.LeaseRenewInterval.Duration <= 0 {
			allErrs = append(allErrs, field.Invalid(scopePath.Child("leaseRenewInterval"), ns.Scope.LeaseRenewInterval.Duration, "must be greater than 0 in namespace-restricted mode"))
		}
		if ns.Scope.LeaseRenewInterval.Duration > 0 && ns.Scope.LeaseDuration.Duration > 0 &&
			ns.Scope.LeaseRenewInterval.Duration >= ns.Scope.LeaseDuration.Duration {
			allErrs = append(allErrs, field.Invalid(scopePath.Child("leaseRenewInterval"), ns.Scope.LeaseRenewInterval.Duration, "must be less than leaseDuration"))
		}
	}

	return allErrs
}

func validateMPI(mpi *configv1alpha1.MPIConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if mpi.SSHSecretName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("sshSecretName"), "MPI SSH secret name is required"))
	}
	if mpi.SSHSecretNamespace == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("sshSecretNamespace"), "MPI SSH secret namespace is required"))
	}

	return allErrs
}

func validateInfrastructure(infra *configv1alpha1.InfrastructureConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if infra.ModelExpressURL != "" {
		if _, err := url.Parse(infra.ModelExpressURL); err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("modelExpressURL"), infra.ModelExpressURL, "must be a valid URL"))
		}
	}

	// TLS client identity pairs must be set together (both or neither).
	if (infra.TCPTLSClientCertPath != "") != (infra.TCPTLSClientKeyPath != "") {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("tcpTLSClientCertPath"), infra.TCPTLSClientCertPath,
			"tcpTLSClientCertPath and tcpTLSClientKeyPath must be set together"))
	}
	if (infra.NATSTLSClientCertPath != "") != (infra.NATSTLSClientKeyPath != "") {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("natsTLSClientCertPath"), infra.NATSTLSClientCertPath,
			"natsTLSClientCertPath and natsTLSClientKeyPath must be set together"))
	}

	// TCP server certificate and key must be set together (both or neither).
	if (infra.TCPTLSCertPath != "") != (infra.TCPTLSKeyPath != "") {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("tcpTLSCertPath"), infra.TCPTLSCertPath,
			"tcpTLSCertPath and tcpTLSKeyPath must be set together"))
	}

	// TCP server cert/key require a server CA so the client side also uses TLS.
	// The operator does not expose DYN_TCP_TLS_INSECURE, so without a CA the
	// TCP client stays plaintext while the server is TLS, and connections fail.
	if infra.TCPTLSCertPath != "" && infra.TCPTLSKeyPath != "" && infra.TCPTLSCAPath == "" {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("tcpTLSCAPath"),
			"tcpTLSCAPath is required when tcpTLSCertPath and tcpTLSKeyPath are set"))
	}

	// TCP client-side TLS (CA or client identity) requires the server certificate
	// and key. The operator injects the same config into every DGD pod, each of
	// which is both a TCP client and server, so enabling client-side TLS without
	// server-side TLS leaves peer servers plaintext and handshakes fail.
	tcpClientTLS := infra.TCPTLSCAPath != "" || infra.TCPTLSClientCertPath != "" || infra.TCPTLSClientKeyPath != ""
	if tcpClientTLS && (infra.TCPTLSCertPath == "" || infra.TCPTLSKeyPath == "") {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("tcpTLSCertPath"),
			"tcpTLSCertPath and tcpTLSKeyPath are required when any client-side TCP TLS field is set"))
	}

	// TCP client-CA (mTLS) requires the server certificate and key.
	if infra.TCPTLSClientCAPath != "" && (infra.TCPTLSCertPath == "" || infra.TCPTLSKeyPath == "") {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("tcpTLSClientCAPath"), infra.TCPTLSClientCAPath,
			"tcpTLSClientCAPath requires tcpTLSCertPath and tcpTLSKeyPath to also be set"))
	}

	// TCP client identity requires a server CA (the operator does not expose insecure mode).
	if infra.TCPTLSClientCertPath != "" && infra.TCPTLSCAPath == "" {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("tcpTLSClientCertPath"), infra.TCPTLSClientCertPath,
			"tcpTLSClientCertPath requires tcpTLSCAPath to also be set"))
	}

	// NATS client identity requires a server CA.
	if infra.NATSTLSClientCertPath != "" && infra.NATSTLSCAPath == "" {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("natsTLSClientCertPath"), infra.NATSTLSClientCertPath,
			"natsTLSClientCertPath requires natsTLSCAPath to also be set"))
	}

	// tcpTLSServerName is inert without a TLS connector (which requires a CA).
	if infra.TCPTLSServerName != "" && infra.TCPTLSCAPath == "" {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("tcpTLSServerName"), infra.TCPTLSServerName,
			"tcpTLSServerName requires tcpTLSCAPath to also be set"))
	}

	// NATS TLS requires a tls:// server address (the runtime fails closed otherwise).
	natsTLS := infra.NATSTLSCAPath != "" || infra.NATSTLSClientCertPath != "" || infra.NATSTLSClientKeyPath != ""
	if natsTLS {
		if infra.NATSAddress == "" {
			allErrs = append(allErrs, field.Required(
				fldPath.Child("natsAddress"),
				"natsAddress is required when NATS TLS is configured"))
		} else if !strings.HasPrefix(infra.NATSAddress, "tls://") {
			allErrs = append(allErrs, field.Invalid(
				fldPath.Child("natsAddress"), infra.NATSAddress,
				"natsAddress must use the tls:// scheme when NATS TLS is configured"))
		}
	}

	return allErrs
}

func validateDiscovery(discovery *configv1alpha1.DiscoveryConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if discovery.Backend != configv1alpha1.DiscoveryBackendKubernetes && discovery.Backend != configv1alpha1.DiscoveryBackendEtcd {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("backend"), discovery.Backend, []string{"kubernetes", "etcd"}))
	}

	return allErrs
}

// validateRBAC is mode-aware: validates RBAC fields based on namespace mode.
func validateRBAC(config *configv1alpha1.OperatorConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}

	// RBAC validation only applies in cluster-wide mode
	if config.Namespace.Restricted != "" {
		return allErrs
	}

	fldPath := field.NewPath("rbac")
	if config.Namespace.Restricted == "" && config.RBAC.PlannerClusterRoleName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("plannerClusterRoleName"), "planner ClusterRole name is required in cluster-wide mode"))
	}
	if config.Namespace.Restricted == "" && config.RBAC.DGDRProfilingClusterRoleName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("dgdrProfilingClusterRoleName"), "DGDR profiling ClusterRole name is required in cluster-wide mode"))
	}
	if config.Namespace.Restricted == "" && config.RBAC.EPPClusterRoleName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("eppClusterRoleName"), "EPP ClusterRole name is required in cluster-wide mode"))
	}

	return allErrs
}

func validateOrchestrators(orch *configv1alpha1.OrchestratorConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if orch.Grove.TerminationDelay.Duration < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("grove", "terminationDelay"), orch.Grove.TerminationDelay.Duration, "must not be negative"))
	}

	return allErrs
}

func validateIngress(ingress *configv1alpha1.IngressConfiguration, fldPath *field.Path) field.ErrorList {
	// No required fields — all ingress configuration is optional
	_ = fldPath
	_ = ingress
	return nil
}

// validateServiceMesh validates the service mesh configuration. The most
// important guard is that "MUTUAL" TLS mode requires a client certificate and
// private key (and optionally a CA certificates file); without them Istio's
// validation webhook rejects the EPP DestinationRule and the operator can
// never finish reconciling the DGD.
func validateServiceMesh(sm *configv1alpha1.ServiceMeshConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if !sm.IsEnabled() {
		return allErrs
	}

	// IsEnabled() only checks Provider. If the user set provider="istio" but
	// omitted the istio block, the controller still treats the mesh as
	// enabled and GenerateEPPDestinationRule (graph.go) silently emits a
	// stub DestinationRule with no Host/TrafficPolicy — useless and
	// confusing. Fail fast here instead of letting reconcile proceed with
	// an incomplete mesh config. Defaulting normally populates this block,
	// but validation must not depend on the defaulter having run (e.g.,
	// hand-written configs, programmatic loaders).
	istioPath := fldPath.Child("istio")
	if sm.Istio == nil {
		allErrs = append(allErrs, field.Required(
			istioPath,
			`istio configuration is required when serviceMesh.provider is "istio"`,
		))
		return allErrs
	}

	switch sm.Istio.TLSMode {
	case "", "SIMPLE", "DISABLE", "ISTIO_MUTUAL":
		// No additional fields required.
	case "MUTUAL":
		if sm.Istio.ClientCertificate == "" {
			allErrs = append(allErrs, field.Required(
				istioPath.Child("clientCertificate"),
				`clientCertificate is required when tlsMode is "MUTUAL"`,
			))
		}
		if sm.Istio.PrivateKey == "" {
			allErrs = append(allErrs, field.Required(
				istioPath.Child("privateKey"),
				`privateKey is required when tlsMode is "MUTUAL"`,
			))
		}
	default:
		allErrs = append(allErrs, field.NotSupported(
			istioPath.Child("tlsMode"),
			sm.Istio.TLSMode,
			[]string{"DISABLE", "SIMPLE", "ISTIO_MUTUAL", "MUTUAL"},
		))
	}

	return allErrs
}
