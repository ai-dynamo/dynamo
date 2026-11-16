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

package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +kubebuilder:object:root=true

// OperatorConfiguration is the Schema for the operator configuration.
type OperatorConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Server configuration (metrics, health probes, webhooks)
	// +required
	Server ServerConfiguration `json:"server"`

	// Leader election configuration
	// +required
	LeaderElection LeaderElectionConfiguration `json:"leaderElection"`

	// Namespace configuration (restricted vs cluster-wide)
	// +required
	Namespace NamespaceConfiguration `json:"namespace"`

	// Orchestrator configuration with optional overrides
	// +required
	Orchestrators OrchestratorConfiguration `json:"orchestrators"`

	// DRA (Dynamic Resource Allocation) settings with optional override
	// +optional
	DRA DRAConfiguration `json:"dra,omitempty"`

	// Service mesh and infrastructure addresses
	// +required
	Infrastructure InfrastructureConfiguration `json:"infrastructure"`

	// Ingress configuration
	// +required
	Ingress IngressConfiguration `json:"ingress"`

	// ServiceMesh configures automatic generation of service-mesh resources
	// (e.g., Istio DestinationRules) for EPP components.
	// +required
	ServiceMesh ServiceMeshConfiguration `json:"serviceMesh"`

	// RBAC configuration for cross-namespace resource management (cluster-wide mode)
	// +required
	RBAC RBACConfiguration `json:"rbac"`

	// MPI SSH secret configuration
	// +required
	MPI MPIConfiguration `json:"mpi"`

	// Checkpoint/restore configuration
	// +required
	Checkpoint CheckpointConfiguration `json:"checkpoint"`

	// Discovery backend configuration
	// +required
	Discovery DiscoveryConfiguration `json:"discovery"`

	// GPU discovery configuration
	// +required
	GPU GPUConfiguration `json:"gpu"`

	// Logging configuration
	// +required
	Logging LoggingConfiguration `json:"logging"`

	// HTTP/2 and TLS settings
	// +required
	Security SecurityConfiguration `json:"security"`

	// LPX scheduler integration and model-registry configuration.
	// +required
	LPX LPXConfiguration `json:"lpx"`
}

// ServerConfiguration holds server bind addresses and ports.
type ServerConfiguration struct {
	// Metrics server configuration
	// +kubebuilder:default={bindAddress: "0.0.0.0", port: 8080, secure: true}
	// +optional
	Metrics MetricsServer `json:"metrics"`
	// Health probe server configuration
	// +kubebuilder:default={bindAddress: "0.0.0.0", port: 8081}
	// +optional
	HealthProbe Server `json:"healthProbe"`
	// Webhook server configuration
	// +kubebuilder:default={host: "0.0.0.0", port: 9443, certDir: "/tmp/k8s-webhook-server/serving-certs"}
	// +optional
	Webhook WebhookServer `json:"webhook"`
}

// Server holds a bind address and port.
type Server struct {
	// BindAddress is the address the server binds to
	// +required
	BindAddress string `json:"bindAddress"`
	// Port is the port the server listens on
	// +required
	Port int `json:"port"`
}

// MetricsServer extends Server with secure serving option.
type MetricsServer struct {
	Server `json:",inline"`
	// Secure enables secure serving for the metrics endpoint.
	// nil = default to true (secure by default).
	// +optional
	Secure *bool `json:"secure,omitempty"`
}

// CertProvisionMode controls how webhook TLS certificates are managed.
type CertProvisionMode string

const (
	// CertProvisionModeAuto uses the built-in cert-controller to generate and rotate certificates.
	CertProvisionModeAuto CertProvisionMode = "auto"
	// CertProvisionModeManual expects certificates to be provided externally (e.g., cert-manager, admin).
	CertProvisionModeManual CertProvisionMode = "manual"
)

// WebhookServer extends Server with host and certificate directory.
type WebhookServer struct {
	Server `json:",inline"`
	// Host is the address the webhook server binds to
	// +required
	Host string `json:"host"`
	// CertDir is the directory containing TLS certificates
	// +required
	CertDir string `json:"certDir"`
	// CertProvisionMode controls certificate management: "auto" (built-in cert-controller) or "manual" (external)
	// +kubebuilder:default="auto"
	// +optional
	CertProvisionMode CertProvisionMode `json:"certProvisionMode"`
	// SecretName is the name of the Kubernetes Secret holding webhook TLS certificates
	// +kubebuilder:default="webhook-server-cert"
	// +optional
	SecretName string `json:"secretName"`
	// ServiceName is the name of the Kubernetes Service fronting the webhook server.
	// Used to generate certificate SANs. Set by the Helm chart.
	// +required
	ServiceName string `json:"serviceName"`
}

// LeaderElectionConfiguration holds leader election settings.
type LeaderElectionConfiguration struct {
	// Enabled enables leader election for controller manager
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled"`
	// ID is the leader election resource identity
	// +required
	ID string `json:"id"`
	// Namespace is the namespace for the leader election resource
	// +required
	Namespace string `json:"namespace"`
}

// NamespaceConfiguration determines operator namespace mode.
type NamespaceConfiguration struct {
	// Restricted enables namespace-restricted mode for development and testing.
	// Namespace-restricted mode is not supported for production.
	// +required
	Restricted string `json:"restricted"`
	// Scope configures the namespace ownership claim in namespace-restricted mode.
	// +required
	Scope NamespaceScopeConfiguration `json:"scope"`
}

// NamespaceScopeConfiguration configures the development/test namespace ownership claim.
type NamespaceScopeConfiguration struct {
	// LeaseDuration is the duration of namespace scope marker lease before expiration
	// +kubebuilder:default="30s"
	// +optional
	LeaseDuration metav1.Duration `json:"leaseDuration"`
	// LeaseRenewInterval is the interval for renewing namespace scope marker lease
	// +kubebuilder:default="10s"
	// +optional
	LeaseRenewInterval metav1.Duration `json:"leaseRenewInterval"`
}

// OrchestratorConfiguration holds orchestrator override settings.
type OrchestratorConfiguration struct {
	// Grove orchestrator configuration
	// +required
	Grove GroveConfiguration `json:"grove"`
	// LWS orchestrator configuration
	// +required
	LWS LWSConfiguration `json:"lws"`
	// KaiScheduler configuration
	// +required
	KaiScheduler KaiSchedulerConfiguration `json:"kaiScheduler"`
	// VolcanoScheduler configuration
	// +required
	VolcanoScheduler VolcanoSchedulerConfiguration `json:"volcanoScheduler"`
}

// GroveConfiguration holds Grove orchestrator settings.
type GroveConfiguration struct {
	// Enabled overrides auto-detection. nil = auto-detect.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
	// TerminationDelay configures the termination delay for Grove PodCliqueSets
	// +kubebuilder:default="15m"
	// +optional
	TerminationDelay metav1.Duration `json:"terminationDelay"`
}

// LWSConfiguration holds LWS orchestrator settings.
type LWSConfiguration struct {
	// Enabled overrides auto-detection. nil = auto-detect.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
}

// KaiSchedulerConfiguration holds Kai-scheduler settings.
type KaiSchedulerConfiguration struct {
	// Enabled overrides auto-detection. nil = auto-detect.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
}

// VolcanoSchedulerConfiguration holds Volcano scheduler settings.
type VolcanoSchedulerConfiguration struct {
	// EXPERIMENTAL: Enabled controls Volcano scheduler integration for Grove PodCliqueSets.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
}

// DRAConfiguration holds Dynamic Resource Allocation (resource.k8s.io/v1) settings.
//
// NOTE: auto-detection here only verifies that the resource.k8s.io/v1 API is
// registered on the apiserver (Kubernetes 1.34+). It does NOT verify that a
// GPU-specific DRA resource driver (e.g. nvidia/k8s-dra-driver-gpu) is
// installed, that its DeviceClass exists, or that node-level GPU drivers are
// compatible. An admin can use `enabled: false` to force-off DRA integration
// on clusters where the API is present but the GPU driver stack is not wired
// up — this makes the operator fail GMS / inter-pod failover admissions early
// with a clear error instead of letting pods Pend with a confusing
// "resourceclaim not found" at schedule time.
type DRAConfiguration struct {
	// Enabled overrides auto-detection of the resource.k8s.io/v1 API.
	// nil = auto-detect. Setting true requires detection to also succeed (the
	// operator will exit at startup otherwise).
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
}

// InfrastructureConfiguration holds service mesh and backend addresses.
type InfrastructureConfiguration struct {
	// NATSAddress is the address of the NATS server
	// +required
	NATSAddress string `json:"natsAddress"`
	// ETCDAddress is the address of the etcd server
	// +required
	ETCDAddress string `json:"etcdAddress"`
	// ModelExpressURL is the URL of the Model Express server to inject into all pods
	// +required
	ModelExpressURL string `json:"modelExpressURL"`
	// PrometheusEndpoint is the URL of the Prometheus endpoint to use for metrics
	// +required
	PrometheusEndpoint string `json:"prometheusEndpoint"`
	// NATSTLSCAPath is the CA certificate path for verifying the NATS server
	// +optional
	NATSTLSCAPath string `json:"natsTLSCAPath,omitempty"`
	// NATSTLSClientCertPath is the client certificate path for NATS mTLS
	// +optional
	NATSTLSClientCertPath string `json:"natsTLSClientCertPath,omitempty"`
	// NATSTLSClientKeyPath is the client private key path for NATS mTLS
	// +optional
	NATSTLSClientKeyPath string `json:"natsTLSClientKeyPath,omitempty"`
	// TCPTLSCertPath is the server certificate path for TCP TLS
	// +optional
	TCPTLSCertPath string `json:"tcpTLSCertPath,omitempty"`
	// TCPTLSKeyPath is the server private key path for TCP TLS
	// +optional
	TCPTLSKeyPath string `json:"tcpTLSKeyPath,omitempty"`
	// TCPTLSCAPath is the CA certificate path for verifying TCP peers
	// +optional
	TCPTLSCAPath string `json:"tcpTLSCAPath,omitempty"`
	// TCPTLSClientCertPath is the client certificate path for TCP mTLS
	// +optional
	TCPTLSClientCertPath string `json:"tcpTLSClientCertPath,omitempty"`
	// TCPTLSClientKeyPath is the client private key path for TCP mTLS
	// +optional
	TCPTLSClientKeyPath string `json:"tcpTLSClientKeyPath,omitempty"`
	// TCPTLSClientCAPath is the CA certificate path for verifying TCP client certificates (mTLS)
	// +optional
	TCPTLSClientCAPath string `json:"tcpTLSClientCAPath,omitempty"`
	// TCPTLSServerName overrides the TLS SNI hostname used by TCP clients when
	// verifying the server certificate. Useful when dialing by IP (pod address)
	// to a server whose certificate has a DNS SAN.
	// +optional
	TCPTLSServerName string `json:"tcpTLSServerName,omitempty"`
}

// IngressConfiguration holds ingress settings.
type IngressConfiguration struct {
	// VirtualServiceGateway is the name of the Istio virtual service gateway
	// +required
	VirtualServiceGateway string `json:"virtualServiceGateway"`
	// ControllerClassName is the ingress controller class name
	// +required
	ControllerClassName string `json:"controllerClassName"`
	// ControllerTLSSecretName is the TLS secret for the ingress controller
	// +required
	ControllerTLSSecretName string `json:"controllerTLSSecretName"`
	// HostSuffix is the suffix for ingress hostnames
	// +required
	HostSuffix string `json:"hostSuffix"`
}

// UseVirtualService returns true if a VirtualService gateway is configured.
func (i *IngressConfiguration) UseVirtualService() bool {
	return i.VirtualServiceGateway != ""
}

// ServiceMeshProvider enumerates the supported service mesh implementations.
type ServiceMeshProvider string

const (
	// ServiceMeshProviderIstio selects Istio as the service mesh.
	ServiceMeshProviderIstio ServiceMeshProvider = "istio"
)

// ServiceMeshConfiguration holds service mesh integration settings.
// The operator uses this to generate mesh-specific resources (e.g., Istio
// DestinationRules) for EPP components so that sidecar proxies connect
// correctly without double-TLS issues.
type ServiceMeshConfiguration struct {
	// Enabled overrides service mesh auto-detection. nil = auto-detect.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`
	// Provider selects the service mesh implementation. Supported: "istio", "".
	// Empty string disables service mesh resource generation.
	// +required
	Provider string `json:"provider"`
	// Istio holds Istio-specific settings. Only used when Provider is "istio".
	// +optional
	Istio *IstioMeshConfiguration `json:"istio,omitempty"`
}

// IsEnabled returns true if service mesh resources should be created or updated.
// Cleanup of previously owned resources is handled separately during reconcile.
func (s *ServiceMeshConfiguration) IsEnabled() bool {
	if s.Enabled != nil && !*s.Enabled {
		return false
	}
	return ServiceMeshProvider(s.Provider) == ServiceMeshProviderIstio
}

// IstioMeshConfiguration holds Istio-specific mesh settings.
type IstioMeshConfiguration struct {
	// TLSMode is the Istio TLS mode for DestinationRules.
	// Supported values: "DISABLE", "SIMPLE", "ISTIO_MUTUAL", "MUTUAL".
	// Defaults to "SIMPLE".
	// +required
	TLSMode string `json:"tlsMode"`
	// InsecureSkipVerify skips TLS certificate verification in DestinationRules.
	// Defaults to true (matching upstream GAIE behavior with self-signed certs).
	// +optional
	InsecureSkipVerify *bool `json:"insecureSkipVerify,omitempty"`
	// ClientCertificate is the path (in the istio-proxy sidecar's filesystem)
	// to the file holding the client-side TLS certificate used for mTLS.
	// REQUIRED when TLSMode is "MUTUAL"; ignored for other modes.
	// +optional
	ClientCertificate string `json:"clientCertificate,omitempty"`
	// PrivateKey is the path (in the istio-proxy sidecar's filesystem) to the
	// file holding the client-side TLS private key used for mTLS.
	// REQUIRED when TLSMode is "MUTUAL"; ignored for other modes.
	// +optional
	PrivateKey string `json:"privateKey,omitempty"`
	// CaCertificates is the optional path (in the istio-proxy sidecar's
	// filesystem) to the file holding CA certificates used to verify the
	// server certificate. Used only when TLSMode is "MUTUAL"; for other modes
	// the field is ignored.
	// +optional
	CaCertificates string `json:"caCertificates,omitempty"`
}

// RBACConfiguration holds RBAC settings for cluster-wide mode.
type RBACConfiguration struct {
	// PlannerClusterRoleName is the ClusterRole for planner
	// +required
	PlannerClusterRoleName string `json:"plannerClusterRoleName"`
	// DGDRProfilingClusterRoleName is the ClusterRole for DGDR profiling jobs
	// +required
	DGDRProfilingClusterRoleName string `json:"dgdrProfilingClusterRoleName"`
	// EPPClusterRoleName is the ClusterRole for EPP
	// +required
	EPPClusterRoleName string `json:"eppClusterRoleName"`
}

// MPIConfiguration holds MPI SSH secret settings.
type MPIConfiguration struct {
	// SSHSecretName is the name of the secret containing the SSH key for MPI
	// +required
	SSHSecretName string `json:"sshSecretName"`
	// SSHSecretNamespace is the namespace where the MPI SSH secret is located
	// +required
	SSHSecretNamespace string `json:"sshSecretNamespace"`
}

// DefaultSeccompProfile is the localhost seccomp profile applied to checkpoint
// and restore pods when the operator config does not specify one explicitly.
const DefaultSeccompProfile = "profiles/block-iouring.json"

// CheckpointConfiguration holds checkpoint/restore settings.
type CheckpointConfiguration struct {
	// Enabled indicates if checkpoint functionality is enabled
	// +required
	Enabled bool `json:"enabled"`
	// Seccomp controls the localhost seccomp profile applied to checkpoint and
	// restore pods. A nil value means "use the default profile"; set
	// Seccomp.Disabled=true to disable seccomp injection entirely.
	// +optional
	Seccomp *CheckpointSeccompConfiguration `json:"seccomp,omitempty"`
}

// CheckpointSeccompConfiguration controls the localhost seccomp profile applied
// to checkpoint and restore pods. The profile blocks io_uring syscalls (which
// CRIU cannot dump). Default behavior (zero-value substruct, or absent
// substruct) applies DefaultSeccompProfile. Set Disabled=true on OpenShift
// (custom localhost profiles require privileged SCC) or when using a CRIU
// build with io_uring support. Set Profile to override the default path.
type CheckpointSeccompConfiguration struct {
	// Disabled, when true, suppresses seccomp profile injection entirely.
	// Use this for clusters where custom localhost profiles are not allowed
	// (e.g. OpenShift's restricted-v2 SCC) or for CRIU builds that handle
	// io_uring natively.
	// +optional
	Disabled bool `json:"disabled,omitempty"`
	// Profile is the localhost seccomp profile path. Empty falls back to
	// DefaultSeccompProfile. Ignored when Disabled is true.
	// +optional
	Profile string `json:"profile,omitempty"`
}

// EffectiveSeccompProfile returns the seccomp profile to use, or "" to disable.
// A nil substruct or zero-value substruct uses DefaultSeccompProfile. Disabled=true
// disables injection. Profile override takes effect when Disabled is false.
func (c *CheckpointConfiguration) EffectiveSeccompProfile() string {
	if c.Seccomp == nil {
		return DefaultSeccompProfile
	}
	if c.Seccomp.Disabled {
		return ""
	}
	if c.Seccomp.Profile == "" {
		return DefaultSeccompProfile
	}
	return c.Seccomp.Profile
}

// DiscoveryConfiguration holds discovery backend settings.
type DiscoveryConfiguration struct {
	// Backend is the discovery backend: "kubernetes" or "etcd"
	// +kubebuilder:default="kubernetes"
	// +optional
	Backend DiscoveryBackend `json:"backend"`
}

// DiscoveryBackend is the type for the discovery backend.
type DiscoveryBackend string

const (
	// DiscoveryBackendKubernetes is the Kubernetes discovery backend
	DiscoveryBackendKubernetes DiscoveryBackend = "kubernetes"
	// DiscoveryBackendEtcd is the etcd discovery backend
	DiscoveryBackendEtcd DiscoveryBackend = "etcd"
)

// KubeDiscoveryMode is the kube discovery identity granularity.
type KubeDiscoveryMode string

const (
	// KubeDiscoveryModePod is the default: one identity per pod.
	KubeDiscoveryModePod KubeDiscoveryMode = "pod"
	// KubeDiscoveryModeContainer: each container registers independently with the discovery plane.
	KubeDiscoveryModeContainer KubeDiscoveryMode = "container"
)

// GPUConfiguration holds GPU discovery settings.
type GPUConfiguration struct {
	// DiscoveryEnabled indicates whether GPU discovery is enabled
	// +kubebuilder:default=true
	// +optional
	DiscoveryEnabled *bool `json:"discoveryEnabled,omitempty"`
}

// LoggingConfiguration holds logging settings.
type LoggingConfiguration struct {
	// Level is the log level (e.g., "info", "debug")
	// +kubebuilder:default="info"
	// +optional
	Level string `json:"level"`
	// Format is the log format (e.g., "json", "text")
	// +kubebuilder:default="json"
	// +optional
	Format string `json:"format"`
}

// SecurityConfiguration holds HTTP/2 and TLS settings.
type SecurityConfiguration struct {
	// EnableHTTP2 enables HTTP/2 for metrics and webhook servers
	// +kubebuilder:default=false
	// +optional
	EnableHTTP2 bool `json:"enableHTTP2"`
}
