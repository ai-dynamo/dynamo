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

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Checkpoint storage type constants
const (
	CheckpointStorageTypePVC = "pvc"
	CheckpointStorageTypeS3  = "s3"
	CheckpointStorageTypeOCI = "oci"
)

// +kubebuilder:object:root=true

// OperatorConfiguration is the Schema for the operator configuration.
type OperatorConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Server configuration (metrics, health probes, webhooks)
	Server ServerConfiguration `json:"server"`

	// Leader election configuration
	LeaderElection LeaderElectionConfiguration `json:"leaderElection"`

	// Namespace configuration (restricted vs cluster-wide)
	Namespace NamespaceConfiguration `json:"namespace"`

	// Orchestrator configuration with optional overrides
	Orchestrators OrchestratorConfiguration `json:"orchestrators"`

	// Service mesh and infrastructure addresses
	Infrastructure InfrastructureConfiguration `json:"infrastructure"`

	// Ingress configuration
	Ingress IngressConfiguration `json:"ingress"`

	// RBAC configuration for cross-namespace resource management (cluster-wide mode)
	RBAC RBACConfiguration `json:"rbac"`

	// MPI SSH secret configuration
	MPI MPIConfiguration `json:"mpi"`

	// Checkpoint/restore configuration
	Checkpoint CheckpointConfiguration `json:"checkpoint"`

	// Discovery backend configuration
	Discovery DiscoveryConfiguration `json:"discovery"`

	// GPU discovery configuration
	GPU GPUConfiguration `json:"gpu"`

	// Logging configuration
	Logging LoggingConfiguration `json:"logging"`

	// HTTP/2 and TLS settings
	Security SecurityConfiguration `json:"security"`
}

// ServerConfiguration holds server bind addresses and ports.
type ServerConfiguration struct {
	// Metrics server configuration
	// +kubebuilder:default={bindAddress: "127.0.0.1", port: 8080}
	Metrics MetricsServer `json:"metrics"`
	// Health probe server configuration
	// +kubebuilder:default={bindAddress: "127.0.0.1", port: 8081}
	HealthProbe Server `json:"healthProbe"`
	// Webhook server configuration
	// +kubebuilder:default={host: "0.0.0.0", port: 9443, certDir: "/tmp/k8s-webhook-server/serving-certs"}
	Webhook WebhookServer `json:"webhook"`
}

// Server holds a bind address and port.
type Server struct {
	// BindAddress is the address the server binds to
	BindAddress string `json:"bindAddress"`
	// Port is the port the server listens on
	Port int `json:"port"`
}

// MetricsServer extends Server with secure serving option.
type MetricsServer struct {
	Server `json:",inline"`
	// Secure enables secure serving for the metrics endpoint
	Secure bool `json:"secure"`
}

// WebhookServer extends Server with host and certificate directory.
type WebhookServer struct {
	Server `json:",inline"`
	// Host is the address the webhook server binds to
	Host string `json:"host"`
	// CertDir is the directory containing TLS certificates
	CertDir string `json:"certDir"`
}

// LeaderElectionConfiguration holds leader election settings.
type LeaderElectionConfiguration struct {
	// Enabled enables leader election for controller manager
	// +kubebuilder:default=false
	Enabled bool `json:"enabled"`
	// ID is the leader election resource identity
	ID string `json:"id"`
	// Namespace is the namespace for the leader election resource
	Namespace string `json:"namespace"`
}

// NamespaceConfiguration determines operator namespace mode.
type NamespaceConfiguration struct {
	// Restricted is the namespace to restrict to. Empty = cluster-wide mode.
	Restricted string `json:"restricted"`
	// Scope holds namespace scope lease settings (namespace-restricted mode only)
	Scope NamespaceScopeConfiguration `json:"scope"`
}

// NamespaceScopeConfiguration holds lease settings for namespace-restricted mode.
type NamespaceScopeConfiguration struct {
	// LeaseDuration is the duration of namespace scope marker lease before expiration
	// +kubebuilder:default="30s"
	LeaseDuration metav1.Duration `json:"leaseDuration"`
	// LeaseRenewInterval is the interval for renewing namespace scope marker lease
	// +kubebuilder:default="10s"
	LeaseRenewInterval metav1.Duration `json:"leaseRenewInterval"`
}

// OrchestratorConfiguration holds orchestrator override settings.
type OrchestratorConfiguration struct {
	// Grove orchestrator configuration
	Grove GroveConfiguration `json:"grove"`
	// LWS orchestrator configuration
	LWS LWSConfiguration `json:"lws"`
	// KaiScheduler configuration
	KaiScheduler KaiSchedulerConfiguration `json:"kaiScheduler"`
}

// GroveConfiguration holds Grove orchestrator settings.
type GroveConfiguration struct {
	// Enabled overrides auto-detection. nil = auto-detect.
	Enabled *bool `json:"enabled,omitempty"`
	// TerminationDelay configures the termination delay for Grove PodCliqueSets
	// +kubebuilder:default="15m"
	TerminationDelay metav1.Duration `json:"terminationDelay"`
}

// LWSConfiguration holds LWS orchestrator settings.
type LWSConfiguration struct {
	// Enabled overrides auto-detection. nil = auto-detect.
	Enabled *bool `json:"enabled,omitempty"`
}

// KaiSchedulerConfiguration holds Kai-scheduler settings.
type KaiSchedulerConfiguration struct {
	// Enabled overrides auto-detection. nil = auto-detect.
	Enabled *bool `json:"enabled,omitempty"`
}

// InfrastructureConfiguration holds service mesh and backend addresses.
type InfrastructureConfiguration struct {
	// NATSAddress is the address of the NATS server
	NATSAddress string `json:"natsAddress"`
	// ETCDAddress is the address of the etcd server
	ETCDAddress string `json:"etcdAddress"`
	// ModelExpressURL is the URL of the Model Express server to inject into all pods
	ModelExpressURL string `json:"modelExpressURL"`
	// PrometheusEndpoint is the URL of the Prometheus endpoint to use for metrics
	PrometheusEndpoint string `json:"prometheusEndpoint"`
}

// IngressConfiguration holds ingress settings.
type IngressConfiguration struct {
	// VirtualServiceGateway is the name of the Istio virtual service gateway
	VirtualServiceGateway string `json:"virtualServiceGateway"`
	// ControllerClassName is the ingress controller class name
	ControllerClassName string `json:"controllerClassName"`
	// ControllerTLSSecretName is the TLS secret for the ingress controller
	ControllerTLSSecretName string `json:"controllerTLSSecretName"`
	// HostSuffix is the suffix for ingress hostnames
	HostSuffix string `json:"hostSuffix"`
}

// UseVirtualService returns true if a VirtualService gateway is configured.
func (i *IngressConfiguration) UseVirtualService() bool {
	return i.VirtualServiceGateway != ""
}

// RBACConfiguration holds RBAC settings for cluster-wide mode.
type RBACConfiguration struct {
	// PlannerClusterRoleName is the ClusterRole for planner
	PlannerClusterRoleName string `json:"plannerClusterRoleName"`
	// DGDRProfilingClusterRoleName is the ClusterRole for DGDR profiling jobs
	DGDRProfilingClusterRoleName string `json:"dgdrProfilingClusterRoleName"`
	// EPPClusterRoleName is the ClusterRole for EPP
	EPPClusterRoleName string `json:"eppClusterRoleName"`
}

// MPIConfiguration holds MPI SSH secret settings.
type MPIConfiguration struct {
	// SSHSecretName is the name of the secret containing the SSH key for MPI
	SSHSecretName string `json:"sshSecretName"`
	// SSHSecretNamespace is the namespace where the MPI SSH secret is located
	SSHSecretNamespace string `json:"sshSecretNamespace"`
}

// CheckpointConfiguration holds checkpoint/restore settings.
type CheckpointConfiguration struct {
	// Enabled indicates if checkpoint functionality is enabled
	Enabled bool `json:"enabled"`
	// ReadyForCheckpointFilePath signals model readiness for checkpoint jobs
	// +kubebuilder:default="/tmp/ready-for-checkpoint"
	ReadyForCheckpointFilePath string `json:"readyForCheckpointFilePath"`
	// Storage holds storage backend configuration
	Storage CheckpointStorageConfiguration `json:"storage"`
}

// CheckpointStorageConfiguration holds storage backend configuration for checkpoints.
type CheckpointStorageConfiguration struct {
	// Type is the storage backend type: pvc, s3, or oci
	// +kubebuilder:default="pvc"
	Type string `json:"type"`
	// PVC configuration (used when Type=pvc)
	PVC CheckpointPVCConfig `json:"pvc"`
	// S3 configuration (used when Type=s3)
	S3 CheckpointS3Config `json:"s3"`
	// OCI configuration (used when Type=oci)
	OCI CheckpointOCIConfig `json:"oci"`
}

// CheckpointPVCConfig holds PVC storage configuration.
type CheckpointPVCConfig struct {
	// PVCName is the name of the PVC
	// +kubebuilder:default="chrek-pvc"
	PVCName string `json:"pvcName"`
	// BasePath is the base directory within the PVC
	// +kubebuilder:default="/checkpoints"
	BasePath string `json:"basePath"`
}

// CheckpointS3Config holds S3 storage configuration.
type CheckpointS3Config struct {
	// URI is the S3 URI (s3://[endpoint/]bucket/prefix)
	URI string `json:"uri"`
	// CredentialsSecretRef is the name of the credentials secret
	CredentialsSecretRef string `json:"credentialsSecretRef"`
}

// CheckpointOCIConfig holds OCI registry storage configuration.
type CheckpointOCIConfig struct {
	// URI is the OCI URI (oci://registry/repository)
	URI string `json:"uri"`
	// CredentialsSecretRef is the name of the docker config secret
	CredentialsSecretRef string `json:"credentialsSecretRef"`
}

// DiscoveryConfiguration holds discovery backend settings.
type DiscoveryConfiguration struct {
	// Backend is the discovery backend: "kubernetes" or "etcd"
	// +kubebuilder:default="kubernetes"
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

// GPUConfiguration holds GPU discovery settings.
type GPUConfiguration struct {
	// DiscoveryEnabled indicates whether GPU discovery is enabled
	// +kubebuilder:default=true
	DiscoveryEnabled *bool `json:"discoveryEnabled,omitempty"`
}

// LoggingConfiguration holds logging settings.
type LoggingConfiguration struct {
	// Level is the log level (e.g., "info", "debug")
	// +kubebuilder:default="info"
	Level string `json:"level"`
	// Format is the log format (e.g., "json", "text")
	// +kubebuilder:default="json"
	Format string `json:"format"`
}

// SecurityConfiguration holds HTTP/2 and TLS settings.
type SecurityConfiguration struct {
	// EnableHTTP2 enables HTTP/2 for metrics and webhook servers
	// +kubebuilder:default=false
	EnableHTTP2 bool `json:"enableHTTP2"`
}
