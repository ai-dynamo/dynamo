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
	"encoding/json"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validConfig returns a minimal valid OperatorConfiguration for cluster-wide mode.
func validConfig() *configv1alpha1.OperatorConfiguration {
	cfg := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(cfg)
	cfg.MPI.SSHSecretName = "mpi-ssh"
	cfg.MPI.SSHSecretNamespace = "default"
	// Cluster-wide validation requires chart-provided RBAC names.
	cfg.RBAC.PlannerClusterRoleName = "planner-role"
	cfg.RBAC.DGDRProfilingClusterRoleName = "dgdr-profiling-role"
	cfg.RBAC.EPPClusterRoleName = "epp-role"
	return cfg
}

// validNamespaceScopedConfig returns a minimal valid OperatorConfiguration for namespace-restricted mode.
func validNamespaceScopedConfig() *configv1alpha1.OperatorConfiguration {
	cfg := validConfig()
	cfg.Namespace.Restricted = "my-namespace"
	// RBAC not required in namespace mode
	cfg.RBAC.PlannerClusterRoleName = ""
	cfg.RBAC.DGDRProfilingClusterRoleName = ""
	cfg.RBAC.EPPClusterRoleName = ""
	return cfg
}

func TestValidateOperatorConfiguration_Valid(t *testing.T) {
	errs := ValidateOperatorConfiguration(validConfig())
	if len(errs) != 0 {
		t.Errorf("expected no errors for valid config, got: %v", errs)
	}
}

func TestValidateOperatorConfiguration_ValidNamespaceScoped(t *testing.T) {
	errs := ValidateOperatorConfiguration(validNamespaceScopedConfig())
	if len(errs) != 0 {
		t.Errorf("expected no errors for valid namespace-scoped config, got: %v", errs)
	}
}

func TestValidateOperatorConfiguration_MissingMPISecret(t *testing.T) {
	cfg := validConfig()
	cfg.MPI.SSHSecretName = ""
	cfg.MPI.SSHSecretNamespace = ""

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 2 {
		t.Errorf("expected 2 errors for missing MPI secret, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_InvalidDiscoveryBackend(t *testing.T) {
	cfg := validConfig()
	cfg.Discovery.Backend = "consul"

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for invalid discovery backend, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_ClusterWideMissingPlannerRole(t *testing.T) {
	cfg := validConfig()
	cfg.RBAC.PlannerClusterRoleName = ""

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing planner role in cluster-wide mode, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_NamespaceScopedNoRBACRequired(t *testing.T) {
	cfg := validNamespaceScopedConfig()
	// Verify that RBAC is not required in namespace mode
	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors for namespace-scoped config without RBAC, got: %v", errs)
	}
}

func TestValidateOperatorConfiguration_NamespaceScopedInvalidLease(t *testing.T) {
	cfg := validNamespaceScopedConfig()
	cfg.Namespace.Scope.LeaseDuration = metav1.Duration{Duration: 0}
	cfg.Namespace.Scope.LeaseRenewInterval = metav1.Duration{Duration: 0}

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 2 {
		t.Errorf("expected 2 errors for zero lease values, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_NamespaceScopedLeaseRenewExceedsDuration(t *testing.T) {
	cfg := validNamespaceScopedConfig()
	cfg.Namespace.Scope.LeaseDuration = metav1.Duration{Duration: 10 * time.Second}
	cfg.Namespace.Scope.LeaseRenewInterval = metav1.Duration{Duration: 15 * time.Second}

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for lease renew > duration, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_CheckpointEnabledRequiresNoStorageConfig(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = true

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors for checkpoint config without storage settings, got: %v", errs)
	}
}

func TestValidateOperatorConfiguration_CheckpointDeprecatedStorageConfigIsAccepted(t *testing.T) {
	cfg := validConfig()
	rawConfig := []byte(`{
		"checkpoint": {
			"enabled": true,
			"storage": {
				"type": "s3",
				"s3": {
					"uri": "s3://legacy-bucket/checkpoints"
				}
			}
		}
	}`)
	if err := json.Unmarshal(rawConfig, cfg); err != nil {
		t.Fatalf("failed to unmarshal compatibility config: %v", err)
	}

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors for deprecated checkpoint storage config, got: %v", errs)
	}
}

func TestValidateOperatorConfiguration_CheckpointDisabledSkipsValidation(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = false

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors when checkpoint is disabled, got: %v", errs)
	}
}

func TestValidateOperatorConfiguration_InvalidModelExpressURL(t *testing.T) {
	cfg := validConfig()
	cfg.Infrastructure.ModelExpressURL = "://bad-url"

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for invalid model express URL, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_InvalidPort(t *testing.T) {
	cfg := validConfig()
	cfg.Server.Metrics.Port = 99999

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for invalid port, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_LeaderElectionEnabledMissingID(t *testing.T) {
	cfg := validConfig()
	cfg.LeaderElection.Enabled = true
	cfg.LeaderElection.ID = ""

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing leader election ID, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_NegativeTerminationDelay(t *testing.T) {
	cfg := validConfig()
	cfg.Orchestrators.Grove.TerminationDelay = metav1.Duration{Duration: -1 * time.Second}

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for negative termination delay, got %d: %v", len(errs), errs)
	}
}

// hasFieldError reports whether errs contains an entry whose field path ends with
// the given suffix, making the test resilient to the parent "infrastructure." prefix.
func hasFieldError(t *testing.T, errs field.ErrorList, fieldSuffix string) bool {
	t.Helper()
	for _, e := range errs {
		if e.Field == fieldSuffix || e.Field == "infrastructure."+fieldSuffix {
			return true
		}
	}
	return false
}

func TestValidateInfrastructure_TLS(t *testing.T) {
	// Valid: no TLS fields set → no TLS errors.
	t.Run("empty config has no TLS errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{}, field.NewPath("infrastructure"))
		if len(errs) != 0 {
			t.Fatalf("expected no errors, got %v", errs)
		}
	})

	// Valid: TCP server cert+key together, with CA.
	t.Run("TCP server cert+key+CA is valid", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSCertPath: "/c.pem",
			TCPTLSKeyPath:  "/k.pem",
			TCPTLSCAPath:   "/ca.pem",
		}, field.NewPath("infrastructure"))
		if len(errs) != 0 {
			t.Fatalf("expected no errors, got %v", errs)
		}
	})

	// Invalid: TCP server cert without key.
	t.Run("TCP server cert without key errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSCertPath: "/c.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "tcpTLSCertPath") {
			t.Fatalf("expected tcpTLSCertPath error, got %v", errs)
		}
	})

	// Invalid: TCP server cert+key without CA (client stays plaintext, server is TLS).
	t.Run("TCP server cert+key without CA errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSCertPath: "/c.pem",
			TCPTLSKeyPath:  "/k.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "tcpTLSCAPath") {
			t.Fatalf("expected tcpTLSCAPath error, got %v", errs)
		}
	})

	// Invalid: TCP client-CA without server cert/key.
	t.Run("TCP client-CA without server cert errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSClientCAPath: "/client-ca.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "tcpTLSClientCAPath") {
			t.Fatalf("expected tcpTLSClientCAPath error, got %v", errs)
		}
	})

	// Invalid: TCP client cert without CA.
	t.Run("TCP client cert without CA errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSClientCertPath: "/client.pem",
			TCPTLSClientKeyPath:  "/client-key.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "tcpTLSClientCertPath") {
			t.Fatalf("expected tcpTLSClientCertPath error, got %v", errs)
		}
	})

	// Invalid: TCP CA without server cert/key (client-side TLS without server-side TLS).
	t.Run("TCP CA without server cert errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSCAPath: "/ca.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "tcpTLSCertPath") {
			t.Fatalf("expected tcpTLSCertPath error, got %v", errs)
		}
	})

	// Invalid: TCP client cert/key without server cert/key.
	t.Run("TCP client cert without server cert errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSClientCertPath: "/client.pem",
			TCPTLSClientKeyPath:  "/client-key.pem",
			TCPTLSCAPath:         "/ca.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "tcpTLSCertPath") {
			t.Fatalf("expected tcpTLSCertPath error, got %v", errs)
		}
	})

	// Invalid: NATS client cert without CA.
	t.Run("NATS client cert without CA errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			NATSAddress:           "tls://nats:4222",
			NATSTLSClientCertPath: "/client.pem",
			NATSTLSClientKeyPath:  "/client-key.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "natsTLSClientCertPath") {
			t.Fatalf("expected natsTLSClientCertPath error, got %v", errs)
		}
	})

	// Invalid: NATS TLS with nats:// address.
	t.Run("NATS TLS with nats:// address errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			NATSAddress:   "nats://nats:4222",
			NATSTLSCAPath: "/ca.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "natsAddress") {
			t.Fatalf("expected natsAddress error, got %v", errs)
		}
	})

	// Invalid: NATS TLS without natsAddress (runtime would default to nats:// and fail).
	t.Run("NATS TLS without natsAddress errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			NATSTLSCAPath: "/ca.pem",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "natsAddress") {
			t.Fatalf("expected natsAddress error, got %v", errs)
		}
	})

	// Valid: NATS TLS with tls:// address and CA.
	t.Run("NATS TLS with tls:// address and CA is valid", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			NATSAddress:   "tls://nats:4222",
			NATSTLSCAPath: "/ca.pem",
		}, field.NewPath("infrastructure"))
		if len(errs) != 0 {
			t.Fatalf("expected no errors, got %v", errs)
		}
	})

	// Invalid: tcpTLSServerName without tcpTLSCAPath (SNI override is inert without a TLS connector).
	t.Run("TCP server name without CA errors", func(t *testing.T) {
		errs := validateInfrastructure(&configv1alpha1.InfrastructureConfiguration{
			TCPTLSServerName: "dynamo-worker.dynamo-system.svc.cluster.local",
		}, field.NewPath("infrastructure"))
		if !hasFieldError(t, errs, "tcpTLSServerName") {
			t.Fatalf("expected tcpTLSServerName error, got %v", errs)
		}
	})
}
