/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package config

import (
	"encoding/json"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func TestValidate_Valid(t *testing.T) {
	errs := validate(validConfig())
	if len(errs) != 0 {
		t.Errorf("expected no errors for valid config, got: %v", errs)
	}
}

func TestValidate_ValidNamespaceScoped(t *testing.T) {
	errs := validate(validNamespaceScopedConfig())
	if len(errs) != 0 {
		t.Errorf("expected no errors for valid namespace-scoped config, got: %v", errs)
	}
}

func TestValidate_MissingMPISecret(t *testing.T) {
	cfg := validConfig()
	cfg.MPI.SSHSecretName = ""
	cfg.MPI.SSHSecretNamespace = ""

	errs := validate(cfg)
	if len(errs) != 2 {
		t.Errorf("expected 2 errors for missing MPI secret, got %d: %v", len(errs), errs)
	}
}

func TestValidate_InvalidDiscoveryBackend(t *testing.T) {
	cfg := validConfig()
	cfg.Discovery.Backend = "consul"

	errs := validate(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for invalid discovery backend, got %d: %v", len(errs), errs)
	}
}

func TestValidate_ClusterWideMissingPlannerRole(t *testing.T) {
	cfg := validConfig()
	cfg.RBAC.PlannerClusterRoleName = ""

	errs := validate(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing planner role in cluster-wide mode, got %d: %v", len(errs), errs)
	}
}

func TestValidate_NamespaceScopedNoRBACRequired(t *testing.T) {
	cfg := validNamespaceScopedConfig()
	// Verify that RBAC is not required in namespace mode
	errs := validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors for namespace-scoped config without RBAC, got: %v", errs)
	}
}

func TestValidate_NamespaceScopedInvalidLease(t *testing.T) {
	cfg := validNamespaceScopedConfig()
	cfg.Namespace.Scope.LeaseDuration = metav1.Duration{Duration: 0}
	cfg.Namespace.Scope.LeaseRenewInterval = metav1.Duration{Duration: 0}

	errs := validate(cfg)
	if len(errs) != 2 {
		t.Errorf("expected 2 errors for zero lease values, got %d: %v", len(errs), errs)
	}
}

func TestValidate_NamespaceScopedLeaseRenewExceedsDuration(t *testing.T) {
	cfg := validNamespaceScopedConfig()
	cfg.Namespace.Scope.LeaseDuration = metav1.Duration{Duration: 10 * time.Second}
	cfg.Namespace.Scope.LeaseRenewInterval = metav1.Duration{Duration: 15 * time.Second}

	errs := validate(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for lease renew > duration, got %d: %v", len(errs), errs)
	}
}

func TestValidate_CheckpointEnabledRequiresNoStorageConfig(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = true

	errs := validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors for checkpoint config without storage settings, got: %v", errs)
	}
}

func TestValidate_CheckpointDeprecatedStorageConfigIsAccepted(t *testing.T) {
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

	errs := validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors for deprecated checkpoint storage config, got: %v", errs)
	}
}

func TestValidate_CheckpointDisabledSkipsValidation(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = false

	errs := validate(cfg)
	if len(errs) != 0 {
		t.Errorf("expected no errors when checkpoint is disabled, got: %v", errs)
	}
}

func TestValidate_InvalidModelExpressURL(t *testing.T) {
	cfg := validConfig()
	cfg.Infrastructure.ModelExpressURL = "://bad-url"

	errs := validate(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for invalid model express URL, got %d: %v", len(errs), errs)
	}
}

func TestValidate_InvalidPort(t *testing.T) {
	cfg := validConfig()
	cfg.Server.Metrics.Port = 99999

	errs := validate(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for invalid port, got %d: %v", len(errs), errs)
	}
}

func TestValidate_LeaderElectionEnabledMissingID(t *testing.T) {
	cfg := validConfig()
	cfg.LeaderElection.Enabled = true
	cfg.LeaderElection.ID = ""

	errs := validate(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing leader election ID, got %d: %v", len(errs), errs)
	}
}

func TestValidate_NegativeTerminationDelay(t *testing.T) {
	cfg := validConfig()
	cfg.Orchestrators.Grove.TerminationDelay = metav1.Duration{Duration: -1 * time.Second}

	errs := validate(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for negative termination delay, got %d: %v", len(errs), errs)
	}
}
