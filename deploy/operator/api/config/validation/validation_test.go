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
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// validConfig returns a minimal valid OperatorConfiguration for cluster-wide mode.
func validConfig() *configv1alpha1.OperatorConfiguration {
	cfg := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaults_OperatorConfiguration(cfg)
	cfg.MPI.SSHSecretName = "mpi-ssh"
	cfg.MPI.SSHSecretNamespace = "default"
	cfg.RBAC.PlannerClusterRoleName = "planner-role"
	return cfg
}

// validNamespaceScopedConfig returns a minimal valid OperatorConfiguration for namespace-restricted mode.
func validNamespaceScopedConfig() *configv1alpha1.OperatorConfiguration {
	cfg := validConfig()
	cfg.Namespace.Restricted = "my-namespace"
	// RBAC not required in namespace mode
	cfg.RBAC.PlannerClusterRoleName = ""
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

func TestValidateOperatorConfiguration_CheckpointS3MissingURI(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = true
	cfg.Checkpoint.Storage.Type = configv1alpha1.CheckpointStorageTypeS3
	cfg.Checkpoint.Storage.S3.URI = ""

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing S3 URI, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_CheckpointOCIMissingURI(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = true
	cfg.Checkpoint.Storage.Type = configv1alpha1.CheckpointStorageTypeOCI
	cfg.Checkpoint.Storage.OCI.URI = ""

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing OCI URI, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_CheckpointInvalidStorageType(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = true
	cfg.Checkpoint.Storage.Type = "nfs"

	errs := ValidateOperatorConfiguration(cfg)
	if len(errs) != 1 {
		t.Errorf("expected 1 error for invalid storage type, got %d: %v", len(errs), errs)
	}
}

func TestValidateOperatorConfiguration_CheckpointDisabledSkipsValidation(t *testing.T) {
	cfg := validConfig()
	cfg.Checkpoint.Enabled = false
	cfg.Checkpoint.Storage.Type = "invalid"

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
