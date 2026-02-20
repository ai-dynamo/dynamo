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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateOperatorConfiguration validates an OperatorConfiguration object.
func ValidateOperatorConfiguration(config *configv1alpha1.OperatorConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, validateServer(&config.Server, field.NewPath("server"))...)
	allErrs = append(allErrs, validateLeaderElection(&config.LeaderElection, field.NewPath("leaderElection"))...)
	allErrs = append(allErrs, validateNamespace(&config.Namespace, field.NewPath("namespace"))...)
	allErrs = append(allErrs, validateMPI(&config.MPI, field.NewPath("mpi"))...)
	allErrs = append(allErrs, validateInfrastructure(&config.Infrastructure, field.NewPath("infrastructure"))...)
	allErrs = append(allErrs, validateDiscovery(&config.Discovery, field.NewPath("discovery"))...)
	allErrs = append(allErrs, validateCheckpoint(&config.Checkpoint, field.NewPath("checkpoint"))...)
	allErrs = append(allErrs, validateRBAC(config)...)
	allErrs = append(allErrs, validateOrchestrators(&config.Orchestrators, field.NewPath("orchestrators"))...)
	allErrs = append(allErrs, validateIngress(&config.Ingress, field.NewPath("ingress"))...)

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

	return allErrs
}

func validateDiscovery(discovery *configv1alpha1.DiscoveryConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if discovery.Backend != configv1alpha1.DiscoveryBackendKubernetes && discovery.Backend != configv1alpha1.DiscoveryBackendEtcd {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("backend"), discovery.Backend, []string{"kubernetes", "etcd"}))
	}

	return allErrs
}

func validateCheckpoint(checkpoint *configv1alpha1.CheckpointConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if !checkpoint.Enabled {
		return allErrs
	}

	storagePath := fldPath.Child("storage")
	switch checkpoint.Storage.Type {
	case configv1alpha1.CheckpointStorageTypePVC:
		// PVC is the default, no additional required fields
	case configv1alpha1.CheckpointStorageTypeS3:
		if checkpoint.Storage.S3.URI == "" {
			allErrs = append(allErrs, field.Required(storagePath.Child("s3", "uri"), "S3 URI is required when storage type is s3"))
		}
	case configv1alpha1.CheckpointStorageTypeOCI:
		if checkpoint.Storage.OCI.URI == "" {
			allErrs = append(allErrs, field.Required(storagePath.Child("oci", "uri"), "OCI URI is required when storage type is oci"))
		}
	default:
		allErrs = append(allErrs, field.NotSupported(storagePath.Child("type"), checkpoint.Storage.Type,
			[]string{configv1alpha1.CheckpointStorageTypePVC, configv1alpha1.CheckpointStorageTypeS3, configv1alpha1.CheckpointStorageTypeOCI}))
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
	if config.RBAC.PlannerClusterRoleName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("plannerClusterRoleName"), "planner ClusterRole name is required in cluster-wide mode"))
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
