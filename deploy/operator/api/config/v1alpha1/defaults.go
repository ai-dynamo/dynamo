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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SetDefaults_OperatorConfiguration sets default values for OperatorConfiguration.
func SetDefaults_OperatorConfiguration(obj *OperatorConfiguration) {
	// Server defaults
	if obj.Server.Metrics.Port == 0 {
		obj.Server.Metrics.Port = 8080
	}
	if obj.Server.HealthProbe.Port == 0 {
		obj.Server.HealthProbe.Port = 8081
	}
	if obj.Server.Webhook.Host == "" {
		obj.Server.Webhook.Host = "0.0.0.0"
	}
	if obj.Server.Webhook.Port == 0 {
		obj.Server.Webhook.Port = 9443
	}
	if obj.Server.Webhook.CertDir == "" {
		obj.Server.Webhook.CertDir = "/tmp/k8s-webhook-server/serving-certs"
	}

	// Orchestrator defaults
	if obj.Orchestrators.Grove.TerminationDelay.Duration == 0 {
		obj.Orchestrators.Grove.TerminationDelay = metav1.Duration{Duration: 15 * time.Minute}
	}

	// Namespace scope defaults
	if obj.Namespace.Scope.LeaseDuration.Duration == 0 {
		obj.Namespace.Scope.LeaseDuration = metav1.Duration{Duration: 30 * time.Second}
	}
	if obj.Namespace.Scope.LeaseRenewInterval.Duration == 0 {
		obj.Namespace.Scope.LeaseRenewInterval = metav1.Duration{Duration: 10 * time.Second}
	}

	// Discovery defaults
	if obj.Discovery.Backend == "" {
		obj.Discovery.Backend = DiscoveryBackendKubernetes
	}

	// Checkpoint defaults
	if obj.Checkpoint.InitContainerImage == "" {
		obj.Checkpoint.InitContainerImage = "busybox:latest"
	}
	if obj.Checkpoint.ReadyForCheckpointFilePath == "" {
		obj.Checkpoint.ReadyForCheckpointFilePath = "/tmp/ready-for-checkpoint"
	}
	if obj.Checkpoint.RestoreMarkerFilePath == "" {
		obj.Checkpoint.RestoreMarkerFilePath = "/tmp/dynamo-restored"
	}
	if obj.Checkpoint.Storage.Type == "" {
		obj.Checkpoint.Storage.Type = CheckpointStorageTypePVC
	}
	if obj.Checkpoint.Storage.SignalHostPath == "" {
		obj.Checkpoint.Storage.SignalHostPath = "/var/lib/chrek/signals"
	}
	if obj.Checkpoint.Storage.PVC.PVCName == "" {
		obj.Checkpoint.Storage.PVC.PVCName = "chrek-pvc"
	}
	if obj.Checkpoint.Storage.PVC.BasePath == "" {
		obj.Checkpoint.Storage.PVC.BasePath = "/checkpoints"
	}

	// Logging defaults
	if obj.Logging.Level == "" {
		obj.Logging.Level = "info"
	}
	if obj.Logging.Format == "" {
		obj.Logging.Format = "json"
	}
}
