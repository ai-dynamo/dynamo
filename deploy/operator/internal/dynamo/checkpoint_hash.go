/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package dynamo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// ComputeDGDWorkerCheckpointCompatHash computes the checkpoint compatibility
// hash for one worker component of dgd, which must be non-nil and must declare
// componentName as a worker.
//
// This answers a different question from ComputeDGDWorkersSpecHash. That hash
// identifies a worker *generation* within one DGD and drives managed rollout.
// This one identifies a worker *shape*: whether a PodSnapshot captured from one
// worker can be restored into another. The two differ on three points, each of
// which makes the rollout hash unusable as a restore-compatibility token:
//
//   - Scope. The rollout hash covers every worker component at once, so editing
//     a prefill worker would invalidate a decode worker's checkpoint. This hash
//     covers the one component being restored.
//   - Graph identity. The rollout hash carries the owning DGD's name, so a
//     checkpoint captured by one DGD could never be restored into another,
//     however identically configured.
//   - Checkpoint selection. The rollout hash carries checkpointRef, which is
//     necessarily empty at capture time and set at restore time, so an explicit
//     reference could never match the value recorded at capture.
//
// Keeping this separate leaves ComputeDGDWorkersSpecHash byte-identical, so
// deployments that do not use checkpointing are not rolled by an operator
// upgrade.
func ComputeDGDWorkerCheckpointCompatHash(dgd *v1beta1.DynamoGraphDeployment, componentName string) (string, error) {
	dcds, err := GenerateDynamoComponentsDeployments(
		dgd,
		nil,
		nil,
		RollingUpdateContext{NewWorkerHash: dgdWorkerHashPlaceholderValue},
	)
	if err != nil {
		return "", err
	}

	for _, dcd := range dcds {
		if dcd == nil || GetDCDComponentName(dcd) != componentName {
			continue
		}
		if !IsWorkerComponent(string(dcd.Spec.ComponentType)) {
			return "", fmt.Errorf("component %q is not a worker component", componentName)
		}

		spec := workerHashSpec(dcd)
		canonicalizeCheckpointForCompatHash(spec.Experimental)
		data, err := json.Marshal(struct {
			Labels         map[string]string                     `json:"labels,omitempty"`
			Annotations    map[string]string                     `json:"annotations,omitempty"`
			RuntimeVersion string                                `json:"runtimeVersion,omitempty"`
			Spec           v1beta1.DynamoComponentDeploymentSpec `json:"spec"`
		}{
			Labels:         canonicalizeGraphIdentityForCompatHash(dcd, GetDCDKubeLabels(dcd)),
			Annotations:    GetDCDKubeAnnotations(dcd),
			RuntimeVersion: resolvedRuntimeVersionForHash(&dcd.Spec.DynamoComponentDeploymentSharedSpec),
			Spec:           spec,
		})
		if err != nil {
			return "", fmt.Errorf("marshal generated worker DCD %q: %w", componentName, err)
		}

		hash := sha256.Sum256(data)
		return hex.EncodeToString(hash[:])[:8], nil
	}

	return "", fmt.Errorf("no generated DCD for component %q", componentName)
}

// canonicalizeCheckpointForCompatHash removes the checkpoint fields that select
// a PodSnapshot or govern its lifecycle from the spec copy returned by
// workerHashSpec.
//
// checkpointRef is the field that makes explicit reuse impossible: the worker
// that captured a checkpoint could not have referenced its own not-yet-existing
// PodSnapshot. deletionPolicy has the same defect — retaining a checkpoint past
// its producing DGD is exactly what a later consumer needs, yet the consumer
// has no reason to repeat the producer's Retain. startupPolicy only gates the
// replica count, which workerHashSpec already excludes.
//
// enabled, targetContainerName, and job all stay: they describe how the worker
// Pod is rendered, which container is captured, and what the capture Pod ran.
func canonicalizeCheckpointForCompatHash(experimental *v1beta1.ExperimentalSpec) {
	if experimental == nil || experimental.Checkpoint == nil {
		return
	}
	checkpoint := experimental.Checkpoint

	checkpoint.CheckpointRef = nil
	checkpoint.Identity = nil
	checkpoint.Mode = ""
	checkpoint.StartupPolicy = ""
	checkpoint.DeletionPolicy = ""
}

// canonicalizeGraphIdentityForCompatHash removes the owning DGD's name from the
// labels rendered for dcd, where it otherwise appears twice: directly, and as
// the second segment of the Dynamo namespace.
//
// The Kubernetes namespace stays. A checkpointRef resolves a PodSnapshot in the
// consumer's own namespace, so dropping it would widen compatibility without
// enabling any supported reuse. globalDynamoNamespace remains part of the hash
// through its effect on this label.
//
// Removing the name narrows what the hash covers: getCommonContainer injects
// DYN_NAMESPACE and DYN_PARENT_DGD_K8S_NAME downstream of the DCD spec, so two
// workers that share this hash still render Pods differing in those values.
func canonicalizeGraphIdentityForCompatHash(dcd *v1beta1.DynamoComponentDeployment, labels map[string]string) map[string]string {
	delete(labels, commonconsts.KubeLabelDynamoGraphDeploymentName)
	if _, ok := labels[commonconsts.KubeLabelDynamoNamespace]; ok {
		labels[commonconsts.KubeLabelDynamoNamespace] = v1beta1.ComputeDynamoNamespace(
			dcd.Spec.GlobalDynamoNamespace,
			dcd.GetNamespace(),
			"",
		)
	}
	return labels
}
