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
	"reflect"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// checkpointDGD builds a single-worker DGD named name whose worker carries an
// enabled checkpoint block, mutated by mutate when supplied.
func checkpointDGD(t testing.TB, name string, mutate func(*v1beta1.ComponentCheckpointConfig)) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	src := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"decode": {ComponentType: commonconsts.ComponentTypeDecode},
	})
	src.Name = name
	dgd := betaDGD(t, src)

	checkpoint := &v1beta1.ComponentCheckpointConfig{
		Enabled:             true,
		TargetContainerName: commonconsts.MainContainerName,
		StartupPolicy:       v1beta1.CheckpointStartupPolicyImmediate,
		DeletionPolicy:      v1beta1.CheckpointDeletionPolicyDelete,
	}
	if mutate != nil {
		mutate(checkpoint)
	}
	dgd.Spec.Components[0].Experimental = &v1beta1.ExperimentalSpec{Checkpoint: checkpoint}

	return dgd
}

func mustComputeCheckpointCompatHash(t testing.TB, dgd *v1beta1.DynamoGraphDeployment, componentName string) string {
	t.Helper()
	hash, err := ComputeDGDWorkerCheckpointCompatHash(dgd, componentName)
	require.NoError(t, err)
	return hash
}

func TestComputeDGDWorkerCheckpointCompatHash_ReportedScenario(t *testing.T) {
	t.Log("Build the probe DGD that captures an automatic checkpoint and retains it")
	probe := mustComputeCheckpointCompatHash(t, checkpointDGD(t, "tc-probe", func(c *v1beta1.ComponentCheckpointConfig) {
		c.DeletionPolicy = v1beta1.CheckpointDeletionPolicyRetain
	}), "decode")

	t.Log("Build a separately named consumer DGD with a byte-identical worker restoring that PodSnapshot")
	consumer := mustComputeCheckpointCompatHash(t, checkpointDGD(t, "tc-consumer", func(c *v1beta1.ComponentCheckpointConfig) {
		c.CheckpointRef = ptr.To("tc-standalone-snapjob")
	}), "decode")

	t.Log("Verify the consumer matches the compatibility identity recorded at capture")
	assert.Equal(t, probe, consumer)
}

func TestComputeDGDWorkerCheckpointCompatHash_IgnoresNonRestoreFields(t *testing.T) {
	baseline := mustComputeCheckpointCompatHash(t, checkpointDGD(t, "tc-probe", nil), "decode")

	tests := map[string]func(*v1beta1.ComponentCheckpointConfig){
		"checkpointRef":  func(c *v1beta1.ComponentCheckpointConfig) { c.CheckpointRef = ptr.To("other-snapjob") },
		"deletionPolicy": func(c *v1beta1.ComponentCheckpointConfig) { c.DeletionPolicy = v1beta1.CheckpointDeletionPolicyRetain },
		"startupPolicy": func(c *v1beta1.ComponentCheckpointConfig) {
			c.StartupPolicy = v1beta1.CheckpointStartupPolicyWaitForCheckpoint
		},
		"mode": func(c *v1beta1.ComponentCheckpointConfig) { c.Mode = v1beta1.CheckpointModeManual },
		"identity": func(c *v1beta1.ComponentCheckpointConfig) {
			c.Identity = &v1beta1.DynamoCheckpointIdentity{Model: "Qwen/Qwen3-0.6B", BackendFramework: "vllm"}
		},
	}
	for name, mutate := range tests {
		assert.Equal(t, baseline, mustComputeCheckpointCompatHash(t, checkpointDGD(t, "tc-probe", mutate), "decode"), name)
	}
}

func TestComputeDGDWorkerCheckpointCompatHash_TracksRestoreShape(t *testing.T) {
	baseline := mustComputeCheckpointCompatHash(t, checkpointDGD(t, "tc-probe", nil), "decode")

	tests := map[string]func(*v1beta1.ComponentCheckpointConfig){
		"disabled":               func(c *v1beta1.ComponentCheckpointConfig) { c.Enabled = false },
		"other target container": func(c *v1beta1.ComponentCheckpointConfig) { c.TargetContainerName = "sidecar" },
		// The capture Pod determines what the produced PodSnapshot contains, so
		// a snapshot taken under a different capture template is a different
		// artifact even when the worker Pod is unchanged.
		"capture job template": func(c *v1beta1.ComponentCheckpointConfig) {
			c.Job = &v1beta1.ComponentCheckpointJobConfig{GMSClientContainers: []string{"gms-saver"}}
		},
	}
	for name, mutate := range tests {
		assert.NotEqual(t, baseline, mustComputeCheckpointCompatHash(t, checkpointDGD(t, "tc-probe", mutate), "decode"), name)
	}
}

func TestComputeDGDWorkerCheckpointCompatHash_TracksWorkerPodShape(t *testing.T) {
	baseline := mustComputeCheckpointCompatHash(t, checkpointDGD(t, "tc-probe", nil), "decode")

	t.Log("Verify a different worker image is not a compatible restore target")
	image := checkpointDGD(t, "tc-probe", nil)
	image.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "other:1.0"}},
		},
	}
	assert.NotEqual(t, baseline, mustComputeCheckpointCompatHash(t, image, "decode"))

	t.Log("Verify the Kubernetes namespace stays part of the identity, since a checkpointRef never crosses one")
	namespace := checkpointDGD(t, "tc-probe", nil)
	namespace.Namespace = "other"
	assert.NotEqual(t, baseline, mustComputeCheckpointCompatHash(t, namespace, "decode"))

	t.Log("Verify the user-settable Dynamo namespace input stays part of the identity")
	global := checkpointDGD(t, "tc-probe", nil)
	global.Spec.Components[0].GlobalDynamoNamespace = true
	assert.NotEqual(t, baseline, mustComputeCheckpointCompatHash(t, global, "decode"))
}

func TestComputeDGDWorkerCheckpointCompatHash_IsScopedToOneComponent(t *testing.T) {
	t.Log("Build a disaggregated DGD whose decode worker is checkpointed")
	disagg := func() *v1beta1.DynamoGraphDeployment {
		src := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
			"decode":  {ComponentType: commonconsts.ComponentTypeDecode},
			"prefill": {ComponentType: commonconsts.ComponentTypePrefill},
		})
		src.Name = "tc-probe"
		dgd := betaDGD(t, src)
		for i := range dgd.Spec.Components {
			if dgd.Spec.Components[i].ComponentName == "decode" {
				dgd.Spec.Components[i].Experimental = &v1beta1.ExperimentalSpec{
					Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}
		}
		return dgd
	}
	baseline := mustComputeCheckpointCompatHash(t, disagg(), "decode")

	t.Log("Verify editing the prefill worker does not invalidate the decode checkpoint")
	changed := disagg()
	for i := range changed.Spec.Components {
		if changed.Spec.Components[i].ComponentName == "prefill" {
			changed.Spec.Components[i].PodTemplate = &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "other:1.0"}},
				},
			}
		}
	}
	assert.Equal(t, baseline, mustComputeCheckpointCompatHash(t, changed, "decode"))
	assert.NotEqual(t, baseline, mustComputeCheckpointCompatHash(t, changed, "prefill"))
}

func TestComputeDGDWorkerCheckpointCompatHash_RejectsNonWorkerComponents(t *testing.T) {
	dgd := betaDGD(t, baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"decode":   {ComponentType: commonconsts.ComponentTypeDecode},
		"frontend": {ComponentType: commonconsts.ComponentTypeFrontend},
	}))

	_, err := ComputeDGDWorkerCheckpointCompatHash(dgd, "frontend")
	assert.ErrorContains(t, err, "not a worker component")

	_, err = ComputeDGDWorkerCheckpointCompatHash(dgd, "absent")
	assert.ErrorContains(t, err, "no generated DCD")
}

// TestCheckpointCompatHashFieldsReviewed fails when ComponentCheckpointConfig
// gains a field, forcing an explicit decision about whether it identifies a
// restorable worker. Add it to canonicalizeCheckpointForCompatHash when it only
// selects a PodSnapshot or governs checkpoint CR lifecycle; otherwise leave it
// hashed.
func TestCheckpointCompatHashFieldsReviewed(t *testing.T) {
	reviewed := []string{
		"Enabled", "Mode", "StartupPolicy", "DeletionPolicy",
		"CheckpointRef", "Identity", "TargetContainerName", "Job",
	}

	configType := reflect.TypeOf(v1beta1.ComponentCheckpointConfig{})
	actual := make([]string, 0, configType.NumField())
	for i := range configType.NumField() {
		actual = append(actual, configType.Field(i).Name)
	}

	assert.ElementsMatch(t, reviewed, actual,
		"ComponentCheckpointConfig changed: decide whether each new field belongs in the checkpoint compatibility hash")
}

// TestComputeDGDWorkersSpecHash_UnaffectedByCheckpointCompatHash pins the
// rollout hash against the compatibility hash's canonicalization. An
// operator-only upgrade must not roll unchanged workloads, so these two must
// stay independent.
func TestComputeDGDWorkersSpecHash_UnaffectedByCheckpointCompatHash(t *testing.T) {
	t.Log("Verify checkpoint selection still creates a worker generation in the rollout hash")
	baseline := mustComputeBetaDGDWorkersSpecHash(t, checkpointDGD(t, "tc-probe", nil))
	withRef := mustComputeBetaDGDWorkersSpecHash(t, checkpointDGD(t, "tc-probe", func(c *v1beta1.ComponentCheckpointConfig) {
		c.CheckpointRef = ptr.To("tc-standalone-snapjob")
	}))
	assert.NotEqual(t, baseline, withRef)

	t.Log("Verify graph identity still creates a worker generation in the rollout hash")
	assert.NotEqual(t, baseline, mustComputeBetaDGDWorkersSpecHash(t, checkpointDGD(t, "tc-consumer", nil)))
}
