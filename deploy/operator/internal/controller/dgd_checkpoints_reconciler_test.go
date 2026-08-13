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

package controller

import (
	"context"
	"errors"
	"fmt"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpointjob"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func newTestDGDCheckpointsReconciler(
	reconciler *DynamoGraphDeploymentReconciler,
) *dgdCheckpointsReconciler {
	return newDGDCheckpointsReconciler(
		newTestDGDResourceSyncer(reconciler),
		reconciler.Config,
		reconciler.RuntimeConfig,
		reconciler.DockerSecretRetriever,
	)
}

func TestDGDCheckpointsReconciler_CreateDoesNotReuseExistingCapture(t *testing.T) {
	t.Log("Build an existing checkpoint and a DGD-managed checkpoint request")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	existing := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "existing-worker-checkpoint",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "keep-existing:latest",
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(existing).
			Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{},
		Recorder:      events.NewFakeRecorder(10),
	}

	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
	})
	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: string(commonconsts.ComponentTypeWorker),
		Checkpoint: &v1alpha1.ServiceCheckpointConfig{
			Enabled: true,
			Mode:    v1alpha1.CheckpointModeAuto,
			Identity: &v1alpha1.DynamoCheckpointIdentity{
				Model:                identity.Model,
				BackendFramework:     identity.BackendFramework,
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				ExtraParameters:      map[string]string{},
			},
		},
		ExtraPodSpec: &v1alpha1.ExtraPodSpec{
			MainContainer: &corev1.Container{
				Name:  "main",
				Image: "new-writer:latest",
			},
		},
	}

	t.Log("Create the DGD-managed checkpoint")
	ckpt, err := newTestDGDCheckpointsReconciler(reconciler).createCheckpointCR(ctx, dgd, "worker", betaComponent(t, component))
	if err != nil {
		t.Fatalf("createCheckpointCR() error = %v", err)
	}

	t.Log("Verify a deterministic owned checkpoint was created without mutating the existing one")
	if ckpt.Name == "existing-worker-checkpoint" {
		t.Fatalf("createCheckpointCR() reused existing checkpoint")
	}
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	if err != nil {
		t.Fatalf("checkpointWorkerHashForComponent() error = %v", err)
	}
	expectedID := checkpoint.DGDCheckpointID(
		dgd.Namespace,
		dgd.Name,
		string(dgd.UID),
		"worker",
		workerHash,
	)
	expectedName := fmt.Sprintf("checkpoint-%s", expectedID)
	if ckpt.Name != expectedName {
		t.Fatalf("createCheckpointCR() returned checkpoint %s, want %s", ckpt.Name, expectedName)
	}
	if got := ckpt.Labels[snapshotprotocol.CheckpointIDLabel]; got != expectedID {
		t.Fatalf("checkpoint ID label = %s, want %s", got, expectedID)
	}

	updated := &v1alpha1.DynamoCheckpoint{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "existing-worker-checkpoint", Namespace: "default"}, updated); err != nil {
		t.Fatalf("Failed to get checkpoint: %v", err)
	}
	if len(updated.Spec.Job.PodTemplateSpec.Spec.Containers) != 1 {
		t.Fatalf("expected one job container, got %d", len(updated.Spec.Job.PodTemplateSpec.Spec.Containers))
	}
	if updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image != "keep-existing:latest" {
		t.Fatalf("existing job image was mutated to %s", updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image)
	}
	created := &v1alpha1.DynamoCheckpoint{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: ckpt.Name, Namespace: "default"}, created); err != nil {
		t.Fatalf("Failed to get created checkpoint: %v", err)
	}
	if len(created.OwnerReferences) != 1 || created.OwnerReferences[0].UID != dgd.UID {
		t.Fatalf("expected created checkpoint to be owned by DGD UID %q, got %#v", dgd.UID, created.OwnerReferences)
	}
}

func TestDGDCheckpointsReconciler_CreateDoesNotAdoptLegacyIdentityTemplate(t *testing.T) {
	t.Log("Build a legacy checkpoint and ResourceClaimTemplate with existing ownership")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	require.NoError(t, err)

	existing := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "existing-worker-checkpoint",
			Namespace: "default",
			UID:       types.UID("checkpoint-uid"),
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity:         identity,
			GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{Enabled: true},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			IdentityHash: hash,
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
	}
	claimTemplateName := checkpointGMSResourceClaimTemplateName(hash)
	template := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      claimTemplateName,
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(dgd, v1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
			},
		},
	}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(existing, dgd, template).
			Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{
			Gate: features.Gates{},
		},
	}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		ComponentType: v1beta1.ComponentTypeWorker,
		Experimental: &v1beta1.ExperimentalSpec{
			Checkpoint: &v1beta1.ComponentCheckpointConfig{
				Enabled: true,
				Mode:    v1beta1.CheckpointModeAuto,
				Identity: &v1beta1.DynamoCheckpointIdentity{
					Model:            identity.Model,
					BackendFramework: identity.BackendFramework,
				},
			},
		},
	}

	t.Log("Create the DGD-managed checkpoint")
	ckpt, err := newTestDGDCheckpointsReconciler(reconciler).createCheckpointCR(ctx, dgd, "worker", component)
	require.NoError(t, err)

	t.Log("Verify the new checkpoint does not adopt the legacy template")
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	checkpointID := checkpoint.DGDCheckpointID(
		dgd.Namespace,
		dgd.Name,
		string(dgd.UID),
		"worker",
		workerHash,
	)
	assert.Equal(t, "checkpoint-"+checkpointID, ckpt.Name)
	assert.NotEqual(t, existing.Name, ckpt.Name)

	updatedTemplate := &resourcev1.ResourceClaimTemplate{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Name: claimTemplateName, Namespace: "default"}, updatedTemplate))
	controllerRef := metav1.GetControllerOf(updatedTemplate)
	require.NotNil(t, controllerRef)
	assert.Equal(t, "DynamoGraphDeployment", controllerRef.Kind)
	assert.Equal(t, dgd.Name, controllerRef.Name)
}

func TestDGDCheckpointsReconciler_CreatePreservesGMSSaverClient(t *testing.T) {
	t.Log("Build a GMS checkpoint component with a saver client")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	deviceClass := &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: dra.DefaultDeviceClassName}}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(deviceClass).
			Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{
			Gate: features.Gates{},
		},
	}

	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
	})
	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: string(commonconsts.ComponentTypeWorker),
		Resources: &v1alpha1.Resources{
			Limits: &v1alpha1.ResourceItem{GPU: "1"},
		},
		GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{
			Enabled: true,
			Mode:    v1alpha1.GMSModeIntraPod,
		},
		Checkpoint: &v1alpha1.ServiceCheckpointConfig{
			Enabled: true,
			Mode:    v1alpha1.CheckpointModeAuto,
			Identity: &v1alpha1.DynamoCheckpointIdentity{
				Model:                identity.Model,
				BackendFramework:     identity.BackendFramework,
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				ExtraParameters:      map[string]string{},
			},
		},
		ExtraPodSpec: &v1alpha1.ExtraPodSpec{
			MainContainer: &corev1.Container{
				Name:  commonconsts.MainContainerName,
				Image: "checkpoint-writer:latest",
			},
		},
	}
	component.Checkpoint.Job = &v1alpha1.ServiceCheckpointJobConfig{
		GMSClientContainers: []string{"gms-saver"},
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name:    "gms-saver",
					Image:   "custom-saver:latest",
					Command: []string{"/bin/custom-saver"},
				}},
			},
		},
	}

	t.Log("Create the GMS-backed checkpoint")
	ckpt, err := newTestDGDCheckpointsReconciler(reconciler).createCheckpointCR(ctx, dgd, "worker", betaComponent(t, component))
	if err != nil {
		t.Fatalf("createCheckpointCR() error = %v", err)
	}

	t.Log("Verify GMS clients, containers, claims, and templates are preserved")
	if ckpt.Spec.GPUMemoryService == nil || !ckpt.Spec.GPUMemoryService.Enabled {
		t.Fatalf("expected auto-created checkpoint to carry enabled GMS spec, got %#v", ckpt.Spec.GPUMemoryService)
	}
	if diff := cmp.Diff([]string{"gms-saver"}, ckpt.Spec.GPUMemoryService.ExtraClientContainers); diff != "" {
		t.Fatalf("checkpoint GMS extra clients mismatch (-want +got):\n%s", diff)
	}
	saver := findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.Containers, "gms-saver")
	if saver == nil {
		t.Fatalf("expected checkpoint job pod template to include saver")
	}
	if got := saver.Image; got != "custom-saver:latest" {
		t.Fatalf("checkpoint saver image = %q, want custom-saver:latest", got)
	}
	if got := saver.Command; len(got) != 1 || got[0] != "/bin/custom-saver" {
		t.Fatalf("checkpoint saver command = %#v, want [/bin/custom-saver]", got)
	}
	main := findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.Containers, commonconsts.MainContainerName)
	require.NotNil(t, main)
	assert.Contains(t, main.Resources.Claims, corev1.ResourceClaim{Name: dra.ClaimName})
	assert.Contains(t, saver.VolumeMounts, corev1.VolumeMount{Name: gms.SharedVolumeName, MountPath: gms.SharedMountPath})
	assert.NotNil(t, findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.InitContainers, gms.ServerContainerName))
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	checkpointID := checkpoint.DGDCheckpointID(
		dgd.Namespace,
		dgd.Name,
		string(dgd.UID),
		"worker",
		workerHash,
	)
	claimTemplateName := checkpointGMSResourceClaimTemplateName(checkpointID)
	assert.Contains(t, ckpt.Spec.Job.PodTemplateSpec.Spec.ResourceClaims, corev1.PodResourceClaim{
		Name:                      dra.ClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	})

	template := &resourcev1.ResourceClaimTemplate{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{Name: claimTemplateName, Namespace: "default"}, template))
	require.Len(t, template.Spec.Spec.Devices.Requests, 1)
	request := template.Spec.Spec.Devices.Requests[0]
	require.NotNil(t, request.Exactly)
	assert.Equal(t, int64(1), request.Exactly.Count)
	assert.Equal(t, dra.DefaultDeviceClassName, request.Exactly.DeviceClassName)
	controllerRef := metav1.GetControllerOf(template)
	require.NotNil(t, controllerRef)
	assert.Equal(t, "DynamoCheckpoint", controllerRef.Kind)
	assert.Equal(t, ckpt.Name, controllerRef.Name)

	t.Log("Render the same GMS checkpoint contract for SGLang V1")
	sglangDGD := dgd.DeepCopy()
	sglangDGD.Name = "test-sglang-dgd"
	sglangDGD.UID = types.UID("sglang-dgd-uid")
	sglangComponent := component.DeepCopy()
	sglangComponent.Checkpoint.Identity.BackendFramework = string(dynamo.BackendFrameworkSGLang)

	sglangCheckpoint, err := newTestDGDCheckpointsReconciler(reconciler).createCheckpointCR(
		ctx, sglangDGD, "worker", betaComponent(t, sglangComponent),
	)
	require.NoError(t, err)
	sglangMain := findContainer(
		sglangCheckpoint.Spec.Job.PodTemplateSpec.Spec.Containers,
		commonconsts.MainContainerName,
	)
	require.NotNil(t, sglangMain)
	assert.Contains(t, sglangMain.Env, corev1.EnvVar{
		Name:  "DYN_SGL_ENABLE_GMS_V1",
		Value: "true",
	})
	sglangServer := findContainer(
		sglangCheckpoint.Spec.Job.PodTemplateSpec.Spec.InitContainers,
		gms.ServerContainerName,
	)
	require.NotNil(t, sglangServer)
	assert.Equal(t, []string{"--use-v1"}, sglangServer.Args)
}

func TestDGDCheckpointsReconciler_SyncGMSResourceClaimTemplateUsesTemporaryDGDOwner(t *testing.T) {
	t.Log("Build a DGD and the GPU DeviceClass required by the checkpoint template")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
	}
	deviceClass := &resourcev1.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: dra.DefaultDeviceClassName},
	}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(dgd, deviceClass).
			Build(),
		Recorder: events.NewFakeRecorder(10),
	}
	checkpointReconciler := newTestDGDCheckpointsReconciler(reconciler)

	t.Log("Synchronize the ResourceClaimTemplate before its checkpoint exists")
	err := checkpointReconciler.syncCheckpointGMSResourceClaimTemplate(
		ctx,
		dgd,
		"checkpoint-template",
		1,
		dra.DefaultDeviceClassName,
	)
	require.NoError(t, err)

	t.Log("Verify the DGD temporarily controls the new template")
	template := &resourcev1.ResourceClaimTemplate{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKey{
		Name:      "checkpoint-template",
		Namespace: "default",
	}, template))
	assert.True(t, metav1.IsControlledBy(template, dgd))
}

func TestDGDCheckpointsReconciler_CreateAppliesDGDDefaults(t *testing.T) {
	t.Log("Build a checkpoint component with graph-level defaults")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config: &configv1alpha1.OperatorConfiguration{
			Discovery: configv1alpha1.DiscoveryConfiguration{
				Backend: configv1alpha1.DiscoveryBackendKubernetes,
			},
		},
		RuntimeConfig: &controller_common.RuntimeConfig{},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Env: []corev1.EnvVar{
				{Name: "HF_HOME", Value: "/models/huggingface"},
				{Name: "OVERRIDE_ME", Value: "graph"},
			},
		},
	}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		ComponentType: v1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name:  commonconsts.MainContainerName,
					Image: "checkpoint-writer:latest",
					Env:   []corev1.EnvVar{{Name: "OVERRIDE_ME", Value: "component"}},
				}},
			},
		},
		Experimental: &v1beta1.ExperimentalSpec{
			Checkpoint: &v1beta1.ComponentCheckpointConfig{
				Enabled: true,
				Mode:    v1beta1.CheckpointModeAuto,
				Identity: &v1beta1.DynamoCheckpointIdentity{
					Model:                identity.Model,
					BackendFramework:     identity.BackendFramework,
					TensorParallelSize:   1,
					PipelineParallelSize: 1,
					ExtraParameters:      map[string]string{},
				},
			},
		},
	}

	t.Log("Create the checkpoint job pod template")
	ckpt, err := newTestDGDCheckpointsReconciler(reconciler).createCheckpointCR(ctx, dgd, "worker", component)
	require.NoError(t, err)

	t.Log("Verify graph defaults reach the checkpoint job")
	main := findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.Containers, commonconsts.MainContainerName)
	require.NotNil(t, main)
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "HF_HOME", Value: "/models/huggingface"})
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "OVERRIDE_ME", Value: "component"})
	assert.Equal(t,
		discovery.GetK8sDiscoveryServiceAccountName("test-dgd"),
		ckpt.Spec.Job.PodTemplateSpec.Spec.ServiceAccountName,
	)
}

func TestDGDCheckpointsReconciler_CreateUsesTargetContainer(t *testing.T) {
	t.Log("Build a checkpoint component with an explicit target container")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config: &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{
			Gate: features.Gates{},
		},
	}
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default", UID: types.UID("dgd-uid")},
	})
	checkpointIdentity := v1beta1.DynamoCheckpointIdentity{
		Model:            identity.Model,
		BackendFramework: identity.BackendFramework,
	}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		ComponentType: v1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: commonconsts.MainContainerName, Image: "main:latest"},
					{Name: "snapshot-me", Image: "target:latest"},
					{Name: "serve-sidecar", Image: "serve-sidecar:latest"},
				},
			},
		},
		Experimental: &v1beta1.ExperimentalSpec{
			GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
				Mode: v1beta1.GMSModeIntraPod,
			},
			Checkpoint: &v1beta1.ComponentCheckpointConfig{
				Enabled:             true,
				Mode:                v1beta1.CheckpointModeAuto,
				TargetContainerName: "snapshot-me",
				Identity:            &checkpointIdentity,
				Job: &v1beta1.ComponentCheckpointJobConfig{
					GMSClientContainers: []string{"gms-saver"},
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  "gms-saver",
								Image: "saver:latest",
							}},
						},
					},
				},
			},
		},
	}

	t.Log("Create the target-container checkpoint")
	ckpt, err := newTestDGDCheckpointsReconciler(reconciler).createCheckpointCR(ctx, dgd, "worker", component)
	require.NoError(t, err)

	t.Log("Verify target and GMS containers are retained")
	assert.Equal(t, "snapshot-me", ckpt.Spec.Job.TargetContainerName)
	assert.NotNil(t, findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.Containers, "snapshot-me"))
	assert.NotNil(t, findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.Containers, "gms-saver"))
	assert.Equal(t, []string{"gms-saver"}, ckpt.Spec.GPUMemoryService.ExtraClientContainers)
	assert.Nil(t, findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.Containers, commonconsts.MainContainerName))
	assert.Nil(t, findContainer(ckpt.Spec.Job.PodTemplateSpec.Spec.Containers, "serve-sidecar"))
}

func TestDGDCheckpointsReconciler_AutoUsesTargetContainerWithoutIdentity(t *testing.T) {
	t.Log("Build an auto-checkpoint component without an explicit identity")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: commonconsts.MainContainerName, Image: "main:latest"},
							{Name: "snapshot-me", Image: "target:latest"},
						},
					},
				},
				Experimental: &v1beta1.ExperimentalSpec{
					Checkpoint: &v1beta1.ComponentCheckpointConfig{
						Enabled:             true,
						Mode:                v1beta1.CheckpointModeAuto,
						TargetContainerName: "snapshot-me",
					},
				},
			}},
		},
	}

	t.Log("Reconcile the auto checkpoint")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	require.NoError(t, err)

	t.Log("Verify target-container restore state and managed checkpoint identity")
	info := checkpointInfos["worker"]
	require.NotNil(t, info)
	assert.Equal(t, []string{"snapshot-me"}, info.RestoreTargetContainers)
	require.NotEmpty(t, checkpointStatuses["worker"].CheckpointName)
	require.NotEmpty(t, checkpointStatuses["worker"].CheckpointID)

	ckpt := &v1alpha1.DynamoCheckpoint{}
	require.NoError(t, reconciler.Get(ctx, types.NamespacedName{Name: checkpointStatuses["worker"].CheckpointName, Namespace: "default"}, ckpt))
	assert.Equal(t, "snapshot-me", ckpt.Spec.Job.TargetContainerName)
	assert.Equal(t, string(dynamo.BackendFrameworkVLLM), ckpt.Spec.Identity.BackendFramework)
	assert.Equal(t, checkpointStatuses["worker"].CheckpointID, ckpt.Spec.Identity.ExtraParameters["checkpointID"])
}

func TestDGDCheckpointsReconciler_RejectsDisabledFeatureBeforeCreatingResources(t *testing.T) {
	t.Log("Build a checkpoint-enabled DGD while the checkpoint feature is disabled")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config: &configv1alpha1.OperatorConfiguration{
			Checkpoint: configv1alpha1.CheckpointConfiguration{
				Storage: configv1alpha1.CheckpointStorageConfiguration{
					Type: configv1alpha1.CheckpointStorageTypePVC,
					PVC: configv1alpha1.CheckpointPVCConfig{
						PVCName: "checkpoint-storage",
						Create:  true,
						Size:    "1Gi",
					},
				},
			},
		},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				Experimental: &v1beta1.ExperimentalSpec{
					Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true},
				},
			}},
		},
	}

	t.Log("Reconcile checkpoint resources")
	_, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.ErrorContains(t, err, "checkpoint functionality is disabled")

	t.Log("Verify rejection happens before checkpoint or storage resources are created")
	checkpoints := &v1alpha1.DynamoCheckpointList{}
	require.NoError(t, reconciler.List(ctx, checkpoints, client.InNamespace("default")))
	assert.Empty(t, checkpoints.Items)
	pvcs := &corev1.PersistentVolumeClaimList{}
	require.NoError(t, reconciler.List(ctx, pvcs, client.InNamespace("default")))
	assert.Empty(t, pvcs.Items)
}

func TestDGDCheckpointsReconciler_PropagatesManagedCheckpointResolveError(t *testing.T) {
	t.Log("Build an auto-checkpoint DGD and inject a checkpoint read failure")
	ctx := context.Background()
	resolveErr := errors.New("checkpoint read failed")
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	kubeClient := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if _, ok := obj.(*v1alpha1.DynamoCheckpoint); ok {
					return resolveErr
				}
				return c.Get(ctx, key, obj, opts...)
			},
		}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeWorker,
				Experimental: &v1beta1.ExperimentalSpec{
					Checkpoint: &v1beta1.ComponentCheckpointConfig{
						Enabled: true,
						Mode:    v1beta1.CheckpointModeAuto,
					},
				},
			}},
		},
	}

	t.Log("Reconcile the managed checkpoint")
	_, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)

	t.Log("Verify the read failure is returned instead of dereferencing a nil result")
	require.ErrorIs(t, err, resolveErr)
	require.ErrorContains(t, err, "failed to resolve checkpoint for component worker")
}

func TestDGDCheckpointsReconciler_AutoPreservesPodTemplateMetadata(t *testing.T) {
	t.Log("Build an auto-checkpoint component with pod-template metadata")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							"workload-label": "keep-me",
						},
						Annotations: map[string]string{
							commonconsts.KubeAnnotationIstioSidecarInject: "false",
							"policy.example.com/keep":                     "yes",
						},
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: commonconsts.MainContainerName, Image: "main:latest"},
						},
					},
				},
				Experimental: &v1beta1.ExperimentalSpec{
					Checkpoint: &v1beta1.ComponentCheckpointConfig{
						Enabled: true,
						Mode:    v1beta1.CheckpointModeAuto,
					},
				},
			}},
		},
	}

	t.Log("Reconcile the auto checkpoint")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	require.NoError(t, err)
	require.NotEmpty(t, checkpointStatuses["worker"].CheckpointName)

	t.Log("Verify workload metadata and managed labels on the checkpoint job")
	ckpt := &v1alpha1.DynamoCheckpoint{}
	require.NoError(t, reconciler.Get(ctx, types.NamespacedName{Name: checkpointStatuses["worker"].CheckpointName, Namespace: "default"}, ckpt))

	jobMeta := ckpt.Spec.Job.PodTemplateSpec.ObjectMeta
	assert.Equal(t, "keep-me", jobMeta.Labels["workload-label"])
	assert.Equal(t, "false", jobMeta.Annotations[commonconsts.KubeAnnotationIstioSidecarInject])
	assert.Equal(t, "yes", jobMeta.Annotations["policy.example.com/keep"])
	assert.Equal(t, "worker", jobMeta.Labels[commonconsts.KubeLabelDynamoComponent])
}

func TestDGDCheckpointsReconciler_SyncsExistingAutoLifecycle(t *testing.T) {
	t.Log("Build a DGD-managed checkpoint with lifecycle fields requiring synchronization")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	reconciler := &DynamoGraphDeploymentReconciler{
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  commonconsts.MainContainerName,
							Image: "main:latest",
						}},
					},
				},
				Experimental: &v1beta1.ExperimentalSpec{
					Checkpoint: &v1beta1.ComponentCheckpointConfig{
						Enabled:        true,
						Mode:           v1beta1.CheckpointModeAuto,
						DeletionPolicy: v1beta1.CheckpointDeletionPolicyRetain,
					},
				},
			}},
		},
	}
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	checkpointID := checkpoint.DGDCheckpointID(
		dgd.Namespace,
		dgd.Name,
		string(dgd.UID),
		"worker",
		workerHash,
	)
	existing := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("checkpoint-%s", checkpointID),
			Namespace: "default",
			Labels: map[string]string{
				snapshotprotocol.CheckpointIDLabel:              checkpointID,
				commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				commonconsts.KubeLabelDynamoComponent:           "worker",
				commonconsts.KubeLabelDynamoWorkerHash:          workerHash,
			},
			Annotations: map[string]string{
				commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
				commonconsts.CheckpointDeletionPolicyAnnotation: string(v1alpha1.CheckpointDeletionPolicyDelete),
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: v1beta1.GroupVersion.String(),
				Kind:       "DynamoGraphDeployment",
				Name:       dgd.Name,
				UID:        dgd.UID,
				Controller: ptr.To(true),
			}},
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: v1alpha1.DynamoCheckpointIdentity{
				Model:            "default/test-dgd",
				BackendFramework: string(dynamo.BackendFrameworkVLLM),
			},
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  commonconsts.MainContainerName,
							Image: "existing:latest",
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			CheckpointID: checkpointID,
			Phase:        v1alpha1.DynamoCheckpointPhaseCreating,
		},
	}
	reconciler.Client = fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(existing).
		WithStatusSubresource(existing).
		Build()

	t.Log("Reconcile the existing managed checkpoint")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	require.NoError(t, err)
	assert.Equal(t, existing.Name, checkpointStatuses["worker"].CheckpointName)
	assert.Equal(t, checkpointID, checkpointStatuses["worker"].CheckpointID)
	require.NotNil(t, checkpointInfos["worker"])
	assert.True(t, checkpointInfos["worker"].Exists)

	t.Log("Verify lifecycle annotations, ownership, finalizer, and labels were synchronized")
	updated := &v1alpha1.DynamoCheckpoint{}
	require.NoError(t, reconciler.Get(ctx, types.NamespacedName{Name: existing.Name, Namespace: "default"}, updated))
	assert.Equal(t, string(v1alpha1.CheckpointDeletionPolicyRetain),
		updated.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation])
	assert.Empty(t, updated.OwnerReferences)
	assert.True(t, controller_common.ContainsFinalizer(updated))
	assert.Equal(t, "test-dgd", updated.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName])
	assert.Equal(t, "worker", updated.Labels[commonconsts.KubeLabelDynamoComponent])
}

func TestDGDCheckpointsReconciler_CheckpointRefSkipsAutoCreateWhileReferencedCRIsNotReady(t *testing.T) {
	t.Log("Build a DGD referencing a checkpoint that is not ready")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "keep-existing:latest",
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseCreating,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	t.Log("Reconcile the not-ready checkpoint reference")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	t.Log("Verify the reference remains pending without creating an automatic checkpoint")
	info, ok := checkpointInfos["worker"]
	if !ok {
		t.Fatalf("expected checkpoint info for worker service")
	}
	if info.Ready {
		t.Fatalf("expected referenced checkpoint to remain not ready")
	}
	if !info.Exists {
		t.Fatalf("expected referenced checkpoint to exist")
	}
	if info.Hash != hash {
		t.Fatalf("checkpoint hash = %s, want %s", info.Hash, hash)
	}
	if checkpointStatuses["worker"].CheckpointName != friendlyCheckpointName {
		t.Fatalf("checkpoint status name = %s, want friendly-checkpoint", checkpointStatuses["worker"].CheckpointName)
	}

	checkpoints := &v1alpha1.DynamoCheckpointList{}
	if err := reconciler.List(ctx, checkpoints, client.InNamespace("default")); err != nil {
		t.Fatalf("failed to list checkpoints: %v", err)
	}
	if len(checkpoints.Items) != 1 {
		t.Fatalf("expected only the referenced checkpoint to exist, found %d", len(checkpoints.Items))
	}
	if checkpoints.Items[0].Name != friendlyCheckpointName {
		t.Fatalf("unexpected checkpoint %s", checkpoints.Items[0].Name)
	}
}

func TestDGDCheckpointsReconciler_CheckpointRefUsesReadyReferencedCR(t *testing.T) {
	t.Log("Build a DGD referencing a ready checkpoint")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	t.Log("Reconcile the ready checkpoint reference")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	t.Log("Verify ready checkpoint information and status are projected")
	info, ok := checkpointInfos["worker"]
	if !ok {
		t.Fatalf("expected checkpoint info for worker service")
	}
	if !info.Ready {
		t.Fatalf("expected referenced checkpoint to be ready")
	}
	if !info.Exists {
		t.Fatalf("expected referenced checkpoint to exist")
	}
	if info.Hash != hash {
		t.Fatalf("checkpoint hash = %s, want %s", info.Hash, hash)
	}
	if checkpointStatuses["worker"].CheckpointName != friendlyCheckpointName {
		t.Fatalf("checkpoint status name = %s, want friendly-checkpoint", checkpointStatuses["worker"].CheckpointName)
	}
	if !checkpointStatuses["worker"].Ready {
		t.Fatalf("expected checkpoint status to be ready")
	}
}

func TestDGDCheckpointsReconciler_OverlaysServiceGMSLoader(t *testing.T) {
	t.Log("Build a checkpoint reference with component-level GMS clients")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity:         identity,
			GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{Enabled: true},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{
			Gate: features.Gates{Checkpoint: true},
		},
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{
						Enabled:               true,
						Mode:                  v1alpha1.GMSModeIntraPod,
						ExtraClientContainers: []string{"gms-loader"},
					},
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeManual,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	t.Log("Reconcile the referenced GMS checkpoint")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointInfos := checkpointResult.Infos
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	t.Log("Verify component-level GMS clients overlay the resolved checkpoint")
	info := checkpointInfos["worker"]
	if info == nil || info.GPUMemoryService == nil {
		t.Fatalf("expected resolved GMS checkpoint info, got %#v", info)
	}
	if diff := cmp.Diff([]string{"gms-loader"}, info.GPUMemoryService.ExtraClientContainers); diff != "" {
		t.Fatalf("restore GMS extra clients mismatch (-want +got):\n%s", diff)
	}
}

func TestDGDCheckpointsReconciler_RejectsServiceGMSWithNonGMSCheckpoint(t *testing.T) {
	t.Log("Build a GMS component referencing a non-GMS checkpoint")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	require.NoError(t, err)

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{
						Enabled:               true,
						Mode:                  v1alpha1.GMSModeIntraPod,
						ExtraClientContainers: []string{"gms-loader"},
					},
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeManual,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	t.Log("Reconcile the incompatible checkpoint reference")
	_, err = newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)

	t.Log("Verify the incompatibility is returned with checkpoint context")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "gpuMemoryService restore requires resolved checkpoint")
	assert.Contains(t, err.Error(), friendlyCheckpointName)
}

func TestDGDCheckpointsReconciler_CreatesCheckpointStoragePVC(t *testing.T) {
	t.Log("Build checkpoint storage configuration that creates a PVC")
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	ctx := context.Background()
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config: &configv1alpha1.OperatorConfiguration{
			Checkpoint: configv1alpha1.CheckpointConfiguration{
				Storage: configv1alpha1.CheckpointStorageConfiguration{
					Type: configv1alpha1.CheckpointStorageTypePVC,
					PVC: configv1alpha1.CheckpointPVCConfig{
						PVCName:          "snapshot-pvc",
						BasePath:         "/checkpoints",
						Create:           true,
						Size:             "2Gi",
						StorageClassName: "efs-sc",
						AccessMode:       string(corev1.ReadWriteMany),
					},
				},
			},
		},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	t.Log("Reconcile checkpoint resources")
	if _, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd); err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	t.Log("Verify the checkpoint storage PVC settings")
	pvc := &corev1.PersistentVolumeClaim{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "snapshot-pvc", Namespace: "default"}, pvc); err != nil {
		t.Fatalf("expected checkpoint storage PVC to be created: %v", err)
	}
	storageRequest := pvc.Spec.Resources.Requests[corev1.ResourceStorage]
	if storageRequest.String() != "2Gi" {
		t.Fatalf("PVC storage request = %s, want 2Gi", storageRequest.String())
	}
	if pvc.Spec.StorageClassName == nil || *pvc.Spec.StorageClassName != "efs-sc" {
		t.Fatalf("PVC storageClassName = %v, want efs-sc", pvc.Spec.StorageClassName)
	}
	if len(pvc.Spec.AccessModes) != 1 || pvc.Spec.AccessModes[0] != corev1.ReadWriteMany {
		t.Fatalf("PVC accessModes = %v, want [ReadWriteMany]", pvc.Spec.AccessModes)
	}
}

func TestDGDCheckpointsReconciler_AutoModeWaitsForExistingCreatingCheckpoint(t *testing.T) {
	t.Log("Build an auto-checkpoint DGD with an existing checkpoint still creating")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	existing := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "existing-worker-checkpoint",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "keep-existing:latest",
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseCreating,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(existing).
			WithStatusSubresource(existing).
			Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}

	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled: true,
						Mode:    v1alpha1.CheckpointModeAuto,
						Identity: &v1alpha1.DynamoCheckpointIdentity{
							Model:            identity.Model,
							BackendFramework: identity.BackendFramework,
						},
					},
					ExtraPodSpec: &v1alpha1.ExtraPodSpec{
						MainContainer: &corev1.Container{
							Name:  "main",
							Image: "new-writer:latest",
						},
					},
				},
			},
		},
	})

	t.Log("Reconcile the automatic checkpoint")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	t.Log("Verify the DGD-owned checkpoint remains pending without reusing the legacy identity")
	info, ok := checkpointInfos["worker"]
	if !ok {
		t.Fatalf("expected checkpoint info for worker service")
	}
	if info.Ready {
		t.Fatalf("expected existing checkpoint to remain not ready")
	}
	if !info.Exists {
		t.Fatalf("expected auto checkpoint to exist")
	}
	if info.Hash == hash {
		t.Fatalf("auto checkpoint unexpectedly reused legacy identity hash %s", hash)
	}
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	if err != nil {
		t.Fatalf("checkpointWorkerHashForComponent() error = %v", err)
	}
	expectedName := fmt.Sprintf("checkpoint-%s", checkpoint.DGDCheckpointID(
		dgd.Namespace,
		dgd.Name,
		string(dgd.UID),
		"worker",
		workerHash,
	))
	if checkpointStatuses["worker"].CheckpointName != expectedName {
		t.Fatalf("checkpoint status name = %s, want %s", checkpointStatuses["worker"].CheckpointName, expectedName)
	}

	updated := &v1alpha1.DynamoCheckpoint{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "existing-worker-checkpoint", Namespace: "default"}, updated); err != nil {
		t.Fatalf("Failed to get checkpoint: %v", err)
	}
	if len(updated.Spec.Job.PodTemplateSpec.Spec.Containers) != 1 {
		t.Fatalf("expected one job container, got %d", len(updated.Spec.Job.PodTemplateSpec.Spec.Containers))
	}
	if updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image != "keep-existing:latest" {
		t.Fatalf("existing job image was mutated to %s", updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image)
	}
	created := &v1alpha1.DynamoCheckpoint{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: expectedName, Namespace: "default"}, created); err != nil {
		t.Fatalf("failed to get auto checkpoint %s: %v", expectedName, err)
	}
}

func TestCheckpointWorkerHashForComponentUsesActiveGeneration(t *testing.T) {
	t.Log("Build a rolling-update DGD with an active worker generation")
	rollout := &dgdWorkerRolloutReconciler{}
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Envs:          []corev1.EnvVar{{Name: "GENERATION", Value: "next"}},
				},
			},
		},
	})
	rollout.setCurrentWorkerHashes(dgd, workerGenerationHashes{v1: "oldhash"})

	t.Log("Compute the desired and checkpoint worker hashes")
	desired, err := desiredWorkerHashes(dgd)
	if err != nil {
		t.Fatalf("desiredWorkerHashes() error = %v", err)
	}

	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	if err != nil {
		t.Fatalf("checkpointWorkerHashForComponent() error = %v", err)
	}
	want := activeWorkerHashForDCDGeneration(dgd, desired)

	t.Log("Verify the checkpoint follows the active generation")
	if workerHash != want {
		t.Fatalf("checkpoint worker hash = %s, want active generated hash %s", workerHash, want)
	}
	if workerHash == "oldhash" {
		t.Fatalf("checkpoint worker hash used previous current-worker-hash annotation")
	}
}

func TestDGDCheckpointsReconciler_DeleteAutoCheckpointsForDGD(t *testing.T) {
	t.Log("Build automatic, retained, manual, and foreign checkpoint fixtures")
	ctx := context.Background()
	s := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
	})

	auto := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "auto",
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
			Annotations: map[string]string{
				commonconsts.CheckpointAutoAnnotation: commonconsts.KubeLabelValueTrue,
			},
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: v1alpha1.DynamoCheckpointIdentity{Model: "m", BackendFramework: "vllm"},
		},
	}
	manual := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "manual",
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: v1alpha1.DynamoCheckpointIdentity{Model: "m", BackendFramework: "vllm"},
		},
	}
	retained := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "retained",
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: v1beta1.GroupVersion.String(),
				Kind:       "DynamoGraphDeployment",
				Name:       "test-dgd",
				UID:        dgd.UID,
			}},
			Annotations: map[string]string{
				commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
				commonconsts.CheckpointDeletionPolicyAnnotation: string(v1alpha1.CheckpointDeletionPolicyRetain),
			},
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: v1alpha1.DynamoCheckpointIdentity{Model: "m", BackendFramework: "vllm"},
		},
	}
	otherDGD := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "other-dgd",
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: "other-dgd",
			},
			Annotations: map[string]string{
				commonconsts.CheckpointAutoAnnotation: commonconsts.KubeLabelValueTrue,
			},
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: v1alpha1.DynamoCheckpointIdentity{Model: "m", BackendFramework: "vllm"},
		},
	}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(s).
			WithObjects(auto, manual, retained, otherDGD).
			Build(),
	}

	t.Log("Delete automatic checkpoints owned by the DGD")
	if err := newTestDGDCheckpointsReconciler(reconciler).deleteAutoCheckpointsForDGD(ctx, dgd); err != nil {
		t.Fatalf("deleteAutoCheckpointsForDGD() error = %v", err)
	}

	t.Log("Verify only eligible automatic checkpoints were deleted")
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "auto", Namespace: "default"}, &v1alpha1.DynamoCheckpoint{}); !apierrors.IsNotFound(err) {
		t.Fatalf("auto checkpoint get err = %v, want not found", err)
	}
	for _, name := range []string{"manual", "retained", "other-dgd"} {
		if err := reconciler.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, &v1alpha1.DynamoCheckpoint{}); err != nil {
			t.Fatalf("checkpoint %s should remain, get error = %v", name, err)
		}
	}
	retainedAfter := &v1alpha1.DynamoCheckpoint{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "retained", Namespace: "default"}, retainedAfter); err != nil {
		t.Fatalf("retained checkpoint should remain, get error = %v", err)
	}
	if len(retainedAfter.OwnerReferences) != 0 {
		t.Fatalf("retained checkpoint should be detached from DGD owner references, got %#v", retainedAfter.OwnerReferences)
	}
	if _, ok := retainedAfter.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName]; ok {
		t.Fatalf("retained checkpoint should not keep DGD label after finalizer detach")
	}
}
