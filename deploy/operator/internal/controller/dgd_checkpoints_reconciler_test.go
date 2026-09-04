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
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
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

const friendlyCheckpointName = "friendly-checkpoint"

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

func dgdTestPodSnapshot(name string, workerHash string, ready bool) *snapshotv1alpha1.PodSnapshot {
	snapshot := &snapshotv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			UID:       types.UID("snapshot-uid"),
			Annotations: map[string]string{
				commonconsts.SnapshotCompatibilityVersionAnnotation: commonconsts.SnapshotCompatibilityVersion,
				commonconsts.SnapshotWorkerHashAnnotation:           workerHash,
				commonconsts.SnapshotGMSModeAnnotation:              commonconsts.SnapshotGMSModeDisabled,
			},
		},
		Spec: snapshotv1alpha1.PodSnapshotSpec{
			Source: snapshotv1alpha1.PodSnapshotSource{
				PodRef: snapshotv1alpha1.PodReference{
					Name:       "capture-worker",
					Containers: []string{"main"},
				},
			},
		},
	}
	if ready {
		snapshot.Status = snapshotv1alpha1.PodSnapshotStatus{
			BoundPodSnapshotContentName: ptr.To("content-a"),
			Conditions: []metav1.Condition{{
				Type:   snapshotv1alpha1.PodSnapshotConditionReady,
				Status: metav1.ConditionTrue,
			}},
		}
	}
	return snapshot
}

func reconcileAutomaticSnapshotJobForTest(
	t *testing.T,
	reconciler *DynamoGraphDeploymentReconciler,
	dgd *v1beta1.DynamoGraphDeployment,
	componentName string,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
) *snapshotv1alpha1.SnapshotJob {
	t.Helper()
	workerHash, err := checkpointWorkerHashForComponent(dgd, componentName)
	require.NoError(t, err)
	_, err = newTestDGDCheckpointsReconciler(reconciler).reconcileAutomaticSnapshotJob(
		context.Background(),
		dgd,
		componentName,
		component,
		workerHash,
		v1alpha1.CheckpointStartupPolicyImmediate,
	)
	require.NoError(t, err)

	checkpointID := checkpoint.DGDCheckpointID(
		dgd.Namespace,
		dgd.Name,
		string(dgd.UID),
		componentName,
		workerHash,
	)
	job := &snapshotv1alpha1.SnapshotJob{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{
		Namespace: dgd.Namespace,
		Name:      "checkpoint-" + checkpointID,
	}, job))
	return job
}

func TestDGDCheckpointsReconciler_SnapshotJobPreservesGMSSaverClient(t *testing.T) {
	t.Log("Build a GMS checkpoint component with a saver client")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
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
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
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

	t.Log("Create the GMS-backed SnapshotJob")
	job := reconcileAutomaticSnapshotJobForTest(t, reconciler, dgd, "worker", betaComponent(t, component))

	t.Log("Verify GMS clients, containers, claims, and templates are preserved")
	saver := findContainer(job.Spec.PodTemplate.Spec.Containers, "gms-saver")
	if saver == nil {
		t.Fatalf("expected checkpoint job pod template to include saver")
	}
	if got := saver.Image; got != "custom-saver:latest" {
		t.Fatalf("checkpoint saver image = %q, want custom-saver:latest", got)
	}
	if got := saver.Command; len(got) != 1 || got[0] != "/bin/custom-saver" {
		t.Fatalf("checkpoint saver command = %#v, want [/bin/custom-saver]", got)
	}
	main := findContainer(job.Spec.PodTemplate.Spec.Containers, commonconsts.MainContainerName)
	require.NotNil(t, main)
	assert.Contains(t, main.Resources.Claims, corev1.ResourceClaim{Name: dra.ClaimName})
	assert.Contains(t, saver.VolumeMounts, corev1.VolumeMount{Name: gms.SharedVolumeName, MountPath: gms.SharedMountPath})
	server := findContainer(job.Spec.PodTemplate.Spec.InitContainers, gms.ServerContainerName)
	require.NotNil(t, server)
	assert.Empty(t, server.Args)
	assert.Contains(t, server.Env, corev1.EnvVar{Name: gms.EnvUseV1, Value: "true"})
	assert.Contains(t, main.Env, corev1.EnvVar{Name: gms.EnvUseV1, Value: "true"})
	assert.Equal(t, string(v1alpha1.GMSModeIntraPod),
		job.Spec.PodSnapshotTemplate.Metadata.Annotations[commonconsts.SnapshotGMSModeAnnotation])
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
	assert.Contains(t, job.Spec.PodTemplate.Spec.ResourceClaims, corev1.PodResourceClaim{
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
	assert.Equal(t, "SnapshotJob", controllerRef.Kind)
	assert.Equal(t, job.Name, controllerRef.Name)
}

func TestPrepareCheckpointGMSPodTemplateRejectsMissingClient(t *testing.T) {
	t.Log("Build a capture Pod template that omits a requested GMS client")
	podTemplate := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
		Containers: []corev1.Container{{Name: commonconsts.MainContainerName}},
	}}
	gmsSpec := &v1alpha1.GPUMemoryServiceSpec{
		Enabled:               true,
		Mode:                  v1alpha1.GMSModeIntraPod,
		ExtraClientContainers: []string{"missing-saver"},
	}

	t.Log("Render the GMS capture wiring")
	err := prepareCheckpointGMSPodTemplate(
		podTemplate,
		commonconsts.MainContainerName,
		"checkpoint-id",
		gmsSpec,
	)

	t.Log("Verify the typo fails before partially mutating the Pod template")
	require.ErrorContains(t, err, `gpuMemoryService client container "missing-saver"`)
	require.ErrorContains(t, err, `pod spec has no container named "missing-saver"`)
	assert.Empty(t, podTemplate.Spec.InitContainers)
	assert.Empty(t, podTemplate.Spec.ResourceClaims)
	assert.Empty(t, podTemplate.Spec.Volumes)
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

func TestDGDCheckpointsReconciler_SnapshotJobAppliesDGDDefaults(t *testing.T) {
	t.Log("Build a checkpoint component with graph-level defaults")
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
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
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
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
			},
		},
	}

	t.Log("Create the SnapshotJob pod template")
	job := reconcileAutomaticSnapshotJobForTest(t, reconciler, dgd, "worker", component)

	t.Log("Verify graph defaults reach the SnapshotJob")
	main := findContainer(job.Spec.PodTemplate.Spec.Containers, commonconsts.MainContainerName)
	require.NotNil(t, main)
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "HF_HOME", Value: "/models/huggingface"})
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "OVERRIDE_ME", Value: "component"})
	assert.Equal(t,
		discovery.GetK8sDiscoveryServiceAccountName("test-dgd"),
		job.Spec.PodTemplate.Spec.ServiceAccountName,
	)
}

func TestDGDCheckpointsReconciler_SnapshotJobUsesTargetContainer(t *testing.T) {
	t.Log("Build a checkpoint component with an explicit target container")
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config: &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{
			Gate: features.Gates{},
		},
	}
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default", UID: types.UID("dgd-uid")},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
		},
	})
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

	t.Log("Create the target-container SnapshotJob")
	job := reconcileAutomaticSnapshotJobForTest(t, reconciler, dgd, "worker", component)

	t.Log("Verify target and GMS containers are retained")
	assert.Equal(t, []string{"snapshot-me"}, job.Spec.PodSnapshotTemplate.TargetContainers)
	assert.NotNil(t, findContainer(job.Spec.PodTemplate.Spec.Containers, "snapshot-me"))
	assert.NotNil(t, findContainer(job.Spec.PodTemplate.Spec.Containers, "gms-saver"))
	assert.Nil(t, findContainer(job.Spec.PodTemplate.Spec.Containers, commonconsts.MainContainerName))
	assert.Nil(t, findContainer(job.Spec.PodTemplate.Spec.Containers, "serve-sidecar"))
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
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}

	t.Log("Reconcile the auto checkpoint")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	require.NoError(t, err)

	t.Log("Verify the pending native capture and generated SnapshotJob")
	info := checkpointInfos["worker"]
	require.NotNil(t, info)
	assert.Equal(t, []string{"snapshot-me"}, info.RestoreTargetContainers)
	assert.Empty(t, checkpointStatuses["worker"].CheckpointName)
	assert.Empty(t, checkpointStatuses["worker"].CheckpointID)

	jobs := &snapshotv1alpha1.SnapshotJobList{}
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	require.Len(t, jobs.Items, 1)
	assert.Equal(t, []string{"snapshot-me"}, jobs.Items[0].Spec.PodSnapshotTemplate.TargetContainers)
	assert.NotNil(t, findContainer(jobs.Items[0].Spec.PodTemplate.Spec.Containers, "snapshot-me"))
	firstJobName := jobs.Items[0].Name

	t.Log("Change a worker-hash input and verify automatic capture rotates")
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[1].Image = "target:v2"
	_, err = newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	require.Len(t, jobs.Items, 2)
	rotatedJobName := jobs.Items[0].Name
	if rotatedJobName == firstJobName {
		rotatedJobName = jobs.Items[1].Name
	}
	assert.NotEqual(t, firstJobName, rotatedJobName)
}

func TestDGDCheckpointsReconciler_AutomaticCaptureWaitsForActiveWorkerHash(t *testing.T) {
	t.Log("Build a first-generation worker before its active hash is initialized")
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
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "worker:latest"}},
				}},
				Experimental: &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{
					Enabled: true,
					Mode:    v1beta1.CheckpointModeAuto,
				}},
			}},
		},
	}

	t.Log("Reconcile while the worker generation has no durable identity")
	result, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)

	t.Log("Verify automatic capture remains pending without creating a generation-less SnapshotJob")
	info := result.Infos["worker"]
	require.NotNil(t, info)
	assert.True(t, info.AutomaticCapture)
	assert.False(t, info.Ready)
	jobs := &snapshotv1alpha1.SnapshotJobList{}
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	assert.Empty(t, jobs.Items)

	t.Log("Record the active worker hash and reconcile capture again")
	workerHash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: workerHash,
	}
	_, err = newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)

	t.Log("Verify exactly one generation-bound SnapshotJob is created")
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	require.Len(t, jobs.Items, 1)
	assert.Equal(t, workerHash, jobs.Items[0].Labels[commonconsts.KubeLabelDynamoWorkerHash])
}

func TestDGDCheckpointsReconciler_ExplicitRestoreWaitsForActiveWorkerHash(t *testing.T) {
	t.Log("Build a first-generation worker with an explicit reference before its active hash is initialized")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}
	ref := friendlyCheckpointName
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
				Experimental: &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{
					Enabled:       true,
					CheckpointRef: &ref,
				}},
			}},
		},
	}

	t.Log("Reconcile while the worker generation has no durable identity")
	result, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)

	t.Log("Verify explicit restore remains fail-closed instead of failing reconciliation or cold-starting")
	info := result.Infos["worker"]
	require.NotNil(t, info)
	assert.Equal(t, ref, info.CheckpointName)
	assert.False(t, info.Exists)
	assert.False(t, info.Ready)
	assert.Equal(t, v1alpha1.CheckpointStartupPolicyWaitForCheckpoint, info.StartupPolicy)

	t.Log("Record the active hash while the referenced PodSnapshot is still missing")
	workerHash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{commonconsts.AnnotationCurrentWorkerHashV2: workerHash}

	t.Log("Verify a missing explicit reference fails reconciliation without synthesizing pending state")
	_, err = newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.ErrorContains(t, err, "get referenced PodSnapshot")

	t.Log("Create the compatible referenced PodSnapshot")
	referenced := dgdTestPodSnapshot(ref, workerHash, true)
	require.NoError(t, reconciler.Create(ctx, referenced))

	t.Log("Verify the same explicit reference resolves once compatibility identity is available")
	result, err = newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)
	info = result.Infos["worker"]
	require.NotNil(t, info)
	assert.True(t, info.Exists)
	assert.True(t, info.Ready)
	require.NotNil(t, info.NativeSnapshot)
	assert.Equal(t, referenced.UID, info.NativeSnapshot.UID)
}

func TestDGDCheckpointsReconciler_RejectsDisabledFeatureBeforeCreatingResources(t *testing.T) {
	t.Log("Build a checkpoint-enabled DGD while the checkpoint feature is disabled")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
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

	t.Log("Verify rejection happens before capture or storage resources are created")
	jobs := &snapshotv1alpha1.SnapshotJobList{}
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	assert.Empty(t, jobs.Items)
	pvcs := &corev1.PersistentVolumeClaimList{}
	require.NoError(t, reconciler.List(ctx, pvcs, client.InNamespace("default")))
	assert.Empty(t, pvcs.Items)
}

func TestDGDCheckpointsReconciler_PropagatesSnapshotJobReadError(t *testing.T) {
	t.Log("Build an auto-checkpoint DGD and inject a SnapshotJob read failure")
	ctx := context.Background()
	resolveErr := errors.New("checkpoint read failed")
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	kubeClient := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if _, ok := obj.(*snapshotv1alpha1.SnapshotJob); ok {
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
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
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
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}

	t.Log("Reconcile the managed capture")
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
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}

	t.Log("Reconcile the auto checkpoint")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)
	assert.Empty(t, checkpointResult.Statuses["worker"].CheckpointName)

	t.Log("Verify workload metadata and managed labels on the SnapshotJob")
	jobs := &snapshotv1alpha1.SnapshotJobList{}
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	require.Len(t, jobs.Items, 1)

	jobMeta := jobs.Items[0].Spec.PodTemplate.ObjectMeta
	assert.Equal(t, "keep-me", jobMeta.Labels["workload-label"])
	assert.Equal(t, "false", jobMeta.Annotations[commonconsts.KubeAnnotationIstioSidecarInject])
	assert.Equal(t, "yes", jobMeta.Annotations["policy.example.com/keep"])
	assert.Equal(t, "worker", jobMeta.Labels[commonconsts.KubeLabelDynamoComponent])
}

func TestDGDCheckpointsReconciler_SyncsExistingAutoLifecycle(t *testing.T) {
	t.Log("Build an existing SnapshotJob with lifecycle fields requiring synchronization")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: v1beta1.GroupVersion.String(),
			Kind:       "DynamoGraphDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
	}
	desired := buildAutomaticSnapshotJob(
		dgd,
		"worker",
		"capture-id",
		"worker-hash",
		corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name:  commonconsts.MainContainerName,
			Image: "main:latest",
		}}}},
		commonconsts.MainContainerName,
		v1alpha1.CheckpointDeletionPolicyRetain,
		commonconsts.SnapshotGMSModeDisabled,
	)
	existing := desired.DeepCopy()
	existing.Spec.PodTemplate.Spec.Containers[0].Image = "captured:latest"
	existing.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation] = string(v1alpha1.CheckpointDeletionPolicyDelete)
	existing.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(
		dgd,
		v1beta1.GroupVersion.WithKind("DynamoGraphDeployment"),
	)}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(existing).
			Build(),
	}

	t.Log("Synchronize the retained SnapshotJob lifecycle metadata")
	updated, err := newTestDGDCheckpointsReconciler(reconciler).syncAutomaticSnapshotJob(
		ctx,
		dgd,
		desired,
		v1alpha1.CheckpointDeletionPolicyRetain,
	)
	require.NoError(t, err)

	t.Log("Verify the desired policy is recorded and DGD ownership is removed")
	assert.Equal(t, string(v1alpha1.CheckpointDeletionPolicyRetain),
		updated.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation])
	assert.Empty(t, updated.OwnerReferences)
	assert.Equal(t, "test-dgd", updated.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName])
	assert.Equal(t, "worker", updated.Labels[commonconsts.KubeLabelDynamoComponent])
	assert.Equal(t, "captured:latest", updated.Spec.PodTemplate.Spec.Containers[0].Image,
		"non-invalidating input changes must not replace the one-shot capture")
}

func TestDGDCheckpointsReconciler_CheckpointRefSkipsAutoCreateWhilePodSnapshotIsNotReady(t *testing.T) {
	t.Log("Build a DGD referencing a PodSnapshot that is not Ready")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
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
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	require.NotEmpty(t, workerHash)
	referenced := dgdTestPodSnapshot(ref, workerHash, false)
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

	t.Log("Reconcile the not-Ready native reference")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	t.Log("Verify the reference remains pending without creating a legacy checkpoint")
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
	require.NotNil(t, info.NativeSnapshot)
	assert.Equal(t, referenced.UID, info.NativeSnapshot.UID)
	if checkpointStatuses["worker"].CheckpointName != friendlyCheckpointName {
		t.Fatalf("checkpoint status name = %s, want friendly-checkpoint", checkpointStatuses["worker"].CheckpointName)
	}

	jobs := &snapshotv1alpha1.SnapshotJobList{}
	if err := reconciler.List(ctx, jobs, client.InNamespace("default")); err != nil {
		t.Fatalf("failed to list SnapshotJobs: %v", err)
	}
	assert.Empty(t, jobs.Items)
}

func TestDGDCheckpointsReconciler_CheckpointRefUsesReadyPodSnapshot(t *testing.T) {
	t.Log("Build a DGD referencing a Ready compatible PodSnapshot")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)

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
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	require.NotEmpty(t, workerHash)
	referenced := dgdTestPodSnapshot(ref, workerHash, true)
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

	t.Log("Reconcile the Ready PodSnapshot reference")
	checkpointResult, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	checkpointStatuses := checkpointResult.Statuses
	checkpointInfos := checkpointResult.Infos
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	t.Log("Verify native artifact information and status are projected without legacy IDs")
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
	require.NotNil(t, info.NativeSnapshot)
	assert.Equal(t, "content-a", info.NativeSnapshot.BoundContentName)
	if checkpointStatuses["worker"].CheckpointName != friendlyCheckpointName {
		t.Fatalf("checkpoint status name = %s, want friendly-checkpoint", checkpointStatuses["worker"].CheckpointName)
	}
	if !checkpointStatuses["worker"].Ready {
		t.Fatalf("expected checkpoint status to be ready")
	}
	assert.Empty(t, checkpointStatuses["worker"].CheckpointID)
	assert.Empty(t, checkpointStatuses["worker"].IdentityHash)

	t.Log("Verify explicit native restore does not create Dynamo legacy storage")
	pvc := &corev1.PersistentVolumeClaim{}
	err = reconciler.Get(ctx, types.NamespacedName{Name: "legacy-storage", Namespace: "default"}, pvc)
	assert.True(t, apierrors.IsNotFound(err), "expected no legacy PVC, got %v", err)
}

func TestDGDCheckpointsReconciler_OverlaysServiceGMSLoader(t *testing.T) {
	t.Log("Build a native GMS snapshot reference with component-level clients")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)

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
					ExtraPodSpec: &v1alpha1.ExtraPodSpec{
						PodSpec:       &corev1.PodSpec{Containers: []corev1.Container{{Name: "gms-loader", Image: "loader:latest"}}},
						MainContainer: &corev1.Container{Name: commonconsts.MainContainerName, Image: "worker:latest"},
					},
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
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	referenced := dgdTestPodSnapshot(ref, workerHash, true)
	referenced.Annotations[commonconsts.SnapshotGMSModeAnnotation] = string(v1alpha1.GMSModeIntraPod)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:   fake.NewClientBuilder().WithScheme(testScheme).WithObjects(referenced).Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{
			Gate: features.Gates{Checkpoint: true},
		},
	}

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
	t.Log("Build a GMS component referencing a snapshot captured without GMS")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					ExtraPodSpec: &v1alpha1.ExtraPodSpec{
						PodSpec:       &corev1.PodSpec{Containers: []corev1.Container{{Name: "gms-loader", Image: "loader:latest"}}},
						MainContainer: &corev1.Container{Name: commonconsts.MainContainerName, Image: "worker:latest"},
					},
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
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	referenced := dgdTestPodSnapshot(ref, workerHash, true)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(testScheme).WithObjects(referenced).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}

	t.Log("Reconcile the incompatible checkpoint reference")
	_, err = newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)

	t.Log("Verify the incompatibility is returned with checkpoint context")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "gpuMemoryService topology for PodSnapshot")
	assert.Contains(t, err.Error(), "snapshot enabled=false, workload enabled=true")
	assert.Contains(t, err.Error(), friendlyCheckpointName)
}

func TestDGDCheckpointsReconciler_AutomaticRestoreWaitsForSnapshotJobCompletion(t *testing.T) {
	t.Log("Build an automatic checkpoint DGD")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					ExtraPodSpec: &v1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{
						Name:  commonconsts.MainContainerName,
						Image: "worker:latest",
					}},
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled: true,
						Mode:    v1alpha1.CheckpointModeAuto,
					},
				},
			},
		},
	})
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(testScheme).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}
	checkpointReconciler := newTestDGDCheckpointsReconciler(reconciler)

	t.Log("Create the SnapshotJob and simulate a Ready capture before helper completion")
	_, err := checkpointReconciler.Reconcile(ctx, dgd)
	require.NoError(t, err)
	jobs := &snapshotv1alpha1.SnapshotJobList{}
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	require.Len(t, jobs.Items, 1)
	job := jobs.Items[0].DeepCopy()
	// controller-runtime's fake client does not assign API-server UIDs.
	job.UID = types.UID("snapshot-job-uid")
	workerHash, err := checkpointWorkerHashForComponent(dgd, "worker")
	require.NoError(t, err)
	snapshot := dgdTestPodSnapshot(job.Name, workerHash, true)
	snapshot.Labels = map[string]string{
		snapshotv1alpha1.SnapshotJobOwnerLabel:    job.Name,
		snapshotv1alpha1.SnapshotJobOwnerUIDLabel: string(job.UID),
	}
	snapshot.Annotations[commonconsts.CheckpointAutoAnnotation] = commonconsts.KubeLabelValueTrue
	snapshot.Annotations[commonconsts.CheckpointOwnerUIDAnnotation] = string(dgd.UID)
	job.Status.PodSnapshotName = snapshot.Name
	job.Status.PodSnapshotUID = snapshot.UID
	job.Status.Conditions = []metav1.Condition{{
		Type:   snapshotv1alpha1.SnapshotJobConditionCaptured,
		Status: metav1.ConditionTrue,
	}}
	require.NoError(t, reconciler.Update(ctx, job))
	require.NoError(t, reconciler.Create(ctx, snapshot))

	t.Log("Verify Captured alone never makes the artifact restorable")
	result, err := checkpointReconciler.Reconcile(ctx, dgd)
	require.NoError(t, err)
	info := result.Infos["worker"]
	require.NotNil(t, info)
	assert.False(t, info.Exists)
	assert.False(t, info.Ready)
	assert.Nil(t, info.NativeSnapshot)
	assert.Equal(t, snapshot.Name, result.Statuses["worker"].CheckpointName)
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(snapshot), snapshot))
	assert.Equal(t, string(v1alpha1.CheckpointDeletionPolicyDelete),
		snapshot.Annotations[commonconsts.CheckpointDeletionPolicyAnnotation])

	t.Log("Complete the SnapshotJob and verify the same PodSnapshot becomes the restore source")
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(job), job))
	job.Status.Conditions = append(job.Status.Conditions, metav1.Condition{
		Type:   snapshotv1alpha1.SnapshotJobConditionCompleted,
		Status: metav1.ConditionTrue,
	})
	require.NoError(t, reconciler.Update(ctx, job))
	result, err = checkpointReconciler.Reconcile(ctx, dgd)
	require.NoError(t, err)
	info = result.Infos["worker"]
	require.NotNil(t, info)
	assert.True(t, info.Exists)
	assert.True(t, info.Ready)
	require.NotNil(t, info.NativeSnapshot)
	assert.Equal(t, snapshot.UID, info.NativeSnapshot.UID)
}

func TestDGDCheckpointsReconciler_AutomaticCaptureReportsSnapshotJobFailure(t *testing.T) {
	t.Log("Build a terminally failed automatic SnapshotJob")
	job := &snapshotv1alpha1.SnapshotJob{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-worker", Namespace: "default"},
		Status: snapshotv1alpha1.SnapshotJobStatus{Conditions: []metav1.Condition{{
			Type:    snapshotv1alpha1.SnapshotJobConditionFailed,
			Status:  metav1.ConditionTrue,
			Reason:  snapshotv1alpha1.ReasonDeadlineExceeded,
			Message: "capture exceeded its active deadline",
		}}},
	}

	t.Log("Resolve the automatic capture")
	_, err := (&dgdCheckpointsReconciler{}).resolveAutomaticSnapshotJob(
		context.Background(),
		job,
		&v1alpha1.ServiceCheckpointConfig{},
		ptr.To("worker-hash"),
		v1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
	)

	t.Log("Verify the terminal reason is surfaced instead of reported as pending")
	require.ErrorContains(t, err, "automatic SnapshotJob default/checkpoint-worker failed")
	require.ErrorContains(t, err, snapshotv1alpha1.ReasonDeadlineExceeded)
	require.ErrorContains(t, err, "capture exceeded its active deadline")
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
	rollout.setCurrentWorkerHashes(dgd, workerGenerationHashes{v2: "oldhash"})

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

func TestDGDCheckpointsReconciler_DeleteAutomaticSnapshotResourcesForDGD(t *testing.T) {
	t.Log("Build delete, retain, and same-name foreign native snapshot resources")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name:      "test-dgd",
		Namespace: "default",
		UID:       types.UID("dgd-uid"),
	}}
	metadata := func(policy v1alpha1.CheckpointDeletionPolicy, ownerUID string) metav1.ObjectMeta {
		return metav1.ObjectMeta{
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			},
			Annotations: map[string]string{
				commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
				commonconsts.CheckpointDeletionPolicyAnnotation: string(policy),
				commonconsts.CheckpointOwnerUIDAnnotation:       ownerUID,
			},
		}
	}
	deleteJob := &snapshotv1alpha1.SnapshotJob{ObjectMeta: metadata(v1alpha1.CheckpointDeletionPolicyDelete, string(dgd.UID))}
	deleteJob.Name = "delete-job"
	retainedJob := &snapshotv1alpha1.SnapshotJob{ObjectMeta: metadata(v1alpha1.CheckpointDeletionPolicyRetain, string(dgd.UID))}
	retainedJob.Name = "retained-job"
	retainedJob.UID = types.UID("retained-job-uid")
	retainedJob.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: v1beta1.GroupVersion.String(),
		Kind:       "DynamoGraphDeployment",
		Name:       dgd.Name,
		UID:        dgd.UID,
		Controller: ptr.To(true),
	}}
	foreignJob := &snapshotv1alpha1.SnapshotJob{ObjectMeta: metadata(v1alpha1.CheckpointDeletionPolicyDelete, "other-dgd-uid")}
	foreignJob.Name = "foreign-job"
	deleteSnapshot := &snapshotv1alpha1.PodSnapshot{ObjectMeta: metadata(v1alpha1.CheckpointDeletionPolicyDelete, string(dgd.UID))}
	deleteSnapshot.Name = "delete-snapshot"
	retainedSnapshot := &snapshotv1alpha1.PodSnapshot{ObjectMeta: metadata(v1alpha1.CheckpointDeletionPolicyRetain, string(dgd.UID))}
	retainedSnapshot.Name = "retained-snapshot"
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(deleteJob, retainedJob, foreignJob, deleteSnapshot, retainedSnapshot).
			Build(),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Checkpoint: false}},
	}
	checkpointReconciler := newTestDGDCheckpointsReconciler(reconciler)

	t.Log("Delete SnapshotJobs before allowing artifact cleanup")
	err := checkpointReconciler.deleteAutoCheckpointsForDGD(ctx, dgd)
	require.ErrorIs(t, err, errAutomaticSnapshotCleanupPending)
	assert.True(t, apierrors.IsNotFound(reconciler.Get(ctx, client.ObjectKeyFromObject(deleteJob), &snapshotv1alpha1.SnapshotJob{})))
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(deleteSnapshot), &snapshotv1alpha1.PodSnapshot{}))

	t.Log("Delete artifacts after capture jobs are gone and detach retained resources")
	require.NoError(t, checkpointReconciler.deleteAutoCheckpointsForDGD(ctx, dgd))
	assert.True(t, apierrors.IsNotFound(reconciler.Get(ctx, client.ObjectKeyFromObject(deleteSnapshot), &snapshotv1alpha1.PodSnapshot{})))
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(foreignJob), &snapshotv1alpha1.SnapshotJob{}))

	retainedJobAfter := &snapshotv1alpha1.SnapshotJob{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(retainedJob), retainedJobAfter))
	assert.Empty(t, retainedJobAfter.OwnerReferences)
	assert.Equal(t, dgd.Name, retainedJobAfter.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName])
	retainedSnapshotAfter := &snapshotv1alpha1.PodSnapshot{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(retainedSnapshot), retainedSnapshotAfter))
	assert.Equal(t, dgd.Name, retainedSnapshotAfter.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName])
}

func TestDGDCheckpointsReconciler_DeletingSnapshotJobDoesNotBlockFinalization(t *testing.T) {
	t.Log("Build a delete-policy SnapshotJob held by an external finalizer")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name:      "test-dgd",
		Namespace: "default",
		UID:       types.UID("dgd-uid"),
	}}
	job := &snapshotv1alpha1.SnapshotJob{ObjectMeta: metav1.ObjectMeta{
		Name:       "delete-job",
		Namespace:  dgd.Namespace,
		UID:        types.UID("delete-job-uid"),
		Finalizers: []string{"snapshot.nvidia.com/external-cleanup"},
		Labels: map[string]string{
			commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
		Annotations: map[string]string{
			commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
			commonconsts.CheckpointDeletionPolicyAnnotation: string(v1alpha1.CheckpointDeletionPolicyDelete),
			commonconsts.CheckpointOwnerUIDAnnotation:       string(dgd.UID),
		},
	}}
	snapshot := &snapshotv1alpha1.PodSnapshot{ObjectMeta: metav1.ObjectMeta{
		Name:      "delete-artifact",
		Namespace: dgd.Namespace,
		Labels: map[string]string{
			commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			snapshotv1alpha1.SnapshotJobOwnerLabel:          job.Name,
			snapshotv1alpha1.SnapshotJobOwnerUIDLabel:       string(job.UID),
		},
		Annotations: map[string]string{
			commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
			commonconsts.CheckpointDeletionPolicyAnnotation: string(v1alpha1.CheckpointDeletionPolicyDelete),
			commonconsts.CheckpointOwnerUIDAnnotation:       string(dgd.UID),
		},
	}}
	kubeClient := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(job, snapshot).
		Build()
	checkpointReconciler := newTestDGDCheckpointsReconciler(&DynamoGraphDeploymentReconciler{Client: kubeClient})

	t.Log("Request SnapshotJob deletion and observe cleanup pending once")
	err := checkpointReconciler.deleteAutoCheckpointsForDGD(ctx, dgd)
	require.ErrorIs(t, err, errAutomaticSnapshotCleanupPending)
	deletingJob := &snapshotv1alpha1.SnapshotJob{}
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(job), deletingJob))
	require.NotNil(t, deletingJob.DeletionTimestamp)

	t.Log("Reconcile again while Snapshot still owns final Job cleanup")
	require.NoError(t, checkpointReconciler.deleteAutoCheckpointsForDGD(ctx, dgd))

	t.Log("Verify Dynamo deletes its artifact without waiting on the foreign finalizer")
	assert.True(t, apierrors.IsNotFound(kubeClient.Get(
		ctx,
		client.ObjectKeyFromObject(snapshot),
		&snapshotv1alpha1.PodSnapshot{},
	)))
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(job), &snapshotv1alpha1.SnapshotJob{}))
}

func TestDGDCheckpointsReconciler_RetainProtectsArtifactCreatedDuringFinalization(t *testing.T) {
	t.Log("Build a retained SnapshotJob whose artifact appears after lifecycle synchronization")
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name:      "test-dgd",
		Namespace: "default",
		UID:       types.UID("dgd-uid"),
	}}
	job := &snapshotv1alpha1.SnapshotJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "retained-job",
			Namespace: dgd.Namespace,
			UID:       types.UID("retained-job-uid"),
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			},
			Annotations: map[string]string{
				commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
				commonconsts.CheckpointDeletionPolicyAnnotation: string(v1alpha1.CheckpointDeletionPolicyRetain),
				commonconsts.CheckpointOwnerUIDAnnotation:       string(dgd.UID),
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: v1beta1.GroupVersion.String(),
				Kind:       "DynamoGraphDeployment",
				Name:       dgd.Name,
				UID:        dgd.UID,
				Controller: ptr.To(true),
			}},
		},
		Status: snapshotv1alpha1.SnapshotJobStatus{PodSnapshotName: "retained-artifact"},
	}
	snapshot := &snapshotv1alpha1.PodSnapshot{ObjectMeta: metav1.ObjectMeta{
		Name:      job.Status.PodSnapshotName,
		Namespace: dgd.Namespace,
		Labels: map[string]string{
			commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			snapshotv1alpha1.SnapshotJobOwnerLabel:          job.Name,
			snapshotv1alpha1.SnapshotJobOwnerUIDLabel:       string(job.UID),
		},
		Annotations: map[string]string{
			commonconsts.CheckpointAutoAnnotation:     commonconsts.KubeLabelValueTrue,
			commonconsts.CheckpointOwnerUIDAnnotation: string(dgd.UID),
		},
	}}
	hideArtifactFromLifecycleSync := true
	kubeClient := fake.NewClientBuilder().
		WithScheme(testScheme).
		WithObjects(job, snapshot).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				if _, ok := obj.(*snapshotv1alpha1.PodSnapshot); ok && key == client.ObjectKeyFromObject(snapshot) && hideArtifactFromLifecycleSync {
					hideArtifactFromLifecycleSync = false
					return apierrors.NewNotFound(
						snapshotv1alpha1.GroupVersion.WithResource("podsnapshots").GroupResource(),
						key.Name,
					)
				}
				return c.Get(ctx, key, obj, opts...)
			},
		}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient}

	t.Log("Finalize while the retained artifact is absent from the point read but present in the following list")
	require.NoError(t, newTestDGDCheckpointsReconciler(reconciler).deleteAutoCheckpointsForDGD(ctx, dgd))

	t.Log("Verify SnapshotJob identity protects the artifact without relying on its mutable policy annotation")
	retainedSnapshot := &snapshotv1alpha1.PodSnapshot{}
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(snapshot), retainedSnapshot))
	assert.NotContains(t, retainedSnapshot.Annotations, commonconsts.CheckpointDeletionPolicyAnnotation)
	retainedJob := &snapshotv1alpha1.SnapshotJob{}
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(job), retainedJob))
	assert.Empty(t, retainedJob.OwnerReferences)
}
