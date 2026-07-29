/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
)

func TestProjectedPodSupportsControllerContract(t *testing.T) {
	controllerOwner := true
	pod := podcache.Project(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-0",
			Namespace: "inference",
			UID:       types.UID("pod-uid"),
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName:       "graph",
				consts.KubeLabelDynamoComponent:                 "prefill",
				consts.KubeLabelDynamoSelector:                  "graph-prefill-old",
				consts.KubeLabelDynamoComponentType:             consts.ComponentTypeWorker,
				consts.KubeLabelDynamoSubComponentType:          consts.ComponentTypePrefill,
				consts.KubeLabelDynamoFailoverEngineGroupMember: consts.KubeLabelValueTrue,
				batchv1.JobNameLabel:                            "profiling-job",
				"job-name":                                      "profiling-job",
			},
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			OwnerReferences: []metav1.OwnerReference{{
				Name: "graph-prefill-old", UID: types.UID("owner-uid"), Controller: &controllerOwner,
			}},
		},
		Spec: corev1.PodSpec{
			NodeName: "node-a",
			Containers: []corev1.Container{{
				Name:    consts.MainContainerName,
				Command: []string{"python"},
				Args:    []string{"-m", "dynamo.vllm"},
			}},
			Volumes: []corev1.Volume{{
				Name: "topology",
				VolumeSource: corev1.VolumeSource{DownwardAPI: &corev1.DownwardAPIVolumeSource{
					Items: []corev1.DownwardAPIVolumeFile{{
						FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['nvidia.com/dynamo-topology.zone']"},
					}},
				}},
			}},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodFailed,
			Conditions: []corev1.PodCondition{{
				Type: corev1.PodReady, Status: corev1.ConditionTrue,
			}},
			ContainerStatuses: []corev1.ContainerStatus{{
				Name: ContainerNameProfiler,
				State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
					ExitCode: 23, Reason: "Error", Message: "profiling failed",
				}},
			}},
		},
	})

	t.Run("topology", func(t *testing.T) {
		assert.True(t, needsTopologyLabelCopy(pod))
		assert.Equal(t, []string{"nvidia.com/dynamo-topology.zone"}, expectedDynamoTopologyLabelKeys(pod))
	})

	t.Run("failover", func(t *testing.T) {
		assert.True(t, failoverCascadePredicate().Create(event.CreateEvent{Object: pod}))
	})

	t.Run("checkpoint and snapshot", func(t *testing.T) {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: pod.Namespace}}
		snapshot := buildPodSnapshot(ckpt, "checkpoint-id", pod)
		assert.Equal(t, pod.Name, snapshot.Spec.Source.PodRef.Name)
		assert.Equal(t, pod.UID, snapshot.Spec.Source.PodRef.UID)
		require.NoError(t, validateSourcePod(snapshot, pod))
	})

	t.Run("model", func(t *testing.T) {
		identified := withPodIdentity(modelendpoint.Candidate{}, pod)
		assert.True(t, identified.KubernetesReady)
		assert.Equal(t, "graph", identified.GraphDeploymentName)
		assert.Equal(t, "graph-prefill-old", identified.WorkloadName)
		assert.Equal(t, "graph:graph", identified.LoRAFallbackGroup)
		assert.True(t, identified.AllowLoRAManagementUnavailable)
	})

	t.Run("DGDR diagnostics", func(t *testing.T) {
		scheme := runtime.NewScheme()
		require.NoError(t, corev1.AddToScheme(scheme))
		client := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod).Build()
		reconciler := &DynamoGraphDeploymentRequestReconciler{
			Client:   client,
			Recorder: record.NewFakeRecorder(1),
		}
		dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace}}
		job := &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "profiling-job", Namespace: pod.Namespace}}

		detail := reconciler.getProfilingJobErrorDetails(context.Background(), dgdr, job)
		assert.Contains(t, detail, "ExitCode: 23")
		assert.Contains(t, detail, "profiling failed")
	})

	t.Run("Recreate", func(t *testing.T) {
		// PR #11909 lists old worker Pods by these labels and treats only
		// Failed/Succeeded phases (or cache disappearance) as terminated.
		assert.Equal(t, "graph", pod.Labels[consts.KubeLabelDynamoGraphDeploymentName])
		assert.Equal(t, "prefill", pod.Labels[consts.KubeLabelDynamoComponent])
		assert.Equal(t, "graph-prefill-old", pod.Labels[consts.KubeLabelDynamoSelector])
		assert.True(t, isTerminalPhase(pod.Status.Phase))
	})
}
