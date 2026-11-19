/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	podcontract "github.com/ai-dynamo/snapshot/api/podcontract"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
)

func TestProjectedPodSupportsControllerContract(t *testing.T) {
	controllerOwner := true
	pod := podcache.Project(&corev1.Pod{
		Name:      "worker-0",
		Namespace: "inference",
		UID:       types.UID("pod-uid"),
		Labels: map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName:       "graph",
			consts.KubeLabelDynamoComponent:                 "prefill",
			consts.KubeLabelDynamoNamespace:                 "inference-graph",
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
		Spec: corev1.PodSpec{
			NodeName: "node-a",
			Containers: []corev1.Container{{
				Name:    consts.MainContainerName,
				Command: []string{"python"},
				Args:    []string{"-m", "dynamo.vllm"},
			}},
			InitContainers: []corev1.Container{{
				Name:          gms.ServerContainerName,
				Image:         "discarded-image",
				RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
			}},
			Volumes: []corev1.Volume{{
				Name: "topology",
				DownwardAPI: &corev1.DownwardAPIVolumeSource{
					Items: []corev1.DownwardAPIVolumeFile{{
						FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['nvidia.com/dynamo-topology.zone']"},
					}},
				},
			}},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodFailed,
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
				{
					Type:   corev1.PodConditionType(podcontract.RestoredCondition),
					Status: corev1.ConditionTrue,
					Reason: podcontract.RestoreReasonSucceeded,
				},
			},
			ContainerStatuses: []corev1.ContainerStatus{{
				Name: ContainerNameProfiler,
				State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
					ExitCode: 23, Reason: "Error", Message: "profiling failed",
				}},
			}},
			InitContainerStatuses: []corev1.ContainerStatus{{
				Name: gms.ServerContainerName, RestartCount: 1,
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

	t.Run("GMS Pod replacement", func(t *testing.T) {
		t.Log("Build a projected Dynamo Pod carrying standalone Snapshot restore state")
		nativeRestorePod := pod.DeepCopy()
		nativeRestorePod.Annotations[podcontract.RestoreFromAnnotation] = "snapshot-a"

		t.Log("Verify GMS replacement observes the projected native sidecar restart")
		assert.True(t, gmsPodReplacementPredicate().Create(event.CreateEvent{Object: nativeRestorePod}))
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
			Recorder: events.NewFakeRecorder(1),
		}
		dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{Namespace: pod.Namespace}
		job := &batchv1.Job{Name: "profiling-job", Namespace: pod.Namespace}

		detail := reconciler.getProfilingJobErrorDetails(context.Background(), dgdr, job)
		assert.Contains(t, detail, "ExitCode: 23")
		assert.Contains(t, detail, "profiling failed")
	})

	t.Run("DGDR Run diagnostics", func(t *testing.T) {
		t.Log("Project a pod of a Run's Job whose sweeper waits on a mount error")
		job := &batchv1.Job{Name: "search-1-aisimulate", Namespace: pod.Namespace, UID: types.UID("job-uid")}
		runPod := podcache.Project(&corev1.Pod{
			Name:            "search-1-aisimulate-abcde",
			Namespace:       pod.Namespace,
			Labels:          map[string]string{batchv1.JobNameLabel: job.Name},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(job, batchv1.SchemeGroupVersion.WithKind("Job"))},
			Spec: corev1.PodSpec{
				NodeName:   "node-a",
				Containers: []corev1.Container{{Name: dgdrRunSweeperContainer}, {Name: dgdrRunPublisherContainer}},
			},
			Status: corev1.PodStatus{
				Phase: corev1.PodPending,
				ContainerStatuses: []corev1.ContainerStatus{
					{Name: dgdrRunPublisherContainer, State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}},
					{Name: dgdrRunSweeperContainer, State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{
						Reason: "CreateContainerConfigError", Message: `failed to prepare subPath for volumeMount "trace"`,
					}}},
				},
			},
		})

		t.Log("The Run controller finds the pod of its Job and reports the waiting sweeper verbatim")
		scheme := runtime.NewScheme()
		require.NoError(t, corev1.AddToScheme(scheme))
		reconciler := &DynamoGraphDeploymentRunV1beta2Reconciler{
			Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(runPod).Build(),
		}
		pods, err := reconciler.jobPods(context.Background(), job)
		require.NoError(t, err)
		require.Len(t, pods, 1)
		container, waiting, found := dgdrRunWaitingContainer(pods)
		assert.True(t, found)
		assert.Equal(t, dgdrRunSweeperContainer, container)
		assert.Equal(t, corev1.ContainerStateWaiting{
			Reason: "CreateContainerConfigError", Message: `failed to prepare subPath for volumeMount "trace"`,
		}, waiting)
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
