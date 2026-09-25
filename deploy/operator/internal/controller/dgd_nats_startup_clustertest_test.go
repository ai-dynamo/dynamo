//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/clusterenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestClusterDynamoGraphDeploymentToleratesLateNATS(t *testing.T) {
	t.Log("Create an isolated namespace and use the production planner image for all three processes")
	ctx := t.Context()
	image := clusterenv.RequireEnv(t, "DYNAMO_CLUSTERTEST_PROFILER_IMAGE")
	env := clusterTestEnv.RunT(t)
	kubeClient, err := kubernetes.NewForConfig(env.RESTConfig())
	require.NoError(t, err)

	t.Log("Mount the same local model metadata in the worker and frontend without model downloads")
	const modelPath = "/opt/dynamo/test-model"
	modelFiles := []string{"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}
	modelConfig := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "mock-model", Namespace: env.Namespace()},
		Data:       make(map[string]string, len(modelFiles)),
	}
	for _, name := range modelFiles {
		contents, err := os.ReadFile(filepath.Join("../../../../lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct", name))
		require.NoError(t, err)
		modelConfig.Data[name] = string(contents)
	}
	require.NoError(t, env.Client().Create(ctx, modelConfig))

	t.Log("Create the NATS Service with no endpoints and keep the NATS Deployment at zero replicas")
	natsLabels := map[string]string{"app": "nats"}
	require.NoError(t, env.Client().Create(ctx, &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "nats", Namespace: env.Namespace()},
		Spec: corev1.ServiceSpec{
			Selector: natsLabels,
			Ports:    []corev1.ServicePort{{Name: "nats", Port: 4222, TargetPort: intstr.FromInt(4222)}},
		},
	}))
	natsDeployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "nats", Namespace: env.Namespace()},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To(int32(0)),
			Selector: &metav1.LabelSelector{MatchLabels: natsLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: natsLabels},
				Spec: corev1.PodSpec{
					TerminationGracePeriodSeconds: ptr.To(int64(5)),
					SecurityContext:               &corev1.PodSecurityContext{FSGroup: ptr.To(int64(1000))},
					Containers: []corev1.Container{{
						Name: "nats", Image: image, ImagePullPolicy: corev1.PullIfNotPresent,
						Command: []string{"/usr/local/bin/nats-server"},
						Args:    []string{"-js", "-sd", "/data", "-p", "4222", "-m", "8222"},
						Resources: corev1.ResourceRequirements{Requests: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse("100m"), corev1.ResourceMemory: resource.MustParse("128Mi"),
						}},
						VolumeMounts: []corev1.VolumeMount{{Name: "data", MountPath: "/data"}},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler:  corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstr.FromInt(8222)}},
							PeriodSeconds: 1, TimeoutSeconds: 1,
						},
					}},
					Volumes: []corev1.Volume{{Name: "data", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}},
				},
			},
		},
	}
	require.NoError(t, env.Client().Create(ctx, natsDeployment))

	t.Log("Create a frontend and mock worker with runtime 1.6.0 defaults and the default NATS timeout")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "late-nats", Namespace: env.Namespace(),
			Annotations: map[string]string{commonconsts.KubeAnnotationEnableGrove: "false"},
		},
	}
	for _, componentType := range []nvidiacomv1beta1.ComponentType{nvidiacomv1beta1.ComponentTypeFrontend, nvidiacomv1beta1.ComponentTypeWorker} {
		component := nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: string(componentType), ComponentType: componentType,
			RuntimeVersionOverride: "1.6.0", Replicas: ptr.To(int32(1)),
			SharedMemorySize: ptr.To(resource.MustParse("0")),
			PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				TerminationGracePeriodSeconds: ptr.To(int64(5)),
				Containers: []corev1.Container{{
					Name: commonconsts.MainContainerName, Image: image, ImagePullPolicy: corev1.PullIfNotPresent,
					Resources: corev1.ResourceRequirements{Requests: corev1.ResourceList{
						corev1.ResourceCPU: resource.MustParse("200m"), corev1.ResourceMemory: resource.MustParse("256Mi"),
					}},
					Env: []corev1.EnvVar{
						{Name: "HF_HUB_OFFLINE", Value: "1"},
						{Name: "DYN_SDK_DISABLE_ANSI_LOGGING", Value: "1"},
					},
					VolumeMounts: []corev1.VolumeMount{{Name: "model", MountPath: modelPath, ReadOnly: true}},
				}},
				Volumes: []corev1.Volume{{Name: "model", VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: modelConfig.Name}},
				}}},
			}},
		}
		if componentType == nvidiacomv1beta1.ComponentTypeWorker {
			component.PodTemplate.Spec.Containers[0].Command = []string{"python3"}
			component.PodTemplate.Spec.Containers[0].Args = []string{"-m", "dynamo.mocker", "--model-path", modelPath, "--model-name", "mock-llama"}
		}
		dgd.Spec.Components = append(dgd.Spec.Components, component)
	}
	require.NoError(t, env.Client().Create(ctx, dgd))

	t.Log("Start the production DGD and DCD controllers with NATS enabled and Kubernetes discovery")
	natsAddress := fmt.Sprintf("nats://nats.%s.svc.cluster.local:4222", env.Namespace())
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	operatorConfig.Infrastructure.NATSAddress = natsAddress
	env.StartManager(func(mgr ctrl.Manager) error {
		setupOptions := SetupOptions{Config: operatorConfig, RuntimeConfig: &commoncontroller.RuntimeConfig{}}
		if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: setupOptions}); err != nil {
			return err
		}
		return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{SetupOptions: setupOptions})
	})

	t.Log("Wait for both main containers to run and record their identities and later start time")
	podSelector := client.MatchingLabels{commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name}
	podIDs := make(map[string]types.UID)
	var laterStart time.Time
	var frontendPod string
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		pods := &corev1.PodList{}
		require.NoError(c, env.Client().List(ctx, pods, client.InNamespace(env.Namespace()), podSelector))
		require.Len(c, pods.Items, 2)
		for _, pod := range pods.Items {
			require.Len(c, pod.Status.ContainerStatuses, 1)
			status := pod.Status.ContainerStatuses[0]
			require.Equal(c, commonconsts.MainContainerName, status.Name)
			require.NotNil(c, status.State.Running)
			require.Zero(c, status.RestartCount)
			podIDs[pod.Name] = pod.UID
			if status.State.Running.StartedAt.Time.After(laterStart) {
				laterStart = status.State.Running.StartedAt.Time
			}
			if pod.Labels[commonconsts.KubeLabelDynamoComponentType] == string(nvidiacomv1beta1.ComponentTypeFrontend) {
				frontendPod = pod.Name
			}
		}
	}, 2*time.Minute, time.Second)
	require.NotEmpty(t, frontendPod)

	t.Log("Keep NATS unavailable for 60 seconds while both original pods stay unready without restarting")
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		pods := &corev1.PodList{}
		require.NoError(t, env.Client().List(ctx, pods, client.InNamespace(env.Namespace()), podSelector))
		require.Len(t, pods.Items, 2)
		for _, pod := range pods.Items {
			require.Equal(t, podIDs[pod.Name], pod.UID)
			require.Len(t, pod.Status.ContainerStatuses, 1)
			status := pod.Status.ContainerStatuses[0]
			require.NotNil(t, status.State.Running)
			require.Nil(t, status.State.Terminated)
			require.Nil(t, status.LastTerminationState.Terminated)
			require.Zero(t, status.RestartCount)
			require.False(t, status.Ready)
			for _, condition := range pod.Status.Conditions {
				if condition.Type == corev1.PodReady {
					require.NotEqual(t, corev1.ConditionTrue, condition.Status)
				}
			}
		}
		if time.Since(laterStart) >= 60*time.Second {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-ticker.C:
		}
	}

	t.Log("Start NATS and wait for a ready Service endpoint inside the default startup deadline")
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(natsDeployment), natsDeployment))
	natsDeployment.Spec.Replicas = ptr.To(int32(1))
	require.NoError(t, env.Client().Update(ctx, natsDeployment))
	require.Eventually(t, func() bool {
		slices := &discoveryv1.EndpointSliceList{}
		if err := env.Client().List(ctx, slices, client.InNamespace(env.Namespace()), client.MatchingLabels{discoveryv1.LabelServiceName: "nats"}); err != nil {
			return false
		}
		for _, slice := range slices.Items {
			for _, endpoint := range slice.Endpoints {
				if ptr.Deref(endpoint.Conditions.Ready, false) {
					return true
				}
			}
		}
		return false
	}, 30*time.Second, time.Second)

	t.Log("Wait for both original pods to become ready with no restarts")
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		pods := &corev1.PodList{}
		require.NoError(c, env.Client().List(ctx, pods, client.InNamespace(env.Namespace()), podSelector))
		require.Len(c, pods.Items, 2)
		for _, pod := range pods.Items {
			require.Equal(c, podIDs[pod.Name], pod.UID)
			require.Len(c, pod.Status.ContainerStatuses, 1)
			status := pod.Status.ContainerStatuses[0]
			require.Zero(c, status.RestartCount)
			require.Nil(c, status.LastTerminationState.Terminated)
			require.True(c, status.Ready)
		}
	}, 2*time.Minute, time.Second)

	t.Log("Verify worker discovery through the frontend models API using the Kubernetes pod proxy")
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		body, err := kubeClient.CoreV1().RESTClient().Get().Namespace(env.Namespace()).Resource("pods").
			Name(frontendPod + ":8000").SubResource("proxy").Suffix("v1/models").Timeout(5 * time.Second).DoRaw(ctx)
		require.NoError(c, err)
		var models struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		require.NoError(c, json.Unmarshal(body, &models))
		var modelNames []string
		for _, model := range models.Data {
			modelNames = append(modelNames, model.ID)
		}
		require.Contains(c, modelNames, "mock-llama")
	}, time.Minute, time.Second)

	t.Log("Check the retry diagnostics and recovery log without exposing the NATS address")
	logs, err := kubeClient.CoreV1().Pods(env.Namespace()).GetLogs(frontendPod, &corev1.PodLogOptions{Container: commonconsts.MainContainerName}).DoRaw(ctx)
	require.NoError(t, err)
	for _, field := range []string{"attempt=", "error_kind=", "remaining=", "retry_in=", "NATS startup connection established after retry"} {
		require.Contains(t, string(logs), field)
	}
	require.NotContains(t, string(logs), natsAddress)
}
