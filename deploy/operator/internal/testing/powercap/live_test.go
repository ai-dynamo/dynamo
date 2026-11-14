//go:build clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package powercap

import (
	"context"
	"os"
	"testing"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// These black-box tests require the deployed operator and real Grove controller.
// They intentionally do not import the controller package's mock-backed TestMain.
func TestLivePowerCapsDeployment(t *testing.T) { testLivePowerCaps(t, "component") }
func TestLivePowerCapsGrove(t *testing.T)      { testLivePowerCaps(t, "grove") }

func testLivePowerCaps(t *testing.T, provider string) {
	t.Helper()
	t.Log("Connect to the explicitly selected real-controller test cluster")
	configPath := os.Getenv("KUBECONFIG")
	require.NotEmpty(t, configPath, "KUBECONFIG must explicitly select the smoke cluster")
	config, err := clientcmd.BuildConfigFromFlags("", configPath)
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))
	c, err := client.New(config, client.Options{Scheme: scheme})
	require.NoError(t, err)
	clientset, err := kubernetes.NewForConfig(config)
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	namespace := os.Getenv("DYNAMO_POWERCAP_TEST_NAMESPACE")
	if namespace == "" {
		namespace = "default"
	}
	// Admission derives the provider from routing intent, so request it through enable-grove.
	enableGrove := consts.KubeLabelValueFalse
	if provider == consts.WorkloadProviderGrove {
		enableGrove = consts.KubeLabelValueTrue
	}
	probe := &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"/bin/true"}}}, PeriodSeconds: 1}
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{GenerateName: "live-power-" + provider + "-", Namespace: namespace, Annotations: map[string]string{consts.KubeAnnotationEnableGrove: enableGrove}}, Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker", RuntimeVersionOverride: "1.6.0", Replicas: new(int32(2)), PodTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}}, Spec: corev1.PodSpec{TerminationGracePeriodSeconds: new(int64(1)), Containers: []corev1.Container{{Name: "main", Image: "busybox:1.37", Command: []string{"/bin/sh", "-c"}, Args: []string{"sleep 3600"}, StartupProbe: probe.DeepCopy(), LivenessProbe: probe.DeepCopy(), ReadinessProbe: probe.DeepCopy()}}}}}}}}
	require.NoError(t, c.Create(ctx, dgd))
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 20*time.Second)
		defer stop()
		if err := c.Delete(cleanup, dgd); err != nil {
			t.Errorf("delete smoke DGD: %v", err)
		}
		if err := wait.PollUntilContextCancel(cleanup, 200*time.Millisecond, true, func(ctx context.Context) (bool, error) {
			pods := &corev1.PodList{}
			if err := c.List(ctx, pods, client.InNamespace(namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}); err != nil {
				return false, err
			}
			return len(pods.Items) == 0, nil
		}); err != nil {
			t.Errorf("cleanup smoke workers: %v", err)
		}
	})

	t.Log("Confirm admission selected the requested workload provider")
	require.Equal(t, provider, dgd.Annotations[consts.KubeAnnotationWorkloadProvider])

	t.Log("Wait for two real CPU-only worker Pods and record their identities and creation templates")
	pods := waitForPowerCaps(t, ctx, c, dgd, "300")
	initialUIDs := map[types.UID]bool{}
	for _, pod := range pods {
		initialUIDs[pod.UID] = true
	}
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	initialHash := dgd.Annotations[consts.AnnotationCurrentWorkerHashV2]
	initialDCDs := workerDCDIdentities(t, ctx, c, dgd)
	var templates any
	if provider == "grove" {
		pcs := &grovev1alpha1.PodCliqueSet{}
		require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(dgd), pcs))
		templates = pcs.Spec.DeepCopy()
	} else {
		deployments := &appsv1.DeploymentList{}
		require.NoError(t, c.List(ctx, deployments, client.InNamespace(namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}))
		require.Len(t, deployments.Items, 1)
		templates = deployments.Items[0].Spec.Template.DeepCopy()
	}

	t.Log("Edit 300 to 350 W without replacing warm Pods or changing creation templates")
	previous := dgd.DeepCopy()
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	require.NoError(t, c.Patch(ctx, dgd, client.MergeFrom(previous)))
	pods = waitForPowerCaps(t, ctx, c, dgd, "350")
	for _, pod := range pods {
		require.True(t, initialUIDs[pod.UID], "cap edit must preserve Pod UIDs")
	}
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	require.Equal(t, initialHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
	requireNoWorkerReplacement(t, ctx, c, dgd, initialDCDs)
	if provider == "grove" {
		pcs := &grovev1alpha1.PodCliqueSet{}
		require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(dgd), pcs))
		require.Equal(t, templates, pcs.Spec.DeepCopy())
	} else {
		deployments := &appsv1.DeploymentList{}
		require.NoError(t, c.List(ctx, deployments, client.InNamespace(namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}))
		require.Len(t, deployments.Items, 1)
		require.Equal(t, templates, deployments.Items[0].Spec.Template.DeepCopy())
	}

	t.Log("Delete one Pod: a real controller must replace it, retain the survivor and preserve its patched annotation")
	deleted, survivor := pods[0], pods[1]
	// Ordered API watch events expose transient reversion even if Dynamo repairs it immediately.
	var survivorEvents <-chan watch.Event
	if provider == "grove" {
		require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(&survivor), &survivor))
		stream, err := clientset.CoreV1().Pods(namespace).Watch(ctx, metav1.ListOptions{FieldSelector: fields.OneTermEqualSelector("metadata.name", survivor.Name).String(), ResourceVersion: survivor.ResourceVersion})
		require.NoError(t, err)
		defer stream.Stop()
		survivorEvents = stream.ResultChan()
	}
	require.NoError(t, c.Delete(ctx, &deleted))
	require.NoError(t, wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		current := &corev1.PodList{}
		if err := c.List(ctx, current, client.InNamespace(namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}); err != nil {
			return false, err
		}
		if len(current.Items) != 2 {
			return false, nil
		}
		foundSurvivor := false
		for _, pod := range current.Items {
			if pod.UID == deleted.UID || pod.Annotations[consts.KubeAnnotationGPUPowerLimit] != "350" {
				return false, nil
			}
			foundSurvivor = foundSurvivor || pod.UID == survivor.UID
		}
		return foundSurvivor, nil
	}))
	// A sentinel after real Grove progress gives the watch a causal drain barrier.
	if provider == "grove" {
		// A child availability-only update confirms a completed Grove pass without changing its Pod template.
		owner := metav1.GetControllerOf(&survivor)
		require.NotNil(t, owner)
		require.Equal(t, "PodClique", owner.Kind)
		clique := &grovev1alpha1.PodClique{}
		require.NoError(t, c.Get(ctx, client.ObjectKey{Namespace: namespace, Name: owner.Name}, clique))
		beforeClique := clique.DeepCopy()
		minimum := int32(1)
		if clique.Spec.MinAvailable != nil && *clique.Spec.MinAvailable == minimum {
			minimum = 2
		}
		clique.Spec.MinAvailable = &minimum
		require.NoError(t, c.Patch(ctx, clique, client.MergeFrom(beforeClique)))
		targetGeneration := clique.Generation
		require.NoError(t, wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, 60*time.Second, true, func(ctx context.Context) (bool, error) {
			if err := c.Get(ctx, client.ObjectKeyFromObject(clique), clique); err != nil {
				return false, err
			}
			return clique.Status.ObservedGeneration != nil && *clique.Status.ObservedGeneration >= targetGeneration && clique.Status.ReadyReplicas == 2, nil
		}))
		require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(&survivor), &survivor))
		before := survivor.DeepCopy()
		const barrierKey = "dynamo.nvidia.com/power-cap-test-barrier"
		survivor.Annotations[barrierKey] = string(deleted.UID)
		require.NoError(t, c.Patch(ctx, &survivor, client.MergeFrom(before)))
		barrierSeen := false
		for !barrierSeen {
			select {
			case <-ctx.Done():
				t.Fatal("timed out draining survivor watch through Grove progress barrier")
			case event, open := <-survivorEvents:
				require.True(t, open, "survivor watch must not close before the barrier")
				require.NotEqual(t, watch.Error, event.Type, "watch error cannot establish non-reversion: %v", event.Object)
				observed, ok := event.Object.(*corev1.Pod)
				require.True(t, ok, "expected survivor Pod event")
				require.Equal(t, survivor.UID, observed.UID)
				require.Equal(t, "350", observed.Annotations[consts.KubeAnnotationGPUPowerLimit], "Grove must never revert the survivor, even transiently")
				barrierSeen = observed.Annotations[barrierKey] == string(deleted.UID)
			}
		}
	}
	waitForPowerCaps(t, ctx, c, dgd, "350")
	requireNoWorkerReplacement(t, ctx, c, dgd, initialDCDs)
}

// workerDCDIdentities records every DCD of the graph; Grove graphs legitimately have none.
func workerDCDIdentities(t *testing.T, ctx context.Context, c client.Client, dgd *v1beta1.DynamoGraphDeployment) map[types.UID]string {
	t.Helper()
	dcds := &v1beta1.DynamoComponentDeploymentList{}
	require.NoError(t, c.List(ctx, dcds, client.InNamespace(dgd.Namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}))
	identities := map[types.UID]string{}
	for _, dcd := range dcds.Items {
		identities[dcd.UID] = dcd.Name
	}
	return identities
}

// requireNoWorkerReplacement fails if a replacement DCD exists or a managed rollout is pending,
// so propagation to old Pods cannot mask an incorrect cap-triggered rollout.
func requireNoWorkerReplacement(t *testing.T, ctx context.Context, c client.Client, dgd *v1beta1.DynamoGraphDeployment, initial map[types.UID]string) {
	t.Helper()
	require.Equal(t, initial, workerDCDIdentities(t, ctx, c, dgd), "cap edit must not create or replace worker DCDs")
	current := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(dgd), current))
	if rollout := current.Status.RollingUpdate; rollout != nil {
		require.Contains(t, []v1beta1.RollingUpdatePhase{"", v1beta1.RollingUpdatePhaseNone, v1beta1.RollingUpdatePhaseCompleted}, rollout.Phase, "cap edit must not start a managed rollout")
	}
}

func waitForPowerCaps(t *testing.T, ctx context.Context, c client.Client, dgd *v1beta1.DynamoGraphDeployment, cap string) []corev1.Pod {
	t.Helper()
	var matching []corev1.Pod
	require.NoError(t, wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		pods := &corev1.PodList{}
		if err := c.List(ctx, pods, client.InNamespace(dgd.Namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}); err != nil {
			return false, err
		}
		if len(pods.Items) != 2 {
			return false, nil
		}
		for _, pod := range pods.Items {
			if pod.Annotations[consts.KubeAnnotationGPUPowerLimit] != cap || pod.Status.Phase != corev1.PodRunning {
				return false, nil
			}
		}
		current := &v1beta1.DynamoGraphDeployment{}
		if err := c.Get(ctx, client.ObjectKeyFromObject(dgd), current); err != nil {
			return false, err
		}
		if current.Annotations[consts.AnnotationCurrentWorkerHashV2] == "" && current.Annotations[consts.AnnotationCurrentWorkerHash] == "" {
			return false, nil
		}

		// Preservation checks are meaningful only after the workload controller has processed this generation.
		if current.Status.ObservedGeneration != current.Generation {
			return false, nil
		}
		condition := meta.FindStatusCondition(current.Status.Conditions, "PowerCapPropagationReady")
		if condition == nil || condition.Status != metav1.ConditionTrue || condition.ObservedGeneration != current.Generation {
			return false, nil
		}
		matching = pods.Items
		return true, nil
	}))
	return matching
}
