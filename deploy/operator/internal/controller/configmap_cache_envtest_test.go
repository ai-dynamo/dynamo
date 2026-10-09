//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

func TestConfigMapMetadataWatchReadsProfilingPayload(t *testing.T) {
	t.Log("Seed a profiling request and its output before starting the controller")
	env := operatorenv.New(operatorenv.Options{SetupWebhooks: setupProductionWebhooks}).RunT(t)
	ctx := t.Context()
	dgdr := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "configmap-watch", Namespace: env.Namespace()},
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Model: "test-model", Backend: "vllm", AutoApply: ptr.To(false),
		},
	}
	commoncontroller.AddFinalizer(dgdr)
	require.NoError(t, env.Client().Create(ctx, dgdr))
	dgdr.Status.Phase = v1beta1.DGDRPhaseProfiling
	require.NoError(t, env.Client().Status().Update(ctx, dgdr))
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: getProfilingJobName(dgdr), Namespace: env.Namespace()},
		Spec: batchv1.JobSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "profiler", Image: "unused"}}, RestartPolicy: corev1.RestartPolicyNever,
		}}},
	}
	require.NoError(t, env.Client().Create(ctx, job))
	output := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: getOutputConfigMapName(dgdr), Namespace: env.Namespace(),
			Labels: map[string]string{
				v1beta1.LabelDGDRName: dgdr.Name, v1beta1.LabelDGDRNamespace: dgdr.Namespace,
			},
		},
		Data:       map[string]string{"phase": string(v1beta1.ProfilingPhaseSweepingPrefill)},
		BinaryData: map[string][]byte{"archive": {0, 1, 255}},
	}
	require.NoError(t, env.Client().Create(ctx, output))

	t.Log("Start the real controller with ConfigMap payload reads bypassing its metadata cache")
	mgr, err := ctrl.NewManager(env.RESTConfig(), ctrl.Options{
		Scheme: env.Scheme(), Metrics: metricsserver.Options{BindAddress: "0"},
		Cache: cache.Options{ReaderFailOnMissingInformer: true},
		Client: client.Options{Cache: &client.CacheOptions{
			DisableFor: []client.Object{&corev1.ConfigMap{}},
		}},
	})
	require.NoError(t, err)
	reconciler := &DynamoGraphDeploymentRequestReconciler{
		Client: mgr.GetClient(), APIReader: mgr.GetAPIReader(),
		Config: env.OperatorConfig(), RuntimeConfig: env.RuntimeConfig(),
		Recorder: events.NewFakeRecorder(100),
	}
	require.NoError(t, reconciler.SetupWithManager(mgr))
	managerCtx, cancel := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- mgr.Start(managerCtx) }()
	t.Cleanup(func() {
		cancel()
		require.NoError(t, <-done)
	})
	key := client.ObjectKeyFromObject(dgdr)
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		observed := &v1beta1.DynamoGraphDeploymentRequest{}
		assert.NoError(c, env.Client().Get(ctx, key, observed))
		assert.Equal(c, v1beta1.ProfilingPhaseSweepingPrefill, observed.Status.ProfilingPhase)
	}, 15*time.Second, 50*time.Millisecond)

	t.Log("A payload-only update still wakes the request through the metadata watch")
	output.Data["phase"] = string(v1beta1.ProfilingPhaseSweepingDecode)
	require.NoError(t, env.Client().Update(ctx, output))
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		metadata := &metav1.PartialObjectMetadata{
			TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "ConfigMap"},
		}
		assert.NoError(c, mgr.GetCache().Get(ctx, client.ObjectKeyFromObject(output), metadata))
		assert.Equal(c, output.ResourceVersion, metadata.ResourceVersion)
		observed := &v1beta1.DynamoGraphDeploymentRequest{}
		assert.NoError(c, env.Client().Get(ctx, key, observed))
		assert.Equal(c, v1beta1.ProfilingPhaseSweepingDecode, observed.Status.ProfilingPhase)
	}, 15*time.Second, 50*time.Millisecond)

	t.Log("Full reads preserve the payload without creating a second, typed informer")
	configMapKey := client.ObjectKeyFromObject(output)
	observed := &corev1.ConfigMap{}
	require.NoError(t, mgr.GetClient().Get(ctx, configMapKey, observed))
	require.Equal(t, output.Data, observed.Data)
	require.Equal(t, output.BinaryData, observed.BinaryData)
	listed := &corev1.ConfigMapList{}
	require.NoError(t, mgr.GetClient().List(ctx, listed, client.InNamespace(env.Namespace()),
		client.MatchingLabels{v1beta1.LabelDGDRName: dgdr.Name}))
	require.Len(t, listed.Items, 1)
	require.Equal(t, output.Data, listed.Items[0].Data)
	require.Equal(t, output.BinaryData, listed.Items[0].BinaryData)
	var notCached *cache.ErrResourceNotCached
	require.ErrorAs(t, mgr.GetCache().Get(ctx, configMapKey, &corev1.ConfigMap{}), &notCached)
}
