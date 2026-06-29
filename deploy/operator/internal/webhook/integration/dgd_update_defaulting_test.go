/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package integration

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	webhookregistration "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/registration"
	"github.com/stretchr/testify/require"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

const webhookTestOperatorVersion = "1.3.0"

func TestWebhookFixtureMatchesHelmTemplate(t *testing.T) {
	if _, err := exec.LookPath("helm"); err != nil {
		t.Skip("helm is not installed")
	}

	cmd := exec.Command(
		"helm", "template", "dynamo-operator",
		filepath.Join("..", "..", "..", "..", "helm", "charts", "platform", "components", "operator"),
		"--namespace", "dynamo-system",
		"--show-only", "templates/webhook-configuration.yaml",
		"--set", "discoveryBackend=kubernetes",
	)
	got, err := cmd.CombinedOutput()
	require.NoError(t, err, string(got))

	want, err := os.ReadFile(filepath.Join("testdata", "webhook-configuration.yaml"))
	require.NoError(t, err)
	require.Equal(t, string(want), string(got), "regenerate testdata/webhook-configuration.yaml with helm template")
}

func TestDGDUpdateRunsHelmInstalledMutatingWebhook(t *testing.T) {
	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, admissionregistrationv1.AddToScheme(scheme))
	require.NoError(t, apiextensionsv1.AddToScheme(scheme))

	testEnv := &envtest.Environment{
		CRDDirectoryPaths: []string{filepath.Join("..", "..", "..", "config", "crd", "bases")},
		WebhookInstallOptions: envtest.WebhookInstallOptions{
			Paths: []string{filepath.Join("testdata", "webhook-configuration.yaml")},
		},
		ErrorIfCRDPathMissing: true,
	}
	t.Cleanup(func() {
		require.NoError(t, testEnv.Stop())
	})

	cfg, err := testEnv.Start()
	require.NoError(t, err)
	require.NotNil(t, cfg)

	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:  scheme,
		Metrics: metricsserver.Options{BindAddress: "0"},
		WebhookServer: webhook.NewServer(webhook.Options{
			Host:    testEnv.WebhookInstallOptions.LocalServingHost,
			Port:    testEnv.WebhookInstallOptions.LocalServingPort,
			CertDir: testEnv.WebhookInstallOptions.LocalServingCertDir,
		}),
	})
	require.NoError(t, err)

	require.NoError(t, webhookregistration.RegisterHandlers(
		mgr,
		&configv1alpha1.OperatorConfiguration{},
		&commoncontroller.RuntimeConfig{GroveEnabled: true},
		webhookTestOperatorVersion,
	))

	managerCtx, cancelManager := context.WithCancel(context.Background())
	managerDone := make(chan error, 1)
	t.Cleanup(func() {
		cancelManager()
		select {
		case err := <-managerDone:
			require.NoError(t, err)
		case <-time.After(10 * time.Second):
			t.Fatal("manager did not stop")
		}
	})
	go func() {
		managerDone <- mgr.Start(managerCtx)
	}()
	waitForWebhookServer(t, testEnv.WebhookInstallOptions.LocalServingHost, testEnv.WebhookInstallOptions.LocalServingPort)

	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme})
	require.NoError(t, err)

	ctx := context.Background()
	name := "dgd-minavailable-update"
	create := &nvidiacomv1alpha1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoGraphDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {Replicas: ptr.To(int32(1))},
			},
		},
	}
	require.NoError(t, k8sClient.Create(ctx, create))

	stored := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	require.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Namespace: "default", Name: name}, stored))
	require.NotNil(t, stored.Spec.Services["worker"].MinAvailable)
	require.Equal(t, int32(1), *stored.Spec.Services["worker"].MinAvailable)

	// Simulate re-applying an older manifest after upgrade: the stored object has
	// the defaulted minAvailable value, but the incoming update omits the field.
	update := &nvidiacomv1alpha1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoGraphDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       "default",
			ResourceVersion: stored.ResourceVersion,
			Annotations: map[string]string{
				"test.nvidia.com/update": "true",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {Replicas: ptr.To(int32(1))},
			},
		},
	}
	require.NoError(t, k8sClient.Update(ctx, update))

	updated := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	require.NoError(t, k8sClient.Get(ctx, client.ObjectKey{Namespace: "default", Name: name}, updated))
	require.Equal(t, "true", updated.Annotations["test.nvidia.com/update"])
	require.NotNil(t, updated.Spec.Services["worker"].MinAvailable)
	require.Equal(t, int32(1), *updated.Spec.Services["worker"].MinAvailable)
}

func waitForWebhookServer(t *testing.T, host string, port int) {
	t.Helper()
	addr := net.JoinHostPort(host, fmt.Sprintf("%d", port))
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 200*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("webhook server did not listen on %s", addr)
}
