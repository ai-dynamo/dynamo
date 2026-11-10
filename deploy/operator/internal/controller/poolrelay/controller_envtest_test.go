//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/yaml"
)

//nolint:gocyclo // This test checks the same Deployment through defaulting, drift, and field removal.
func TestDeploymentAPIDefaultsDoNotCauseRolloutLoop(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS is needed for the API defaulting test")
	}
	ctx := context.Background()
	environment := &envtest.Environment{}
	config, err := environment.Start()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := environment.Stop(); err != nil {
			t.Error(err)
		}
	})
	scheme := runtime.NewScheme()
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	apiClient, err := client.New(config, client.Options{Scheme: scheme})
	if err != nil {
		t.Fatal(err)
	}
	relayMeta := metav1.ObjectMeta{Name: "green", Namespace: "default", UID: "relay-uid"}
	relay := &v1alpha1.DynamoPoolRelay{
		ObjectMeta: relayMeta,
		Spec:       relaySpec("east"),
	}
	r := &reconciler{Client: apiClient, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	key := types.NamespacedName{Namespace: relay.Namespace, Name: serviceName(relay.Name)}
	var firstResourceVersion string
	for attempt := range 2 {
		_, wanted, err := render(relay, []string{"green-runtime"})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := r.applyDeployment(ctx, relay, wanted); err != nil {
			t.Fatal(err)
		}
		var deployment appsv1.Deployment
		if err := apiClient.Get(ctx, key, &deployment); err != nil {
			t.Fatal(err)
		}
		if deployment.Spec.Template.Spec.DNSPolicy == "" {
			t.Fatal("the API server did not apply Pod defaults")
		}
		if attempt == 0 {
			firstResourceVersion = deployment.ResourceVersion
		} else if deployment.ResourceVersion != firstResourceVersion {
			t.Fatalf("unchanged Deployment was rewritten: %s -> %s", firstResourceVersion, deployment.ResourceVersion)
		}
	}
	t.Log("Restore controller-owned Pod settings with Server-Side Apply")
	var drifted appsv1.Deployment
	if err := apiClient.Get(ctx, key, &drifted); err != nil {
		t.Fatal(err)
	}
	drifted.Spec.Template.Spec.HostNetwork = true
	drifted.Spec.Template.Spec.Volumes[0].ConfigMap.Items = []corev1.KeyToPath{{Key: "wrong", Path: "wrong.json"}}
	if err := apiClient.Update(ctx, &drifted); err != nil {
		t.Fatal(err)
	}
	_, wanted, err := render(relay, []string{"green-runtime"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.applyDeployment(ctx, relay, wanted); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, key, &drifted); err != nil {
		t.Fatal(err)
	}
	if drifted.Spec.Template.Spec.HostNetwork || len(drifted.Spec.Template.Spec.Volumes[0].ConfigMap.Items) != 1 ||
		drifted.Spec.Template.Spec.Volumes[0].ConfigMap.Items[0].Key != "sources.json" {
		t.Fatalf("Server-Side Apply did not restore Relay Pod settings: %+v", drifted.Spec.Template.Spec)
	}
	t.Log("Remove user PodTemplate fields when the resource no longer requests them")
	relay.Spec.PodTemplate.Spec.Containers = append(relay.Spec.PodTemplate.Spec.Containers,
		corev1.Container{Name: "metrics", Image: "metrics:test"})
	relay.Spec.PodTemplate.Spec.Containers[0].Env = append(relay.Spec.PodTemplate.Spec.Containers[0].Env,
		corev1.EnvVar{Name: "DYN_EVENT_PLANE", Value: "zmq"})
	_, wanted, err = render(relay, []string{"green-runtime"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.applyDeployment(ctx, relay, wanted); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, key, &drifted); err != nil {
		t.Fatal(err)
	}
	if len(drifted.Spec.Template.Spec.Containers) != 2 {
		t.Fatalf("user sidecar was not added: %+v", drifted.Spec.Template.Spec.Containers)
	}
	relay.Spec.PodTemplate.Spec.Containers = relay.Spec.PodTemplate.Spec.Containers[:1]
	relay.Spec.PodTemplate.Spec.Containers[0].Env = nil
	_, wanted, err = render(relay, []string{"green-runtime"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.applyDeployment(ctx, relay, wanted); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, key, &drifted); err != nil {
		t.Fatal(err)
	}
	if len(drifted.Spec.Template.Spec.Containers) != 1 ||
		len(drifted.Spec.Template.Spec.Containers[0].Env) != len(wanted.Spec.Template.Spec.Containers[0].Env) {
		t.Fatalf("removed PodTemplate fields remained in Deployment: %+v", drifted.Spec.Template.Spec.Containers)
	}
}

//nolint:gocyclo // This test follows one Relay through publication and withdrawal against the API server.
func TestControllerPublishesAndWithdrawsOneDGDService(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS is needed for the controller API test")
	}
	environment := &envtest.Environment{CRDDirectoryPaths: []string{
		filepath.Join("..", "..", "..", "config", "crd", "bases"),
	}}
	config, err := environment.Start()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := environment.Stop(); err != nil {
			t.Error(err)
		}
	})
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		v1alpha1.AddToScheme, v1beta1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme, rbacv1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatal(err)
		}
	}
	manager, err := ctrl.NewManager(config, ctrl.Options{Scheme: scheme, Metrics: server.Options{BindAddress: "0"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := Setup(manager, &configv1alpha1.OperatorConfiguration{
		Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
	}, &commoncontroller.RuntimeConfig{}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	managerErr := make(chan error, 1)
	go func() { managerErr <- manager.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		if err := <-managerErr; err != nil {
			t.Error(err)
		}
	})
	apiClient, err := client.New(config, client.Options{Scheme: scheme})
	if err != nil {
		t.Fatal(err)
	}
	t.Log("Apply the checked-in Relay sample through the API")
	sample, err := os.ReadFile(filepath.Join("..", "..", "..", "samples", "dynamo-pool-relay.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var relay v1alpha1.DynamoPoolRelay
	if err := yaml.UnmarshalStrict(sample, &relay); err != nil {
		t.Fatal(err)
	}
	key := types.NamespacedName{Namespace: relay.Namespace, Name: relay.Name}
	namespace := &corev1.Namespace{}
	namespace.Name = key.Namespace
	if err := apiClient.Create(ctx, namespace); err != nil {
		t.Fatal(err)
	}
	t.Log("Reject Unicode whitespace at API admission")
	invalidRelay := relay.DeepCopy()
	invalidRelay.Name = "invalid-cluster-id"
	invalidRelay.Spec.ClusterID = "\u00a0east"
	if err := apiClient.Create(ctx, invalidRelay); !apierrors.IsInvalid(err) {
		t.Fatalf("expected invalid cluster ID to be rejected, got %v", err)
	}
	dgdMeta := metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: dgdMeta,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
	}
	if err := apiClient.Create(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Create(ctx, &relay); err != nil {
		t.Fatal(err)
	}
	serviceKey := types.NamespacedName{Namespace: key.Namespace, Name: serviceName(key.Name)}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		var service corev1.Service
		if err := apiClient.Get(ctx, serviceKey, &service); err == nil {
			break
		} else if !apierrors.IsNotFound(err) {
			t.Fatal(err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	var service corev1.Service
	if err := apiClient.Get(ctx, serviceKey, &service); err != nil {
		t.Fatalf("Relay Service was not published: %v", err)
	}
	var deployment appsv1.Deployment
	deadline = time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if err := apiClient.Get(ctx, serviceKey, &deployment); err == nil {
			break
		} else if !apierrors.IsNotFound(err) {
			t.Fatal(err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err := apiClient.Get(ctx, serviceKey, &deployment); err != nil {
		t.Fatalf("Relay Deployment was not published: %v", err)
	}
	if relay.Spec.Replicas == nil || deployment.Spec.Replicas == nil ||
		*deployment.Spec.Replicas != *relay.Spec.Replicas ||
		len(deployment.Spec.Template.Spec.Containers) != 1 ||
		deployment.Spec.Template.Spec.Containers[0].Image != relay.Spec.PodTemplate.Spec.Containers[0].Image {
		t.Fatalf("Relay Deployment does not match the sample: %+v", deployment.Spec)
	}
	if err := apiClient.Delete(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	deadline = time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if err := apiClient.Get(ctx, serviceKey, &service); apierrors.IsNotFound(err) {
			return
		} else if err != nil {
			t.Fatal(err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatal("Relay Service remained discoverable after DGD deletion")
}
