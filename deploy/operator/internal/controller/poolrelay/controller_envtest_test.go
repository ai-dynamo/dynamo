//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"maps"
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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
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
		_, wanted, err := render(relay, []string{"green-runtime"}, configv1alpha1.GlobalDiscoveryConfiguration{})
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
	_, wanted, err := render(relay, []string{"green-runtime"}, configv1alpha1.GlobalDiscoveryConfiguration{})
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
	_, wanted, err = render(relay, []string{"green-runtime"}, configv1alpha1.GlobalDiscoveryConfiguration{})
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
	_, wanted, err = render(relay, []string{"green-runtime"}, configv1alpha1.GlobalDiscoveryConfiguration{})
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

func TestAnOperatorUpgradeRollsOnlyRelaysThatDriftedFromTheirSpec(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS is needed for the operator upgrade test")
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
	for _, add := range []func(*runtime.Scheme) error{v1alpha1.AddToScheme, appsv1.AddToScheme} {
		if err := add(scheme); err != nil {
			t.Fatal(err)
		}
	}
	apiClient, err := client.New(config, client.Options{Scheme: scheme})
	if err != nil {
		t.Fatal(err)
	}
	r := &reconciler{Client: apiClient, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	none := configv1alpha1.GlobalDiscoveryConfiguration{}
	for _, tc := range []struct {
		name string
		// createdEnv is env that the relay had when release 1.7.0 created its
		// Deployment, and has dropped since.
		createdEnv []corev1.EnvVar
		wantRoll   bool
	}{
		{name: "unchanged"},
		{name: "drifted", createdEnv: []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "zmq"}}, wantRoll: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Release 1.7.0 creates the Deployment, then applies the relay's current spec")
			relay := &v1alpha1.DynamoPoolRelay{
				Name: tc.name, Namespace: "default", UID: types.UID(tc.name + "-uid"),
				Spec: relaySpec("east"),
			}
			relay.Spec.PodTemplate.Spec.Containers[0].Env = tc.createdEnv
			_, created, err := render(relay, []string{"green-runtime"}, none)
			if err != nil {
				t.Fatal(err)
			}
			if err := controllerutil.SetControllerReference(relay, created, scheme); err != nil {
				t.Fatal(err)
			}
			if err := apiClient.Create(ctx, created, client.FieldOwner("dynamo-pool-relay")); err != nil {
				t.Fatal(err)
			}
			relay.Spec.PodTemplate.Spec.Containers[0].Env = nil
			_, current, err := render(relay, []string{"green-runtime"}, none)
			if err != nil {
				t.Fatal(err)
			}
			applied := current.DeepCopy()
			if err := controllerutil.SetControllerReference(relay, applied, scheme); err != nil {
				t.Fatal(err)
			}
			applied.TypeMeta = metav1.TypeMeta{APIVersion: appsv1.SchemeGroupVersion.String(), Kind: "Deployment"}
			fields, err := runtime.DefaultUnstructuredConverter.ToUnstructured(applied)
			if err != nil {
				t.Fatal(err)
			}
			if err := apiClient.Apply(ctx, client.ApplyConfigurationFromUnstructured(&unstructured.Unstructured{Object: fields}),
				client.FieldOwner("dynamo-pool-relay"), client.ForceOwnership); err != nil {
				t.Fatal(err)
			}
			key := client.ObjectKeyFromObject(current)
			var before appsv1.Deployment
			if err := apiClient.Get(ctx, key, &before); err != nil {
				t.Fatal(err)
			}

			t.Log("This operator keeps an unchanged Deployment and drops what the spec no longer sets")
			if _, err := r.applyDeployment(ctx, relay, current); err != nil {
				t.Fatal(err)
			}
			var after appsv1.Deployment
			if err := apiClient.Get(ctx, key, &after); err != nil {
				t.Fatal(err)
			}
			if rolled := after.Generation != before.Generation; rolled != tc.wantRoll ||
				len(after.Spec.Template.Spec.Containers[0].Env) != len(current.Spec.Template.Spec.Containers[0].Env) {
				t.Fatalf("rolled = %v, want %v; env %v", rolled, tc.wantRoll, after.Spec.Template.Spec.Containers[0].Env)
			}
		})
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
		Discovery:       configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
		GlobalDiscovery: configv1alpha1.GlobalDiscoveryConfiguration{URL: "http://discovery:8080/", Region: "us-east-1"},
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
	t.Log("The API server applies the frontends' URL rule and rejects a repeated frontend")
	for _, frontends := range [][]string{{"http://frontend.123:8000/"}, {"http://green:8000", "http://green:8000"}} {
		invalid := relay.DeepCopy()
		invalid.Name = "invalid-frontends"
		invalid.Spec.GlobalDiscovery = &v1alpha1.DynamoPoolRelayGlobalDiscovery{Frontends: frontends}
		if err := apiClient.Create(ctx, invalid, client.DryRunAll); !apierrors.IsInvalid(err) {
			t.Fatalf("expected frontends %q to be rejected, got %v", frontends, err)
		}
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

	t.Log("Create the Relay with a globalDiscovery block")
	relay.Spec.GlobalDiscovery = &v1alpha1.DynamoPoolRelayGlobalDiscovery{
		Frontends: []string{"http://frontend-a.example.com:8000", "https://frontend-b.example.com"},
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
	t.Log("The Relay registers the pool with the operator's settings")
	want := map[string]string{
		"DYN_RELAY_GLOBAL_DISCOVERY_URL":   "http://discovery:8080/",
		"DYN_RELAY_ROUTING_POOL":           relay.Spec.ClusterID + "/" + key.Namespace + "/" + key.Name,
		"DYN_RELAY_ROUTING_POOL_REGION":    "us-east-1",
		"DYN_RELAY_ROUTING_POOL_FRONTENDS": "http://frontend-a.example.com:8000,https://frontend-b.example.com",
	}
	if got := registrationEnvOf(t, &deployment); !maps.Equal(got, want) {
		t.Fatalf("Relay registers with %v, want %v", got, want)
	}

	t.Log("Removing the block stops the registration of a Relay that had it from creation")
	registered := relay.DeepCopy()
	relay.Spec.GlobalDiscovery = nil
	if err := apiClient.Patch(ctx, &relay, client.MergeFrom(registered)); err != nil {
		t.Fatal(err)
	}
	deadline = time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if err := apiClient.Get(ctx, serviceKey, &deployment); err != nil {
			t.Fatal(err)
		}
		if len(registrationEnvOf(t, &deployment)) == 0 {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if got := registrationEnvOf(t, &deployment); len(got) != 0 {
		t.Fatalf("the Relay still registers after the block was removed: %v", got)
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
