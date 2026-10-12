/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/yaml"
)

func TestDynamoPoolRelayEnvtest(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("set KUBEBUILDER_ASSETS to run the real API-server relay test")
	}

	t.Log("Install the generated single-version CRD on an isolated API server")
	raw, err := os.ReadFile("../../../config/crd/bases/nvidia.com_dynamopoolrelays.yaml")
	require.NoError(t, err)
	crd := &apiextensionsv1.CustomResourceDefinition{}
	require.NoError(t, yaml.Unmarshal(raw, crd))
	require.Equal(t, "dynamopoolrelays.nvidia.com", crd.Name)
	require.Equal(t, "DynamoPoolRelay", crd.Spec.Names.Kind)
	require.Equal(t, []string{"dprelay"}, crd.Spec.Names.ShortNames)
	dgdRaw, err := os.ReadFile("../../../config/crd/bases/nvidia.com_dynamographdeployments.yaml")
	require.NoError(t, err)
	dgdCRD := &apiextensionsv1.CustomResourceDefinition{}
	require.NoError(t, yaml.Unmarshal(dgdRaw, dgdCRD))
	testEnv := &envtest.Environment{CRDs: []*apiextensionsv1.CustomResourceDefinition{crd, dgdCRD}}
	t.Cleanup(func() { require.NoError(t, testEnv.Stop()) })
	cfg, err := testEnv.Start()
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, v1alpha1.AddToScheme(scheme))
	require.NoError(t, v1beta1.AddToScheme(scheme))
	c, err := client.New(cfg, client.Options{Scheme: scheme})
	require.NoError(t, err)
	ctx := context.Background()
	namespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "relay-test"}}
	require.NoError(t, c.Create(ctx, namespace))

	t.Log("Admission validates required source objects and preserves nested selection")
	for _, tc := range []struct {
		name    string
		spec    map[string]any
		invalid bool
	}{
		{name: "missing-sources", invalid: true},
		{name: "null-sources", spec: map[string]any{"sources": nil}, invalid: true},
		{name: "missing-dgds", spec: map[string]any{"sources": map[string]any{}}, invalid: true},
		{name: "null-dgds", spec: map[string]any{"sources": map[string]any{"dgds": nil}}, invalid: true},
		{name: "empty-refs", spec: map[string]any{"sources": map[string]any{"dgds": map[string]any{"refs": []any{}}}}, invalid: true},
		{name: "all-dgds", spec: map[string]any{"sources": map[string]any{"dgds": map[string]any{"labelSelector": map[string]any{}}}}},
		{name: "refs-only", spec: map[string]any{"sources": map[string]any{"dgds": map[string]any{"refs": []any{map[string]any{"name": "model"}}}}}},
		{name: "selector-and-refs", spec: map[string]any{"sources": map[string]any{"dgds": map[string]any{
			"labelSelector": map[string]any{"matchLabels": map[string]any{"serve": "yes"}},
			"refs":          []any{map[string]any{"name": "model"}},
		}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Submit the wire-format selection to the real API server")
			spec := map[string]any{
				"clusterId": "dc-a",
				"podTemplate": map[string]any{"spec": map[string]any{"containers": []any{
					map[string]any{"name": "main", "image": "example/relay:v1"},
				}}},
			}
			for key, value := range tc.spec {
				spec[key] = value
			}
			relay := &unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "nvidia.com/v1alpha1", "kind": "DynamoPoolRelay",
				"metadata": map[string]any{"name": tc.name, "namespace": namespace.Name},
				"spec":     spec,
			}}
			err := c.Create(ctx, relay)
			if tc.invalid {
				require.True(t, apierrors.IsInvalid(err), "expected invalid selection, got %v", err)
				return
			}
			require.NoError(t, err)

			t.Log("Read the persisted selection and ensure the schema retained both mechanisms")
			require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(relay), relay))
			sources, found, err := unstructured.NestedMap(relay.Object, "spec", "sources")
			require.NoError(t, err)
			require.True(t, found)
			require.Equal(t, tc.spec["sources"], sources)
			require.NoError(t, c.Delete(ctx, relay))
		})
	}

	t.Log("Admission rejects missing selectors, duplicate refs, and invalid DGD names")
	for _, tc := range []struct {
		name string
		refs []v1alpha1.DynamoPoolRelayDGDRef
	}{
		{name: "missing-selection"},
		{name: "duplicate-refs", refs: []v1alpha1.DynamoPoolRelayDGDRef{{Name: "a"}, {Name: "a"}}},
		{name: "empty-name", refs: []v1alpha1.DynamoPoolRelayDGDRef{{Name: ""}}},
		{name: "invalid-name", refs: []v1alpha1.DynamoPoolRelayDGDRef{{Name: "a,b"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Submit an invalid selection to the real API server")
			relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: tc.name, Namespace: namespace.Name}, Spec: v1alpha1.DynamoPoolRelaySpec{ClusterID: "dc-a", PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}}}}, Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{Refs: tc.refs}}}}
			require.True(t, apierrors.IsInvalid(c.Create(ctx, relay)))
		})
	}

	t.Log("Admission requires main and images and rejects duplicate container names")
	for _, tc := range []struct {
		name       string
		containers []corev1.Container
	}{
		{name: "no-main", containers: []corev1.Container{{Name: "sidecar", Image: "example/sidecar:v1"}}},
		{name: "no-image", containers: []corev1.Container{{Name: "main"}}},
		{name: "no-containers"},
		{name: "duplicate-main", containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}, {Name: "main", Image: "example/relay:v1"}}},
		{name: "sidecar-no-image", containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}, {Name: "sidecar"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			invalid := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: tc.name, Namespace: namespace.Name}, Spec: v1alpha1.DynamoPoolRelaySpec{ClusterID: "dc-a", Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{LabelSelector: &metav1.LabelSelector{}}}, PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: tc.containers}}}}
			require.True(t, apierrors.IsInvalid(c.Create(ctx, invalid)))
		})
	}
	t.Log("Apply defaults and reject changes to a persisted cluster identity")
	relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "dc-a", Namespace: namespace.Name}, Spec: v1alpha1.DynamoPoolRelaySpec{
		ClusterID: "dc-a", PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}}}}, Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{LabelSelector: &metav1.LabelSelector{}}},
	}}
	relay.Spec.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("512Mi")}}
	require.NoError(t, c.Create(ctx, relay))
	require.Equal(t, int32(2), *relay.Spec.Replicas)
	zero := relay.DeepCopy()
	zero.Spec.Replicas = ptr.To(int32(0))
	require.True(t, apierrors.IsInvalid(c.Update(ctx, zero)))
	changed := relay.DeepCopy()
	changed.Spec.ClusterID = "dc-b"
	require.True(t, apierrors.IsInvalid(c.Update(ctx, changed)))

	t.Log("Reconcile real resources and prove defaulting does not create a patch loop")
	reconciler := &relayReconciler{Client: c, config: &configv1alpha1.OperatorConfiguration{}}
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)}
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	key := client.ObjectKey{Name: relayResourceName(relay.Name), Namespace: namespace.Name}
	deployment := &appsv1.Deployment{}
	require.NoError(t, c.Get(ctx, key, deployment))
	generation, version := deployment.Generation, deployment.ResourceVersion
	require.Equal(t, "512Mi", deployment.Spec.Template.Spec.Containers[0].Resources.Requests.Memory().String())
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, generation, deployment.Generation)
	require.Equal(t, version, deployment.ResourceVersion)

	t.Log("API dry-run rejects invalid main resources without altering the Deployment")
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	savedResources := relay.Spec.PodTemplate.Spec.Containers[0].Resources.DeepCopy()
	relay.Spec.PodTemplate.Spec.Containers[0].Resources.Requests = corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("-1")}
	require.NoError(t, c.Update(ctx, relay))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.True(t, meta.IsStatusConditionFalse(relay.Status.Conditions, "ConfigurationResolved"))
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, version, deployment.ResourceVersion)
	relay.Spec.PodTemplate.Spec.Containers[0].Resources = *savedResources
	require.NoError(t, c.Update(ctx, relay))
	t.Log("Mark the current workload available, then verify an image update clears Ready")
	deployment.Status = appsv1.DeploymentStatus{ObservedGeneration: deployment.Generation, Replicas: 2, UpdatedReplicas: 2, ReadyReplicas: 2, AvailableReplicas: 2}
	require.NoError(t, c.Status().Update(ctx, deployment))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.True(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready"))
	relay.Spec.PodTemplate.Spec.Containers[0].Image = "example/relay:v2"
	require.NoError(t, c.Update(ctx, relay))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.False(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready"))
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Greater(t, deployment.Generation, generation)
	require.Equal(t, "example/relay:v2", deployment.Spec.Template.Spec.Containers[0].Image)

	t.Log("Start the real manager with cached clients and registered dependency watches")
	relay.Spec.Sources.DGDs.LabelSelector = &metav1.LabelSelector{MatchLabels: map[string]string{"serve": "yes"}}
	require.NoError(t, c.Update(ctx, relay))
	managerOptions := ctrl.Options{Scheme: scheme, Metrics: metricsserver.Options{BindAddress: "0"}, HealthProbeBindAddress: "0"}
	require.NoError(t, podcache.Configure(&managerOptions.Cache))
	mgr, err := ctrl.NewManager(cfg, managerOptions)
	require.NoError(t, err)
	require.NoError(t, Setup(mgr, &configv1alpha1.OperatorConfiguration{}, &commoncontroller.RuntimeConfig{}))
	managerCtx, cancel := context.WithCancel(ctx)
	managerDone := make(chan error, 1)
	go func() { managerDone <- mgr.Start(managerCtx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-managerDone:
			require.NoError(t, err)
		case <-time.After(10 * time.Second):
			t.Error("Relay manager did not stop")
		}
	})
	require.True(t, mgr.GetCache().WaitForCacheSync(managerCtx))

	t.Log("A selected DGD with a missing Secret becomes unresolved through its watch")
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: namespace.Name, Labels: map[string]string{"serve": "yes"}}, Spec: v1beta1.DynamoGraphDeploymentSpec{
		Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.5.0", Env: []corev1.EnvVar{
			{Name: "DYN_EVENT_PLANE", Value: "nats"},
			{Name: "NATS_SERVER", Value: "nats://test:4222"},
			{Name: "NATS_AUTH_TOKEN", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "auth"}, Key: "token"}}},
		}}}}}}},
	}}

	t.Log("A backend envFrom Secret is indexed without inheriting its contents")
	backend := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "backend", Namespace: namespace.Name}, Data: map[string][]byte{"HF_TOKEN": []byte("private-backend")}}
	require.NoError(t, c.Create(ctx, backend))
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].EnvFrom = []corev1.EnvFromSource{{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: backend.Name}}}}
	require.NoError(t, c.Create(ctx, dgd))
	require.Eventually(t, func() bool {
		var observed v1alpha1.DynamoPoolRelay
		return c.Get(ctx, request.NamespacedName, &observed) == nil && observed.Status.SelectedDGDs == 1 && meta.IsStatusConditionFalse(observed.Status.Conditions, "ConfigurationResolved")
	}, 10*time.Second, 50*time.Millisecond)

	t.Log("Creating the indexed Secret unblocks reconciliation without a manual reconcile")
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "auth", Namespace: namespace.Name}, Data: map[string][]byte{"token": []byte("test-token-a")}}
	require.NoError(t, c.Create(ctx, secret))
	require.Eventually(t, func() bool {
		var observed v1alpha1.DynamoPoolRelay
		return c.Get(ctx, request.NamespacedName, &observed) == nil && meta.IsStatusConditionTrue(observed.Status.Conditions, "ConfigurationResolved") && c.Get(ctx, key, deployment) == nil && deployment.Generation > generation+1
	}, 10*time.Second, 50*time.Millisecond)
	generation = deployment.Generation
	beforeRotation := deployment.Spec.Template.DeepCopy()

	t.Log("An envFrom key change requeues the Relay and blocks hidden transport settings")
	backend.Data["NATS_SERVER"] = []byte("private-transport")
	require.NoError(t, c.Update(ctx, backend))
	require.Eventually(t, func() bool {
		var observed v1alpha1.DynamoPoolRelay
		return c.Get(ctx, request.NamespacedName, &observed) == nil && meta.IsStatusConditionFalse(observed.Status.Conditions, "ConfigurationResolved")
	}, 10*time.Second, 50*time.Millisecond)
	delete(backend.Data, "NATS_SERVER")
	require.NoError(t, c.Update(ctx, backend))
	require.Eventually(t, func() bool {
		var observed v1alpha1.DynamoPoolRelay
		return c.Get(ctx, request.NamespacedName, &observed) == nil && meta.IsStatusConditionTrue(observed.Status.Conditions, "ConfigurationResolved")
	}, 10*time.Second, 50*time.Millisecond)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, beforeRotation, &deployment.Spec.Template)

	t.Log("Rotating credentials triggers a rolling update while keeping Secret references")
	secret.Data["token"] = []byte("test-token-b")
	require.NoError(t, c.Update(ctx, secret))
	require.Eventually(t, func() bool {
		return c.Get(ctx, key, deployment) == nil && deployment.Generation > generation
	}, 10*time.Second, 50*time.Millisecond)
	require.NotEqual(t, *beforeRotation, deployment.Spec.Template)
	require.Equal(t, appsv1.RollingUpdateDeploymentStrategyType, deployment.Spec.Strategy.Type)
	for _, entry := range deployment.Spec.Template.Spec.Containers[0].Env {
		if entry.Name == "NATS_AUTH_TOKEN" {
			require.NotNil(t, entry.ValueFrom.SecretKeyRef)
			require.Empty(t, entry.Value)
		}
	}
	generation = deployment.Generation

	t.Log("A label-only deselection empties the projected sources without rolling the process")
	dgdGeneration := dgd.Generation
	dgd.Labels["serve"] = "no"
	require.NoError(t, c.Update(ctx, dgd))
	require.Equal(t, dgdGeneration, dgd.Generation)
	require.Eventually(t, func() bool {
		var observed v1alpha1.DynamoPoolRelay
		var sourcesMap corev1.ConfigMap
		var sources relaySources
		return c.Get(ctx, request.NamespacedName, &observed) == nil && observed.Status.SelectedDGDs == 0 && c.Get(ctx, key, &sourcesMap) == nil && json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources) == nil && len(sources.Sources) == 0
	}, 10*time.Second, 50*time.Millisecond)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, generation, deployment.Generation)

	t.Log("Apply sidecars, pod metadata and security settings through the real manager")
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	relay.Spec.PodTemplate.Annotations = map[string]string{"example.com/managed": "yes"}
	relay.Spec.PodTemplate.Spec.NodeSelector = map[string]string{"zone": "a"}
	relay.Spec.PodTemplate.Spec.SecurityContext = &corev1.PodSecurityContext{FSGroup: ptr.To(int64(101))}
	relay.Spec.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "prepare", Image: "example/init:1", Command: []string{"true"}}}
	relay.Spec.PodTemplate.Spec.Containers = append(relay.Spec.PodTemplate.Spec.Containers, corev1.Container{Name: "envoy", Image: "example/envoy:1", Ports: []corev1.ContainerPort{{Name: "mtls", ContainerPort: 5560}}})
	relay.Spec.Service = &v1alpha1.DynamoPoolRelayServiceSpec{Ports: []corev1.ServicePort{{Name: "grpc", Port: 5560, TargetPort: intstr.FromString("mtls")}}}
	require.NoError(t, c.Update(ctx, relay))
	require.Eventually(t, func() bool {
		var observed v1alpha1.DynamoPoolRelay
		return c.Get(ctx, key, deployment) == nil && len(deployment.Spec.Template.Spec.Containers) == 2 && deployment.Spec.Template.Annotations["example.com/managed"] == "yes" && c.Get(ctx, request.NamespacedName, &observed) == nil && strings.HasSuffix(observed.Status.Endpoint, ":5560")
	}, 10*time.Second, 50*time.Millisecond)
	require.Equal(t, int64(101), *deployment.Spec.Template.Spec.SecurityContext.FSGroup)
	require.Equal(t, "a", deployment.Spec.Template.Spec.NodeSelector["zone"])
	require.Len(t, deployment.Spec.Template.Spec.InitContainers, 1)

	t.Log("An unrelated injected annotation survives a later user template removal")
	beforeInjection := deployment.DeepCopy()
	deployment.Spec.Template.Annotations["example.com/injected"] = "keep"
	require.NoError(t, c.Patch(ctx, deployment, client.MergeFrom(beforeInjection), client.FieldOwner("test-admission")))
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	relay.Spec.PodTemplate.Annotations = nil
	relay.Spec.PodTemplate.Spec.NodeSelector = nil
	relay.Spec.PodTemplate.Spec.SecurityContext = nil
	relay.Spec.PodTemplate.Spec.InitContainers = nil
	relay.Spec.PodTemplate.Spec.Containers = relay.Spec.PodTemplate.Spec.Containers[:1]
	relay.Spec.Service = nil
	require.NoError(t, c.Update(ctx, relay))
	require.Eventually(t, func() bool {
		return c.Get(ctx, key, deployment) == nil && len(deployment.Spec.Template.Spec.Containers) == 1 && len(deployment.Spec.Template.Spec.InitContainers) == 0 && len(deployment.Spec.Template.Spec.NodeSelector) == 0 && deployment.Spec.Template.Annotations["example.com/managed"] == "" && deployment.Spec.Template.Annotations["example.com/injected"] == "keep" && (deployment.Spec.Template.Spec.SecurityContext == nil || deployment.Spec.Template.Spec.SecurityContext.FSGroup == nil)
	}, 10*time.Second, 50*time.Millisecond)
}
