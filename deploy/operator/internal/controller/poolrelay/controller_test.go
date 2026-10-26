/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

//nolint:gocyclo // This scenario verifies one Relay's dependent lifecycle transitions.
func TestOneDGDRelayLifecycle(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		v1alpha1.AddToScheme, v1beta1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme, rbacv1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatal(err)
		}
	}
	key := types.NamespacedName{Namespace: "inference", Name: "green"}
	relayMeta := metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace, UID: "relay-uid"}
	dgdMeta := metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace}
	relay := &v1alpha1.DynamoPoolRelay{
		ObjectMeta: relayMeta,
		Spec:       v1alpha1.DynamoPoolRelaySpec{ClusterID: "east", Image: "relay:test"},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: dgdMeta,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend},
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
		Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{
			"decode": {RuntimeNamespace: "inference-green-hash"},
		}},
	}
	client := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(relay, dgd, &appsv1.Deployment{}).
		WithObjects(relay, dgd).Build()
	r := &reconciler{Client: client, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	reconcile := func() {
		t.Helper()
		if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
			t.Fatal(err)
		}
	}
	childKey := types.NamespacedName{Namespace: key.Namespace, Name: "green-pool-relay"}

	reconcile()
	var service corev1.Service
	if err := client.Get(ctx, childKey, &service); err != nil {
		t.Fatal(err)
	}
	if service.Spec.Ports[0].Port != relayPort || service.Spec.Selector["nvidia.com/relay-uid"] != string(relay.UID) {
		t.Fatalf("unexpected Relay service: %+v", service.Spec)
	}
	var deployment appsv1.Deployment
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	if *deployment.Spec.Replicas != 2 || deployment.Spec.Template.Spec.Containers[0].Image != "relay:test" {
		t.Fatalf("unexpected Relay deployment: %+v", deployment.Spec)
	}
	container := deployment.Spec.Template.Spec.Containers[0]
	if !slices.Equal(container.Command, []string{"python3", "-m", "dynamo.pool_relay"}) ||
		!slices.Equal(container.Args, []string{"--cluster-id", "east", "--sources-file", sourcesFilePath, "--bind", "0.0.0.0:5561"}) {
		t.Fatalf("unexpected Relay process command: %v %v", container.Command, container.Args)
	}
	t.Log("Relay listens for frontend load in the DGD base namespace while workers use a suffixed namespace")
	foundNamespace := false
	for _, item := range deployment.Spec.Template.Spec.Containers[0].Env {
		if item.Name == "DYN_NAMESPACE" {
			foundNamespace = true
			if item.Value != "inference-green" {
				t.Fatalf("unexpected Relay runtime namespace: %q", item.Value)
			}
		}
	}
	if !foundNamespace {
		t.Fatal("Relay runtime namespace was not configured")
	}
	var sources corev1.ConfigMap
	if err := client.Get(ctx, childKey, &sources); err != nil {
		t.Fatal(err)
	}
	var sourceDocument sourcesDocument
	if err := json.Unmarshal([]byte(sources.Data["sources.json"]), &sourceDocument); err != nil {
		t.Fatal(err)
	}
	if len(sourceDocument.Sources) != 1 || sourceDocument.Sources[0].Namespace != "inference-green-hash" || sourceDocument.ConnectionRevision == "" {
		t.Fatalf("unexpected Relay source set: %s", sources.Data["sources.json"])
	}
	connectionRevision := sourceDocument.ConnectionRevision
	if !slices.ContainsFunc(container.Env, func(item corev1.EnvVar) bool {
		return item.Name == "DYN_RELAY_CONNECTION_REVISION" && item.Value == connectionRevision
	}) {
		t.Fatal("Relay Pod and source file have different connection revisions")
	}
	serviceVersion, deploymentVersion := service.ResourceVersion, deployment.ResourceVersion
	reconcile()
	if err := client.Get(ctx, childKey, &service); err != nil {
		t.Fatal(err)
	}
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	if service.ResourceVersion != serviceVersion || deployment.ResourceVersion != deploymentVersion {
		t.Fatal("unchanged resources were rewritten on the second reconcile")
	}
	if err := client.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	dgd.Status.Components["decode"] = v1beta1.ComponentReplicaStatus{RuntimeNamespace: "green-runtime-next"}
	if err := client.Status().Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &sources); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(sources.Data["sources.json"]), &sourceDocument); err != nil {
		t.Fatal(err)
	}
	if len(sourceDocument.Sources) != 1 || sourceDocument.Sources[0].Namespace != "green-runtime-next" || sourceDocument.ConnectionRevision != connectionRevision {
		t.Fatalf("Relay did not follow the worker runtime namespace: %s", sources.Data["sources.json"])
	}
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	if deployment.ResourceVersion != deploymentVersion {
		t.Fatal("changing the source file unexpectedly rolled the Relay Deployment")
	}
	t.Log("Restore a source volume that no longer projects sources.json")
	deployment.Spec.Template.Spec.Volumes[0].ConfigMap.Items = []corev1.KeyToPath{{Key: "wrong", Path: "wrong.json"}}
	if err := client.Update(ctx, &deployment); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	if len(deployment.Spec.Template.Spec.Volumes[0].ConfigMap.Items) != 0 {
		t.Fatal("Relay source projection was not restored")
	}
	t.Log("Restore listener safety settings after manual drift")
	deployment.Spec.Template.Spec.HostNetwork = true
	if err := client.Update(ctx, &deployment); err != nil {
		t.Fatal(err)
	}
	if err := client.Get(ctx, childKey, &service); err != nil {
		t.Fatal(err)
	}
	service.Spec.PublishNotReadyAddresses = true
	if err := client.Update(ctx, &service); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	if deployment.Spec.Template.Spec.HostNetwork {
		t.Fatal("Relay Pod still uses the host network")
	}
	if err := client.Get(ctx, childKey, &service); err != nil {
		t.Fatal(err)
	}
	if service.Spec.PublishNotReadyAddresses {
		t.Fatal("Relay Service still publishes unready Pod addresses")
	}
	t.Log("Pair the source file with the new Pod connection revision")
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	relay.Spec.Env = []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "zmq"}}
	if err := client.Update(ctx, relay); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &sources); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(sources.Data["sources.json"]), &sourceDocument); err != nil {
		t.Fatal(err)
	}
	if sourceDocument.ConnectionRevision == connectionRevision {
		t.Fatal("connection change retained the old source revision")
	}
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(deployment.Spec.Template.Spec.Containers[0].Env, func(item corev1.EnvVar) bool {
		return item.Name == "DYN_RELAY_CONNECTION_REVISION" && item.Value == sourceDocument.ConnectionRevision
	}) {
		t.Fatal("new Relay Pod does not match the updated source file")
	}

	t.Log("Withdraw the Service when the DGD switches to container-mode discovery")
	if err := client.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	dgd.Annotations = map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}
	if err := client.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &service); err == nil {
		t.Fatal("Relay Service remained published for container-mode discovery")
	}
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); condition == nil || condition.Reason != "UnsupportedDiscovery" {
		t.Fatalf("unexpected status for container-mode discovery: %+v", relay.Status)
	}
	if err := client.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	dgd.Annotations = nil
	if err := client.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	reconcile()

	// The replica count is independent of local frontend replicas.
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	three := int32(3)
	relay.Spec.Replicas = &three
	if err := client.Update(ctx, relay); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	if *deployment.Spec.Replicas != 3 {
		t.Fatalf("expected 3 Relay replicas, got %d", *deployment.Spec.Replicas)
	}

	// A DGD finalizer keeps the object readable while it terminates. Withdraw
	// its listener immediately, before final deletion completes.
	if err := client.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	dgd.Finalizers = []string{"test.nvidia.com/protect"}
	if err := client.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	if err := client.Delete(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	if err := client.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	if dgd.DeletionTimestamp.IsZero() {
		t.Fatal("DGD was removed before its finalizer completed")
	}
	reconcile()
	if err := client.Get(ctx, childKey, &service); err == nil {
		t.Fatal("Service remained while the DGD was terminating")
	}
	if err := client.Get(ctx, childKey, &deployment); err == nil {
		t.Fatal("Deployment remained while the DGD was terminating")
	}
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); condition == nil || condition.Reason != "DGDTerminating" {
		t.Fatalf("unexpected status for terminating DGD: %+v", relay.Status)
	}

	// Losing the named DGD keeps the listener withdrawn.
	dgd.Finalizers = nil
	if err := client.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &service); err == nil {
		t.Fatal("Service remained after the DGD disappeared")
	}
	if err := client.Get(ctx, childKey, &deployment); err == nil {
		t.Fatal("Deployment remained after the DGD disappeared")
	}
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); relay.Status.Endpoint != "" || condition == nil || condition.Reason != "DGDNotFound" || meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready") {
		t.Fatalf("Relay still advertises readiness after DGD deletion: %+v", relay.Status)
	}

	// A shared global runtime namespace cannot identify one DGD by namespace alone.
	globalDGD := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: dgdMeta,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker, GlobalDynamoNamespace: true},
		}},
		Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{
			"decode": {RuntimeNamespace: "inference-green-previoushash"},
		}},
	}
	if err := client.Create(ctx, globalDGD); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &service); err == nil {
		t.Fatal("Relay Service was published for a shared global namespace")
	}
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); condition == nil || condition.Reason != "UnsupportedSourceScope" {
		t.Fatalf("unexpected status for shared global namespace: %+v", relay.Status)
	}

	t.Log("A global frontend with local workers must not advertise a Relay without frontend load")
	if err := client.Get(ctx, key, globalDGD); err != nil {
		t.Fatal(err)
	}
	globalDGD.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, GlobalDynamoNamespace: true},
		{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
	}
	if err := client.Update(ctx, globalDGD); err != nil {
		t.Fatal(err)
	}
	reconcile()
	if err := client.Get(ctx, childKey, &service); err == nil {
		t.Fatal("Relay Service was published for a global frontend")
	}
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); condition == nil || condition.Reason != "UnsupportedSourceScope" || !strings.Contains(condition.Message, "frontend load") {
		t.Fatalf("unexpected status for global frontend: %+v", relay.Status)
	}

	dottedKey := types.NamespacedName{Namespace: key.Namespace, Name: "green.blue"}
	dottedRelayMeta := metav1.ObjectMeta{Name: dottedKey.Name, Namespace: dottedKey.Namespace, UID: "dotted-relay-uid"}
	dottedDGDMeta := metav1.ObjectMeta{Name: dottedKey.Name, Namespace: dottedKey.Namespace}
	dottedRelay := &v1alpha1.DynamoPoolRelay{
		ObjectMeta: dottedRelayMeta,
		Spec:       v1alpha1.DynamoPoolRelaySpec{ClusterID: "east", Image: "relay:test"},
	}
	dottedDGD := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: dottedDGDMeta,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
	}
	if err := client.Create(ctx, dottedRelay); err != nil {
		t.Fatal(err)
	}
	if err := client.Create(ctx, dottedDGD); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: dottedKey}); err != nil {
		t.Fatal(err)
	}
	if err := client.Get(ctx, dottedKey, dottedRelay); err != nil {
		t.Fatal(err)
	}
	condition := meta.FindStatusCondition(dottedRelay.Status.Conditions, "Ready")
	if condition == nil || condition.Reason != "InvalidConfiguration" || !strings.Contains(condition.Message, "green.blue-pool-relay") {
		t.Fatalf("unexpected status for invalid Service name: %+v", dottedRelay.Status)
	}
}

func TestDiscoveryModeFollowsPodTemplatePrecedence(t *testing.T) {
	t.Log("DGD spec annotations override metadata discovery defaults")
	dgdMeta := metav1.ObjectMeta{Annotations: map[string]string{
		commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "pod",
	}}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: dgdMeta,
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Annotations: map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"},
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker,
			}},
		},
	}
	if supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes) {
		t.Fatal("container-mode workers were accepted as pod-mode")
	}

	t.Log("Reject mixed discovery settings even if a component PodTemplate overrides them")
	podMeta := metav1.ObjectMeta{Annotations: map[string]string{
		commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "pod",
	}}
	dgd.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{ObjectMeta: podMeta}
	if supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes) {
		t.Fatal("mixed discovery settings were accepted")
	}
	dgd.Spec.Annotations = nil
	if !supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes) {
		t.Fatal("consistent pod-mode discovery was rejected")
	}
	dgd.Annotations[commonconsts.KubeAnnotationDynamoDiscoveryBackend] = "etcd"
	if supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes) {
		t.Fatal("Grove's non-Kubernetes DGD backend was accepted")
	}
	dgd.Annotations = nil
	dgd.Spec.Components[0].PodTemplate.Spec.Containers = []corev1.Container{{Name: commonconsts.MainContainerName, Env: []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"}}}}
	if supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes) {
		t.Fatal("component environment override to container mode was accepted")
	}
	dgd.Spec.Components[0].PodTemplate.Spec.Containers = nil
	dgd.Spec.Env = []corev1.EnvVar{{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "etcd"}}
	if supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes) {
		t.Fatal("DGD environment override to etcd was accepted")
	}
}

func TestRelayEnvironmentOwnership(t *testing.T) {
	relayMeta := metav1.ObjectMeta{Name: "green", Namespace: "inference"}
	relay := &v1alpha1.DynamoPoolRelay{
		ObjectMeta: relayMeta,
		Spec: v1alpha1.DynamoPoolRelaySpec{
			ClusterID: "east", Image: "relay:test",
			Env: []corev1.EnvVar{{Name: "DYN_RELAY_PUBLICATION_THRESHOLD", Value: "32"}},
		},
	}
	objects, err := render(relay, []string{"green-runtime"})
	if err != nil {
		t.Fatalf("supported Relay tuning was rejected: %v", err)
	}
	deployment := objects[len(objects)-1].(*appsv1.Deployment)
	if !slices.ContainsFunc(deployment.Spec.Template.Spec.Containers[0].Env, func(item corev1.EnvVar) bool {
		return item.Name == "DYN_RELAY_PUBLICATION_THRESHOLD" && item.Value == "32"
	}) {
		t.Fatal("supported Relay tuning was not forwarded")
	}
	relay.Spec.Env = []corev1.EnvVar{{Name: "DYN_RELAY_CONNECTION_REVISION", Value: "wrong"}}
	if _, err := render(relay, []string{"green-runtime"}); err == nil {
		t.Fatal("operator-owned connection revision was accepted")
	}
}

func TestRelayClusterIDRejectsSurroundingWhitespace(t *testing.T) {
	for _, clusterID := range []string{"", " ", " east", "east ", "\teast", "\u00a0east", "\x1ceast"} {
		t.Run(clusterID, func(t *testing.T) {
			t.Log("Reject a cluster ID that the Relay CLI cannot accept")
			relayMeta := metav1.ObjectMeta{Name: "green", Namespace: "inference"}
			relay := &v1alpha1.DynamoPoolRelay{
				ObjectMeta: relayMeta,
				Spec:       v1alpha1.DynamoPoolRelaySpec{ClusterID: clusterID, Image: "relay:test"},
			}
			if _, err := render(relay, []string{"green-runtime"}); err == nil {
				t.Fatalf("accepted cluster ID %q", clusterID)
			}
		})
	}
}

func TestWithdrawCleansOwnedDeploymentAfterForeignService(t *testing.T) {
	t.Log("Keep the foreign Service but delete the owned Deployment and withdraw Relay readiness")
	ctx := context.Background()
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{v1alpha1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme} {
		if err := add(scheme); err != nil {
			t.Fatal(err)
		}
	}
	relayMeta := metav1.ObjectMeta{Name: "green", Namespace: "inference", UID: "relay-uid"}
	relay := &v1alpha1.DynamoPoolRelay{
		ObjectMeta: relayMeta,
		Spec:       v1alpha1.DynamoPoolRelaySpec{ClusterID: "east", Image: "relay:test"},
	}
	key := types.NamespacedName{Name: serviceName(relay.Name), Namespace: relay.Namespace}
	resourceMeta := metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace}
	foreignService := &corev1.Service{ObjectMeta: resourceMeta}
	deployment := &appsv1.Deployment{ObjectMeta: resourceMeta}
	if err := controllerutil.SetControllerReference(relay, deployment, scheme); err != nil {
		t.Fatal(err)
	}
	apiClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(relay).
		WithObjects(relay, foreignService, deployment).Build()
	r := &reconciler{Client: apiClient, scheme: scheme}
	if _, err := r.withdraw(ctx, relay, "DGDNotFound", "DGD is missing"); err == nil || !strings.Contains(err.Error(), "not owned") {
		t.Fatalf("expected foreign Service conflict, got %v", err)
	}
	if err := apiClient.Get(ctx, key, foreignService); err != nil {
		t.Fatalf("foreign Service was removed: %v", err)
	}
	if err := apiClient.Get(ctx, key, deployment); !apierrors.IsNotFound(err) {
		t.Fatalf("owned Deployment remains: %v", err)
	}
	if err := apiClient.Get(ctx, types.NamespacedName{Name: relay.Name, Namespace: relay.Namespace}, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != "DGDNotFound" {
		t.Fatalf("Relay still advertises readiness: %+v", relay.Status)
	}
}
