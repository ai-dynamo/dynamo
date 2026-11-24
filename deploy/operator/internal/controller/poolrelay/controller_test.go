/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"errors"
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
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func relaySpec(clusterID string) v1alpha1.DynamoPoolRelaySpec {
	return v1alpha1.DynamoPoolRelaySpec{
		ClusterID: clusterID,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: mainContainer, Image: "relay:test"}},
		}},
	}
}

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
		Spec:       relaySpec("east"),
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
	childKey := types.NamespacedName{Namespace: key.Namespace, Name: "green-pool-relay"}

	t.Log("Publish one DGD's Relay Service, Deployment, and source file")
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
	t.Log("Leave unchanged Relay resources untouched")
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
	t.Log("Follow the worker runtime namespace without rolling Relay Pods")
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
	if err := client.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	items := deployment.Spec.Template.Spec.Volumes[0].ConfigMap.Items
	if len(items) != 1 || items[0].Key != "sources.json" || items[0].Path != "sources.json" {
		t.Fatalf("Relay source projection was not restored: %+v", deployment.Spec.Template.Spec.Volumes[0].ConfigMap.Items)
	}
	t.Log("Restore Service publication settings after manual drift")
	if err := client.Get(ctx, childKey, &service); err != nil {
		t.Fatal(err)
	}
	service.Spec.PublishNotReadyAddresses = true
	if err := client.Update(ctx, &service); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
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
	relay.Spec.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "zmq"}}
	if err := client.Update(ctx, relay); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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

	t.Log("Keep the Service when explicit runtime settings override container-mode defaults")
	if err := client.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	dgd.Annotations = map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}
	for i := range dgd.Spec.Components {
		dgd.Spec.Components[i].PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: commonconsts.MainContainerName,
			Env:  []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"}},
		}}}}
	}
	if err := client.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
	if err := client.Get(ctx, childKey, &service); err != nil {
		t.Fatalf("Relay Service was withdrawn despite explicit pod-mode discovery: %v", err)
	}
	t.Log("Withdraw the Service when no runtime override remains")
	if err := client.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	for i := range dgd.Spec.Components {
		dgd.Spec.Components[i].PodTemplate = nil
	}
	if err := client.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}

	// The replica count is independent of local frontend replicas.
	if err := client.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	three := int32(3)
	t.Log("Scale Relay independently of frontend replicas")
	relay.Spec.Replicas = &three
	if err := client.Update(ctx, relay); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
	t.Log("Withdraw the listener while the DGD is terminating")
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
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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

	dgd.Finalizers = nil
	t.Log("Keep the listener withdrawn after DGD deletion")
	if err := client.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
	t.Log("Reject workers in the shared global runtime namespace")
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
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
		Spec:       relaySpec("east"),
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
	t.Log("Report an invalid DGD-derived Service name")
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
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || supported {
		t.Fatal("container-mode workers were accepted as pod-mode")
	}

	t.Log("A component PodTemplate overrides the DGD discovery default")
	podMeta := metav1.ObjectMeta{Annotations: map[string]string{
		commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "pod",
	}}
	dgd.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{ObjectMeta: podMeta}
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || !supported {
		t.Fatalf("effective pod-mode setting was rejected: %v", err)
	}
	dgd.Spec.Annotations = nil
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || !supported {
		t.Fatal("consistent pod-mode discovery was rejected")
	}
	dgd.Spec.Components[0].PodTemplate.Annotations[commonconsts.KubeAnnotationDynamoKubeDiscoveryMode] = "container"
	dgd.Spec.Components[0].PodTemplate.Spec.Containers = []corev1.Container{{Name: commonconsts.MainContainerName, Env: []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"}}}}
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || !supported {
		t.Fatalf("explicit pod mode did not override container-mode annotation: supported=%t err=%v", supported, err)
	}
	dgd.Spec.Components[0].PodTemplate.Annotations[commonconsts.KubeAnnotationDynamoKubeDiscoveryMode] = "pod"
	dgd.Spec.Components[0].PodTemplate.Spec.Containers = nil
	dgd.Annotations[commonconsts.KubeAnnotationDynamoDiscoveryBackend] = "etcd"
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || supported {
		t.Fatal("Grove's non-Kubernetes DGD backend was accepted")
	}
	dgd.Annotations = nil
	dgd.Spec.Components[0].PodTemplate.Spec.Containers = []corev1.Container{{Name: commonconsts.MainContainerName, Env: []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"}}}}
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || supported {
		t.Fatal("component environment override to container mode was accepted")
	}
	dgd.Spec.Components[0].PodTemplate.Spec.Containers = nil
	dgd.Spec.Env = []corev1.EnvVar{{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "etcd"}}
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || supported {
		t.Fatal("DGD environment override to etcd was accepted")
	}
	dgd.Spec.Components[0].PodTemplate.Spec.Containers = []corev1.Container{{Name: commonconsts.MainContainerName, Env: []corev1.EnvVar{{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"}}}}
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || !supported {
		t.Fatalf("runtime override of DGD environment was rejected: %v", err)
	}

	t.Log("Check every authored role template and its runtime init container")
	dgd.Spec.Env = nil
	dgd.Spec.Components[0].PodTemplate = nil
	dgd.Spec.Components[0].Roles = []v1beta1.ComponentRoleSpec{
		{Name: "leader", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: commonconsts.MainContainerName}},
		}}},
		{Name: "worker", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: commonconsts.MainContainerName}},
			InitContainers: []corev1.Container{{Name: commonconsts.RuntimeContainerName, Env: []corev1.EnvVar{
				{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"},
			}}},
		}}},
	}
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || supported {
		t.Fatalf("container-mode role runtime was accepted: supported=%t err=%v", supported, err)
	}
	dgd.Spec.Components[0].Roles[1].PodTemplate.Spec.InitContainers[0].Env[0].Value = "pod"
	if supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes); err != nil || !supported {
		t.Fatalf("pod-mode role runtimes were rejected: supported=%t err=%v", supported, err)
	}
}

func TestDiscoveryUsesPreservedServiceAnnotationsAndHubEnvironment(t *testing.T) {
	alpha := &v1alpha1.DynamoGraphDeployment{
		Spec: v1alpha1.DynamoGraphDeploymentSpec{Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
			"decode": {
				ServiceName:   "decode",
				ComponentType: "worker",
				Annotations:   map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"},
				Envs:          []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"}},
			},
		}},
	}
	var hub v1beta1.DynamoGraphDeployment
	if err := alpha.ConvertTo(&hub); err != nil {
		t.Fatal(err)
	}
	if supported, err := supportsPodDiscovery(&hub, configv1alpha1.DiscoveryBackendKubernetes); err != nil || supported {
		t.Fatalf("legacy service container-mode annotation was ignored: supported=%t err=%v", supported, err)
	}
	if len(hub.Spec.Components) != 1 {
		t.Fatalf("expected one hub component, got %d", len(hub.Spec.Components))
	}
	hub.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{Annotations: map[string]string{
		commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "pod",
	}, Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Name: commonconsts.MainContainerName,
		Env:  []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"}},
	}}}}
	if supported, err := supportsPodDiscovery(&hub, configv1alpha1.DiscoveryBackendKubernetes); err != nil || !supported {
		t.Fatalf("hub PodTemplate did not override preserved alpha settings: supported=%t err=%v", supported, err)
	}
}

func TestDiscoveryEnvFromHonorsExplicitSettings(t *testing.T) {
	for _, test := range []struct {
		name      string
		mode      string
		env       []corev1.EnvVar
		supported bool
	}{
		{
			name: "explicit pod mode overrides container annotation and envFrom",
			mode: "container",
			env: []corev1.EnvVar{
				{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
				{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"},
			},
			supported: true,
		},
		{
			name: "pod discovery overrides envFrom",
			mode: "pod",
			env: []corev1.EnvVar{
				{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
				{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"},
			},
			supported: true,
		},
		{
			name: "explicit container mode is rejected",
			mode: "pod",
			env: []corev1.EnvVar{
				{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
				{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"},
			},
		},
		{
			name: "unspecified mode could come from envFrom",
			mode: "pod",
			env:  []corev1.EnvVar{{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"}},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			dgd := &v1beta1.DynamoGraphDeployment{Spec: v1beta1.DynamoGraphDeploymentSpec{
				Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
					ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{
						commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: test.mode,
					}, Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: commonconsts.MainContainerName,
						Env:  test.env,
						EnvFrom: []corev1.EnvFromSource{{SecretRef: &corev1.SecretEnvSource{
							Name: "worker-credentials",
						}}},
					}}}},
				}},
			}}
			supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes)
			if err != nil || supported != test.supported {
				t.Fatalf("supported=%t, want %t, err=%v", supported, test.supported, err)
			}
		})
	}
}

func TestDeploymentApplyUsesResponseWhenCacheIsStale(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		v1alpha1.AddToScheme, v1beta1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme, rbacv1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatal(err)
		}
	}
	relay := &v1alpha1.DynamoPoolRelay{
		Name: "green", Namespace: "inference", UID: "relay-uid",
		Spec: relaySpec("east")}
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: relay.Name, Namespace: relay.Namespace,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
	}
	_, old, err := render(relay, []string{"green-runtime"})
	if err != nil {
		t.Fatal(err)
	}
	if err := controllerutil.SetControllerReference(relay, old, scheme); err != nil {
		t.Fatal(err)
	}
	old.UID = "deployment-uid"
	old.Generation = 1
	old.Status = appsv1.DeploymentStatus{
		ObservedGeneration: 1, Replicas: 2, UpdatedReplicas: 2, AvailableReplicas: 2,
	}
	stale := old.DeepCopy()
	relay.Spec.PodTemplate.Spec.Containers[0].Image = "relay:new"
	_, wanted, err := render(relay, []string{"green-runtime"})
	if err != nil {
		t.Fatal(err)
	}
	reads := 0
	apiClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&appsv1.Deployment{}, relay, dgd).
		WithObjects(old, relay, dgd).WithInterceptorFuncs(interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			if deployment, ok := obj.(*appsv1.Deployment); ok {
				reads++
				*deployment = *stale.DeepCopy()
				return nil
			}
			return c.Get(ctx, key, obj, opts...)
		},
		Apply: func(ctx context.Context, c client.WithWatch, obj runtime.ApplyConfiguration, opts ...client.ApplyOption) error {
			if err := c.Apply(ctx, obj, opts...); err != nil {
				return err
			}
			obj.(metav1.Object).SetGeneration(2)
			return nil
		},
	}).Build()
	r := &reconciler{Client: apiClient, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	observed, err := r.applyDeployment(ctx, relay, wanted)
	if err != nil {
		t.Fatal(err)
	}
	if reads != 1 || observed.Spec.Template.Spec.Containers[0].Image != "relay:new" ||
		observed.Generation != 2 || observed.Status.ObservedGeneration != 1 {
		t.Fatalf("used stale cached Deployment after apply: reads=%d, observed=%+v", reads, observed)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{
		Name: relay.Name, Namespace: relay.Namespace}); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, client.ObjectKeyFromObject(relay), relay); err != nil {
		t.Fatal(err)
	}
	condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready")
	if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != "DeploymentProgressing" {
		t.Fatalf("Relay reported the old listeners ready after apply: %+v", relay.Status)
	}
}

func TestDiscoveryConversionErrorDoesNotWithdrawPublishedRelay(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{
		v1alpha1.AddToScheme, v1beta1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatal(err)
		}
	}
	key := types.NamespacedName{Name: "green", Namespace: "inference"}
	relay := &v1alpha1.DynamoPoolRelay{Name: key.Name, Namespace: key.Namespace, UID: "relay-uid", Spec: relaySpec("east")}
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: key.Name, Namespace: key.Namespace, Annotations: map[string]string{"nvidia.com/dgd-status": "{"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
	}
	childKey := types.NamespacedName{Name: serviceName(key.Name), Namespace: key.Namespace}
	service := &corev1.Service{Name: childKey.Name, Namespace: childKey.Namespace}
	deployment := &appsv1.Deployment{Name: childKey.Name, Namespace: childKey.Namespace}
	for _, obj := range []client.Object{service, deployment} {
		if err := controllerutil.SetControllerReference(relay, obj, scheme); err != nil {
			t.Fatal(err)
		}
	}
	apiClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(relay).WithObjects(relay, dgd, service, deployment).Build()
	r := &reconciler{Client: apiClient, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err == nil || !strings.Contains(err.Error(), "nvidia.com/dgd-status") {
		t.Fatalf("expected preserved DGD annotation error, got %v", err)
	}
	if err := apiClient.Get(ctx, childKey, service); err != nil {
		t.Fatalf("Service was withdrawn for an unreadable DGD annotation: %v", err)
	}
	if err := apiClient.Get(ctx, childKey, deployment); err != nil {
		t.Fatalf("Deployment was withdrawn for an unreadable DGD annotation: %v", err)
	}
}

func TestRelayEnvironmentOwnership(t *testing.T) {
	relayMeta := metav1.ObjectMeta{Name: "green", Namespace: "inference"}
	relay := &v1alpha1.DynamoPoolRelay{
		ObjectMeta: relayMeta,
		Spec:       relaySpec("east"),
	}
	relay.Spec.PodTemplate.Spec.ImagePullSecrets = []corev1.LocalObjectReference{{Name: "relay-registry"}}
	relay.Spec.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "DYN_RELAY_PUBLICATION_THRESHOLD", Value: "32"}}
	_, deployment, err := render(relay, []string{"green-runtime"})
	if err != nil {
		t.Fatalf("supported Relay tuning was rejected: %v", err)
	}
	if !slices.ContainsFunc(deployment.Spec.Template.Spec.Containers[0].Env, func(item corev1.EnvVar) bool {
		return item.Name == "DYN_RELAY_PUBLICATION_THRESHOLD" && item.Value == "32"
	}) {
		t.Fatal("supported Relay tuning was not forwarded")
	}
	if !slices.Equal(deployment.Spec.Template.Spec.ImagePullSecrets, relay.Spec.PodTemplate.Spec.ImagePullSecrets) {
		t.Fatal("PodTemplate image pull secrets were not preserved")
	}
	relay.Spec.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "DYN_RELAY_CONNECTION_REVISION", Value: "wrong"}}
	if _, _, err := render(relay, []string{"green-runtime"}); err == nil {
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
				Spec:       relaySpec(clusterID),
			}
			if _, _, err := render(relay, []string{"green-runtime"}); err == nil {
				t.Fatalf("accepted cluster ID %q", clusterID)
			}
		})
	}
}

func TestApplyFailureWithdrawsStaleReadiness(t *testing.T) {
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
	relay := &v1alpha1.DynamoPoolRelay{
		Name: key.Name, Namespace: key.Namespace, UID: "relay-uid", Generation: 2,
		Spec: relaySpec("east"),
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: key.Name, Namespace: key.Namespace,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
	}
	childKey := types.NamespacedName{Namespace: key.Namespace, Name: serviceName(key.Name)}
	denyDeployment := false
	apiClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(relay, dgd, &appsv1.Deployment{}).
		WithObjects(relay, dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
				if _, ok := obj.(*appsv1.Deployment); ok && denyDeployment {
					return errors.New("admission denied Relay Deployment")
				}
				return c.Create(ctx, obj, opts...)
			},
		}).Build()
	r := &reconciler{Client: apiClient, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	t.Log("Publish the Relay and mark the current generation ready")
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
	var deployment appsv1.Deployment
	if err := apiClient.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	deployment.Status.ObservedGeneration = deployment.Generation
	deployment.Status.Replicas = 2
	deployment.Status.UpdatedReplicas = 2
	deployment.Status.AvailableReplicas = 2
	if err := apiClient.Status().Update(ctx, &deployment); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if !meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready") {
		t.Fatal("Relay did not become ready")
	}
	observedGeneration := relay.Status.ObservedGeneration

	t.Log("An admission failure after Deployment loss clears stale readiness")
	if err := apiClient.Delete(ctx, &deployment); err != nil {
		t.Fatal(err)
	}
	denyDeployment = true
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err == nil || !strings.Contains(err.Error(), "admission denied") {
		t.Fatalf("expected Deployment admission error, got %v", err)
	}
	if err := apiClient.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready")
	if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != "ReconcileFailed" ||
		relay.Status.Endpoint != "" || relay.Status.AvailableReplicas != 0 || relay.Status.ObservedGeneration != observedGeneration {
		t.Fatalf("Relay kept stale readiness after apply failure: %+v", relay.Status)
	}
}

func TestWithdrawUsesVerifiedUIDAndPreservesGenerationOnFailure(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{v1alpha1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme} {
		if err := add(scheme); err != nil {
			t.Fatal(err)
		}
	}
	relay := &v1alpha1.DynamoPoolRelay{
		Name: "green", Namespace: "inference", UID: "relay-uid", Generation: 3,
		Spec:   relaySpec("east"),
		Status: v1alpha1.DynamoPoolRelayStatus{ObservedGeneration: 2},
	}
	childKey := types.NamespacedName{Name: serviceName(relay.Name), Namespace: relay.Namespace}
	service := &corev1.Service{Name: childKey.Name, Namespace: childKey.Namespace, UID: "old-service"}
	if err := controllerutil.SetControllerReference(relay, service, scheme); err != nil {
		t.Fatal(err)
	}
	var deleteUID *types.UID
	apiClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(relay).
		WithObjects(relay, service).
		WithInterceptorFuncs(interceptor.Funcs{
			Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
				deleteUID = (&client.DeleteOptions{}).ApplyOptions(opts).Preconditions.UID
				return errors.New("delete failed")
			},
		}).Build()
	r := &reconciler{Client: apiClient, scheme: scheme}
	if _, err := r.withdraw(ctx, relay, "DGDNotFound", "DGD is missing"); err == nil || !strings.Contains(err.Error(), "delete failed") {
		t.Fatalf("expected deletion error, got %v", err)
	}
	if deleteUID == nil || *deleteUID != service.UID {
		t.Fatalf("deletion was not bound to the checked Service UID: %v", deleteUID)
	}
	if err := apiClient.Get(ctx, types.NamespacedName{Name: relay.Name, Namespace: relay.Namespace}, relay); err != nil {
		t.Fatal(err)
	}
	if relay.Status.ObservedGeneration != 2 || meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready") {
		t.Fatalf("failed withdrawal acknowledged generation 3: %+v", relay.Status)
	}
}

func TestPrerequisiteOwnershipConflictWithdrawsRelay(t *testing.T) {
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
	childKey := types.NamespacedName{Namespace: key.Namespace, Name: serviceName(key.Name)}
	relay := &v1alpha1.DynamoPoolRelay{
		Name: key.Name, Namespace: key.Namespace, UID: "relay-uid",
		Spec: relaySpec("east"),
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: key.Name, Namespace: key.Namespace,
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeWorker},
		}},
		Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{
			"decode": {RuntimeNamespace: "green-runtime"},
		}},
	}
	apiClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(relay, dgd, &appsv1.Deployment{}).
		WithObjects(relay, dgd).Build()
	r := &reconciler{Client: apiClient, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	reconcile := func() error {
		_, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key})
		return err
	}
	if err := reconcile(); err != nil {
		t.Fatal(err)
	}
	var deployment appsv1.Deployment
	if err := apiClient.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	deployment.Status.ObservedGeneration = deployment.Generation
	deployment.Status.UpdatedReplicas = 2
	deployment.Status.Replicas = 3
	deployment.Status.AvailableReplicas = 2
	if err := apiClient.Status().Update(ctx, &deployment); err != nil {
		t.Fatal(err)
	}
	if err := reconcile(); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready") {
		t.Fatal("old surge replica was counted as current listener readiness")
	}
	if err := apiClient.Get(ctx, childKey, &deployment); err != nil {
		t.Fatal(err)
	}
	deployment.Status.Replicas = 2
	if err := apiClient.Status().Update(ctx, &deployment); err != nil {
		t.Fatal(err)
	}
	if err := reconcile(); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); condition == nil || condition.Status != metav1.ConditionTrue {
		t.Fatalf("Relay was not ready before the collision: %+v", relay.Status)
	}

	var serviceAccount corev1.ServiceAccount
	if err := apiClient.Get(ctx, childKey, &serviceAccount); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Delete(ctx, &serviceAccount); err != nil {
		t.Fatal(err)
	}
	foreign := &corev1.ServiceAccount{Name: childKey.Name, Namespace: childKey.Namespace}
	if err := apiClient.Create(ctx, foreign); err != nil {
		t.Fatal(err)
	}
	if err := apiClient.Get(ctx, key, dgd); err != nil {
		t.Fatal(err)
	}
	dgd.Status.Components["decode"] = v1beta1.ComponentReplicaStatus{RuntimeNamespace: "green-runtime-next"}
	if err := apiClient.Status().Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}
	if err := reconcile(); err == nil || !strings.Contains(err.Error(), "not owned") {
		t.Fatalf("expected ServiceAccount ownership conflict, got %v", err)
	}
	var service corev1.Service
	if err := apiClient.Get(ctx, childKey, &service); !apierrors.IsNotFound(err) {
		t.Fatalf("Relay Service remained published: %v", err)
	}
	if err := apiClient.Get(ctx, childKey, &deployment); !apierrors.IsNotFound(err) {
		t.Fatalf("Relay Deployment remained active: %v", err)
	}
	if err := apiClient.Get(ctx, childKey, foreign); err != nil {
		t.Fatalf("foreign ServiceAccount was removed: %v", err)
	}
	if err := apiClient.Get(ctx, key, relay); err != nil {
		t.Fatal(err)
	}
	if condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready"); condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != "ResourceOwnershipConflict" || relay.Status.Endpoint != "" {
		t.Fatalf("Relay still advertises readiness: %+v", relay.Status)
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
		Spec:       relaySpec("east"),
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
