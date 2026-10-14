/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoPoolRelayLifecycle(t *testing.T) {
	t.Log("Create a relay with an explicit logical sources and reconcile its independent resources")
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, v1alpha1.AddToScheme(scheme))
	require.NoError(t, v1beta1.AddToScheme(scheme))
	relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "dc-a", Namespace: "workers", UID: "relay-uid", Generation: 1}, Spec: v1alpha1.DynamoPoolRelaySpec{
		ClusterID: "dc-a", PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}}}}, Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{Refs: []v1alpha1.DynamoPoolRelayDGDRef{{Name: "model-b"}, {Name: "model-a"}}}},
	}}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithStatusSubresource(relay, &appsv1.Deployment{}).WithObjects(relay).Build()
	for _, name := range []string{"model-a", "model-b"} {
		dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "workers", UID: types.UID(name)}, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.5.0"}}}}}}}}
		require.NoError(t, c.Create(ctx, dgd))
	}
	reconciler := &relayReconciler{Client: c, config: &configv1alpha1.OperatorConfiguration{}}
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)}
	result, err := reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.Positive(t, result.RequeueAfter)
	key := types.NamespacedName{Name: relayResourceName(relay.Name), Namespace: relay.Namespace}
	deployment := &appsv1.Deployment{}
	require.NoError(t, c.Get(ctx, key, deployment))
	require.True(t, metav1.IsControlledBy(deployment, relay))
	require.Equal(t, int32(2), *deployment.Spec.Replicas)
	require.Equal(t, appsv1.RollingUpdateDeploymentStrategyType, deployment.Spec.Strategy.Type)
	require.Contains(t, deployment.Spec.Template.Spec.Containers[0].Args, relaySourcesPath)
	sourcesMap := &corev1.ConfigMap{}
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	var sources relaySources
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Len(t, sources.Sources, 2)
	require.Equal(t, "workers-model-a", sources.Sources[0].Namespace)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.False(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready"))
	require.Equal(t, key.Name+".workers.svc:5561", relay.Status.Endpoint)

	t.Log("Reconciliation is a no-op and preserves Service allocation and API-defaulted fields")
	service := &corev1.Service{}
	require.NoError(t, c.Get(ctx, key, service))
	service.Spec.ClusterIP = "10.0.0.42"
	service.Spec.ClusterIPs = []string{"10.0.0.42"}
	require.NoError(t, c.Update(ctx, service))
	deployment.Spec.RevisionHistoryLimit = new(int32(10))
	require.NoError(t, c.Update(ctx, deployment))
	version := deployment.ResourceVersion
	statusVersion := relay.ResourceVersion
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, version, deployment.ResourceVersion)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.Equal(t, statusVersion, relay.ResourceVersion)
	require.NoError(t, c.Get(ctx, key, service))
	require.Equal(t, "10.0.0.42", service.Spec.ClusterIP)

	t.Log("Current Deployment availability makes the listener Ready without claiming model readiness")
	deployment.Generation = 2
	require.NoError(t, c.Update(ctx, deployment))
	deployment.Status = appsv1.DeploymentStatus{ObservedGeneration: 2, Replicas: 2, UpdatedReplicas: 2, ReadyReplicas: 2, AvailableReplicas: 2}
	require.NoError(t, c.Status().Update(ctx, deployment))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.True(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready"))
	require.Contains(t, meta.FindStatusCondition(relay.Status.Conditions, "Ready").Message, "model readiness are not evaluated")

	t.Log("Stale Deployment status must not mark a new generation Ready")
	deployment.Generation = 3
	require.NoError(t, c.Update(ctx, deployment))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.False(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready"))

	t.Log("Deleted children are recreated and drift is repaired")
	role := &rbacv1.Role{}
	require.NoError(t, c.Get(ctx, key, role))
	require.NoError(t, c.Delete(ctx, role))
	deployment.Spec.Replicas = new(int32(3))
	require.NoError(t, c.Update(ctx, deployment))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, role))
	require.True(t, metav1.IsControlledBy(role, relay))
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, int32(2), *deployment.Spec.Replicas)

	t.Log("Scaling adds an active replica without changing the Pod template")
	beforeScale := deployment.Spec.Template.DeepCopy()
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	relay.Spec.Replicas = new(int32(3))
	require.NoError(t, c.Update(ctx, relay))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, int32(3), *deployment.Spec.Replicas)
	require.Equal(t, *beforeScale, deployment.Spec.Template)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.False(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "Ready"))
}

func TestDynamoPoolRelayRejectsForeignResources(t *testing.T) {
	for _, controlled := range []bool{false, true} {
		t.Run(map[bool]string{false: "unowned", true: "other-controller"}[controlled], func(t *testing.T) {
			t.Log("Place an unrelated ServiceAccount at the desired name")
			scheme := runtime.NewScheme()
			require.NoError(t, clientgoscheme.AddToScheme(scheme))
			require.NoError(t, v1alpha1.AddToScheme(scheme))
			require.NoError(t, v1beta1.AddToScheme(scheme))
			relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "workers", UID: "relay", Generation: 2}, Spec: v1alpha1.DynamoPoolRelaySpec{ClusterID: "dc", PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}}}}, Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{LabelSelector: &metav1.LabelSelector{}}}}}
			account := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: relayResourceName(relay.Name), Namespace: relay.Namespace}}
			if controlled {
				account.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1alpha1", Kind: "DynamoPoolRelay", Name: "another", UID: "other", Controller: new(true)}}
			}
			c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithObjects(relay, account).WithStatusSubresource(relay).Build()
			reconciler := &relayReconciler{Client: c, config: &configv1alpha1.OperatorConfiguration{}}

			t.Log("Refuse adoption, report the failure, and leave the foreign resource unchanged")
			_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)})
			require.ErrorContains(t, err, "not owned by this DynamoPoolRelay")
			actual := &corev1.ServiceAccount{}
			require.NoError(t, c.Get(context.Background(), client.ObjectKeyFromObject(account), actual))
			require.Equal(t, account.OwnerReferences, actual.OwnerReferences)
			require.NoError(t, c.Get(context.Background(), client.ObjectKeyFromObject(relay), relay))
			require.Equal(t, "ReconcileFailed", meta.FindStatusCondition(relay.Status.Conditions, "Ready").Reason)
			require.Zero(t, relay.Status.ObservedGeneration)
		})
	}
}

func TestDynamoPoolRelayRendering(t *testing.T) {
	t.Log("Render NATS credentials as a required Secret reference without mutating the CR")
	relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "a." + strings.Repeat("a", 251), Namespace: "workers", UID: "relay"}, Spec: v1alpha1.DynamoPoolRelaySpec{
		ClusterID: "cluster-a", PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}}}}, Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{LabelSelector: &metav1.LabelSelector{}}},
	}}
	connection := relayConnection{Env: []corev1.EnvVar{{Name: "NATS_SERVER", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "nats"}, Key: "url", Optional: new(false)}}}}}
	sources := newRelaySources(connection.revision(), []relaySource{{Namespace: "workers-z"}, {Namespace: "workers-a"}})
	before := relay.DeepCopy()
	resources, err := renderRelayResources(relay, connection, sources)
	require.NoError(t, err)
	deployment := resources[5].(*appsv1.Deployment)
	require.LessOrEqual(t, len(deployment.Name), 63)
	require.Empty(t, validation.IsDNS1035Label(deployment.Name))
	t.Log("Pass the CR cluster identity through the runtime compatibility entry point")
	main := deployment.Spec.Template.Spec.Containers[0]
	require.Equal(t, []string{"python3", "-m", "dynamo.kv_dc_relay"}, main.Command)
	require.Equal(t, []string{"--dc-id", "cluster-a", "--sources-file", relaySourcesPath, "--bind", "0.0.0.0:5561"}, main.Args)
	require.Equal(t, "dynamo.kvrelay.v1.KvEventRelay", *main.ReadinessProbe.GRPC.Service)

	t.Log("Render the source connection revision into the Relay process environment")
	require.Contains(t, deployment.Spec.Template.Spec.Containers[0].Env, corev1.EnvVar{Name: "DYN_RELAY_CONNECTION_REVISION", Value: sources.ConnectionRevision})

	var server corev1.EnvVar
	for _, env := range deployment.Spec.Template.Spec.Containers[0].Env {
		if env.Name == "NATS_SERVER" {
			server = env
		}
	}
	require.Empty(t, server.Value)
	require.NotNil(t, server.ValueFrom)
	require.Equal(t, "nats", server.ValueFrom.SecretKeyRef.Name)
	require.False(t, *server.ValueFrom.SecretKeyRef.Optional)
	require.Equal(t, before, relay)

	t.Log("Changing the selected sources has no effect on the rendered Deployment")
	sources = newRelaySources(connection.revision(), []relaySource{{Namespace: "workers-a"}})
	updated, err := renderRelayResources(relay, connection, sources)
	require.NoError(t, err)
	require.Equal(t, deployment, updated[5])
}

func TestDynamoPoolRelaySelectionUpdatesDoNotRollDeployment(t *testing.T) {
	t.Log("Reconcile a selected worker with a compatible connection")
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, v1alpha1.AddToScheme(scheme))
	require.NoError(t, v1beta1.AddToScheme(scheme))
	relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "relay", Namespace: "workers", UID: "relay", Generation: 1}, Spec: v1alpha1.DynamoPoolRelaySpec{ClusterID: "dc", PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}}}}, Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"selected": "yes"}}}}}}
	first := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "first", Namespace: "workers", UID: "first", Labels: map[string]string{"selected": "yes"}}, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.5.0"}}}}}}}}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithStatusSubresource(relay, &appsv1.Deployment{}).WithObjects(relay, first).Build()
	reconciler := &relayReconciler{Client: c, config: &configv1alpha1.OperatorConfiguration{}}
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)}
	_, err := reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	key := client.ObjectKey{Namespace: relay.Namespace, Name: relayResourceName(relay.Name)}
	original := &appsv1.Deployment{}
	require.NoError(t, c.Get(ctx, key, original))

	t.Log("Recreating a DGD with the same namespace and connection preserves the source file")
	beforeRecreate := &corev1.ConfigMap{}
	require.NoError(t, c.Get(ctx, key, beforeRecreate))
	require.NoError(t, c.Delete(ctx, first))
	first.UID = "replacement-first"
	first.ResourceVersion = ""
	require.NoError(t, c.Create(ctx, first))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	afterRecreate := &corev1.ConfigMap{}
	require.NoError(t, c.Get(ctx, key, afterRecreate))
	require.Equal(t, beforeRecreate.Data, afterRecreate.Data)
	afterRecreateDeployment := &appsv1.Deployment{}
	require.NoError(t, c.Get(ctx, key, afterRecreateDeployment))
	require.Equal(t, original.ResourceVersion, afterRecreateDeployment.ResourceVersion)

	t.Log("Adding a second DGD changes only the sources ConfigMap and desired revision")
	second := first.DeepCopy()
	second.Name = "second"
	second.UID = "second"
	second.ResourceVersion = ""
	require.NoError(t, c.Create(ctx, second))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	deployment := &appsv1.Deployment{}
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, original.ResourceVersion, deployment.ResourceVersion)
	sourcesMap := &corev1.ConfigMap{}
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	var sources relaySources
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Len(t, sources.Sources, 2)

	t.Log("An incompatible addition blocks application but cannot retain deselected sources")
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	relay.Status.AppliedSourcesRevision = sources.Revision
	meta.SetStatusCondition(&relay.Status.Conditions, metav1.Condition{Type: "SourcesApplied", Status: metav1.ConditionTrue, Reason: "Applied", Message: "previous sources applied"})
	require.NoError(t, c.Status().Update(ctx, relay))
	first.Labels = nil
	require.NoError(t, c.Update(ctx, first))
	incompatible := second.DeepCopy()
	incompatible.Name = "incompatible"
	incompatible.UID = "incompatible"
	incompatible.ResourceVersion = ""
	incompatible.Spec.Components[0].PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "nats"}, {Name: "NATS_SERVER", Value: "nats://different:4222"}}
	require.NoError(t, c.Create(ctx, incompatible))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Len(t, sources.Sources, 1)
	require.Equal(t, v1beta1.ComputeDynamoNamespace(false, "workers", "second"), sources.Sources[0].Namespace)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, original.ResourceVersion, deployment.ResourceVersion)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	require.False(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "ConfigurationResolved"))
	require.False(t, meta.IsStatusConditionTrue(relay.Status.Conditions, "SourcesApplied"))
	require.Empty(t, relay.Status.AppliedSourcesRevision)

	t.Log("Removing every DGD withdraws the sources while preserving the connection and Pod template")
	second.Labels = nil
	require.NoError(t, c.Update(ctx, second))
	require.NoError(t, c.Delete(ctx, incompatible))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, original.ResourceVersion, deployment.ResourceVersion)
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Empty(t, sources.Sources)

	t.Log("Dependency mapping handles labels and empty selectors within the namespace")
	second.Labels = map[string]string{"selected": "yes"}
	requests := reconciler.relaysForDGD(ctx, second)
	require.Equal(t, []ctrl.Request{request}, requests)
}
