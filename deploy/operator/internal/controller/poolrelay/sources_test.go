/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoPoolRelayWorkerRuntimeNamespaces(t *testing.T) {
	for _, tc := range []struct {
		name       string
		components []v1beta1.DynamoComponentDeploymentSharedSpec
		statuses   map[string]v1beta1.ComponentReplicaStatus
		want       []relaySource
	}{
		{
			name:       "legacy missing status",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker"}},
			want:       []relaySource{{Namespace: "workers-model"}},
		},
		{
			name:       "hashed worker ignores frontend and removed component status",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "frontend", ComponentType: "frontend"}, {ComponentName: "worker", ComponentType: "worker"}},
			statuses: map[string]v1beta1.ComponentReplicaStatus{
				"frontend": {RuntimeNamespace: "workers-model"},
				"worker":   {RuntimeNamespace: "workers-model-newhash"},
				"removed":  {RuntimeNamespace: "workers-model-orphan"},
			},
			want: []relaySource{{Namespace: "workers-model-newhash", LogicalNamespace: "workers-model"}},
		},
		{
			name:       "shared worker namespaces are deduplicated",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "prefill", ComponentType: "prefill"}, {ComponentName: "decode", ComponentType: "decode"}},
			statuses: map[string]v1beta1.ComponentReplicaStatus{
				"prefill": {RuntimeNamespace: "workers-model-hash"},
				"decode":  {RuntimeNamespace: "workers-model-hash"},
			},
			want: []relaySource{{Namespace: "workers-model-hash", LogicalNamespace: "workers-model"}},
		},
		{
			name:       "separate worker namespaces coexist and are sorted",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "prefill", ComponentType: "prefill"}, {ComponentName: "decode", ComponentType: "decode"}},
			statuses: map[string]v1beta1.ComponentReplicaStatus{
				"prefill": {RuntimeNamespace: "workers-model-oldhash"},
				"decode":  {RuntimeNamespace: "workers-model-newhash"},
			},
			want: []relaySource{{Namespace: "workers-model-newhash", LogicalNamespace: "workers-model"}, {Namespace: "workers-model-oldhash", LogicalNamespace: "workers-model"}},
		},
		{
			name:       "partial worker status retains base fallback",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "prefill", ComponentType: "prefill"}, {ComponentName: "decode", ComponentType: "decode"}},
			statuses:   map[string]v1beta1.ComponentReplicaStatus{"prefill": {RuntimeNamespace: "workers-model-hash"}, "decode": {}},
			want:       []relaySource{{Namespace: "workers-model"}, {Namespace: "workers-model-hash", LogicalNamespace: "workers-model"}},
		},
		{
			name:       "rollout follows active status rather than future component names",
			components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker"}},
			statuses: map[string]v1beta1.ComponentReplicaStatus{
				"worker": {RuntimeNamespace: "workers-model-oldhash", ComponentNames: []string{"model-worker-newhash", "model-worker-oldhash"}, Replicas: 5, UpdatedReplicas: 1},
			},
			want: []relaySource{{Namespace: "workers-model-oldhash", LogicalNamespace: "workers-model"}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve the worker namespaces advertised by the current DGD status")
			dgd := v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"},
				Spec:       v1beta1.DynamoGraphDeploymentSpec{Components: tc.components},
				Status:     v1beta1.DynamoGraphDeploymentStatus{Components: tc.statuses},
			}
			for i := range dgd.Spec.Components {
				dgd.Spec.Components[i].PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:v1"}}}}
			}
			before := dgd.DeepCopy()
			reader := fake.NewClientBuilder().Build()
			_, sources, err := resolveRelayConnections(context.Background(), reader, &configv1alpha1.OperatorConfiguration{}, []v1beta1.DynamoGraphDeployment{dgd})
			require.NoError(t, err)
			require.Equal(t, tc.want, sources)
			require.Equal(t, before, &dgd)
		})
	}
}

func TestDynamoPoolRelayRuntimeNamespaceSourceLimits(t *testing.T) {
	for _, tc := range []struct {
		name        string
		dgdCount    int
		workerCount int
		want        string
	}{
		{name: "different DGDs cannot share a published namespace", dgdCount: 2, workerCount: 1, want: "same Dynamo namespace"},
		{name: "source limit counts namespaces after expansion", dgdCount: 1, workerCount: relayMaxSources + 1, want: "sources source limit exceeded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Construct published worker namespaces that exceed a source invariant")
			dgds := make([]v1beta1.DynamoGraphDeployment, tc.dgdCount)
			for i := range dgds {
				dgds[i].ObjectMeta = metav1.ObjectMeta{Name: fmt.Sprintf("model-%d", i), Namespace: "workers"}
				dgds[i].Status.Components = make(map[string]v1beta1.ComponentReplicaStatus)
				for j := 0; j < tc.workerCount; j++ {
					name := fmt.Sprintf("worker-%d", j)
					dgds[i].Spec.Components = append(dgds[i].Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: name, ComponentType: "worker"})
					dgds[i].Status.Components[name] = v1beta1.ComponentReplicaStatus{RuntimeNamespace: fmt.Sprintf("workers-shared-%d", j)}
				}
			}
			reader := fake.NewClientBuilder().Build()

			t.Log("Reject ambiguity or excess sources before publishing any configuration")
			_, _, err := resolveRelayConnections(context.Background(), reader, &configv1alpha1.OperatorConfiguration{}, dgds)
			require.ErrorContains(t, err, tc.want)
		})
	}
}

func TestDynamoPoolRelayRuntimeNamespaceCutoverAndWithdrawal(t *testing.T) {
	t.Log("Create a Relay following the active hashed worker namespace")
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, v1alpha1.AddToScheme(scheme))
	require.NoError(t, v1beta1.AddToScheme(scheme))
	relay := &v1alpha1.DynamoPoolRelay{
		ObjectMeta: metav1.ObjectMeta{Name: "relay", Namespace: "workers", UID: "relay", Generation: 1},
		Spec: v1alpha1.DynamoPoolRelaySpec{
			ClusterID: "dc", Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"selected": "yes"}}}},
			PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/relay:v1"}}}},
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers", UID: "model", Labels: map[string]string{"selected": "yes"}},
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
			ComponentName: "worker", ComponentType: "worker",
			PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:v1"}}}},
		}}},
		Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{"worker": {RuntimeNamespace: "workers-model-oldhash"}}},
	}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithStatusSubresource(relay, dgd, &appsv1.Deployment{}).WithObjects(relay, dgd).Build()
	reconciler := &relayReconciler{Client: c, config: &configv1alpha1.OperatorConfiguration{}}
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)}
	_, err := reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	key := client.ObjectKey{Namespace: relay.Namespace, Name: relayResourceName(relay.Name)}
	original := &appsv1.Deployment{}
	require.NoError(t, c.Get(ctx, key, original))
	sourcesMap := &corev1.ConfigMap{}
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	var sources relaySources
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Equal(t, []relaySource{{Namespace: "workers-model-oldhash", LogicalNamespace: "workers-model"}}, sources.Sources)

	t.Log("Runtime namespace cutover changes sources without rolling the Relay Deployment")
	dgd.Status.Components["worker"] = v1beta1.ComponentReplicaStatus{RuntimeNamespace: "workers-model-newhash"}
	require.NoError(t, c.Status().Update(ctx, dgd))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Equal(t, []relaySource{{Namespace: "workers-model-newhash", LogicalNamespace: "workers-model"}}, sources.Sources)
	deployment := &appsv1.Deployment{}
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, original.ResourceVersion, deployment.ResourceVersion)
	require.Equal(t, original.Spec, deployment.Spec)

	t.Log("An unsupported transport addition retains the selected hashed source")
	incompatible := dgd.DeepCopy()
	incompatible.Name = "unsupported"
	incompatible.UID = "unsupported"
	incompatible.ResourceVersion = ""
	incompatible.Status = v1beta1.DynamoGraphDeploymentStatus{}
	incompatible.Spec.Components[0].PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "unsupported"}}
	require.NoError(t, c.Create(ctx, incompatible))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Equal(t, []relaySource{{Namespace: "workers-model-newhash", LogicalNamespace: "workers-model"}}, sources.Sources)
	require.NoError(t, c.Get(ctx, request.NamespacedName, relay))
	condition := meta.FindStatusCondition(relay.Status.Conditions, "ConfigurationResolved")
	require.NotNil(t, condition)
	require.Equal(t, metav1.ConditionFalse, condition.Status)
	require.Contains(t, condition.Message, "unsupported DYN_EVENT_PLANE")

	t.Log("Deselection withdraws that hashed source even while the unsupported addition remains")
	dgd.Labels = nil
	require.NoError(t, c.Update(ctx, dgd))
	_, err = reconciler.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, c.Get(ctx, key, sourcesMap))
	require.NoError(t, json.Unmarshal([]byte(sourcesMap.Data["sources.json"]), &sources))
	require.Empty(t, sources.Sources)
	require.NoError(t, c.Get(ctx, key, deployment))
	require.Equal(t, original.ResourceVersion, deployment.ResourceVersion)
}

func TestDynamoPoolRelayLogicalIdentityCollisions(t *testing.T) {
	for _, tc := range []struct {
		name          string
		firstName     string
		secondName    string
		firstRuntime  string
		secondRuntime string
	}{
		{name: "same logical identity with distinct physical namespaces", firstName: "model.a", secondName: "model-a", firstRuntime: "workers-first-hash", secondRuntime: "workers-second-hash"},
		{name: "physical namespace collides with another logical identity", firstName: "model", secondName: "model-hash", firstRuntime: "workers-model-hash", secondRuntime: "workers-model-hash-next"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Declare distinct worker revisions whose exported identities are ambiguous")
			dgds := []v1beta1.DynamoGraphDeployment{
				{ObjectMeta: metav1.ObjectMeta{Name: tc.firstName, Namespace: "workers"}},
				{ObjectMeta: metav1.ObjectMeta{Name: tc.secondName, Namespace: "workers"}},
			}
			for i, namespace := range []string{tc.firstRuntime, tc.secondRuntime} {
				dgds[i].Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker"}}
				dgds[i].Status.Components = map[string]v1beta1.ComponentReplicaStatus{"worker": {RuntimeNamespace: namespace}}
			}
			reader := fake.NewClientBuilder().Build()

			t.Log("Reject the conflicting identities before connection resolution")
			_, _, err := resolveRelayConnections(context.Background(), reader, &configv1alpha1.OperatorConfiguration{}, dgds)
			require.ErrorContains(t, err, "same Dynamo namespace")
		})
	}
}

func TestDynamoPoolRelayWithdrawalRequiresCurrentIdentityBinding(t *testing.T) {
	for _, tc := range []struct {
		name             string
		logicalNamespace string
		keep             bool
	}{
		{name: "current physical and logical binding", logicalNamespace: "workers-model", keep: true},
		{name: "stale alias for the same physical namespace", logicalNamespace: "workers-other-model"},
		{name: "missing alias for a now-mapped physical namespace"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Persist a source binding for a still-selected hashed worker namespace")
			ctx := context.Background()
			scheme := runtime.NewScheme()
			require.NoError(t, clientgoscheme.AddToScheme(scheme))
			require.NoError(t, v1alpha1.AddToScheme(scheme))
			relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "relay", Namespace: "workers", UID: "relay"}}
			source := relaySource{Namespace: "workers-model-hash", LogicalNamespace: tc.logicalNamespace}
			document := newRelaySources("connection", []relaySource{source})
			data, err := json.Marshal(document)
			require.NoError(t, err)
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: relayResourceName(relay.Name), Namespace: relay.Namespace, OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(relay, v1alpha1.GroupVersion.WithKind("DynamoPoolRelay"))}},
				Data:       map[string]string{"sources.json": string(data)},
			}
			dgd := v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"},
				Spec:       v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker"}}},
				Status:     v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{"worker": {RuntimeNamespace: source.Namespace}}},
			}
			c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cm).Build()
			reconciler := &relayReconciler{Client: c, config: &configv1alpha1.OperatorConfiguration{}}

			t.Log("Retain only the complete current physical-to-logical association")
			require.NoError(t, reconciler.withdrawDeselectedRelaySources(ctx, relay, []v1beta1.DynamoGraphDeployment{dgd}))
			require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(cm), cm))
			var retained relaySources
			require.NoError(t, json.Unmarshal([]byte(cm.Data["sources.json"]), &retained))
			require.Equal(t, "connection", retained.ConnectionRevision)
			if tc.keep {
				require.Equal(t, []relaySource{source}, retained.Sources)
			} else {
				require.Empty(t, retained.Sources)
			}
		})
	}
}

func TestDynamoPoolRelaySourceIdentityChangesRevision(t *testing.T) {
	t.Log("Keep the old canonical bytes when physical and logical identities coincide")
	legacy := newRelaySources("", []relaySource{{Namespace: "workers-model"}})
	encoded, err := json.Marshal(legacy)
	require.NoError(t, err)
	require.Equal(t, `{"version":1,"sources":[{"namespace":"workers-model"}]}`, string(encoded))

	t.Log("Track either side of a source binding in its acknowledgement revision")
	mapped := newRelaySources("", []relaySource{{Namespace: "workers-model-hash", LogicalNamespace: "workers-model"}})
	retargeted := newRelaySources("", []relaySource{{Namespace: "workers-model-hash", LogicalNamespace: "workers-other"}})
	cutover := newRelaySources("", []relaySource{{Namespace: "workers-model-next", LogicalNamespace: "workers-model"}})
	require.NotEqual(t, mapped.Revision, retargeted.Revision)
	require.NotEqual(t, mapped.Revision, cutover.Revision)

	t.Log("Canonical ordering includes the logical namespace without mutating input")
	entries := []relaySource{{Namespace: "workers-physical", LogicalNamespace: "workers-z"}, {Namespace: "workers-physical", LogicalNamespace: "workers-a"}}
	ordered := newRelaySources("", entries)
	require.Equal(t, []relaySource{entries[1], entries[0]}, ordered.Sources)
	require.Equal(t, "workers-z", entries[0].LogicalNamespace)
}
