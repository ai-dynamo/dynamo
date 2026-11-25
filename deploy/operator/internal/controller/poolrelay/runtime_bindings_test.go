/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"encoding/json"
	"maps"
	"slices"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestRelayUsesPublishedRuntimeBindings(t *testing.T) {
	for _, test := range []struct {
		name        string
		workers     map[string]string
		frontends   map[string]string
		wantRuntime string
		wantSources []string
		wantReason  string
	}{
		{name: "worker-only graph", workers: map[string]string{"decode": "opaque-worker"}, wantRuntime: "opaque-worker", wantSources: []string{"opaque-worker"}},
		{name: "frontend load binding is independent of worker sources", workers: map[string]string{"decode": "worker-hash"}, frontends: map[string]string{"frontend": "load-scope"}, wantRuntime: "load-scope", wantSources: []string{"worker-hash"}},
		{name: "deduplicate and sort worker bindings", workers: map[string]string{"decode": "b", "prefill": "a", "worker": "b"}, wantRuntime: "a", wantSources: []string{"a", "b"}},
		{name: "frontends sharing one binding", workers: map[string]string{"decode": "worker"}, frontends: map[string]string{"http": "load", "grpc": "load"}, wantRuntime: "load", wantSources: []string{"worker"}},
		{name: "missing worker binding", workers: map[string]string{"decode": ""}, wantReason: "RuntimeNamespacePending"},
		{name: "incomplete sources must not publish a partial set", workers: map[string]string{"decode": "known", "prefill": ""}, wantReason: "RuntimeNamespacePending"},
		{name: "missing frontend binding", workers: map[string]string{"decode": "worker"}, frontends: map[string]string{"frontend": ""}, wantReason: "RuntimeNamespacePending"},
		{name: "no declared workers", frontends: map[string]string{"frontend": "load"}, wantReason: "NoWorkers"},
		{name: "distinct frontend scopes are unsupported", workers: map[string]string{"decode": "worker"}, frontends: map[string]string{"http": "first", "grpc": "second"}, wantReason: "UnsupportedSourceScope"},
		{name: "published global worker scope", workers: map[string]string{"decode": consts.GlobalDynamoNamespace}, wantReason: "UnsupportedSourceScope"},
		{name: "published global frontend scope", workers: map[string]string{"decode": "worker"}, frontends: map[string]string{"frontend": consts.GlobalDynamoNamespace}, wantReason: "UnsupportedSourceScope"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish semantic bindings without current role status or interpreting authored discovery settings")
			scheme := runtime.NewScheme()
			for _, add := range []func(*runtime.Scheme) error{
				v1alpha1.AddToScheme, v1beta1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme, rbacv1.AddToScheme,
			} {
				if err := add(scheme); err != nil {
					t.Fatal(err)
				}
			}
			dgd := &v1beta1.DynamoGraphDeployment{
				Name: "graph", Namespace: "ns", Generation: 9,
				Annotations: map[string]string{consts.KubeAnnotationDynamoDiscoveryBackend: "etcd"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Env:         []corev1.EnvVar{{Name: "DYN_DISCOVERY_BACKEND", Value: "etcd"}},
					Annotations: map[string]string{consts.KubeAnnotationDynamoKubeDiscoveryMode: "container"},
				},
				Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{}},
			}
			for _, group := range []struct {
				kind     v1beta1.ComponentType
				bindings map[string]string
			}{{v1beta1.ComponentTypeWorker, test.workers}, {v1beta1.ComponentTypeFrontend, test.frontends}} {
				for _, name := range slices.Sorted(maps.Keys(group.bindings)) {
					dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentName: name, ComponentType: group.kind,
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
							Name: "main", Image: "unrelated-engine:test", Command: []string{"custom-launcher"},
							Env:     []corev1.EnvVar{{Name: "DYN_DISCOVERY_BACKEND", Value: "etcd"}, {Name: "DYN_KUBE_DISCOVERY_MODE", Value: "$(OPAQUE)"}},
							EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "external"}}},
						}}}},
					})
					dgd.Status.Components[name] = v1beta1.ComponentReplicaStatus{RuntimeNamespace: group.bindings[name]}
				}
			}
			relay := &v1alpha1.DynamoPoolRelay{Name: dgd.Name, Namespace: dgd.Namespace, UID: "relay", Spec: relaySpec("cluster")}
			kube := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(relay, dgd).WithObjects(relay, dgd).Build()
			r := &reconciler{Client: kube, scheme: scheme}
			if _, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)}); err != nil {
				t.Fatal(err)
			}
			if err := kube.Get(t.Context(), client.ObjectKeyFromObject(relay), relay); err != nil {
				t.Fatal(err)
			}
			key := types.NamespacedName{Name: serviceName(relay.Name), Namespace: relay.Namespace}
			var deployment appsv1.Deployment
			if test.wantReason != "" {
				t.Log("Do not create listeners when source bindings are missing or incompatible")
				condition := meta.FindStatusCondition(relay.Status.Conditions, "Ready")
				if condition == nil || condition.Reason != test.wantReason {
					t.Fatalf("unexpected condition: %+v", condition)
				}
				if err := kube.Get(t.Context(), key, &deployment); !apierrors.IsNotFound(err) {
					t.Fatalf("unexpected listener Deployment: %v", err)
				}
				return
			}

			t.Log("Wire the published runtime binding and complete source set into the Relay")
			if err := kube.Get(t.Context(), key, &deployment); err != nil {
				t.Fatal(err)
			}
			if !slices.ContainsFunc(deployment.Spec.Template.Spec.Containers[0].Env, func(env corev1.EnvVar) bool {
				return env.Name == "DYN_NAMESPACE" && env.Value == test.wantRuntime
			}) {
				t.Fatalf("Relay did not use the published runtime namespace: %+v", deployment.Spec.Template.Spec.Containers[0].Env)
			}
			var configMap corev1.ConfigMap
			if err := kube.Get(t.Context(), key, &configMap); err != nil {
				t.Fatal(err)
			}
			var document sourcesDocument
			if err := json.Unmarshal([]byte(configMap.Data["sources.json"]), &document); err != nil {
				t.Fatal(err)
			}
			var sources []string
			for _, source := range document.Sources {
				sources = append(sources, source.Namespace)
			}
			if !slices.Equal(sources, test.wantSources) {
				t.Fatalf("sources=%v, want %v", sources, test.wantSources)
			}
		})
	}
}
