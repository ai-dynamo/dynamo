//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestClusterBlockWorkloads(t *testing.T) {
	t.Log("Create an isolated namespace and block downstream workload actuation")
	env := workloadClusterTestEnv.RunT(t)
	env.BlockWorkloads()

	labels := map[string]string{"app": "quota-test"}
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "quota-test", Namespace: env.Namespace()},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To[int32](2),
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "worker", Image: "invalid.example/never-used",
				}}},
			},
		},
	}

	t.Log("Store a Deployment with its production replica count unchanged")
	if err := env.Client().Create(t.Context(), deployment); err != nil {
		t.Fatalf("create Deployment: %v", err)
	}
	stored := &appsv1.Deployment{}
	if err := env.Client().Get(t.Context(), client.ObjectKeyFromObject(deployment), stored); err != nil {
		t.Fatalf("get Deployment: %v", err)
	}
	if replicas := ptr.Deref(stored.Spec.Replicas, 0); replicas != 2 {
		t.Fatalf("stored Deployment replicas = %d, want 2", replicas)
	}

	t.Log("Reject direct ReplicaSet creation at the manifest boundary")
	replicaSet := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "blocked", Namespace: env.Namespace()},
		Spec: appsv1.ReplicaSetSpec{
			Replicas: ptr.To[int32](1),
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: deployment.Spec.Template,
		},
	}
	assertQuotaDenied(t, env.Client().Create(t.Context(), replicaSet), "replicasets")

	t.Log("Reject direct Pod creation as a fallback boundary")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "blocked", Namespace: env.Namespace()},
		Spec:       deployment.Spec.Template.Spec,
	}
	assertQuotaDenied(t, env.Client().Create(t.Context(), pod), "pods")
}

func assertQuotaDenied(t *testing.T, err error, resource string) {
	t.Helper()
	if err == nil {
		t.Fatalf("creating %s succeeded despite workload quota", resource)
	}
	if !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), workloadBlockQuotaName) {
		t.Fatalf("creating %s returned %v, want workload quota rejection", resource, err)
	}
}
