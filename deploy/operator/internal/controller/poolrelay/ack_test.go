/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoPoolRelaySourcesProtocolFixture(t *testing.T) {
	t.Log("Compare the Go sources writer with the same fixture read by Rust")
	fixture, err := os.ReadFile("testdata/relay-sources-v1.json")
	require.NoError(t, err)
	sources := newRelaySources("connection", []relaySource{{Namespace: "workers-model"}})
	encoded, err := json.Marshal(sources)
	require.NoError(t, err)
	require.JSONEq(t, string(fixture), string(encoded))
}

func TestDynamoPoolRelaySourcesAcknowledgement(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		valid      bool
	}{
		{name: "applied", body: `{"mode":"from-file","podUID":"pod","connectionRevision":"connection","sources":{"desiredRevision":"new","appliedRevision":"new","lastError":null}}`, valid: true},
		{name: "wrong-pod", body: `{"mode":"from-file","podUID":"old","connectionRevision":"connection","sources":{}}`},
		{name: "wrong-connection", body: `{"mode":"from-file","podUID":"pod","connectionRevision":"old","sources":{}}`},
		{name: "oversized", body: strings.Repeat("x", 16385)},
		{name: "malformed", body: `{"private-value":`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Read a bounded acknowledgement and verify Pod/connection identity")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, http.MethodGet, r.Method)
				require.Equal(t, "/engine/state", r.URL.Path)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			ack, err := readRelaySourcesAcknowledgement(context.Background(), strings.TrimPrefix(server.URL, "http://"), "pod", "connection")
			if tc.valid {
				require.NoError(t, err)
				require.Equal(t, "new", ack.Sources.AppliedRevision)
			} else {
				require.Error(t, err)
				require.NotContains(t, err.Error(), "private-value")
			}
		})
	}
}

func TestDynamoPoolRelayCanonicalRevisions(t *testing.T) {
	t.Log("Compare canonical documents and SHA256 with shared Rust fixtures")
	data, err := os.ReadFile("testdata/relay-sources-canonical.json")
	require.NoError(t, err)
	var fixtures []struct {
		Document  relaySources `json:"document"`
		Canonical string       `json:"canonical"`
		Revision  string       `json:"revision"`
	}
	require.NoError(t, json.Unmarshal(data, &fixtures))
	for _, fixture := range fixtures {
		document := newRelaySources(fixture.Document.ConnectionRevision, fixture.Document.Sources)
		encoded, err := json.Marshal(document)
		require.NoError(t, err)
		require.Equal(t, fixture.Canonical, string(encoded))
		require.Equal(t, fixture.Revision, document.Revision)
	}
}

func TestDynamoPoolRelayAllReplicaAcknowledgements(t *testing.T) {
	for _, tc := range []struct {
		name     string
		applied  int32
		complete bool
	}{
		{name: "all-current", applied: 2, complete: true},
		{name: "one-lagging", applied: 1},
		{name: "one-reload-failed", applied: 1},
		{name: "wrong-pod-uid", applied: 1},
		{name: "wrong-connection", applied: 1},
		{name: "missing-replica", applied: 1},
		{name: "pending-replica", applied: 1},
		{name: "old-template", applied: 1},
		{name: "old-extra-replica", applied: 2},
		{name: "terminating-extra-replica", applied: 2, complete: true},
		{name: "scale-up-pending", applied: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Build two independently owned replica acknowledgements and a desired Pod template")
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, appsv1.AddToScheme(scheme))
			relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "relay", Namespace: "test", UID: "relay-uid"}}
			labels := map[string]string{"nvidia.com/relay-uid": string(relay.UID)}
			deployment := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "relay", Namespace: "test", UID: "deployment"}, Spec: appsv1.DeploymentSpec{
				Replicas: ptr.To(int32(2)), Template: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: labels}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "relay:new"}}}},
			}}
			if tc.name == "scale-up-pending" {
				deployment.Spec.Replicas = ptr.To(int32(3))
			}
			rs := &appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Name: "current", Namespace: "test", UID: "current", OwnerReferences: []metav1.OwnerReference{{Kind: "Deployment", Name: deployment.Name, UID: deployment.UID, Controller: ptr.To(true)}}}, Spec: appsv1.ReplicaSetSpec{Template: *deployment.Spec.Template.DeepCopy()}}
			rs.Spec.Template.Labels[appsv1.DefaultDeploymentUniqueLabelKey] = "current-hash"
			old := rs.DeepCopy()
			old.Name, old.UID = "old", "old"
			old.Spec.Template.Spec.Containers[0].Image = "relay:old"
			podA := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "test", UID: "a", Labels: labels, OwnerReferences: []metav1.OwnerReference{{Kind: "ReplicaSet", Name: rs.Name, UID: rs.UID, Controller: ptr.To(true)}}}, Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "127.0.0.101"}}
			podB := podA.DeepCopy()
			podB.Name, podB.UID, podB.Status.PodIP = "b", "b", "127.0.0.102"
			if tc.name == "pending-replica" {
				podB.Status.Phase = corev1.PodPending
			}
			if tc.name == "old-template" {
				podB.OwnerReferences[0].Name, podB.OwnerReferences[0].UID = old.Name, old.UID
			}
			objects := []client.Object{rs, old, podA}
			if tc.name != "missing-replica" {
				objects = append(objects, podB)
			}
			if tc.name == "old-extra-replica" || tc.name == "terminating-extra-replica" {
				extra := podA.DeepCopy()
				extra.Name, extra.UID = "extra", "extra"
				extra.OwnerReferences[0].Name, extra.OwnerReferences[0].UID = old.Name, old.UID
				if tc.name == "terminating-extra-replica" {
					extra.DeletionTimestamp = ptr.To(metav1.Now())
					extra.Finalizers = []string{"test/hold"}
				}
				objects = append(objects, extra)
			}

			t.Log("Expose distinct Pod-local HTTP state endpoints with identity and revision guards")
			sources := newRelaySources("connection", nil)
			for _, pod := range []*corev1.Pod{podA, podB} {
				ack := relaySourcesAcknowledgement{Mode: "from-file", PodUID: string(pod.UID), ConnectionRevision: sources.ConnectionRevision}
				ack.Sources.DesiredRevision, ack.Sources.AppliedRevision = sources.Revision, sources.Revision
				if pod == podB {
					switch tc.name {
					case "one-lagging":
						ack.Sources.AppliedRevision = "old"
					case "one-reload-failed":
						ack.Sources.LastError = ptr.To("rejected")
					case "wrong-pod-uid":
						ack.PodUID = "replaced"
					case "wrong-connection":
						ack.ConnectionRevision = "old"
					}
				}
				listener, err := net.Listen("tcp4", net.JoinHostPort(pod.Status.PodIP, relaySourcesStatusPort))
				require.NoError(t, err)
				server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { require.NoError(t, json.NewEncoder(w).Encode(ack)) }))
				server.Listener = listener
				server.Start()
				t.Cleanup(server.Close)
			}

			t.Log("Only every desired current replica can acknowledge the source revision")
			r := &relayReconciler{Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()}
			condition, err := r.observeRelaySources(context.Background(), relay, deployment, sources)
			require.NoError(t, err)
			require.Equal(t, tc.applied, relay.Status.SourcesAppliedReplicas)
			require.Equal(t, tc.complete, condition.Status == metav1.ConditionTrue)
			if tc.complete {
				require.Equal(t, sources.Revision, relay.Status.AppliedSourcesRevision)
			} else {
				require.Empty(t, relay.Status.AppliedSourcesRevision)
			}
			if tc.name == "one-reload-failed" {
				require.Equal(t, "ReloadFailed", condition.Reason)
			}
		})
	}
}
