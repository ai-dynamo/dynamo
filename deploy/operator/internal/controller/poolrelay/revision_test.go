/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoPoolRelayRevisionTracksOnlyReferencedData(t *testing.T) {
	t.Log("Resolve a token Secret and a mounted credentials file")
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "auth", Namespace: "workers"}, Data: map[string][]byte{"token": []byte("token-a"), "creds": []byte("creds-a"), "unrelated": []byte("unrelated-a")}}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithObjects(secret).Build()
	connection := relayConnection{Env: []corev1.EnvVar{{Name: "NATS_AUTH_TOKEN", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "auth"}, Key: "token"}}}}, Volumes: []corev1.Volume{{Name: "auth", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "auth", Items: []corev1.KeyToPath{{Key: "creds", Path: "client.creds"}}}}}}}
	key := []byte("private-revision-key-for-test-only")
	original, err := relayContentRevision(ctx, c, "workers", connection, key)
	require.NoError(t, err)

	t.Log("Metadata and unrelated data changes do not change the rollout revision")
	secret.Labels = map[string]string{"changed": "metadata"}
	secret.Data["unrelated"] = []byte("unrelated-b")
	require.NoError(t, c.Update(ctx, secret))
	current, err := relayContentRevision(ctx, c, "workers", connection, key)
	require.NoError(t, err)
	require.Equal(t, original, current)

	t.Log("Changing either selected authentication input rotates the revision")
	secret.Data["token"] = []byte("token-b")
	require.NoError(t, c.Update(ctx, secret))
	tokenRevision, err := relayContentRevision(ctx, c, "workers", connection, key)
	require.NoError(t, err)
	require.NotEqual(t, original, tokenRevision)
	secret.Data["creds"] = []byte("creds-b")
	require.NoError(t, c.Update(ctx, secret))
	fileRevision, err := relayContentRevision(ctx, c, "workers", connection, key)
	require.NoError(t, err)
	require.NotEqual(t, tokenRevision, fileRevision)

	t.Log("The public revision cannot be reproduced without the per-Relay private key")
	other, err := relayContentRevision(ctx, c, "workers", connection, []byte("other-private-key"))
	require.NoError(t, err)
	require.NotEqual(t, fileRevision, other)
	require.NotContains(t, fileRevision, "token-b")
}
