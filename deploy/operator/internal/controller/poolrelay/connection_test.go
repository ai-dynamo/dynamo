/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"fmt"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoPoolRelayConnectionInheritance(t *testing.T) {
	t.Log("Declare graph defaults, component overrides, and credentials supplied as a file")
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "nats-auth", Namespace: "workers"}, Data: map[string][]byte{"credentials": []byte("test-credential-payload")}}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithObjects(secret).Build()
	config := &configv1alpha1.OperatorConfiguration{}
	config.Infrastructure.NATSAddress = "nats://operator:4222"
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"}, Spec: v1beta1.DynamoGraphDeploymentSpec{Env: []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "nats"}, {Name: "NATS_SERVER", Value: "nats://graph:4222"}}}}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Env: []corev1.EnvVar{{Name: "NATS_SERVER", Value: "nats://component:4222"}, {Name: "NATS_AUTH_CREDENTIALS_FILE", Value: "/secrets/relay.creds"}, {Name: "HF_TOKEN", Value: "unrelated-secret"}}, VolumeMounts: []corev1.VolumeMount{{Name: "auth", MountPath: "/secrets"}, {Name: "unrelated", MountPath: "/model"}}}},
		Volumes:    []corev1.Volume{{Name: "auth", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "nats-auth", Items: []corev1.KeyToPath{{Key: "credentials", Path: "relay.creds"}}}}}},
	}}}
	before := component.DeepCopy()
	result, err := relayWorkerConnection(context.Background(), c, config, dgd, component)
	require.NoError(t, err)

	t.Log("Only connection configuration is inherited and source objects remain unchanged")
	values := map[string]string{}
	for _, entry := range result.Env {
		values[entry.Name] = entry.Value
	}
	require.Equal(t, "nats", values["DYN_EVENT_PLANE"])
	require.Equal(t, "tcp", values["DYN_REQUEST_PLANE"])
	require.Equal(t, "nats://component:4222", values["NATS_SERVER"])
	require.NotContains(t, values, "HF_TOKEN")
	require.Len(t, result.Volumes, 1)
	require.Len(t, result.Mounts, 1)
	require.True(t, result.Mounts[0].ReadOnly)
	require.Equal(t, before, component)
	require.NotContains(t, result.revision(), "test-credential-payload")
}

func TestDynamoPoolRelayConnectionErrorsAreSanitized(t *testing.T) {
	for _, tc := range []struct{ name, key, value, errorText string }{
		{"event", "DYN_EVENT_PLANE", "private-invalid-value", "unsupported DYN_EVENT_PLANE"},
		{"request", "DYN_REQUEST_PLANE", "private-invalid-value", "unsupported DYN_REQUEST_PLANE"},
		{"discovery", "DYN_DISCOVERY_BACKEND", "private-invalid-value", "Kubernetes pod discovery"},
		{"file", "NATS_AUTH_CREDENTIALS_FILE", "/private-missing-file", "no corresponding volume mount"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve a transport value through a Kubernetes Secret")
			scheme := runtime.NewScheme()
			require.NoError(t, clientgoscheme.AddToScheme(scheme))
			secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "config", Namespace: "workers"}, Data: map[string][]byte{"value": []byte(tc.value)}}
			c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithObjects(secret).Build()
			dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"}}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Env: []corev1.EnvVar{{Name: tc.key, ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "config"}, Key: "value"}}}}}}}}}
			_, err := relayWorkerConnection(context.Background(), c, &configv1alpha1.OperatorConfiguration{}, dgd, component)
			require.ErrorContains(t, err, tc.errorText)
			require.NotContains(t, err.Error(), tc.value)
		})
	}
}

func TestDynamoPoolRelayConnectionRetainsKeyReferences(t *testing.T) {
	t.Log("Resolve event selection and NATS authentication through Kubernetes references")
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "auth", Namespace: "workers"}, Data: map[string][]byte{"token": []byte("private-token")}}
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "transport", Namespace: "workers"}, Data: map[string]string{"events": "nats"}}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithObjects(secret, cm).Build()
	env := []corev1.EnvVar{
		{Name: "DYN_EVENT_PLANE", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "transport"}, Key: "events"}}},
		{Name: "NATS_SERVER", Value: "nats://server:4222"},
		{Name: "NATS_AUTH_TOKEN", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "auth"}, Key: "token"}}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"}}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Env: env}}}}}
	result, err := relayWorkerConnection(context.Background(), c, &configv1alpha1.OperatorConfiguration{}, dgd, component)
	require.NoError(t, err)

	t.Log("Rendering inputs preserve references and do not materialize credential contents")
	for _, entry := range result.Env {
		if entry.Name == "NATS_AUTH_TOKEN" {
			require.Equal(t, env[2], entry)
			require.Empty(t, entry.Value)
		}
		require.NotEqual(t, "private-token", entry.Value)
	}
	require.Contains(t, result.Env, env[0])
	require.Contains(t, result.Env, env[2])
}

func TestDynamoPoolRelayLegacyDGDConversionPreservesConnection(t *testing.T) {
	t.Log("Convert the supported legacy DGD representation through the existing API conversion")
	legacy := &v1alpha1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"}, Spec: v1alpha1.DynamoGraphDeploymentSpec{
		Envs:     []corev1.EnvVar{{Name: "DYN_EVENT_PLANE", Value: "nats"}, {Name: "NATS_SERVER", Value: "nats://graph:4222"}},
		Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{"Worker": {ComponentType: "worker", Envs: []corev1.EnvVar{{Name: "NATS_SERVER", Value: "nats://worker:4222"}}, ExtraPodSpec: &v1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{Image: "example/worker:1.5.0"}}}},
	}}
	converted := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, legacy.ConvertTo(converted))
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).Build()
	config := &configv1alpha1.OperatorConfiguration{}
	inherited, _, err := resolveRelayConnections(context.Background(), c, config, []v1beta1.DynamoGraphDeployment{*converted})
	require.NoError(t, err)

	t.Log("Equivalent current-version env configuration resolves to the same profile")
	current := &v1beta1.DynamoGraphDeployment{ObjectMeta: legacy.ObjectMeta, Spec: v1beta1.DynamoGraphDeploymentSpec{
		Env:        legacy.Spec.Envs,
		Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: "worker", PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.5.0", Env: []corev1.EnvVar{{Name: "NATS_SERVER", Value: "nats://worker:4222"}}}}}}}},
	}}
	expected, _, err := resolveRelayConnections(context.Background(), c, config, []v1beta1.DynamoGraphDeployment{*current})
	require.NoError(t, err)
	require.Equal(t, expected, inherited)
}

func TestDynamoPoolRelayRejectsUnrepresentableSources(t *testing.T) {
	for _, tc := range []struct {
		name      string
		count     int
		longNames bool
		want      string
	}{
		{name: "normalized namespace collision", count: 2, want: "same Dynamo namespace"},
		{name: "source count limit", count: relayMaxSources + 1, want: "sources source limit exceeded"},
		{name: "serialized size limit", count: relayMaxSources, longNames: true, want: "sources file size limit exceeded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Construct a selection that the runtime sources protocol cannot accept")
			dgds := make([]v1beta1.DynamoGraphDeployment, tc.count)
			for i := range dgds {
				name := fmt.Sprintf("model-%04d", i)
				if tc.longNames {
					name += strings.Repeat("a", 240)
				}
				dgds[i].ObjectMeta = metav1.ObjectMeta{Name: name, Namespace: "workers", UID: types.UID(fmt.Sprintf("uid-%d", i))}
			}
			if tc.count == 2 {
				dgds[0].Name, dgds[1].Name = "model.a", "model-a"
			}
			scheme := runtime.NewScheme()
			require.NoError(t, clientgoscheme.AddToScheme(scheme))
			reader := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).Build()

			t.Log("Reject the selection before worker configuration or sources publication")
			_, _, err := resolveRelayConnections(context.Background(), reader, &configv1alpha1.OperatorConfiguration{}, dgds)
			require.ErrorContains(t, err, tc.want)
		})
	}
}

func TestDynamoPoolRelayBackendEnvFrom(t *testing.T) {
	t.Log("Backend-only envFrom is not inherited and key rotation is revalidated")
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "backend", Namespace: "workers"}, Data: map[string][]byte{"HF_TOKEN": []byte("private-backend-token")}}
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "settings", Namespace: "workers"}, Data: map[string]string{"LOG_LEVEL": "debug"}}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret, cm).Build()
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"}}
	sources := []corev1.EnvFromSource{{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "backend"}}}, {ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "settings"}}}}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", EnvFrom: sources}}}}}
	before := component.DeepCopy()
	result, err := relayWorkerConnection(context.Background(), c, &configv1alpha1.OperatorConfiguration{}, dgd, component)
	require.NoError(t, err)
	require.Equal(t, before, component)
	for _, entry := range result.Env {
		require.NotEqual(t, "HF_TOKEN", entry.Name)
		require.NotEqual(t, "LOG_LEVEL", entry.Name)
	}
	require.Empty(t, result.Volumes)

	t.Log("Adding a transport key to a referenced Secret blocks configuration without exposing values")
	secret.Data["NATS_SERVER"] = []byte("private-nats-address")
	require.NoError(t, c.Update(context.Background(), secret))
	_, err = relayWorkerConnection(context.Background(), c, &configv1alpha1.OperatorConfiguration{}, dgd, component)
	require.ErrorContains(t, err, "contains transport settings")
	require.NotContains(t, err.Error(), "private-nats-address")

	t.Log("Prefixes are considered, and optional missing sources remain valid")
	for _, tc := range []struct {
		name, prefix, key            string
		optional, missing, wantError bool
	}{
		{name: "prefixed-transport", prefix: "NATS_", key: "SERVER", wantError: true},
		{name: "backend-prefix", prefix: "BACKEND_", key: "NATS_SERVER"},
		{name: "missing-required", missing: true, wantError: true},
		{name: "missing-optional", missing: true, optional: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ref := corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: tc.name}, Optional: &tc.optional}
			if !tc.missing {
				require.NoError(t, c.Create(context.Background(), &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: tc.name, Namespace: "workers"}, Data: map[string]string{tc.key: "private-value"}}))
			}
			err := validateRelayWorkerEnvFrom(context.Background(), c, "workers", []corev1.EnvFromSource{{Prefix: tc.prefix, ConfigMapRef: &ref}})
			if tc.wantError {
				require.Error(t, err)
				require.NotContains(t, err.Error(), "private-value")
			} else {
				require.NoError(t, err)
			}
		})
	}
}
