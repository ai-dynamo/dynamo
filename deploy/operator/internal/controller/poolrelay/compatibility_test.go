/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"slices"
	"testing"
)

func TestDynamoPoolRelayDiscoveryMatchesWorkerDefaults(t *testing.T) {
	t.Log("Select a DGD with metadata discovery-mode annotation")
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	c := fake.NewClientBuilder().WithScheme(scheme).Build()
	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Discovery.Backend = configv1alpha1.DiscoveryBackendKubernetes
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.6.0"}}}}}
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers", Annotations: map[string]string{commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}}, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}}}
	t.Log("Verify worker rendering inherits metadata, and Relay rejects its unsupported discovery mode")
	pod, err := dynamo.GeneratePodSpecForComponent(&component, dynamo.BackendFrameworkNoop, nil, dgd, dynamo.RoleMain, 1, cfg, commonconsts.MultinodeDeploymentTypeGrove, "worker", nil, nil)
	require.NoError(t, err)
	require.Contains(t, pod.Containers[0].Env, corev1.EnvVar{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "container"})
	conn, err := relayWorkerConnection(context.Background(), c, cfg, dgd, &component)
	t.Logf("Worker discovery mode=container; Relay connection=%+v; resolution error=%v", conn.Env, err)
	require.ErrorContains(t, err, "Kubernetes pod discovery")
}

func TestDynamoPoolRelayRolePodTemplates(t *testing.T) {
	t.Log("Declare a multinode worker with complete leader and worker templates")
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	token := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "role-auth", Namespace: "workers"}, Data: map[string][]byte{"token": []byte("fixture-token")}}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(token).Build()
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, Multinode: &v1beta1.MultinodeSpec{NodeCount: 2}}
	for _, role := range []string{v1beta1.ComponentRoleLeader, v1beta1.ComponentRoleWorker} {
		component.Roles = append(component.Roles, v1beta1.ComponentRoleSpec{Name: role, Replicas: ptr.To(int32(1)), PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.6.0", Command: []string{"python3", "-m", "dynamo.vllm"}, Env: []corev1.EnvVar{
			{Name: "DYN_EVENT_PLANE", Value: "nats"}, {Name: "NATS_SERVER", Value: "nats://fixture:4222"},
			{Name: "NATS_AUTH_TOKEN", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: token.Name}, Key: "token"}}},
		}}}}}})
	}
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"}, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}}}
	cfg := &configv1alpha1.OperatorConfiguration{}
	before := dgd.DeepCopy()
	t.Log("The shared role resolver yields compatible supported transports for both roles")
	require.Len(t, dynamo.ComponentPodTemplates(&component), 2)
	for _, role := range []dynamo.Role{dynamo.RoleLeader, dynamo.RoleWorker} {
		effective, err := dynamo.EffectiveComponentForRole(&component, role)
		require.NoError(t, err)
		_, err = relayWorkerConnection(context.Background(), c, cfg, dgd, effective)
		require.NoError(t, err)
	}
	t.Log("Resolve the same authored DGD and collect its transport dependencies")
	_, _, err := resolveRelayConnections(context.Background(), c, cfg, []v1beta1.DynamoGraphDeployment{*dgd})
	require.NoError(t, err)
	require.Equal(t, before, dgd)
	deps := relayDGDDependencies(dgd)
	t.Logf("Relay dependency index=%v", deps)
	require.True(t, slices.Contains(deps, "Secret/role-auth"))

	t.Log("Reject incompatible transports between roles instead of dropping either role")
	dgd.Spec.Components[0].Roles[1].PodTemplate.Spec.Containers[0].Env[0].Value = "zmq"
	_, _, err = resolveRelayConnections(context.Background(), c, cfg, []v1beta1.DynamoGraphDeployment{*dgd})
	require.ErrorContains(t, err, "incompatible transport configuration")
}

func TestDynamoPoolRelayResourceReplacementConflicts(t *testing.T) {
	t.Log("Observe an owned sources ConfigMap before it is replaced at the same name")
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, v1alpha1.AddToScheme(scheme))
	relay := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "relay", Namespace: "workers", UID: "relay-uid"}}
	current := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: relayResourceName(relay.Name), Namespace: relay.Namespace, UID: "owned-cm", ResourceVersion: "10"}, Data: map[string]string{"sources.json": "old-owned-sources"}}
	require.NoError(t, controllerutil.SetControllerReference(relay, current, scheme))
	desired := current.DeepCopy()
	desired.ResourceVersion = ""
	desired.UID = ""
	desired.Data["sources.json"] = "new-owned-sources"
	called := false
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(current).WithInterceptorFuncs(interceptor.Funcs{
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			require.False(t, called)
			called = true
			old := &corev1.ConfigMap{}
			require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(obj), old))
			require.NoError(t, c.Delete(ctx, old))
			replacement := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: obj.GetName(), Namespace: obj.GetNamespace(), UID: "foreign-cm"}, Data: map[string]string{"sources.json": "foreign-sources"}}
			require.NoError(t, c.Create(ctx, replacement))
			return c.Patch(ctx, obj, patch, opts...)
		},
	}).Build()
	t.Log("Inject replacement after the ownership check and before the actual patch")
	r := &relayReconciler{Client: c}
	err := r.reconcileRelayResource(ctx, relay, desired)
	actual := &corev1.ConfigMap{}
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(current), actual))
	require.True(t, called)
	require.True(t, apierrors.IsConflict(err), "write must conflict with the replaced resource: %v", err)
	require.Equal(t, "foreign-cm", string(actual.UID))
	t.Logf("patch error=%v; replacement data=%q", err, actual.Data["sources.json"])
	require.Equal(t, "foreign-sources", actual.Data["sources.json"], "A resource outside Relay ownership must remain unchanged")
}
