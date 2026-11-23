// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"bytes"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	controller_common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	mpiSSHTestSecretName      = "mpi-run-ssh-secret"
	mpiSSHTestSecretNamespace = "dynamo-system"
	mpiSSHTestOriginVersion   = "1.7.0"
)

// newMPISSHTestDGD describes an aggregated TRT-LLM graph whose worker spans two nodes on Grove.
func newMPISSHTestDGD(namespace, name string, uid types.UID, originVersion string) *v1beta1.DynamoGraphDeployment {
	annotations := map[string]string{"nvidia.com/enable-grove": "true"}
	if originVersion != "" {
		annotations["nvidia.com/dynamo-operator-origin-version"] = originVersion
	}
	return &v1beta1.DynamoGraphDeployment{
		Name: name, Namespace: namespace, UID: uid, Annotations: annotations,
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "trtllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeWorker,
				Replicas:      new(int32(1)),
				Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:    "main",
					Image:   "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:1.6.0",
					Command: []string{"python3", "-m", "dynamo.trtllm"},
					Args:    []string{"--model-path", "Qwen/Qwen3-0.6B"},
				}}}},
			}},
		},
	}
}

// newMPISSHTestReconciler runs the production DGD reconciler with Grove and the MPI SSH key manager.
func newMPISSHTestReconciler(t *testing.T, objects ...client.Object) *DynamoGraphDeploymentReconciler {
	t.Helper()
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	require.NoError(t, rbacv1.AddToScheme(scheme))
	plannerRole := &rbacv1.ClusterRole{Name: "planner"}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(append(objects, plannerRole)...).
		WithIndex(&corev1.Pod{}, podcontroller.ComponentIndex, podcontroller.ComponentIndexValues).
		WithStatusSubresource(&v1beta1.DynamoGraphDeployment{}).
		Build()
	mpi := configv1alpha1.MPIConfiguration{SSHSecretName: mpiSSHTestSecretName, SSHSecretNamespace: mpiSSHTestSecretNamespace}
	return &DynamoGraphDeploymentReconciler{
		Client:   kubeClient,
		Recorder: events.NewFakeRecorder(100),
		Config: &configv1alpha1.OperatorConfiguration{
			MPI:  mpi,
			RBAC: configv1alpha1.RBACConfiguration{PlannerClusterRoleName: "planner"},
		},
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{Grove: true}},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) { return nil, nil },
		},
		SSHKeyManager: secret.NewSSHKeyManager(kubeClient, mpi),
	}
}

// reconcileMPISSHTestDGD runs one DGD reconcile, which writes its shared resources and PodCliqueSet.
func reconcileMPISSHTestDGD(t *testing.T, reconciler *DynamoGraphDeploymentReconciler, dgd *v1beta1.DynamoGraphDeployment) {
	t.Helper()
	_, err := reconciler.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
	require.NoError(t, err)
}

// mpiSSHVolumes returns, per rendered Grove clique of the DGD, the Secret volume mounted at /ssh-pk.
func mpiSSHVolumes(t *testing.T, kubeClient client.Client, dgd *v1beta1.DynamoGraphDeployment) map[string]corev1.SecretVolumeSource {
	t.Helper()
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, kubeClient.List(t.Context(), sets, client.InNamespace(dgd.Namespace)))
	volumes := map[string]corev1.SecretVolumeSource{}
	for _, pcs := range sets.Items {
		if !metav1.IsControlledBy(&pcs, dgd) {
			continue
		}
		for _, clique := range pcs.Spec.Template.Cliques {
			spec := clique.Spec.PodSpec
			for _, container := range spec.Containers {
				for _, mount := range container.VolumeMounts {
					if mount.MountPath != "/ssh-pk" {
						continue
					}
					for _, volume := range spec.Volumes {
						if volume.Name == mount.Name {
							require.NotNil(t, volume.Secret, "clique %s mounts /ssh-pk from a non-Secret volume", clique.Name)
							volumes[clique.Name] = *volume.Secret
						}
					}
				}
			}
		}
	}
	require.Len(t, volumes, 2, "expected the leader and worker cliques to mount /ssh-pk")
	return volumes
}

// mountedMPISSHSecret follows the single Secret that all of the DGD's /ssh-pk volumes reference.
func mountedMPISSHSecret(t *testing.T, kubeClient client.Client, dgd *v1beta1.DynamoGraphDeployment) *corev1.Secret {
	t.Helper()
	names := map[string]bool{}
	for _, volume := range mpiSSHVolumes(t, kubeClient, dgd) {
		names[volume.SecretName] = true
	}
	require.Len(t, names, 1, "leader and worker must mount the same Secret")
	mounted := &corev1.Secret{}
	for name := range names {
		require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKey{Namespace: dgd.Namespace, Name: name}, mounted))
	}
	return mounted
}

func TestDGDReconcileGivesEachTRTLLMMultinodeDGDItsOwnMPISSHKey(t *testing.T) {
	t.Run("leader_and_worker_mount_the_same_secret", func(t *testing.T) {
		t.Log("Reconcile one TRT-LLM multinode DGD")
		dgd := newMPISSHTestDGD("team-a", "graph", "uid-a", mpiSSHTestOriginVersion)
		reconciler := newMPISSHTestReconciler(t, dgd)
		reconcileMPISSHTestDGD(t, reconciler, dgd)

		t.Log("The leader and worker cliques mount one Secret that this DGD controls")
		mounted := mountedMPISSHSecret(t, reconciler.Client, dgd)
		require.True(t, metav1.IsControlledBy(mounted, dgd), "Secret %s is not controlled by the DGD", mounted.Name)
		require.NotEmpty(t, mounted.Data["private.key"])
		require.NotEmpty(t, mounted.Data["private.key.pub"])
	})

	t.Run("recreated_dgd_gets_a_new_key_pair", func(t *testing.T) {
		t.Log("Reconcile a DGD and record its mounted key pair")
		original := newMPISSHTestDGD("team-a", "graph", "uid-original", mpiSSHTestOriginVersion)
		reconciler := newMPISSHTestReconciler(t, original)
		reconcileMPISSHTestDGD(t, reconciler, original)
		before := mountedMPISSHSecret(t, reconciler.Client, original)

		t.Log("Recreate the DGD under the same name before garbage collection removes the earlier DGD's Secret")
		leftover := before.DeepCopy()
		leftover.ResourceVersion = ""
		recreated := newMPISSHTestDGD("team-a", "graph", "uid-recreated", mpiSSHTestOriginVersion)
		reconciler = newMPISSHTestReconciler(t, recreated, leftover)
		reconcileMPISSHTestDGD(t, reconciler, recreated)

		t.Log("The recreated DGD mounts a different key pair that it controls")
		after := mountedMPISSHSecret(t, reconciler.Client, recreated)
		require.True(t, metav1.IsControlledBy(after, recreated), "Secret %s is not controlled by the recreated DGD", after.Name)
		require.False(t, bytes.Equal(before.Data["private.key"], after.Data["private.key"]), "the two Secrets hold the same private.key")
		require.False(t, bytes.Equal(before.Data["private.key.pub"], after.Data["private.key.pub"]), "the two Secrets hold the same private.key.pub")
	})

	t.Run("dgds_in_different_namespaces_use_different_private_keys", func(t *testing.T) {
		t.Log("Reconcile two TRT-LLM multinode DGDs in different namespaces")
		first := newMPISSHTestDGD("team-a", "graph", "uid-a", mpiSSHTestOriginVersion)
		second := newMPISSHTestDGD("team-b", "graph", "uid-b", mpiSSHTestOriginVersion)
		reconciler := newMPISSHTestReconciler(t, first, second)
		reconcileMPISSHTestDGD(t, reconciler, first)
		reconcileMPISSHTestDGD(t, reconciler, second)

		t.Log("Their mounted Secrets hold different private keys")
		firstSecret := mountedMPISSHSecret(t, reconciler.Client, first)
		secondSecret := mountedMPISSHSecret(t, reconciler.Client, second)
		require.False(t, bytes.Equal(firstSecret.Data["private.key"], secondSecret.Data["private.key"]), "the two Secrets hold the same private.key")
	})

	t.Run("dgds_in_the_same_namespace_use_different_private_keys", func(t *testing.T) {
		t.Log("Reconcile two TRT-LLM multinode DGDs in one namespace")
		first := newMPISSHTestDGD("team-a", "graph-one", "uid-one", mpiSSHTestOriginVersion)
		second := newMPISSHTestDGD("team-a", "graph-two", "uid-two", mpiSSHTestOriginVersion)
		reconciler := newMPISSHTestReconciler(t, first, second)
		reconcileMPISSHTestDGD(t, reconciler, first)
		reconcileMPISSHTestDGD(t, reconciler, second)

		t.Log("Their mounted Secrets hold different private keys")
		firstSecret := mountedMPISSHSecret(t, reconciler.Client, first)
		secondSecret := mountedMPISSHSecret(t, reconciler.Client, second)
		require.False(t, bytes.Equal(firstSecret.Data["private.key"], secondSecret.Data["private.key"]), "the two Secrets hold the same private.key")
	})

	t.Run("ssh_volume_mode_has_no_group_or_other_bits", func(t *testing.T) {
		t.Log("Reconcile one TRT-LLM multinode DGD")
		dgd := newMPISSHTestDGD("team-a", "graph", "uid-a", mpiSSHTestOriginVersion)
		reconciler := newMPISSHTestReconciler(t, dgd)
		reconcileMPISSHTestDGD(t, reconciler, dgd)

		t.Log("Every /ssh-pk volume grants no group or other permission bits")
		for clique, volume := range mpiSSHVolumes(t, reconciler.Client, dgd) {
			require.NotNil(t, volume.DefaultMode, "clique %s", clique)
			require.Zero(t, *volume.DefaultMode&0o077, "clique %s has DefaultMode %#o", clique, *volume.DefaultMode)
		}
	})

	t.Run("dgd_without_origin_version_keeps_the_shared_secret", func(t *testing.T) {
		t.Log("Reconcile a DGD created before the operator recorded origin versions")
		dgd := newMPISSHTestDGD("team-a", "graph", "uid-a", "")
		reconciler := newMPISSHTestReconciler(t, dgd)
		reconcileMPISSHTestDGD(t, reconciler, dgd)

		t.Log("Its Pods keep the shared replica with the mode they were created with, so an upgrade does not roll them")
		for clique, volume := range mpiSSHVolumes(t, reconciler.Client, dgd) {
			require.Equal(t, mpiSSHTestSecretName, volume.SecretName, "clique %s", clique)
			require.Equal(t, int32(0o644), *volume.DefaultMode, "clique %s", clique)
		}
		require.NoError(t, reconciler.Get(t.Context(), client.ObjectKey{Namespace: dgd.Namespace, Name: mpiSSHTestSecretName}, &corev1.Secret{}))
	})
}
