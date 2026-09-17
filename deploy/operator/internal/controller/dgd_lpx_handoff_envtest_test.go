//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"capnproto.org/go/capnp/v3"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	groveschedulerv1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

func TestLPXPublicationFailureReachesDGDThroughSetup(t *testing.T) {
	t.Log("Encode a real two-partition LPU build owned by this test")
	buildDir := t.TempDir()
	message, segment := capnp.NewSingleSegmentMessage(nil)
	manifest, err := manifestcapnpv2.NewRootManifest(segment)
	require.NoError(t, err)
	manifest.SetContractRevision(manifestcapnpv2.CurrentContractRevision)
	model, err := manifest.NewModel()
	require.NoError(t, err)
	tokenizer, err := model.NewTokenizer()
	require.NoError(t, err)
	require.NoError(t, tokenizer.SetPath("tokenizer"))
	stopTokens, err := tokenizer.NewStopTokens(1)
	require.NoError(t, err)
	stopTokens.Set(0, 1)
	build, err := capnp.NewStruct(manifest.Segment(), capnp.ObjectSize{DataSize: 8, PointerCount: 12})
	require.NoError(t, err)
	require.NoError(t, build.SetText(11, "publication-test"))
	require.NoError(t, manifest.SetReserved3(build.ToPtr()))
	deployment, err := manifest.NewDeployment()
	require.NoError(t, err)
	deployment.SetCompilationMode(manifestcapnpv2.CompilationMode_lpuOnly)
	deployment.SetNumLpuNodes(4)
	program, err := deployment.NewProgram()
	require.NoError(t, err)
	program.SetBatchSize(1)
	program.SetSequenceLength(8192)
	program.SetInputSize(1)
	program.SetOutputSize(1)
	program.SetNumKvCaches(1)
	program.SetNumBatchSplitDivisions(1)
	runtimeIO, err := deployment.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(0)
	runtimeIO.SetReserved1(1)
	runtimeIO.SetIoFpgaCount(1)
	runtimeIO.SetFanoutFactor(1)
	chains, err := deployment.NewSelectedPropSyncChains(1)
	require.NoError(t, err)
	partitionIDs, err := chains.At(0).NewPartitionIds(2)
	require.NoError(t, err)
	partitionIDs.Set(0, 7)
	partitionIDs.Set(1, 8)
	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	partitions, err := artifacts.NewPartitions(2)
	require.NoError(t, err)
	for index := range partitions.Len() {
		partition, err := partitions.At(index).NewPartition()
		require.NoError(t, err)
		partition.SetDeviceType(manifestcapnpv2.DeviceType_lpu)
		partition.SetPartitionId(uint32(index + 7))
		detail, err := partitions.At(index).Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", index+7)))
		require.NoError(t, detail.SetTopology("URSA_V2__Q8__16C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA"))
		detail.SetNumChips(16)
		detail.SetDevicesPerNode(8)
	}
	payload, err := message.Marshal()
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.v2.capnp.bin"), payload, 0o600))

	t.Log("Start an isolated API server and register the external kinds needed only for watches")
	config := &configv1alpha1.OperatorConfiguration{}
	config.LPX.Enabled = true
	config.MPI.SSHSecretName = "ssh-secret"
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LPX: true}}
	env := operatorenv.New(operatorenv.Options{
		Config: config, RuntimeConfig: runtimeConfig,
		Admission: operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
		SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
			if err := lpxv1alpha1.AddToScheme(mgr.GetScheme()); err != nil {
				return err
			}
			if err := groveschedulerv1alpha1.AddToScheme(mgr.GetScheme()); err != nil {
				return err
			}
			return setupProductionWebhooks(mgr, opts)
		},
	}).RunT(t)
	dependencies := make([]*apiextensionsv1.CustomResourceDefinition, 0, 2)
	for _, dependency := range []struct{ group, kind, plural string }{
		{lpxv1alpha1.APIGroup, "LpuPipelineRequest", "lpupipelinerequests"},
		{groveschedulerv1alpha1.SchemeGroupVersion.Group, "PodGang", "podgangs"},
	} {
		dependencies = append(dependencies, &apiextensionsv1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{Name: dependency.plural + "." + dependency.group},
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Group: dependency.group, Scope: apiextensionsv1.NamespaceScoped,
				Names: apiextensionsv1.CustomResourceDefinitionNames{Kind: dependency.kind, ListKind: dependency.kind + "List", Plural: dependency.plural},
				Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
					Name: "v1alpha1", Served: true, Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object", XPreserveUnknownFields: ptr.To(true)}},
				}},
			},
		})
	}
	_, err = envtest.InstallCRDs(env.RESTConfig(), envtest.CRDInstallOptions{CRDs: dependencies})
	require.NoError(t, err)

	t.Log("Deny Grove publication with real namespace quota admission, not a mocked client")
	quotaResource := corev1.ResourceName("count/podcliquesets.grove.io")
	quota := &corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "block-lpx-publication", Namespace: env.Namespace()},
		Spec:       corev1.ResourceQuotaSpec{Hard: corev1.ResourceList{quotaResource: resource.MustParse("0")}},
	}
	require.NoError(t, env.Client().Create(t.Context(), quota))
	quota.Status = corev1.ResourceQuotaStatus{Hard: quota.Spec.Hard.DeepCopy(), Used: corev1.ResourceList{quotaResource: resource.MustParse("0")}}
	require.NoError(t, env.Client().Status().Update(t.Context(), quota))
	config = env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			SetupOptions: SetupOptions{Config: config, RuntimeConfig: runtimeConfig},
		})
	})

	t.Log("Create only the public DGD and let production setup reconcile both controllers")
	source := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "publication-failure", Namespace: env.Namespace()},
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
			ComponentName: "lpx", ComponentType: v1beta1.ComponentTypeLPX,
			LPX: &v1beta1.LPXConfig{BuildID: (&url.URL{Scheme: "file", Path: buildDir}).String()},
			Roles: []v1beta1.ComponentRoleSpec{{Name: v1beta1.ComponentRoleLPXConductor}, {
				Name: v1beta1.ComponentRoleLPXAgent,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: consts.MainContainerName, Image: "example/lpu-runtime:1.4.0",
						VolumeMounts: []corev1.VolumeMount{
							{Name: consts.ModelStorageVolumeName, MountPath: "/models"},
							{Name: "config", MountPath: "/configs"},
							{Name: "host-dev", MountPath: "/dev"},
							{Name: "host-sys", MountPath: "/sys"},
							{Name: "hugepages", MountPath: "/dev/hugepages"},
							{Name: "ssh-secret", MountPath: "/ssh-pk", ReadOnly: true},
							{Name: "single-v2-ssh-key", MountPath: "/tmp/dynamo-lpu-ssh"},
						},
					}},
					Volumes: []corev1.Volume{
						{Name: consts.ModelStorageVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
						{Name: "host-dev", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/dev"}}},
						{Name: "host-sys", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/sys"}}},
						{Name: "hugepages", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumHugePages}}},
						{Name: "ssh-secret", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "ssh-secret"}}},
						{Name: "single-v2-ssh-key", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
					},
				}},
			}},
		}}},
	}
	require.NoError(t, env.Client().Create(t.Context(), source))
	key := client.ObjectKeyFromObject(source)
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, source)) {
			return
		}
		ready := meta.FindStatusCondition(source.Status.Conditions, "Ready")
		if assert.NotNil(c, ready) {
			assert.Contains(c, ready.Message, "exceeded quota: block-lpx-publication")
		}
	}, 20*time.Second, 50*time.Millisecond)

	t.Log("Change LPX intent so the failing child generation is newer than its completed observation")
	require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := env.Client().Get(t.Context(), key, source); err != nil {
			return err
		}
		source.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0].Image = "example/lpu-runtime:1.4.1"
		return env.Client().Update(t.Context(), source)
	}))
	child := &v1alpha1.LPXGraphDeployment{}
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, source)) || !assert.NoError(c, env.Client().Get(t.Context(), key, child)) {
			return
		}
		failed := meta.FindStatusCondition(child.Status.Conditions, "Failed")
		ready := meta.FindStatusCondition(source.Status.Conditions, "Ready")
		if !assert.NotNil(c, failed) || !assert.NotNil(c, ready) {
			return
		}
		assert.Greater(c, child.Generation, child.Status.ObservedGeneration)
		assert.Equal(c, child.Generation, failed.ObservedGeneration)
		assert.Equal(c, metav1.ConditionTrue, failed.Status)
		assert.Contains(c, failed.Message, "exceeded quota: block-lpx-publication")
		assert.Equal(c, v1beta1.DGDStateFailed, source.Status.State)
		assert.Equal(c, source.Generation, ready.ObservedGeneration)
		assert.Equal(c, metav1.ConditionFalse, ready.Status)
		assert.Equal(c, failed.Reason, ready.Reason)
		assert.Equal(c, failed.Message, ready.Message)
		assert.NotNil(c, child.Status.ModelDownload)
		assert.Nil(c, source.Status.LPX)
	}, 20*time.Second, 50*time.Millisecond)

	t.Log("The rejected publication created neither Grove workloads nor scheduler requests")
	cliques := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, env.Client().List(t.Context(), cliques, client.InNamespace(env.Namespace())))
	require.Empty(t, cliques.Items)
	requests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, env.Client().List(t.Context(), requests, client.InNamespace(env.Namespace())))
	require.Empty(t, requests.Items)

	t.Log("Accept an LPX runtime-invalid edit and report its exact rejection on the public DGD")
	require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := env.Client().Get(t.Context(), key, source); err != nil {
			return err
		}
		source.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.NodeName = "manual-placement"
		return env.Client().Update(t.Context(), source)
	}))
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, source)) || !assert.NoError(c, env.Client().Get(t.Context(), key, child)) {
			return
		}
		failed := meta.FindStatusCondition(child.Status.Conditions, "Failed")
		ready := meta.FindStatusCondition(source.Status.Conditions, "Ready")
		if !assert.NotNil(c, failed) || !assert.NotNil(c, ready) {
			return
		}
		assert.Equal(c, child.Generation, failed.ObservedGeneration)
		assert.Equal(c, "LPXRejected", failed.Reason)
		assert.Contains(c, failed.Message, "spec.components[0].roles[1].podTemplate.spec.nodeName")
		assert.Contains(c, failed.Message, "LPX owns role addressing and placement")
		assert.Equal(c, v1beta1.DGDStateFailed, source.Status.State)
		assert.Equal(c, source.Generation, ready.ObservedGeneration)
		assert.Equal(c, metav1.ConditionFalse, ready.Status)
		assert.Equal(c, failed.Reason, ready.Reason)
		assert.Equal(c, failed.Message, ready.Message)
	}, 20*time.Second, 50*time.Millisecond)

	t.Log("Repair the accepted configuration so the same DGD and child can reconcile again")
	require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := env.Client().Get(t.Context(), key, source); err != nil {
			return err
		}
		source.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.NodeName = ""
		return env.Client().Update(t.Context(), source)
	}))

	t.Log("Release quota and observe alpha LGD ownership of the published PCS and runtime resources without waiting for scheduling")
	require.NoError(t, env.Client().Delete(t.Context(), quota))
	pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: dynamo.PCSNameForLPX(child), Namespace: source.Namespace}}
	configMap := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: source.Namespace}}
	service := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: pcs.Name + "-serve", Namespace: source.Namespace}}
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs)) {
			return
		}
		configMap.Name = dynamolpx.LPUConfigMapName(pcs.Name, pcs.Spec.Template.Cliques[0].Annotations[consts.AnnotationExtraResourcesHash])
		for _, resource := range []client.Object{pcs, configMap, service} {
			if assert.NoError(c, env.Client().Get(t.Context(), client.ObjectKeyFromObject(resource), resource)) {
				assert.Equal(c, metav1.NewControllerRef(child, v1alpha1.LPXGraphDeploymentGVK), metav1.GetControllerOf(resource))
			}
		}
		assert.Equal(c, ptr.To(true), configMap.Immutable)
	}, 20*time.Second, 50*time.Millisecond)
}
