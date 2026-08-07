//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("Automatic checkpoint API defaulting", func() {
	It("re-adopts an RBD checkpoint after API-server defaults", func() {
		ctx := context.Background()
		expected, err := checkpoint.ExpectedAutoCheckpoint(
			k8sClient.Scheme(),
			envtestNamespace,
			"envtest-rbd-defaulting",
			nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "test/model",
				BackendFramework: "vllm",
			},
			corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  consts.MainContainerName,
						Image: "worker:expected",
					}},
					Volumes: []corev1.Volume{{
						Name: "checkpoint",
						VolumeSource: corev1.VolumeSource{
							RBD: &corev1.RBDVolumeSource{
								CephMonitors: []string{"10.0.0.1:6789"},
								RBDImage:     "checkpoint",
								FSType:       "ext4",
							},
						},
					}},
				},
			},
			"",
			nvidiacomv1alpha1.CheckpointDeletionPolicyDelete,
			nil,
			nil,
		)
		Expect(err).NotTo(HaveOccurred())
		Expect(admissionBypassClient.Create(ctx, expected.DeepCopy())).To(Succeed())
		DeferCleanup(func() {
			_ = admissionBypassClient.Delete(ctx, expected)
		})

		persisted := &nvidiacomv1alpha1.DynamoCheckpoint{}
		Expect(admissionBypassClient.Get(
			ctx,
			client.ObjectKeyFromObject(expected),
			persisted,
		)).To(Succeed())
		rbd := persisted.Spec.Job.PodTemplateSpec.Spec.Volumes[0].RBD
		Expect(rbd.RBDPool).To(Equal("rbd"))
		Expect(rbd.RadosUser).To(Equal("admin"))
		Expect(rbd.Keyring).To(Equal("/etc/ceph/keyring"))

		_, err = checkpoint.CreateOrGetAutoCheckpoint(ctx, admissionBypassClient, expected)
		Expect(err).NotTo(HaveOccurred())
	})

	It("adopts an equivalent checkpoint after representative nested CRD defaults", func() {
		ctx := context.Background()
		expected, err := checkpoint.ExpectedAutoCheckpoint(
			k8sClient.Scheme(),
			envtestNamespace,
			"envtest-defaulting",
			nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "test/model",
				BackendFramework: "vllm",
			},
			corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"capture": "defaulting"}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:    consts.MainContainerName,
						Image:   "worker:expected",
						Command: []string{"python3"},
						Args:    []string{"-m", "dynamo.vllm"},
						Env: []corev1.EnvVar{{
							Name: "TOKEN",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{Name: "runtime"},
									Key:                  "token",
								},
							},
						}},
						Ports: []corev1.ContainerPort{{Name: "system", ContainerPort: 9090}},
						LivenessProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{
							GRPC: &corev1.GRPCAction{Port: 9090},
						}},
						VolumeMounts: []corev1.VolumeMount{{Name: "runtime", MountPath: "/runtime"}},
					}},
					Volumes: []corev1.Volume{{
						Name: "runtime",
						VolumeSource: corev1.VolumeSource{
							Secret: &corev1.SecretVolumeSource{SecretName: "runtime"},
						},
					}},
				},
			},
			"",
			nvidiacomv1alpha1.CheckpointDeletionPolicyDelete,
			&nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			nil,
		)
		Expect(err).NotTo(HaveOccurred())
		serverShaped := expected.DeepCopy()
		explicitFalse := false
		explicitEmpty := ""
		main := &serverShaped.Spec.Job.PodTemplateSpec.Spec.Containers[0]
		main.Env[0].ValueFrom.SecretKeyRef.Optional = &explicitFalse
		main.LivenessProbe.GRPC.Service = &explicitEmpty
		serverShaped.Spec.Job.PodTemplateSpec.Spec.Volumes[0].Secret.Optional = &explicitFalse
		Expect(admissionBypassClient.Create(ctx, serverShaped)).To(Succeed())
		DeferCleanup(func() {
			_ = admissionBypassClient.Delete(ctx, expected)
		})

		persisted := &nvidiacomv1alpha1.DynamoCheckpoint{}
		Expect(admissionBypassClient.Get(
			ctx,
			client.ObjectKeyFromObject(expected),
			persisted,
		)).To(Succeed())
		Expect(persisted.Spec.Identity.TensorParallelSize).To(Equal(int32(1)))
		Expect(persisted.Spec.Job.ActiveDeadlineSeconds).NotTo(BeNil())
		Expect(persisted.Spec.Job.PodTemplateSpec.Spec.Containers[0].Ports[0].Protocol).
			To(Equal(corev1.ProtocolTCP))
		Expect(persisted.Spec.Job.PodTemplateSpec.Spec.Containers[0].Env[0].
			ValueFrom.SecretKeyRef.Optional).NotTo(BeNil())
		Expect(*persisted.Spec.Job.PodTemplateSpec.Spec.Containers[0].Env[0].
			ValueFrom.SecretKeyRef.Optional).To(BeFalse())
		Expect(persisted.Spec.Job.PodTemplateSpec.Spec.Volumes[0].Secret.Optional).NotTo(BeNil())
		Expect(*persisted.Spec.Job.PodTemplateSpec.Spec.Volumes[0].Secret.Optional).To(BeFalse())
		Expect(persisted.Spec.Job.PodTemplateSpec.Spec.Containers[0].
			LivenessProbe.GRPC.Service).NotTo(BeNil())
		Expect(*persisted.Spec.Job.PodTemplateSpec.Spec.Containers[0].
			LivenessProbe.GRPC.Service).To(BeEmpty())
		Expect(persisted.Spec.GPUMemoryService.Mode).
			To(Equal(nvidiacomv1alpha1.GMSModeIntraPod))

		_, err = checkpoint.CreateOrGetAutoCheckpoint(ctx, admissionBypassClient, expected)
		Expect(err).NotTo(HaveOccurred())
	})
})
