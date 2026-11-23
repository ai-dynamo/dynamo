//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
)

func setupProductionWebhooks(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
	return webhooksetup.Setup(mgr, webhooksetup.Options{
		Config:            opts.OperatorConfig,
		RuntimeConfig:     opts.RuntimeConfig,
		OperatorVersion:   opts.OperatorVersion,
		OperatorPrincipal: opts.OperatorPrincipal,
	})
}

func newEnvtestDSHappyPathDGD(name string) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			UID:       types.UID(fmt.Sprintf("%s-uid", name)),
			Annotations: map[string]string{
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:          "prefill",
					ComponentType:          nvidiacomv1beta1.ComponentTypePrefill,
					RuntimeVersionOverride: "1.0.0",
					Multinode:              &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					PodTemplate:            envtestDSTestPodTemplate(),
				},
				{
					ComponentName:          "decode",
					ComponentType:          nvidiacomv1beta1.ComponentTypeDecode,
					RuntimeVersionOverride: "1.0.0",
					Multinode:              &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					PodTemplate:            envtestDSTestPodTemplate(),
				},
			},
		},
	}
}

func envtestDSTestPodTemplate() *corev1.PodTemplateSpec {
	return &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    consts.MainContainerName,
				Image:   "busybox:1.36",
				Command: []string{"sh"},
				Args:    []string{"-c", "sleep 3600"},
			}},
		},
	}
}
