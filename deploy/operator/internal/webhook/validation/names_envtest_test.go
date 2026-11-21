// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation_test

import (
	"fmt"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func serviceNameAdmissionError(path, name string) string {
	message := fmt.Sprintf("%s: Invalid value: %q: generated Service name is invalid: %s; choose a shorter name starting with a letter", path, name, strings.Join(k8svalidation.IsDNS1035Label(name), "; "))
	if strings.HasSuffix(name, "-00000000") {
		message += "; worker hash is shown as zeros"
	}
	return message
}

func selectorAdmissionError(path, value string) string {
	message := fmt.Sprintf("%s: Invalid value: %q: generated %s label value is invalid: %s; choose a shorter deployment or component name", path, value, consts.KubeLabelDynamoSelector, strings.Join(k8svalidation.IsValidLabelValue(value), "; "))
	if strings.HasSuffix(value, "-00000000") {
		message += "; worker hash is shown as zeros"
	}
	return message
}

func graphNameAdmissionError(name string) string {
	return fmt.Sprintf("metadata.name: Invalid value: %q: invalid as the %s label value: %s; choose a shorter graph name", name, consts.KubeLabelDynamoGraphDeploymentName, strings.Join(k8svalidation.IsValidLabelValue(name), "; "))
}

func betaDSNameForAdmission(name string) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaComponentDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		dgd.Name = name
		dgd.Annotations[consts.KubeAnnotationWorkloadProvider] = consts.WorkloadProviderDisaggregatedSet
		dgd.Annotations[consts.KubeAnnotationEnableDisaggregatedSet] = "true"
		dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = "etcd"
		worker := dgd.Spec.Components[1].DeepCopy()
		worker.ComponentType = nvidiacomv1beta1.ComponentTypePrefill
		worker.ComponentName = "prefill"
		worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
		decode := worker.DeepCopy()
		decode.ComponentType = nvidiacomv1beta1.ComponentTypeDecode
		decode.ComponentName = "decode"
		dgd.Spec.Components = []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{*worker, *decode}
	})
}

func TestAdmissionUsesOperatorDiscoveryDefault(t *testing.T) {
	for _, backend := range []configv1alpha1.DiscoveryBackend{configv1alpha1.DiscoveryBackendEtcd, configv1alpha1.DiscoveryBackendKubernetes} {
		t.Run(string(backend), func(t *testing.T) {
			t.Log("Start production admission with the selected operator discovery default")
			cfg := &configv1alpha1.OperatorConfiguration{}
			configv1alpha1.SetDefaultsOperatorConfiguration(cfg)
			cfg.Discovery.Backend = backend
			env := operatorenv.New(operatorenv.Options{
				Config: cfg, OperatorVersion: "1.6.0",
				Admission: operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
				SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
					return webhooksetup.Setup(mgr, webhooksetup.Options{Config: opts.OperatorConfig, RuntimeConfig: opts.RuntimeConfig, OperatorVersion: opts.OperatorVersion, OperatorPrincipal: opts.OperatorPrincipal})
				},
			}).RunT(t)

			t.Log("Submit unannotated digit-leading workers through both registered resource webhooks")
			dcd := betaDCDForAdmission(nil)
			dcd.Name, dcd.Namespace = "7worker", env.Namespace()
			dgd := betaComponentDGDForAdmission(nil)
			dgd.Name, dgd.Namespace = "7graph", env.Namespace()
			dgd.Spec.Components = dgd.Spec.Components[1:]
			for _, object := range []client.Object{dcd, dgd} {
				err := env.Client().Create(t.Context(), object)
				if backend == configv1alpha1.DiscoveryBackendEtcd {
					require.NoError(t, err)
				} else {
					require.True(t, apierrors.IsInvalid(err), "%v", err)
					require.Contains(t, err.Error(), "generated Service name is invalid")
				}
			}
		})
	}
}
