/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package setup

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
)

func TestSetupRequiresOperatorConfiguration(t *testing.T) {
	wantErr := "operator configuration is required"
	if err := Setup(nil, Options{}); err == nil || err.Error() != wantErr {
		t.Fatalf("Setup() error = %v, want %q", err, wantErr)
	}
}

func TestSetupRequiresRuntimeConfiguration(t *testing.T) {
	wantErr := "runtime configuration is required"
	opts := Options{Config: &configv1alpha1.OperatorConfiguration{}}
	if err := Setup(nil, opts); err == nil || err.Error() != wantErr {
		t.Fatalf("Setup() error = %v, want %q", err, wantErr)
	}
}

// TestSetupDiscoveryDefault verifies the configuration passed to both registered
// handlers. Resource admission semantics remain covered by their admission tables.
func TestSetupDiscoveryDefault(t *testing.T) {
	for _, backend := range []configv1alpha1.DiscoveryBackend{
		configv1alpha1.DiscoveryBackendEtcd,
		configv1alpha1.DiscoveryBackendKubernetes,
	} {
		t.Run(string(backend), func(t *testing.T) {
			t.Log("Register production webhooks with the operator discovery default")
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			server := ctrlwebhook.NewServer(ctrlwebhook.Options{})
			mgr := &setupTestManager{
				scheme: scheme, server: server,
				client: fake.NewClientBuilder().WithScheme(scheme).Build(),
			}
			cfg := &configv1alpha1.OperatorConfiguration{}
			cfg.Namespace.Restricted = "default"
			cfg.Discovery.Backend = backend
			require.NoError(t, Setup(mgr, Options{
				Config: cfg, RuntimeConfig: &commoncontroller.RuntimeConfig{},
				Gate: features.Defaults(), OperatorVersion: "1.6.0",
			}))

			t.Log("Submit unannotated digit-leading single-node workers to both registered handlers")
			component := nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
				RuntimeVersionOverride: "1.6.0", Replicas: new(int32(1)),
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "runtime:1.6.0"}},
				}},
			}
			roleComponent := component.DeepCopy()
			roleComponent.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			roleComponent.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
				{Name: nvidiacomv1beta1.ComponentRoleLeader, PodTemplate: component.PodTemplate.DeepCopy()},
				{Name: nvidiacomv1beta1.ComponentRoleWorker, PodTemplate: component.PodTemplate.DeepCopy()},
			}
			roleComponent.PodTemplate = nil
			override := "kubernetes"
			if backend == configv1alpha1.DiscoveryBackendKubernetes {
				override = "etcd"
			}
			for _, tc := range []struct {
				kind, path, field, name string
				object                  runtime.Object
			}{
				{
					kind: "DynamoGraphDeployment", path: "/validate-nvidia-com-v1beta1-dynamographdeployment",
					field: "spec.components[0].name",
					object: &nvidiacomv1beta1.DynamoGraphDeployment{
						Name: "70b-worker", Namespace: "default",
						Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
							BackendFramework: "vllm",
							Components:       []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{*component.DeepCopy()},
						},
					},
				},
				{
					kind: "DynamoComponentDeployment", path: "/validate-nvidia-com-v1beta1-dynamocomponentdeployment",
					field: "metadata.name",
					object: &nvidiacomv1beta1.DynamoComponentDeployment{
						Name: "70b-worker", Namespace: "default",
						Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
							BackendFramework: "vllm", DynamoComponentDeploymentSharedSpec: *component.DeepCopy(),
						},
					},
				},
				{
					kind: "DynamoGraphDeployment", name: " role templates with opposite metadata override",
					path: "/validate-nvidia-com-v1beta1-dynamographdeployment", field: "spec.components[0].name",
					object: &nvidiacomv1beta1.DynamoGraphDeployment{
						Name: "70b-worker", Namespace: "default",
						Annotations: map[string]string{consts.KubeAnnotationDynamoDiscoveryBackend: override},
						Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
							BackendFramework: "vllm",
							Components:       []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{*roleComponent},
						},
					},
				},
			} {
				t.Run(tc.kind+tc.name, func(t *testing.T) {
					t.Log("Send a create request through the webhook installed by Setup")
					gvk := nvidiacomv1beta1.GroupVersion.WithKind(tc.kind)
					tc.object.GetObjectKind().SetGroupVersionKind(gvk)
					raw, err := json.Marshal(tc.object)
					require.NoError(t, err)
					reviewRequest := admissionv1.AdmissionReview{
						APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview",
						Request: &admissionv1.AdmissionRequest{
							UID: "discovery-default", Operation: admissionv1.Create, Namespace: "default",
							Kind:   metav1.GroupVersionKind{Group: gvk.Group, Version: gvk.Version, Kind: gvk.Kind},
							Object: runtime.RawExtension{Raw: raw},
						},
					}
					body, err := json.Marshal(reviewRequest)
					require.NoError(t, err)
					request := httptest.NewRequest(http.MethodPost, tc.path, bytes.NewReader(body))
					request.Header.Set("Content-Type", "application/json")
					response := httptest.NewRecorder()
					server.WebhookMux().ServeHTTP(response, request)
					require.Equal(t, http.StatusOK, response.Code)
					var review admissionv1.AdmissionReview
					require.NoError(t, json.Unmarshal(response.Body.Bytes(), &review))
					require.NotNil(t, review.Response)

					t.Log("Verify etcd accepts the worker and Kubernetes discovery reports its generated Service name")
					if backend == configv1alpha1.DiscoveryBackendEtcd {
						require.True(t, review.Response.Allowed, "unexpected rejection: %v", review.Response.Result)
					} else {
						require.False(t, review.Response.Allowed)
						require.NotNil(t, review.Response.Result)
						require.Equal(t, metav1.StatusReasonInvalid, review.Response.Result.Reason)
						require.NotNil(t, review.Response.Result.Details)
						for _, cause := range review.Response.Result.Details.Causes {
							require.Equal(t, metav1.CauseTypeFieldValueInvalid, cause.Type)
							require.Equal(t, tc.field, cause.Field)
							require.Contains(t, cause.Message, "Service name")
						}
						require.NotEmpty(t, review.Response.Result.Details.Causes)
					}
				})
			}
		})
	}
}

type setupTestManager struct {
	ctrl.Manager
	scheme *runtime.Scheme
	server ctrlwebhook.Server
	client client.Client
}

func (m *setupTestManager) GetScheme() *runtime.Scheme           { return m.scheme }
func (m *setupTestManager) GetWebhookServer() ctrlwebhook.Server { return m.server }
func (m *setupTestManager) GetClient() client.Client             { return m.client }
func (m *setupTestManager) GetAPIReader() client.Reader          { return m.client }
