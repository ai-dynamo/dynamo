/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"context"
	"fmt"
	"strconv"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/modelendpoint"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

type modelServiceTestReconciler struct {
	client.Client
	recorder events.EventRecorder
}

func (r *modelServiceTestReconciler) GetRecorder() events.EventRecorder {
	return r.recorder
}

func TestModelServiceReconciliationCharacterization(t *testing.T) {
	t.Log("Arrange a DGD with two components sharing one model reference")
	dgd := &v1beta1.DynamoGraphDeployment{
		APIVersion: v1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment",
		Name:      "graph",
		Namespace: "inference",
		UID:       "graph-uid",
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Annotations: map[string]string{"example.com/graph": "true"},
		},
	}
	components := map[string]*v1beta1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentName: "prefill", ModelRef: &v1beta1.ModelReference{Name: "llama-3"}},
		"decode":  {ComponentName: "decode", ModelRef: &v1beta1.ModelReference{Name: "llama-3"}},
	}
	s := scheme.Scheme
	require.NoError(t, v1beta1.AddToScheme(s))
	reconciler := &modelServiceTestReconciler{
		Client:   fake.NewClientBuilder().WithScheme(s).WithObjects(dgd).Build(),
		recorder: events.NewFakeRecorder(10),
	}

	t.Log("Act through the existing model-service reconciliation API")
	require.NoError(t, ReconcileModelServicesForComponents(
		context.Background(), reconciler, dgd, components, dgd.Namespace,
	))

	t.Log("Assert one deterministic headless Service with the current graph contract")
	service := &corev1.Service{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{
		Namespace: dgd.Namespace,
		Name:      GenerateServiceName("llama-3"),
	}, service))
	require.Equal(t, dgd.Namespace, service.Namespace)
	require.Equal(t, corev1.ClusterIPNone, service.Spec.ClusterIP)
	require.Equal(t, map[string]string{
		commonconsts.KubeLabelDynamoBaseModelHash: HashModelName("llama-3"),
	}, service.Spec.Selector)
	require.Equal(t, commonconsts.DynamoSystemPortName, service.Spec.Ports[0].Name)
	require.Equal(t, int32(commonconsts.DynamoSystemPort), service.Spec.Ports[0].Port)
	require.Equal(t, intstr.FromString(commonconsts.DynamoSystemPortName), service.Spec.Ports[0].TargetPort)
	require.Equal(t, HashModelName("llama-3"), service.Labels[commonconsts.KubeLabelDynamoBaseModelHash])
	require.Equal(t, "dynamo-operator", service.Labels["nvidia.com/managed-by"])
	require.Equal(t, "llama-3", service.Annotations[commonconsts.KubeAnnotationDynamoBaseModel])
	require.Equal(t, "true", service.Annotations["example.com/graph"])
	require.Len(t, service.OwnerReferences, 1)
	require.Equal(t, dgd.UID, service.OwnerReferences[0].UID)
	require.True(t, *service.OwnerReferences[0].Controller)
}

func TestGenerateModelServiceForGraphIsPrivateToGraph(t *testing.T) {
	first := GenerateModelServiceForGraph("inference", "llama-3", "graph-a", nil)
	second := GenerateModelServiceForGraph("inference", "llama-3", "graph-b", nil)

	require.NotEqual(t, first.Name, second.Name)
	require.Equal(t, "graph-a", first.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName])
	require.Equal(t, "graph-a", first.Spec.Selector[commonconsts.KubeLabelDynamoGraphDeploymentName])
	require.Equal(t, HashModelName("llama-3"), first.Labels[commonconsts.KubeLabelDynamoBaseModelHash])
	require.Equal(t, intstr.FromString(commonconsts.DynamoSystemPortName), first.Spec.Ports[0].TargetPort)
	require.LessOrEqual(t, len(first.Name), 63)
	require.LessOrEqual(t, len(second.Name), 63)
}

func TestFailoverModelServiceUsesPrimarySystemPort(t *testing.T) {
	for _, port := range []int32{9090, 12345} {
		t.Run(strconv.Itoa(int(port)), func(t *testing.T) {
			t.Log("Render a model Service and two failover engines from the configured base")
			service := GenerateModelServiceForGraph("inference", "base-model", "graph", nil)
			pod := corev1.PodSpec{Containers: []corev1.Container{{
				Name: commonconsts.MainContainerName,
				Env:  []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(int(port))}},
			}}}
			require.NoError(t, buildFailoverEnginePair(&pod))
			require.Len(t, pod.Containers, 2)

			t.Log("Resolve the model target to exactly one primary port, without duplicate aliases")
			var resolved []discoveryv1.EndpointPort
			for i, engine := range pod.Containers {
				require.Len(t, engine.Ports, 1)
				declared := engine.Ports[0]
				require.Equal(t, port+int32(i), declared.ContainerPort)
				if declared.Name == service.Spec.Ports[0].TargetPort.StrVal {
					require.Equal(t, "engine-0", engine.Name)
					resolved = append(resolved, discoveryv1.EndpointPort{Name: new(service.Spec.Ports[0].Name), Port: new(declared.ContainerPort)})
				}
			}
			require.Len(t, resolved, 1)
			require.Equal(t, "system-1", pod.Containers[1].Ports[0].Name)

			t.Log("Model discovery keeps one candidate for the Pod at the configured primary port")
			slices := &discoveryv1.EndpointSliceList{Items: []discoveryv1.EndpointSlice{{
				Labels: map[string]string{discoveryv1.LabelServiceName: service.Name},
				Ports:  resolved,
				Endpoints: []discoveryv1.Endpoint{{
					Addresses:  []string{"10.0.1.5"},
					TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: "worker-0", UID: "worker-uid"},
					Conditions: discoveryv1.EndpointConditions{Ready: new(true)},
				}},
			}}}
			candidates, _ := modelendpoint.ExtractCandidates(slices)
			require.Equal(t, []modelendpoint.Candidate{{
				Address: fmt.Sprintf("http://10.0.1.5:%d", port), PodName: "worker-0", KubernetesReady: true,
			}}, candidates)
		})
	}
}
