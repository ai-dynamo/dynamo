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

package controller

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	kruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	"sigs.k8s.io/yaml"
)

var upgradeScheme = k8sruntime.NewScheme()

func init() {
	kruntime.Must(appsv1.AddToScheme(upgradeScheme))
	kruntime.Must(corev1.AddToScheme(upgradeScheme))
	kruntime.Must(v1beta1.AddToScheme(upgradeScheme))
	kruntime.Must(grovev1alpha1.AddToScheme(upgradeScheme))
	kruntime.Must(leaderworkersetv1.AddToScheme(upgradeScheme))
}

type upgradeCase struct {
	// parent is the persisted pre-upgrade v1alpha1 DCD/DGD.
	parent client.Object

	// child is the workload object created by the old controller.
	child client.Object

	render              func(context.Context, *testing.T, client.Object, client.Object) (child client.Object, serviceSelector map[string]string)
	childPodLabels      func(*testing.T, client.Object) map[string]map[string]string
	expectedWorkerSites map[string]string
}

func TestLegacyWorkerIdentityUpgradeDoesNotTriggerRollout(t *testing.T) {
	ctx := context.Background()
	tests := map[string]upgradeCase{
		"Deployment": func() upgradeCase {
			dynamoNamespace := "default"
			parent := &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "qwen-decode-db6b6891",
					Namespace: "default",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoComponent:           "decode",
						commonconsts.KubeLabelDynamoGraphDeploymentName: "qwen",
						commonconsts.KubeLabelDynamoWorkerHash:          "db6b6891",
						commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
						commonconsts.KubeLabelDynamoSubComponentType:    commonconsts.ComponentTypeDecode,
					},
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: string(dynamo.BackendFrameworkVLLM),
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:           "decode",
							commonconsts.KubeLabelDynamoNamespace:           dynamoNamespace,
							commonconsts.KubeLabelDynamoGraphDeploymentName: "qwen",
							commonconsts.KubeLabelDynamoWorkerHash:          "db6b6891",
							commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
							commonconsts.KubeLabelDynamoSubComponentType:    commonconsts.ComponentTypeDecode,
						},
						ServiceName:      "decode",
						ComponentType:    commonconsts.ComponentTypeWorker,
						SubComponentType: commonconsts.ComponentTypeDecode,
						DynamoNamespace:  &dynamoNamespace,
						ExtraPodSpec: &v1alpha1.ExtraPodSpec{
							MainContainer: &corev1.Container{
								Name:    commonconsts.MainContainerName,
								Image:   "test-image:latest",
								Command: []string{"python3"},
								Args:    []string{"-m", "dynamo.vllm"},
								Resources: corev1.ResourceRequirements{
									Limits: corev1.ResourceList{
										corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
									},
								},
							},
						},
					},
				},
			}
			// The child fixture is the persisted pre-upgrade workload the current renderer discovers.
			child := &appsv1.Deployment{}
			require.NoError(t, yaml.Unmarshal([]byte(`
metadata:
  name: qwen-decode-db6b6891
  namespace: default
spec:
  selector:
    matchLabels:
      nvidia.com/selector: qwen-decode-db6b6891
  template:
    metadata:
      labels:
        nvidia.com/dynamo-component: decode
        nvidia.com/dynamo-component-type: worker
        nvidia.com/dynamo-discovery-backend: kubernetes
        nvidia.com/dynamo-discovery-enabled: "true"
        nvidia.com/dynamo-graph-deployment-name: qwen
        nvidia.com/dynamo-namespace: default
        nvidia.com/dynamo-sub-component-type: decode
        nvidia.com/dynamo-worker-hash: db6b6891
        nvidia.com/metrics-enabled: "true"
        nvidia.com/selector: qwen-decode-db6b6891
    spec:
      volumes:
      - name: shared-memory
        emptyDir:
          medium: Memory
          sizeLimit: 8Gi
      containers:
      - name: main
        image: test-image:latest
        command:
        - python3
        args:
        - -m
        - dynamo.vllm
        ports:
        - name: system
          containerPort: 9090
          protocol: TCP
        - name: nixl
          containerPort: 19090
          protocol: TCP
        env:
        - name: DYN_COMPONENT
          value: worker
        - name: DYN_DISCOVERY_BACKEND
          value: kubernetes
        - name: DYN_FORWARDPASS_METRIC_PORT
          value: "20380"
        - name: DYN_HEALTH_CHECK_ENABLED
          value: "false"
        - name: DYN_NAMESPACE
          value: default-qwen-decode-db6b6891
        - name: DYN_NAMESPACE_WORKER_SUFFIX
          value: db6b6891
        - name: DYN_PARENT_DGD_K8S_NAME
          value: qwen-decode-db6b6891
        - name: DYN_PARENT_DGD_K8S_NAMESPACE
          value: default
        - name: DYN_SYSTEM_ENABLED
          value: "true"
        - name: DYN_SYSTEM_PORT
          value: "9090"
        - name: DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS
          value: '["generate"]'
        - name: NIXL_TELEMETRY_ENABLE
          value: "n"
        - name: NIXL_TELEMETRY_EXPORTER
          value: prometheus
        - name: NIXL_TELEMETRY_PROMETHEUS_PORT
          value: "19090"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_UID
          valueFrom:
            fieldRef:
              fieldPath: metadata.uid
        resources:
          limits:
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: shared-memory
          mountPath: /dev/shm
        livenessProbe:
          httpGet:
            path: /live
            port: system
          timeoutSeconds: 4
          periodSeconds: 5
          failureThreshold: 1
        readinessProbe:
          httpGet:
            path: /health
            port: system
          timeoutSeconds: 4
          periodSeconds: 10
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /live
            port: system
          timeoutSeconds: 5
          periodSeconds: 10
          failureThreshold: 720
      restartPolicy: Always
      terminationGracePeriodSeconds: 60
      serviceAccountName: qwen-decode-db6b6891-k8s-service-discovery
      securityContext:
        fsGroup: 1000
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: "25%"
      maxSurge: "25%"
`), child))
			return upgradeCase{
				parent: parent,
				child:  child,
				render: func(ctx context.Context, t *testing.T, parent, child client.Object) (client.Object, map[string]string) {
					t.Helper()

					t.Log("render Deployment from the converted DCD and existing Deployment")
					dcd := parent.(*v1beta1.DynamoComponentDeployment)
					reconciler := newUpgradeDCDReconciler(t, dcd, child)
					deployment, toDelete, err := reconciler.generateDeployment(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
					require.NoError(t, err)
					require.False(t, toDelete)

					t.Log("generate the DCD service selector from the same converted parent and existing child")
					service, toDelete, err := reconciler.generateService(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
					require.NoError(t, err)
					require.False(t, toDelete)
					return deployment, service.Spec.Selector
				},
				childPodLabels: func(t *testing.T, obj client.Object) map[string]map[string]string {
					t.Helper()
					deployment := obj.(*appsv1.Deployment)
					return map[string]map[string]string{"pod template": deployment.Spec.Template.Labels}
				},
				expectedWorkerSites: map[string]string{"pod template": commonconsts.ComponentTypeDecode},
			}
		}(),
		"LeaderWorkerSet": func() upgradeCase {
			dynamoNamespace := "default"
			parent := &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "qwen-decode-db6b6891",
					Namespace: "default",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoComponent:           "decode",
						commonconsts.KubeLabelDynamoGraphDeploymentName: "qwen",
						commonconsts.KubeLabelDynamoWorkerHash:          "db6b6891",
						commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
						commonconsts.KubeLabelDynamoSubComponentType:    commonconsts.ComponentTypeDecode,
					},
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: string(dynamo.BackendFrameworkVLLM),
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:           "decode",
							commonconsts.KubeLabelDynamoNamespace:           dynamoNamespace,
							commonconsts.KubeLabelDynamoGraphDeploymentName: "qwen",
							commonconsts.KubeLabelDynamoWorkerHash:          "db6b6891",
							commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
							commonconsts.KubeLabelDynamoSubComponentType:    commonconsts.ComponentTypeDecode,
						},
						ServiceName:      "decode",
						ComponentType:    commonconsts.ComponentTypeWorker,
						SubComponentType: commonconsts.ComponentTypeDecode,
						DynamoNamespace:  &dynamoNamespace,
						ExtraPodSpec: &v1alpha1.ExtraPodSpec{
							MainContainer: &corev1.Container{
								Name:    commonconsts.MainContainerName,
								Image:   "test-image:latest",
								Command: []string{"python3"},
								Args:    []string{"-m", "dynamo.vllm"},
								Resources: corev1.ResourceRequirements{
									Limits: corev1.ResourceList{
										corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
									},
								},
							},
						},
					},
				},
			}
			parent.Spec.Multinode = &v1alpha1.MultinodeSpec{NodeCount: 2}
			// The child fixture is the persisted pre-upgrade workload the current renderer discovers.
			child := &leaderworkersetv1.LeaderWorkerSet{}
			require.NoError(t, yaml.Unmarshal([]byte(`
metadata:
  name: qwen-decode-db6b6891-0
  namespace: default
spec:
  replicas: 1
  leaderWorkerTemplate:
    leaderTemplate:
      metadata:
        labels:
          nvidia.com/dynamo-component: decode
          nvidia.com/dynamo-component-type: worker
          nvidia.com/dynamo-discovery-backend: kubernetes
          nvidia.com/dynamo-discovery-enabled: "true"
          nvidia.com/dynamo-graph-deployment-name: qwen
          nvidia.com/dynamo-namespace: default
          nvidia.com/dynamo-sub-component-type: decode
          nvidia.com/dynamo-worker-hash: db6b6891
          nvidia.com/metrics-enabled: "true"
          role: leader
      spec:
        volumes:
        - name: shared-memory
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
        containers:
        - name: main
          image: test-image:latest
          command:
          - python3
          args:
          - -m
          - dynamo.vllm
          ports:
          - name: system
            containerPort: 9090
            protocol: TCP
          - name: nixl
            containerPort: 19090
            protocol: TCP
          env:
          - name: DYN_COMPONENT
            value: worker
          - name: DYN_DISCOVERY_BACKEND
            value: kubernetes
          - name: DYN_FORWARDPASS_METRIC_PORT
            value: "20380"
          - name: DYN_HEALTH_CHECK_ENABLED
            value: "false"
          - name: DYN_NAMESPACE
            value: default-qwen-decode-db6b6891
          - name: DYN_NAMESPACE_WORKER_SUFFIX
            value: db6b6891
          - name: DYN_PARENT_DGD_K8S_NAME
            value: qwen-decode-db6b6891
          - name: DYN_PARENT_DGD_K8S_NAMESPACE
            value: default
          - name: DYN_SYSTEM_ENABLED
            value: "true"
          - name: DYN_SYSTEM_PORT
            value: "9090"
          - name: DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS
            value: '["generate"]'
          - name: NIXL_TELEMETRY_ENABLE
            value: "n"
          - name: NIXL_TELEMETRY_EXPORTER
            value: prometheus
          - name: NIXL_TELEMETRY_PROMETHEUS_PORT
            value: "19090"
          - name: POD_NAME
            valueFrom:
              fieldRef:
                fieldPath: metadata.name
          - name: POD_NAMESPACE
            valueFrom:
              fieldRef:
                fieldPath: metadata.namespace
          - name: POD_UID
            valueFrom:
              fieldRef:
                fieldPath: metadata.uid
          resources:
            limits:
              nvidia.com/gpu: "1"
          volumeMounts:
          - name: shared-memory
            mountPath: /dev/shm
          livenessProbe:
            httpGet:
              path: /live
              port: system
            timeoutSeconds: 4
            periodSeconds: 5
            failureThreshold: 1
          readinessProbe:
            httpGet:
              path: /health
              port: system
            timeoutSeconds: 4
            periodSeconds: 10
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /live
              port: system
            timeoutSeconds: 5
            periodSeconds: 10
            failureThreshold: 720
        restartPolicy: Always
        terminationGracePeriodSeconds: 60
        serviceAccountName: qwen-decode-db6b6891-k8s-service-discovery
        securityContext:
          fsGroup: 1000
    workerTemplate:
      metadata:
        labels:
          nvidia.com/dynamo-component: decode
          nvidia.com/dynamo-component-type: worker
          nvidia.com/dynamo-discovery-backend: kubernetes
          nvidia.com/dynamo-discovery-enabled: "true"
          nvidia.com/dynamo-graph-deployment-name: qwen
          nvidia.com/dynamo-namespace: default
          nvidia.com/dynamo-sub-component-type: decode
          nvidia.com/dynamo-worker-hash: db6b6891
          nvidia.com/metrics-enabled: "true"
          role: worker
      spec:
        volumes:
        - name: shared-memory
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
        containers:
        - name: main
          image: test-image:latest
          command:
          - python3
          args:
          - -m
          - dynamo.vllm
          ports:
          - name: system
            containerPort: 9090
            protocol: TCP
          - name: nixl
            containerPort: 19090
            protocol: TCP
          env:
          - name: DYN_COMPONENT
            value: worker
          - name: DYN_DISCOVERY_BACKEND
            value: kubernetes
          - name: DYN_FORWARDPASS_METRIC_PORT
            value: "20380"
          - name: DYN_HEALTH_CHECK_ENABLED
            value: "false"
          - name: DYN_NAMESPACE
            value: default-qwen-decode-db6b6891
          - name: DYN_NAMESPACE_WORKER_SUFFIX
            value: db6b6891
          - name: DYN_PARENT_DGD_K8S_NAME
            value: qwen-decode-db6b6891
          - name: DYN_PARENT_DGD_K8S_NAMESPACE
            value: default
          - name: DYN_SYSTEM_ENABLED
            value: "true"
          - name: DYN_SYSTEM_PORT
            value: "9090"
          - name: DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS
            value: '["generate"]'
          - name: NIXL_TELEMETRY_ENABLE
            value: "n"
          - name: NIXL_TELEMETRY_EXPORTER
            value: prometheus
          - name: NIXL_TELEMETRY_PROMETHEUS_PORT
            value: "19090"
          - name: POD_NAME
            valueFrom:
              fieldRef:
                fieldPath: metadata.name
          - name: POD_NAMESPACE
            valueFrom:
              fieldRef:
                fieldPath: metadata.namespace
          - name: POD_UID
            valueFrom:
              fieldRef:
                fieldPath: metadata.uid
          resources:
            limits:
              nvidia.com/gpu: "1"
          volumeMounts:
          - name: shared-memory
            mountPath: /dev/shm
        restartPolicy: Always
        terminationGracePeriodSeconds: 60
        serviceAccountName: qwen-decode-db6b6891-k8s-service-discovery
        securityContext:
          fsGroup: 1000
    size: 2
  rolloutStrategy:
    type: ""
  startupPolicy: LeaderCreated
`), child))
			return upgradeCase{
				parent: parent,
				child:  child,
				render: func(ctx context.Context, t *testing.T, parent, child client.Object) (client.Object, map[string]string) {
					t.Helper()

					t.Log("render LeaderWorkerSet from the converted DCD and existing LeaderWorkerSet")
					dcd := parent.(*v1beta1.DynamoComponentDeployment)
					reconciler := newUpgradeDCDReconciler(t, dcd, child)
					lws, toDelete, err := reconciler.generateLeaderWorkerSet(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
					require.NoError(t, err)
					require.False(t, toDelete)

					t.Log("generate the DCD service selector from the same converted parent and existing child")
					service, toDelete, err := reconciler.generateService(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
					require.NoError(t, err)
					require.False(t, toDelete)
					return lws, service.Spec.Selector
				},
				childPodLabels: func(t *testing.T, obj client.Object) map[string]map[string]string {
					t.Helper()
					lws := obj.(*leaderworkersetv1.LeaderWorkerSet)
					return map[string]map[string]string{
						"leader": lws.Spec.LeaderWorkerTemplate.LeaderTemplate.Labels,
						"worker": lws.Spec.LeaderWorkerTemplate.WorkerTemplate.Labels,
					}
				},
				expectedWorkerSites: map[string]string{
					"leader": commonconsts.ComponentTypeDecode,
					"worker": commonconsts.ComponentTypeDecode,
				},
			}
		}(),
		"Grove PodCliqueSet": func() upgradeCase {
			parent := &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "vllm-disagg-planner",
					Namespace: "jsm",
					Annotations: map[string]string{
						commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
					},
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"VllmDecodeWorker": {
							ComponentType:    commonconsts.ComponentTypeWorker,
							SubComponentType: commonconsts.ComponentTypeDecode,
							Replicas:         ptr.To(int32(1)),
						},
						"VllmPrefillWorker": {
							ComponentType:    commonconsts.ComponentTypeWorker,
							SubComponentType: commonconsts.ComponentTypePrefill,
							Replicas:         ptr.To(int32(1)),
						},
					},
				},
			}
			// The child fixture is the persisted pre-upgrade workload the current renderer discovers.
			child := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, yaml.Unmarshal([]byte(`
metadata:
  name: vllm-disagg-planner
  namespace: jsm
spec:
  replicas: 1
  template:
    cliques:
    - name: vllmprefillworker
      labels:
        nvidia.com/dynamo-component: VllmPrefillWorker
        nvidia.com/dynamo-component-type: worker
        nvidia.com/dynamo-graph-deployment-name: vllm-disagg-planner
        nvidia.com/dynamo-namespace: jsm-vllm-disagg-planner
        nvidia.com/dynamo-sub-component-type: prefill
        nvidia.com/metrics-enabled: "true"
        nvidia.com/selector: vllm-disagg-planner-vllmprefillworker
      annotations:
        nvidia.com/dynamo-operator-origin-version: 1.1.0
      spec:
        roleName: vllmprefillworker
        podSpec:
          volumes:
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: 8Gi
          containers:
          - name: main
            command:
            - /bin/sh
            - -c
            ports:
            - name: system
              containerPort: 9090
              protocol: TCP
            - name: nixl
              containerPort: 19090
              protocol: TCP
            env:
            - name: DYN_COMPONENT
              value: worker
            - name: DYN_DISCOVERY_BACKEND
              value: kubernetes
            - name: DYN_FORWARDPASS_METRIC_PORT
              value: "20380"
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: DYN_NAMESPACE
              value: jsm-vllm-disagg-planner
            - name: DYN_PARENT_DGD_K8S_NAME
              value: vllm-disagg-planner
            - name: DYN_PARENT_DGD_K8S_NAMESPACE
              value: jsm
            - name: DYN_SYSTEM_ENABLED
              value: "true"
            - name: DYN_SYSTEM_PORT
              value: "9090"
            - name: DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS
              value: '["generate"]'
            - name: NIXL_TELEMETRY_ENABLE
              value: "n"
            - name: NIXL_TELEMETRY_EXPORTER
              value: prometheus
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            resources: {}
            volumeMounts:
            - name: shared-memory
              mountPath: /dev/shm
            livenessProbe:
              httpGet:
                path: /live
                port: system
              timeoutSeconds: 4
              periodSeconds: 5
              failureThreshold: 1
            readinessProbe:
              httpGet:
                path: /health
                port: system
              timeoutSeconds: 4
              periodSeconds: 10
              failureThreshold: 3
            startupProbe:
              httpGet:
                path: /live
                port: system
              timeoutSeconds: 5
              periodSeconds: 10
              failureThreshold: 720
          restartPolicy: Always
          terminationGracePeriodSeconds: 60
          securityContext:
            fsGroup: 1000
        replicas: 1
        minAvailable: 1
    - name: vllmdecodeworker
      labels:
        nvidia.com/dynamo-component: VllmDecodeWorker
        nvidia.com/dynamo-component-type: worker
        nvidia.com/dynamo-graph-deployment-name: vllm-disagg-planner
        nvidia.com/dynamo-namespace: jsm-vllm-disagg-planner
        nvidia.com/dynamo-sub-component-type: decode
        nvidia.com/metrics-enabled: "true"
        nvidia.com/selector: vllm-disagg-planner-vllmdecodeworker
      annotations:
        nvidia.com/dynamo-operator-origin-version: 1.1.0
      spec:
        roleName: vllmdecodeworker
        podSpec:
          volumes:
          - name: shared-memory
            emptyDir:
              medium: Memory
              sizeLimit: 8Gi
          containers:
          - name: main
            command:
            - /bin/sh
            - -c
            ports:
            - name: system
              containerPort: 9090
              protocol: TCP
            - name: nixl
              containerPort: 19090
              protocol: TCP
            env:
            - name: DYN_COMPONENT
              value: worker
            - name: DYN_DISCOVERY_BACKEND
              value: kubernetes
            - name: DYN_FORWARDPASS_METRIC_PORT
              value: "20380"
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: DYN_NAMESPACE
              value: jsm-vllm-disagg-planner
            - name: DYN_PARENT_DGD_K8S_NAME
              value: vllm-disagg-planner
            - name: DYN_PARENT_DGD_K8S_NAMESPACE
              value: jsm
            - name: DYN_SYSTEM_ENABLED
              value: "true"
            - name: DYN_SYSTEM_PORT
              value: "9090"
            - name: DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS
              value: '["generate"]'
            - name: NIXL_TELEMETRY_ENABLE
              value: "n"
            - name: NIXL_TELEMETRY_EXPORTER
              value: prometheus
            - name: NIXL_TELEMETRY_PROMETHEUS_PORT
              value: "19090"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            resources: {}
            volumeMounts:
            - name: shared-memory
              mountPath: /dev/shm
            livenessProbe:
              httpGet:
                path: /live
                port: system
              timeoutSeconds: 4
              periodSeconds: 5
              failureThreshold: 1
            readinessProbe:
              httpGet:
                path: /health
                port: system
              timeoutSeconds: 4
              periodSeconds: 10
              failureThreshold: 3
            startupProbe:
              httpGet:
                path: /live
                port: system
              timeoutSeconds: 5
              periodSeconds: 10
              failureThreshold: 720
          restartPolicy: Always
          terminationGracePeriodSeconds: 60
          securityContext:
            fsGroup: 1000
        replicas: 1
        minAvailable: 1
    cliqueStartupType: CliqueStartupTypeAnyOrder
    headlessServiceConfig:
      publishNotReadyAddresses: true
`), child))
			return upgradeCase{
				parent: parent,
				child:  child,
				render: func(ctx context.Context, t *testing.T, parent, child client.Object) (client.Object, map[string]string) {
					t.Helper()

					t.Log("prepare Grove render deployment from the converted DGD and existing PodCliqueSet")
					dgd := parent.(*v1beta1.DynamoGraphDeployment)
					reconciler := newUpgradeDGDReconciler(t, dgd, child)
					renderDGD, existing, err := reconciler.prepareGroveRenderDeployment(ctx, dgd)
					require.NoError(t, err)
					require.NotNil(t, existing)

					t.Log("generate the desired Grove PodCliqueSet from the prepared render deployment")
					pcs, err := dynamo.GenerateGrovePodCliqueSet(
						ctx,
						renderDGD,
						&configv1alpha1.OperatorConfiguration{},
						&controller_common.RuntimeConfig{},
						reconciler.Client,
						nil,
						nil,
						nil,
						nil,
					)
					require.NoError(t, err)

					t.Log("preserve the existing PodCliqueSet clique order before comparing specs")
					preserveGrovePodCliqueSetOrder(pcs, existing)

					t.Log("generate the decode service selector from the same prepared Grove component")
					decodeComponent := renderDGD.GetComponentByName("VllmDecodeWorker")
					require.NotNil(t, decodeComponent)
					service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
						ServiceName:     dynamo.GetDCDResourceName(renderDGD, "VllmDecodeWorker", ""),
						Namespace:       renderDGD.Namespace,
						ComponentType:   string(decodeComponent.ComponentType),
						DynamoNamespace: renderDGD.GetDynamoNamespaceForComponent(decodeComponent),
						ComponentName:   "VllmDecodeWorker",
						Labels:          dynamo.GetDGDComponentResourceLabels(renderDGD, "VllmDecodeWorker", decodeComponent),
						Annotations:     dynamo.GetDGDComponentResourceAnnotations(renderDGD, "VllmDecodeWorker", decodeComponent),
						IsK8sDiscovery:  true,
					})
					require.NoError(t, err)
					return pcs, service.Spec.Selector
				},
				childPodLabels: func(t *testing.T, obj client.Object) map[string]map[string]string {
					t.Helper()
					pcs := obj.(*grovev1alpha1.PodCliqueSet)
					return map[string]map[string]string{
						"prefill": requireGroveClique(t, pcs, "vllmprefillworker").Labels,
						"decode":  requireGroveClique(t, pcs, "vllmdecodeworker").Labels,
					}
				},
				expectedWorkerSites: map[string]string{
					"prefill": commonconsts.ComponentTypePrefill,
					"decode":  commonconsts.ComponentTypeDecode,
				},
			}
		}(),
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var parent client.Object

			t.Log("convert the persisted pre-upgrade parent through the v1beta1 conversion path")
			switch alpha := tt.parent.DeepCopyObject().(type) {
			case *v1alpha1.DynamoComponentDeployment:
				dcd := &v1beta1.DynamoComponentDeployment{}
				require.NoError(t, alpha.ConvertTo(dcd))
				parent = dcd
			case *v1alpha1.DynamoGraphDeployment:
				dgd := &v1beta1.DynamoGraphDeployment{}
				require.NoError(t, alpha.ConvertTo(dgd))
				parent = dgd
			default:
				t.Fatalf("unexpected parent type %T", tt.parent)
			}

			oldChild := tt.child.DeepCopyObject().(client.Object)

			t.Logf("seed the fake client with the pre-upgrade %T child", oldChild)

			t.Log("render the desired child and service selector with the current reconciler")
			newChild, selector := tt.render(ctx, t, parent, oldChild)
			oldPodLabels := tt.childPodLabels(t, oldChild)
			newPodLabels := tt.childPodLabels(t, newChild)

			t.Log("compare old and new child specs; a change here would trigger a rollout")
			require.Equal(t, specHash(t, oldChild), specHash(t, newChild), "upgrade should not change the child spec hash")

			t.Log("assert worker pod labels keep the legacy worker identity")
			for site, subComponentType := range tt.expectedWorkerSites {
				oldLabels, ok := oldPodLabels[site]
				require.True(t, ok, "pre-upgrade child labels for %s", site)
				newLabels, ok := newPodLabels[site]
				require.True(t, ok, "post-upgrade child labels for %s", site)

				require.Equal(t, commonconsts.ComponentTypeWorker, oldLabels[commonconsts.KubeLabelDynamoComponentType], "pre-upgrade %s component type", site)
				require.Equal(t, subComponentType, oldLabels[commonconsts.KubeLabelDynamoSubComponentType], "pre-upgrade %s sub-component type", site)
				require.NotContains(t, oldLabels, commonconsts.KubeLabelDynamoComponentClass, "pre-upgrade %s component class", site)
				require.Equal(t, oldLabels[commonconsts.KubeLabelDynamoComponentType], newLabels[commonconsts.KubeLabelDynamoComponentType], "post-upgrade %s component type", site)
				require.Equal(t, oldLabels[commonconsts.KubeLabelDynamoSubComponentType], newLabels[commonconsts.KubeLabelDynamoSubComponentType], "post-upgrade %s sub-component type", site)
				require.NotContains(t, newLabels, commonconsts.KubeLabelDynamoComponentClass, "post-upgrade %s component class", site)
			}

			t.Log("check the rendered service selector stays aligned with the upgraded child")
			require.NotNil(t, selector, "service selector")
			require.Equal(t, commonconsts.ComponentTypeWorker, selector[commonconsts.KubeLabelDynamoComponentType])
		})
	}
}

func TestGroveNativeWorkerIdentityLabelsStayNative(t *testing.T) {
	ctx := context.Background()
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			},
		},
	}
	existingPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "prefill",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:     "prefill",
							commonconsts.KubeLabelDynamoComponentType: commonconsts.ComponentTypePrefill,
						},
					},
				},
			},
		},
	}

	t.Log("seed the fake client with a native v1beta1 DGD and existing PodCliqueSet")
	reconciler := newUpgradeDGDReconciler(t, dgd, existingPCS)

	t.Log("prepare the Grove render deployment without legacy worker selector migration")
	renderDGD, existing, err := reconciler.prepareGroveRenderDeployment(ctx, dgd)
	require.NoError(t, err)
	require.NotNil(t, existing)

	t.Log("generate the desired PodCliqueSet from the prepared native render deployment")
	desired, err := dynamo.GenerateGrovePodCliqueSet(
		ctx,
		renderDGD,
		&configv1alpha1.OperatorConfiguration{},
		&controller_common.RuntimeConfig{},
		reconciler.Client,
		nil,
		nil,
		nil,
		nil,
	)
	require.NoError(t, err)

	t.Log("preserve existing clique order before checking native labels")
	preserveGrovePodCliqueSetOrder(desired, existing)

	t.Log("assert the native prefill component stays prefill instead of legacy worker")
	prefillComponent := renderDGD.GetComponentByName("prefill")
	require.NotNil(t, prefillComponent)
	require.Equal(t, v1beta1.ComponentTypePrefill, prefillComponent.ComponentType)
	prefillClique := requireGroveClique(t, desired, "prefill")
	require.Equal(t, commonconsts.ComponentTypePrefill, prefillClique.Labels[commonconsts.KubeLabelDynamoComponentType])
}

func newUpgradeDGDReconciler(
	t *testing.T,
	dgd *v1beta1.DynamoGraphDeployment,
	objects ...client.Object,
) *DynamoGraphDeploymentReconciler {
	t.Helper()
	objects = append([]client.Object{dgd}, objects...)
	return &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(upgradeScheme).
			WithObjects(objects...).
			Build(),
	}
}

func newUpgradeDCDReconciler(
	t *testing.T,
	dcd *v1beta1.DynamoComponentDeployment,
	objects ...client.Object,
) *DynamoComponentDeploymentReconciler {
	t.Helper()
	objects = append([]client.Object{dcd}, objects...)
	return &DynamoComponentDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(upgradeScheme).
			WithObjects(objects...).
			Build(),
		Config: &configv1alpha1.OperatorConfiguration{
			Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
		},
		RuntimeConfig: &controller_common.RuntimeConfig{LWSEnabled: true},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return nil, nil
			},
		},
	}
}

func requireGroveClique(t *testing.T, pcs *grovev1alpha1.PodCliqueSet, name string) *grovev1alpha1.PodCliqueTemplateSpec {
	t.Helper()
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Name == name {
			return clique
		}
	}
	t.Fatalf("expected rendered grove clique %q", name)
	return nil
}

func specHash(t *testing.T, obj client.Object) string {
	t.Helper()
	hash, err := controller_common.GetSpecHash(obj)
	require.NoError(t, err)
	return hash
}
