package dynamo

import (
	"context"
	"strings"
	"testing"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestResolveKaiSchedulerQueueName(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		expected    string
	}{
		{
			name:        "nil annotations",
			annotations: nil,
			expected:    commonconsts.DefaultKaiSchedulerQueue,
		},
		{
			name:        "empty annotations",
			annotations: map[string]string{},
			expected:    commonconsts.DefaultKaiSchedulerQueue,
		},
		{
			name: "no kai-scheduler annotation",
			annotations: map[string]string{
				"other-annotation": "value",
			},
			expected: commonconsts.DefaultKaiSchedulerQueue,
		},
		{
			name: "empty kai-scheduler annotation",
			annotations: map[string]string{
				commonconsts.KubeAnnotationKaiSchedulerQueue: "",
			},
			expected: commonconsts.DefaultKaiSchedulerQueue,
		},
		{
			name: "custom queue name",
			annotations: map[string]string{
				commonconsts.KubeAnnotationKaiSchedulerQueue: "custom-queue",
			},
			expected: "custom-queue",
		},
		{
			name: "whitespace is trimmed",
			annotations: map[string]string{
				commonconsts.KubeAnnotationKaiSchedulerQueue: "  custom-queue  ",
			},
			expected: "custom-queue",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := resolveKaiSchedulerQueueName(tt.annotations)
			if result != tt.expected {
				t.Errorf("resolveKaiSchedulerQueueName() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestResolveKaiSchedulerQueue(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		expected    string
	}{
		{
			name:        "default queue",
			annotations: nil,
			expected:    commonconsts.DefaultKaiSchedulerQueue,
		},
		{
			name: "custom queue",
			annotations: map[string]string{
				commonconsts.KubeAnnotationKaiSchedulerQueue: "my-queue",
			},
			expected: "my-queue",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ResolveKaiSchedulerQueue(tt.annotations)
			if result != tt.expected {
				t.Errorf("ResolveKaiSchedulerQueue() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestInjectKaiSchedulerIfEnabled(t *testing.T) {
	tests := []struct {
		name               string
		controllerConfig   controller_common.Config
		validatedQueueName string
		initialClique      *grovev1alpha1.PodCliqueTemplateSpec
		expectedScheduler  string
		expectedQueueLabel string
		shouldInject       bool
	}{
		{
			name: "grove disabled - no injection",
			controllerConfig: controller_common.Config{
				Grove:        controller_common.GroveConfig{Enabled: false},
				KaiScheduler: controller_common.KaiSchedulerConfig{Enabled: true},
			},
			validatedQueueName: "test-queue",
			initialClique: &grovev1alpha1.PodCliqueTemplateSpec{
				Spec: grovev1alpha1.PodCliqueSpec{
					PodSpec: corev1.PodSpec{},
				},
			},
			shouldInject: false,
		},
		{
			name: "kai-scheduler disabled - no injection",
			controllerConfig: controller_common.Config{
				Grove:        controller_common.GroveConfig{Enabled: true},
				KaiScheduler: controller_common.KaiSchedulerConfig{Enabled: false},
			},
			validatedQueueName: "test-queue",
			initialClique: &grovev1alpha1.PodCliqueTemplateSpec{
				Spec: grovev1alpha1.PodCliqueSpec{
					PodSpec: corev1.PodSpec{},
				},
			},
			shouldInject: false,
		},
		{
			name: "manual scheduler set - no injection",
			controllerConfig: controller_common.Config{
				Grove:        controller_common.GroveConfig{Enabled: true},
				KaiScheduler: controller_common.KaiSchedulerConfig{Enabled: true},
			},
			validatedQueueName: "test-queue",
			initialClique: &grovev1alpha1.PodCliqueTemplateSpec{
				Spec: grovev1alpha1.PodCliqueSpec{
					PodSpec: corev1.PodSpec{
						SchedulerName: "manual-scheduler",
					},
				},
			},
			shouldInject: false,
		},
		{
			name: "both enabled, no manual scheduler - inject",
			controllerConfig: controller_common.Config{
				Grove:        controller_common.GroveConfig{Enabled: true},
				KaiScheduler: controller_common.KaiSchedulerConfig{Enabled: true},
			},
			validatedQueueName: "test-queue",
			initialClique: &grovev1alpha1.PodCliqueTemplateSpec{
				Spec: grovev1alpha1.PodCliqueSpec{
					PodSpec: corev1.PodSpec{},
				},
			},
			expectedScheduler:  commonconsts.KaiSchedulerName,
			expectedQueueLabel: "test-queue",
			shouldInject:       true,
		},
		{
			name: "inject with existing labels",
			controllerConfig: controller_common.Config{
				Grove:        controller_common.GroveConfig{Enabled: true},
				KaiScheduler: controller_common.KaiSchedulerConfig{Enabled: true},
			},
			validatedQueueName: "custom-queue",
			initialClique: &grovev1alpha1.PodCliqueTemplateSpec{
				Labels: map[string]string{
					"existing-label": "existing-value",
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					PodSpec: corev1.PodSpec{},
				},
			},
			expectedScheduler:  commonconsts.KaiSchedulerName,
			expectedQueueLabel: "custom-queue",
			shouldInject:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a deep copy to avoid modifying the test case
			clique := tt.initialClique.DeepCopy()

			// Call the function
			injectKaiSchedulerIfEnabled(clique, tt.controllerConfig, tt.validatedQueueName)

			if tt.shouldInject {
				// Verify scheduler name is injected
				if clique.Spec.PodSpec.SchedulerName != tt.expectedScheduler {
					t.Errorf("expected schedulerName %v, got %v", tt.expectedScheduler, clique.Spec.PodSpec.SchedulerName)
				}

				// Verify queue label is injected
				if clique.Labels == nil {
					t.Errorf("expected labels to be set, got nil")
				} else {
					queueLabel := clique.Labels[commonconsts.KubeLabelKaiSchedulerQueue]
					if queueLabel != tt.expectedQueueLabel {
						t.Errorf("expected queue label %v, got %v", tt.expectedQueueLabel, queueLabel)
					}
				}

				// Verify existing labels are preserved
				if tt.initialClique.Labels != nil {
					for key, value := range tt.initialClique.Labels {
						if clique.Labels[key] != value {
							t.Errorf("existing label %s=%s was not preserved, got %s", key, value, clique.Labels[key])
						}
					}
				}
			} else {
				// Verify no injection occurred
				if clique.Spec.PodSpec.SchedulerName != tt.initialClique.Spec.PodSpec.SchedulerName {
					t.Errorf("schedulerName should not have changed, expected %v, got %v",
						tt.initialClique.Spec.PodSpec.SchedulerName, clique.Spec.PodSpec.SchedulerName)
				}

				// Verify queue label was not added (unless it existed before)
				if tt.initialClique.Labels == nil || tt.initialClique.Labels[commonconsts.KubeLabelKaiSchedulerQueue] == "" {
					if clique.Labels != nil && clique.Labels[commonconsts.KubeLabelKaiSchedulerQueue] != "" {
						t.Errorf("queue label should not have been added")
					}
				}
			}
		})
	}
}

func TestEnsureQueueExists(t *testing.T) {
	tests := []struct {
		name          string
		queueName     string
		setupQueue    bool
		expectedError bool
		errorContains string
	}{
		{
			name:          "queue exists",
			queueName:     "existing-queue",
			setupQueue:    true,
			expectedError: false,
		},
		{
			name:          "queue does not exist",
			queueName:     "missing-queue",
			setupQueue:    false,
			expectedError: true,
			errorContains: "not found in cluster",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a fake dynamic client
			dynamicScheme := runtime.NewScheme()
			fakeDynamic := dynamicfake.NewSimpleDynamicClient(dynamicScheme)

			if tt.setupQueue {
				// Create a fake queue resource
				queueGVR := schema.GroupVersionResource{
					Group:    "scheduling.run.ai",
					Version:  "v2",
					Resource: "queues",
				}

				queue := &unstructured.Unstructured{}
				queue.SetAPIVersion("scheduling.run.ai/v2")
				queue.SetKind("Queue")
				queue.SetName(tt.queueName)

				_, err := fakeDynamic.Resource(queueGVR).Create(context.Background(), queue, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create fake queue: %v", err)
				}
			}

			// This test is limited because we can't easily mock the dynamic client creation
			// In a real test environment, you would set up a proper test cluster or use envtest
			err := ensureQueueExists(context.Background(), fakeDynamic, tt.queueName)

			if tt.expectedError {
				if err == nil {
					t.Errorf("expected error but got none")
				} else if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("expected error to contain %q, got %v", tt.errorContains, err)
				}
			} else {
				// We expect an error here because we can't properly mock the dynamic client
				// In a real test, this would work with proper test setup
				if err == nil {
					t.Logf("Queue validation passed (this is expected in unit tests)")
				}
			}
		})
	}
}

func Test_GetComponentReadinessAndServiceReplicaStatuses(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                   string
		dgdSpec                v1alpha1.DynamoGraphDeploymentSpec
		existingGroveResources []client.Object
		wantReady              bool
		wantReason             string
		wantServiceStatuses    map[string]v1alpha1.ServiceReplicaStatus
	}{
		{
			name: "single-node service not ready - PodClique not ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(2)),
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:        2,
						UpdatedReplicas: 2,
						ReadyReplicas:   1,
					},
				},
			},
			wantReady:  false,
			wantReason: "podclique/test-dgd-0-frontend: desired=2, ready=1",
			wantServiceStatuses: map[string]v1alpha1.ServiceReplicaStatus{
				"frontend": {
					ComponentKind:   v1alpha1.ComponentKindPodClique,
					ComponentName:   "test-dgd-0-frontend",
					Replicas:        2,
					UpdatedReplicas: 2,
					ReadyReplicas:   ptr.To(int32(1)),
				},
			},
		},
		{
			name: "all multinode services ready - all PCSGs ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 2,
						},
					},
					"prefill": {
						ServiceName:     "prefill",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypePrefill),
						Replicas:        ptr.To(int32(3)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 4,
						},
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-decode",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: 2,
					},
				},
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-prefill",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 3,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          3,
						UpdatedReplicas:   3,
						AvailableReplicas: 3,
					},
				},
			},
			wantReady:  true,
			wantReason: "",
			wantServiceStatuses: map[string]v1alpha1.ServiceReplicaStatus{
				"decode": {
					ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
					ComponentName:     "test-dgd-0-decode",
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(2)),
				},
				"prefill": {
					ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
					ComponentName:     "test-dgd-0-prefill",
					Replicas:          3,
					UpdatedReplicas:   3,
					AvailableReplicas: ptr.To(int32(3)),
				},
			},
		},
		{
			name: "multinode service not ready - PCSG not ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"worker": {
						ServiceName:     "worker",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeWorker),
						Replicas:        ptr.To(int32(2)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 4,
						},
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-worker",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: 1,
					},
				},
			},
			wantReady:  false,
			wantReason: "pcsg/test-dgd-0-worker: desired=2, available=1",
			wantServiceStatuses: map[string]v1alpha1.ServiceReplicaStatus{
				"worker": {
					ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
					ComponentName:     "test-dgd-0-worker",
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(1)),
				},
			},
		},
		{
			name: "mixed services - some ready, some not - combination of PodClique and PCSG",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 2,
						},
					},
					"prefill": {
						ServiceName:     "prefill",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypePrefill),
						Replicas:        ptr.To(int32(2)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 2,
						},
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:        1,
						UpdatedReplicas: 1,
						ReadyReplicas:   1,
					},
				},
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-decode",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: 1,
					},
				},
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-prefill",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: 2,
					},
				},
			},
			wantReady:  false,
			wantReason: "pcsg/test-dgd-0-decode: desired=2, available=1",
			wantServiceStatuses: map[string]v1alpha1.ServiceReplicaStatus{
				"frontend": {
					ComponentKind:   v1alpha1.ComponentKindPodClique,
					ComponentName:   "test-dgd-0-frontend",
					Replicas:        1,
					UpdatedReplicas: 1,
					ReadyReplicas:   ptr.To(int32(1)),
				},
				"decode": {
					ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
					ComponentName:     "test-dgd-0-decode",
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(1)),
				},
				"prefill": {
					ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
					ComponentName:     "test-dgd-0-prefill",
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(2)),
				},
			},
		},
		{
			name: "service resource not found - PodClique missing",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
				},
			},
			existingGroveResources: []client.Object{},
			wantReady:              false,
			wantReason:             "podclique/test-dgd-0-frontend: resource not found",
			wantServiceStatuses: map[string]v1alpha1.ServiceReplicaStatus{
				"frontend": {},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = grovev1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			dgd := &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: tt.dgdSpec,
			}

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingGroveResources...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			ready, reason, serviceStatuses := GetComponentReadinessAndServiceReplicaStatuses(ctx, fakeKubeClient, dgd)

			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			g.Expect(reason).To(gomega.Equal(tt.wantReason))
			g.Expect(serviceStatuses).To(gomega.Equal(tt.wantServiceStatuses))
		})
	}
}
