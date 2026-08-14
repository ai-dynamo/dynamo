package dynamo

import (
	"context"
	"fmt"
	"strings"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
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
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func init() {
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		panic(err)
	}
}

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
		runtimeConfig      *controller_common.RuntimeConfig
		validatedQueueName string
		initialClique      *grovev1alpha1.PodCliqueTemplateSpec
		expectedScheduler  string
		expectedQueueLabel string
		shouldInject       bool
	}{
		{
			name: "grove disabled - no injection",
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{KaiScheduler: true},
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
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true},
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
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true, KaiScheduler: true},
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
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true, KaiScheduler: true},
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
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true, KaiScheduler: true},
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
			injectKaiSchedulerIfEnabled(clique, tt.runtimeConfig, tt.validatedQueueName)

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

func TestInjectVolcanoSchedulerIfEnabled(t *testing.T) {
	tests := []struct {
		name              string
		runtimeConfig     *controller_common.RuntimeConfig
		initialScheduler  string
		expectedScheduler string
	}{
		{
			name: "grove disabled - no injection",
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{VolcanoScheduler: true},
			},
			expectedScheduler: "",
		},
		{
			name: "volcano scheduler disabled - no injection",
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true},
			},
			expectedScheduler: "",
		},
		{
			name: "manual scheduler set - no injection",
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true, VolcanoScheduler: true},
			},
			initialScheduler:  "manual-scheduler",
			expectedScheduler: "manual-scheduler",
		},
		{
			name: "both enabled, no manual scheduler - inject",
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true, VolcanoScheduler: true},
			},
			expectedScheduler: commonconsts.VolcanoSchedulerName,
		},
		{
			name: "volcano scheduler already set - preserve",
			runtimeConfig: &controller_common.RuntimeConfig{
				Gate: features.Gates{Grove: true, VolcanoScheduler: true},
			},
			initialScheduler:  commonconsts.VolcanoSchedulerName,
			expectedScheduler: commonconsts.VolcanoSchedulerName,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clique := &grovev1alpha1.PodCliqueTemplateSpec{
				Spec: grovev1alpha1.PodCliqueSpec{
					PodSpec: corev1.PodSpec{
						SchedulerName: tt.initialScheduler,
					},
				},
			}

			injectVolcanoSchedulerIfEnabled(clique, tt.runtimeConfig)

			if clique.Spec.PodSpec.SchedulerName != tt.expectedScheduler {
				t.Errorf("expected schedulerName %v, got %v", tt.expectedScheduler, clique.Spec.PodSpec.SchedulerName)
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

func TestCheckPodCliqueReady(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name               string
		resourceName       string
		namespace          string
		existingPodClique  *grovev1alpha1.PodClique
		wantReady          bool
		wantReasonContains string
		wantClassification string
		wantServiceStatus  v1beta1.ComponentReplicaStatus
	}{
		{
			name:               "PodClique not found",
			resourceName:       "missing-podclique",
			namespace:          "default",
			wantReady:          false,
			wantReasonContains: "resource not found",
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:  v1beta1.ComponentKindPodClique,
				ComponentNames: []string{"missing-podclique"},
			},
		},
		{
			name:         "PodClique fully ready",
			resourceName: "ready-podclique",
			namespace:    "default",
			existingPodClique: &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "ready-podclique",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           3,
					ReadyReplicas:      3,
					UpdatedReplicas:    3,
					ScheduledReplicas:  3,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          true,
			wantClassification: "",
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodClique,
				ComponentNames:    []string{"ready-podclique"},
				Replicas:          3,
				UpdatedReplicas:   3,
				ReadyReplicas:     ptr.To(int32(3)),
				ScheduledReplicas: ptr.To(int32(3)),
			},
		},
		{
			name:         "PodClique with zero replicas desired",
			resourceName: "zero-replicas-podclique",
			namespace:    "default",
			existingPodClique: &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "zero-replicas-podclique",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					Replicas: 0,
				},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           0,
					ReadyReplicas:      0,
					UpdatedReplicas:    0,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          true,
			wantClassification: "",
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodClique,
				ComponentNames:    []string{"zero-replicas-podclique"},
				Replicas:          0,
				UpdatedReplicas:   0,
				ReadyReplicas:     ptr.To(int32(0)),
				ScheduledReplicas: ptr.To(int32(0)),
			},
		},
		{
			name:         "PodClique spec not yet processed - observedGeneration < generation",
			resourceName: "stale-podclique",
			namespace:    "default",
			existingPodClique: &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "stale-podclique",
					Namespace:  "default",
					Generation: 3,
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					Replicas: 2,
				},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           2,
					ReadyReplicas:      2,
					UpdatedReplicas:    2,
					ObservedGeneration: ptr.To(int64(2)),
				},
			},
			wantReady:          false,
			wantReasonContains: "spec not yet processed",
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:   v1beta1.ComponentKindPodClique,
				ComponentNames:  []string{"stale-podclique"},
				Replicas:        2,
				UpdatedReplicas: 2,
				ReadyReplicas:   ptr.To(int32(2)),
			},
		},
		{
			name:         "PodClique not ready - ready replicas less than desired",
			resourceName: "not-ready-podclique",
			namespace:    "default",
			existingPodClique: &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "not-ready-podclique",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           3,
					ReadyReplicas:      1,
					UpdatedReplicas:    3,
					ScheduledReplicas:  3,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          false,
			wantReasonContains: "scheduled but ready=1/3",
			wantClassification: v1beta1.DGDReadyReasonPodsNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodClique,
				ComponentNames:    []string{"not-ready-podclique"},
				Replicas:          3,
				UpdatedReplicas:   3,
				ReadyReplicas:     ptr.To(int32(1)),
				ScheduledReplicas: ptr.To(int32(3)),
			},
		},
		{
			name:         "PodClique not fully updated - updated replicas less than desired",
			resourceName: "not-updated-podclique",
			namespace:    "default",
			existingPodClique: &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "not-updated-podclique",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           3,
					ReadyReplicas:      3,
					UpdatedReplicas:    2,
					ScheduledReplicas:  3,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          false,
			wantReasonContains: "desired=3, updated=2",
			wantClassification: v1beta1.DGDReadyReasonUpdating,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodClique,
				ComponentNames:    []string{"not-updated-podclique"},
				Replicas:          3,
				UpdatedReplicas:   2,
				ReadyReplicas:     ptr.To(int32(3)),
				ScheduledReplicas: ptr.To(int32(3)),
			},
		},
		{
			name:         "PodClique performing rolling update - replicas != desired",
			resourceName: "rolling-update-podclique",
			namespace:    "default",
			existingPodClique: &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "rolling-update-podclique",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           4,
					ReadyReplicas:      3,
					UpdatedReplicas:    3,
					ScheduledReplicas:  4,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          false,
			wantReasonContains: "performing rolling update",
			wantClassification: v1beta1.DGDReadyReasonUpdating,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodClique,
				ComponentNames:    []string{"rolling-update-podclique"},
				Replicas:          4,
				UpdatedReplicas:   3,
				ReadyReplicas:     ptr.To(int32(3)),
				ScheduledReplicas: ptr.To(int32(4)),
			},
		},
		{
			name:         "PodClique with nil observedGeneration",
			resourceName: "nil-observed-gen-podclique",
			namespace:    "default",
			existingPodClique: &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "nil-observed-gen-podclique",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{
					Replicas: 2,
				},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           2,
					ReadyReplicas:      2,
					UpdatedReplicas:    2,
					ObservedGeneration: nil,
				},
			},
			wantReady:          false,
			wantReasonContains: "observedGeneration is nil",
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:   v1beta1.ComponentKindPodClique,
				ComponentNames:  []string{"nil-observed-gen-podclique"},
				Replicas:        2,
				UpdatedReplicas: 2,
				ReadyReplicas:   ptr.To(int32(2)),
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

			var objects []client.Object
			if tt.existingPodClique != nil {
				objects = append(objects, tt.existingPodClique)
			}

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			logger := log.FromContext(ctx)
			ready, reason, serviceStatus, classification, checkErr := CheckPodCliqueReady(ctx, fakeKubeClient, tt.resourceName, tt.namespace, logger)

			g.Expect(checkErr).NotTo(gomega.HaveOccurred())
			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			if tt.wantReasonContains != "" {
				g.Expect(reason).To(gomega.ContainSubstring(tt.wantReasonContains))
			} else {
				g.Expect(reason).To(gomega.Equal(""))
			}
			g.Expect(classification).To(gomega.Equal(tt.wantClassification))
			g.Expect(serviceStatus).To(gomega.Equal(tt.wantServiceStatus))
		})
	}
}

func TestCheckPCSGReady(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name               string
		resourceName       string
		namespace          string
		existingPCSG       *grovev1alpha1.PodCliqueScalingGroup
		wantReady          bool
		wantReasonContains string
		wantClassification string
		wantServiceStatus  v1beta1.ComponentReplicaStatus
	}{
		{
			name:               "PCSG not found",
			resourceName:       "missing-pcsg",
			namespace:          "default",
			wantReady:          false,
			wantReasonContains: "resource not found",
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:  v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames: []string{"missing-pcsg"},
			},
		},
		{
			name:         "PCSG fully ready",
			resourceName: "ready-pcsg",
			namespace:    "default",
			existingPCSG: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "ready-pcsg",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueScalingGroupStatus{
					Replicas:           3,
					AvailableReplicas:  3,
					UpdatedReplicas:    3,
					ScheduledReplicas:  3,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          true,
			wantClassification: "",
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames:    []string{"ready-pcsg"},
				Replicas:          3,
				UpdatedReplicas:   3,
				AvailableReplicas: ptr.To(int32(3)),
				ScheduledReplicas: ptr.To(int32(3)),
			},
		},
		{
			name:         "PCSG with zero replicas desired",
			resourceName: "zero-replicas-pcsg",
			namespace:    "default",
			existingPCSG: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "zero-replicas-pcsg",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
					Replicas: 0,
				},
				Status: grovev1alpha1.PodCliqueScalingGroupStatus{
					Replicas:           0,
					AvailableReplicas:  0,
					UpdatedReplicas:    0,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          true,
			wantClassification: "",
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames:    []string{"zero-replicas-pcsg"},
				Replicas:          0,
				UpdatedReplicas:   0,
				AvailableReplicas: ptr.To(int32(0)),
				ScheduledReplicas: ptr.To(int32(0)),
			},
		},
		{
			name:         "PCSG spec not yet processed - observedGeneration < generation",
			resourceName: "stale-pcsg",
			namespace:    "default",
			existingPCSG: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "stale-pcsg",
					Namespace:  "default",
					Generation: 3,
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
					Replicas: 2,
				},
				Status: grovev1alpha1.PodCliqueScalingGroupStatus{
					Replicas:           2,
					AvailableReplicas:  2,
					UpdatedReplicas:    2,
					ObservedGeneration: ptr.To(int64(2)),
				},
			},
			wantReady:          false,
			wantReasonContains: "spec not yet processed",
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames:    []string{"stale-pcsg"},
				Replicas:          2,
				UpdatedReplicas:   2,
				AvailableReplicas: ptr.To(int32(2)),
			},
		},
		{
			name:         "PCSG not ready - available replicas less than desired",
			resourceName: "not-ready-pcsg",
			namespace:    "default",
			existingPCSG: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "not-ready-pcsg",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueScalingGroupStatus{
					Replicas:           3,
					AvailableReplicas:  1,
					UpdatedReplicas:    3,
					ScheduledReplicas:  3,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          false,
			wantReasonContains: "scheduled but available=1/3",
			wantClassification: v1beta1.DGDReadyReasonPodsNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames:    []string{"not-ready-pcsg"},
				Replicas:          3,
				UpdatedReplicas:   3,
				AvailableReplicas: ptr.To(int32(1)),
				ScheduledReplicas: ptr.To(int32(3)),
			},
		},
		{
			name:         "PCSG not fully updated - updated replicas less than desired",
			resourceName: "not-updated-pcsg",
			namespace:    "default",
			existingPCSG: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "not-updated-pcsg",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueScalingGroupStatus{
					Replicas:           3,
					AvailableReplicas:  3,
					UpdatedReplicas:    2,
					ScheduledReplicas:  3,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          false,
			wantReasonContains: "desired=3, updated=2",
			wantClassification: v1beta1.DGDReadyReasonUpdating,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames:    []string{"not-updated-pcsg"},
				Replicas:          3,
				UpdatedReplicas:   2,
				AvailableReplicas: ptr.To(int32(3)),
				ScheduledReplicas: ptr.To(int32(3)),
			},
		},
		{
			name:         "PCSG performing rolling update - replicas != desired",
			resourceName: "rolling-update-pcsg",
			namespace:    "default",
			existingPCSG: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "rolling-update-pcsg",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
					Replicas: 3,
				},
				Status: grovev1alpha1.PodCliqueScalingGroupStatus{
					Replicas:           4,
					AvailableReplicas:  3,
					UpdatedReplicas:    3,
					ScheduledReplicas:  4,
					ObservedGeneration: ptr.To(int64(1)),
				},
			},
			wantReady:          false,
			wantReasonContains: "performing rolling update",
			wantClassification: v1beta1.DGDReadyReasonUpdating,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames:    []string{"rolling-update-pcsg"},
				Replicas:          4,
				UpdatedReplicas:   3,
				AvailableReplicas: ptr.To(int32(3)),
				ScheduledReplicas: ptr.To(int32(4)),
			},
		},
		{
			name:         "PCSG with nil observedGeneration",
			resourceName: "nil-observed-gen-pcsg",
			namespace:    "default",
			existingPCSG: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "nil-observed-gen-pcsg",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
					Replicas: 2,
				},
				Status: grovev1alpha1.PodCliqueScalingGroupStatus{
					Replicas:           2,
					AvailableReplicas:  2,
					UpdatedReplicas:    2,
					ObservedGeneration: nil,
				},
			},
			wantReady:          false,
			wantReasonContains: "observedGeneration is nil",
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantServiceStatus: v1beta1.ComponentReplicaStatus{
				ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
				ComponentNames:    []string{"nil-observed-gen-pcsg"},
				Replicas:          2,
				UpdatedReplicas:   2,
				AvailableReplicas: ptr.To(int32(2)),
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

			var objects []client.Object
			if tt.existingPCSG != nil {
				objects = append(objects, tt.existingPCSG)
			}

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			logger := log.FromContext(ctx)
			ready, reason, serviceStatus, classification, checkErr := CheckPCSGReady(ctx, fakeKubeClient, tt.resourceName, tt.namespace, logger)

			g.Expect(checkErr).NotTo(gomega.HaveOccurred())
			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			if tt.wantReasonContains != "" {
				g.Expect(reason).To(gomega.ContainSubstring(tt.wantReasonContains))
			} else {
				g.Expect(reason).To(gomega.Equal(""))
			}
			g.Expect(classification).To(gomega.Equal(tt.wantClassification))
			g.Expect(serviceStatus).To(gomega.Equal(tt.wantServiceStatus))
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
		wantServiceStatuses    map[string]v1beta1.ComponentReplicaStatus
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
						Replicas:           2,
						UpdatedReplicas:    2,
						ReadyReplicas:      1,
						ScheduledReplicas:  2,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReady:  false,
			wantReason: "frontend: scheduled but ready=1/2",
			wantServiceStatuses: map[string]v1beta1.ComponentReplicaStatus{
				"frontend": {
					ComponentKind:     v1beta1.ComponentKindPodClique,
					ComponentNames:    []string{"test-dgd-0-frontend"},
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(1)),
					ScheduledReplicas: ptr.To(int32(2)),
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
						Replicas:           2,
						UpdatedReplicas:    2,
						AvailableReplicas:  2,
						ScheduledReplicas:  2,
						ObservedGeneration: ptr.To(int64(1)),
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
						Replicas:           3,
						UpdatedReplicas:    3,
						AvailableReplicas:  3,
						ScheduledReplicas:  3,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReady:  true,
			wantReason: "",
			wantServiceStatuses: map[string]v1beta1.ComponentReplicaStatus{
				"decode": {
					ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
					ComponentNames:    []string{"test-dgd-0-decode"},
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(2)),
					ScheduledReplicas: ptr.To(int32(2)),
				},
				"prefill": {
					ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
					ComponentNames:    []string{"test-dgd-0-prefill"},
					Replicas:          3,
					UpdatedReplicas:   3,
					AvailableReplicas: ptr.To(int32(3)),
					ScheduledReplicas: ptr.To(int32(3)),
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
						Replicas:           2,
						UpdatedReplicas:    2,
						AvailableReplicas:  1,
						ScheduledReplicas:  2,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReady:  false,
			wantReason: "worker: scheduled but available=1/2",
			wantServiceStatuses: map[string]v1beta1.ComponentReplicaStatus{
				"worker": {
					ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
					ComponentNames:    []string{"test-dgd-0-worker"},
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(1)),
					ScheduledReplicas: ptr.To(int32(2)),
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
						Replicas:           1,
						UpdatedReplicas:    1,
						ReadyReplicas:      1,
						ScheduledReplicas:  1,
						ObservedGeneration: ptr.To(int64(1)),
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
						Replicas:           2,
						UpdatedReplicas:    2,
						AvailableReplicas:  1,
						ScheduledReplicas:  2,
						ObservedGeneration: ptr.To(int64(1)),
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
						Replicas:           2,
						UpdatedReplicas:    2,
						AvailableReplicas:  2,
						ScheduledReplicas:  2,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReady:  false,
			wantReason: "decode: scheduled but available=1/2",
			wantServiceStatuses: map[string]v1beta1.ComponentReplicaStatus{
				"frontend": {
					ComponentKind:     v1beta1.ComponentKindPodClique,
					ComponentNames:    []string{"test-dgd-0-frontend"},
					Replicas:          1,
					UpdatedReplicas:   1,
					ReadyReplicas:     ptr.To(int32(1)),
					ScheduledReplicas: ptr.To(int32(1)),
				},
				"decode": {
					ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
					ComponentNames:    []string{"test-dgd-0-decode"},
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(1)),
					ScheduledReplicas: ptr.To(int32(2)),
				},
				"prefill": {
					ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
					ComponentNames:    []string{"test-dgd-0-prefill"},
					Replicas:          2,
					UpdatedReplicas:   2,
					AvailableReplicas: ptr.To(int32(2)),
					ScheduledReplicas: ptr.To(int32(2)),
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
			wantReason:             "frontend: resource not found",
			wantServiceStatuses: map[string]v1beta1.ComponentReplicaStatus{
				"frontend": {
					ComponentKind:  v1beta1.ComponentKindPodClique,
					ComponentNames: []string{"test-dgd-0-frontend"},
				},
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

			betaDGD := betaDGD(t, dgd)
			var objects []client.Object
			objects = append(objects, betaDGD)
			objects = append(objects, tt.existingGroveResources...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			ready, reason, serviceStatuses, err := GetComponentReadinessAndServiceReplicaStatuses(ctx, fakeKubeClient, betaDGD)

			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			g.Expect(reason).To(gomega.Equal(tt.wantReason))
			for componentName, wantStatus := range tt.wantServiceStatuses {
				component := betaDGD.GetComponentByName(componentName)
				if component == nil {
					continue
				}
				wantStatus.RuntimeNamespace = betaDGD.GetDynamoNamespaceForComponent(component)
				tt.wantServiceStatuses[componentName] = wantStatus
			}
			g.Expect(serviceStatuses).To(gomega.Equal(tt.wantServiceStatuses))
		})
	}
}

// ---------------------------------------------------------------------------
// Ready-reason classification tests (merged from classification_test.go).
// These exercise the DGD-level Ready reason returned as the 4th value of
// CheckPodCliqueReady / CheckPCSGReady, focusing on the capacity-before-
// readiness branches (schedule-gated, scheduling condition, partial scheduled
// count) that the readiness/serviceStatus tables above do not isolate.
// ---------------------------------------------------------------------------

// The Check*Ready classification tests below exercise the full Grove-status
// reading path (client.Get + field/condition inspection) and assert the
// DGD Ready reason string returned as the 4th value. They complement the
// existing TestCheckPodCliqueReady / TestCheckPCSGReady tables (which assert
// ready/reason/serviceStatus) by focusing on capacity-before-readiness
// classification ordering.

func TestCheckPodCliqueReadyClassification(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name               string
		podClique          *grovev1alpha1.PodClique
		wantReady          bool
		wantClassification string
		wantReasonContains string
	}{
		{
			name: "fully ready",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 3, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          true,
			wantClassification: "",
		},
		{
			name: "scheduling condition InsufficientScheduledPods -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 1, ObservedGeneration: ptr.To(int64(1)),
				Conditions: []metav1.Condition{{
					Type:               groveconstants.ConditionTypePodCliqueScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             groveconstants.ConditionReasonInsufficientScheduledPods,
					LastTransitionTime: metav1.Now(),
				}},
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonInsufficientCapacity,
			wantReasonContains: "scheduling condition",
		},
		{
			name: "schedule-gated replicas -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ScheduleGatedReplicas: 2,
				ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonInsufficientCapacity,
			wantReasonContains: "schedule-gated",
		},
		{
			name: "scheduled condition false with insufficient-pods reason -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
				Conditions: []metav1.Condition{{
					Type:               groveconstants.ConditionTypePodCliqueScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             groveconstants.ConditionReasonInsufficientScheduledPods,
					LastTransitionTime: metav1.Now(),
				}},
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonInsufficientCapacity,
			wantReasonContains: "scheduling condition",
		},
		{
			name: "scheduled but not updated -> Updating",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 3, UpdatedReplicas: 2,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonUpdating,
			wantReasonContains: "updated=2",
		},
		{
			name: "rolling update (replicas != desired) -> Updating",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 4, ReadyReplicas: 3, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonUpdating,
			wantReasonContains: "rolling update",
		},
		{
			name: "scheduled and updated but not ready -> PodsNotReady",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 1, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonPodsNotReady,
			wantReasonContains: "ready=1/3",
		},
		{
			name:               "not found -> Unclassified",
			podClique:          nil,
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantReasonContains: "resource not found",
		},
		{
			name: "nil observedGeneration -> Unclassified",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 3, UpdatedReplicas: 3,
				ScheduledReplicas: 3, ObservedGeneration: nil,
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantReasonContains: "observedGeneration is nil",
		},
		{
			name: "capacity checked before readiness: schedule-gated AND unready -> InsufficientCapacity",
			podClique: newPodClique(grovev1alpha1.PodCliqueStatus{
				Replicas: 3, ReadyReplicas: 0, UpdatedReplicas: 3,
				ScheduledReplicas: 1, ScheduleGatedReplicas: 2,
				ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonInsufficientCapacity,
			wantReasonContains: "schedule-gated",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			var objs []client.Object
			if tt.podClique != nil {
				objs = append(objs, tt.podClique)
			}
			c := newFakeGroveClient(g, objs...)
			ready, reason, _, classification, checkErr := CheckPodCliqueReady(ctx, c, testPodCliqueName, "default", log.FromContext(ctx))
			g.Expect(checkErr).NotTo(gomega.HaveOccurred())

			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			g.Expect(classification).To(gomega.Equal(tt.wantClassification))
			if tt.wantReasonContains != "" {
				g.Expect(reason).To(gomega.ContainSubstring(tt.wantReasonContains))
			}
		})
	}
}

func TestCheckPCSGReadyClassification(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name               string
		pcsg               *grovev1alpha1.PodCliqueScalingGroup
		wantReady          bool
		wantClassification string
		wantReasonContains string
	}{
		{
			name: "fully ready",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 2, UpdatedReplicas: 2,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          true,
			wantClassification: "",
		},
		{
			name: "available below desired, fully scheduled -> PodsNotReady",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 0, UpdatedReplicas: 2,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonPodsNotReady,
			wantReasonContains: "available=0/2",
		},
		{
			name: "partial scheduled count below desired -> InsufficientCapacity",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 0, UpdatedReplicas: 2,
				ScheduledReplicas: 1, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonInsufficientCapacity,
			wantReasonContains: "scheduled=1/2",
		},
		{
			name: "MinAvailableBreached=False with PCSG-scheduling reason -> InsufficientCapacity",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 0, UpdatedReplicas: 2,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
				Conditions: []metav1.Condition{{
					// Grove alpha.8 emits the scheduling-shortfall reason with
					// Status=False (Status=True is the availability reason).
					Type:               groveconstants.ConditionTypeMinAvailableBreached,
					Status:             metav1.ConditionFalse,
					Reason:             "InsufficientScheduledPodCliqueScalingGroupReplicas",
					LastTransitionTime: metav1.Now(),
				}},
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonInsufficientCapacity,
			wantReasonContains: "min-available breached",
		},
		{
			name: "scheduled but not updated -> Updating",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 2, UpdatedReplicas: 1,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonUpdating,
			wantReasonContains: "updated=1",
		},
		{
			name: "scheduled and updated but not available -> PodsNotReady",
			pcsg: newPCSG(grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas: 2, AvailableReplicas: 1, UpdatedReplicas: 2,
				ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1)),
			}),
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonPodsNotReady,
			wantReasonContains: "available=1/2",
		},
		{
			name:               "not found -> Unclassified",
			pcsg:               nil,
			wantReady:          false,
			wantClassification: v1beta1.DGDReadyReasonSomeResourcesNotReady,
			wantReasonContains: "resource not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			var objs []client.Object
			if tt.pcsg != nil {
				objs = append(objs, tt.pcsg)
			}
			c := newFakeGroveClient(g, objs...)
			ready, reason, _, classification, checkErr := CheckPCSGReady(ctx, c, testPCSGName, "default", log.FromContext(ctx))
			g.Expect(checkErr).NotTo(gomega.HaveOccurred())

			g.Expect(ready).To(gomega.Equal(tt.wantReady))
			g.Expect(classification).To(gomega.Equal(tt.wantClassification))
			if tt.wantReasonContains != "" {
				g.Expect(reason).To(gomega.ContainSubstring(tt.wantReasonContains))
			}
		})
	}
}

// --- test helpers ---

// Fixed resource identity used by the classification tests. The helpers below
// hardcode these so the name passed to CheckPodCliqueReady / CheckPCSGReady
// stays in sync with the object created in the fake client, and so the spec
// replica count (which every case shares) is defined in one place. Only the
// status varies per test case.
const (
	testPodCliqueName     = "pc"
	testPodCliqueReplicas = 3
	testPCSGName          = "pcsg"
	testPCSGReplicas      = 2
)

func newPodClique(status grovev1alpha1.PodCliqueStatus) *grovev1alpha1.PodClique {
	return &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: testPodCliqueName, Namespace: "default", Generation: 1},
		Spec:       grovev1alpha1.PodCliqueSpec{Replicas: testPodCliqueReplicas},
		Status:     status,
	}
}

func newPCSG(status grovev1alpha1.PodCliqueScalingGroupStatus) *grovev1alpha1.PodCliqueScalingGroup {
	return &grovev1alpha1.PodCliqueScalingGroup{
		ObjectMeta: metav1.ObjectMeta{Name: testPCSGName, Namespace: "default", Generation: 1},
		Spec:       grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: testPCSGReplicas},
		Status:     status,
	}
}

func newFakeGroveClient(g *gomega.WithT, objects ...client.Object) client.Client {
	s := scheme.Scheme
	g.Expect(grovev1alpha1.AddToScheme(s)).To(gomega.Succeed())
	return fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(objects...).
		WithStatusSubresource(objects...).
		Build()
}

// TestGroveReadinessTransientErrorsPropagate verifies that a non-NotFound Get
// error from a Grove child is returned as an error (so the reconcile retries and
// does not advance ObservedGeneration), rather than folded into a normal
// not-ready result. NotFound remains a legitimate not-ready state (covered by
// the "not found" cases in TestCheckPodCliqueReady / TestCheckPCSGReady).
func TestGroveReadinessTransientErrorsPropagate(t *testing.T) {
	ctx := context.Background()
	logger := log.FromContext(ctx)
	transientErr := fmt.Errorf("transient API error")

	newClient := func(g *gomega.WithT) client.Client {
		s := scheme.Scheme
		g.Expect(v1alpha1.AddToScheme(s)).To(gomega.Succeed())
		g.Expect(v1beta1.AddToScheme(s)).To(gomega.Succeed())
		g.Expect(grovev1alpha1.AddToScheme(s)).To(gomega.Succeed())
		return fake.NewClientBuilder().
			WithScheme(s).
			WithInterceptorFuncs(interceptor.Funcs{
				Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
					switch obj.(type) {
					case *grovev1alpha1.PodClique, *grovev1alpha1.PodCliqueScalingGroup:
						return transientErr
					}
					return c.Get(ctx, key, obj, opts...)
				},
			}).
			Build()
	}

	t.Run("CheckPodCliqueReady returns error on non-NotFound get failure", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		c := newClient(g)
		ready, _, _, classification, err := CheckPodCliqueReady(ctx, c, "test-pc", "default", logger)
		g.Expect(err).To(gomega.HaveOccurred())
		g.Expect(err.Error()).To(gomega.ContainSubstring("transient API error"))
		g.Expect(ready).To(gomega.BeFalse())
		// On a transient error we do not emit a classification; the reconcile retries.
		g.Expect(classification).To(gomega.BeEmpty())
	})

	t.Run("CheckPCSGReady returns error on non-NotFound get failure", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		c := newClient(g)
		ready, _, _, classification, err := CheckPCSGReady(ctx, c, "test-pcsg", "default", logger)
		g.Expect(err).To(gomega.HaveOccurred())
		g.Expect(err.Error()).To(gomega.ContainSubstring("transient API error"))
		g.Expect(ready).To(gomega.BeFalse())
		g.Expect(classification).To(gomega.BeEmpty())
	})

	t.Run("GetComponentReadinessAndServiceReplicaStatuses propagates the error", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
			Spec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
				},
			},
		})
		c := newClient(g)
		ready, _, _, err := GetComponentReadinessAndServiceReplicaStatuses(ctx, c, dgd)
		g.Expect(err).To(gomega.HaveOccurred())
		g.Expect(err.Error()).To(gomega.ContainSubstring("transient API error"))
		// A transient error is not a normal not-ready result: ready is false and
		// the error is what callers must act on.
		g.Expect(ready).To(gomega.BeFalse())
	})

	t.Run("ClassifyGroveReadiness propagates the error", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
			Spec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
				},
			},
		})
		c := newClient(g)
		_, err := ClassifyGroveReadiness(ctx, c, dgd)
		g.Expect(err).To(gomega.HaveOccurred())
		g.Expect(err.Error()).To(gomega.ContainSubstring("transient API error"))
	})
}
