package v1alpha1

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestBugDCD_HubSidecarOnlyPodTemplateRoundTrips(t *testing.T) {
	in := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sidecar-only", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "sidecar-only",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						NodeSelector: map[string]string{"accelerator": "h100"},
						Containers: []corev1.Container{{
							Name:  "metrics",
							Image: "busybox:1.36",
							Args:  []string{"sh", "-c", "sleep 3600"},
						}},
					},
				},
			},
		},
	}

	out := dcdRoundTripFromV1beta1(t, in)
	if diff := cmp.Diff(in.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_HubPodTemplateContainerOrderRoundTrips(t *testing.T) {
	in := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "container-order", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "container-order",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "metrics", Image: "busybox:1.36"},
							{Name: "main", Image: "dynamo:latest"},
						},
					},
				},
			},
		},
	}

	out := dcdRoundTripFromV1beta1(t, in)
	if diff := cmp.Diff(in.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_HubEditToEnvFromSecretOptionalRoundTrips(t *testing.T) {
	alpha := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "env-from-secret", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ServiceName:   "env-from-secret",
				EnvFromSecret: ptr.To("secret-a"),
			},
		},
	}

	hub := &v1beta1.DynamoComponentDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	hub.Spec.PodTemplate.Spec.Containers[0].EnvFrom[0].SecretRef.Optional = ptr.To(true)

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	out := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if diff := cmp.Diff(hub.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}
