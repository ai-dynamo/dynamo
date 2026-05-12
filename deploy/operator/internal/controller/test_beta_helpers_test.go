package controller

import (
	"maps"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
)

func betaDCD(t testing.TB, src *v1alpha1.DynamoComponentDeployment) *v1beta1.DynamoComponentDeployment {
	t.Helper()
	if src == nil {
		return nil
	}
	dst := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("convert test DCD to v1beta1: %v", err)
	}
	applyAlphaMetadataToBetaComponent(&dst.Spec.DynamoComponentDeploymentSharedSpec, src.Spec.Annotations, src.Spec.Labels)
	if serviceName := src.Spec.ServiceName; serviceName != "" {
		if dst.Labels == nil {
			dst.Labels = map[string]string{}
		}
		dst.Labels[commonconsts.KubeLabelDynamoComponent] = serviceName
	}
	for _, key := range []string{
		commonconsts.KubeLabelDynamoGraphDeploymentName,
		commonconsts.KubeLabelDynamoNamespace,
		commonconsts.KubeLabelDynamoWorkerHash,
	} {
		if value := src.Spec.Labels[key]; value != "" {
			if dst.Labels == nil {
				dst.Labels = map[string]string{}
			}
			dst.Labels[key] = value
		}
	}
	return dst
}

func betaDGD(t testing.TB, src *v1alpha1.DynamoGraphDeployment) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	if src == nil {
		return nil
	}
	dst := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("convert test DGD to v1beta1: %v", err)
	}
	for i := range dst.Spec.Components {
		component := &dst.Spec.Components[i]
		if alphaSvc := src.Spec.Services[component.ComponentName]; alphaSvc != nil {
			applyAlphaMetadataToBetaComponent(component, alphaSvc.Annotations, alphaSvc.Labels)
		}
	}
	return dst
}

func mustBetaDGD(src *v1alpha1.DynamoGraphDeployment) *v1beta1.DynamoGraphDeployment {
	if src == nil {
		return nil
	}
	dst := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(dst); err != nil {
		panic(err)
	}
	for i := range dst.Spec.Components {
		component := &dst.Spec.Components[i]
		if alphaSvc := src.Spec.Services[component.ComponentName]; alphaSvc != nil {
			applyAlphaMetadataToBetaComponent(component, alphaSvc.Annotations, alphaSvc.Labels)
		}
	}
	return dst
}

func betaDGDWorkersSpecHash(t testing.TB, dgd *v1beta1.DynamoGraphDeployment) string {
	t.Helper()
	hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		t.Fatalf("compute v1beta1 DGD worker hash: %v", err)
	}
	return hash
}

func legacyDGDWorkersSpecHash(t testing.TB, dgd *v1beta1.DynamoGraphDeployment) string {
	t.Helper()
	hash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	if err != nil {
		t.Fatalf("compute v1alpha1-compatible DGD worker hash: %v", err)
	}
	return hash
}

func betaRestartStatus(src *v1alpha1.RestartStatus) *v1beta1.RestartStatus {
	if src == nil {
		return nil
	}
	return &v1beta1.RestartStatus{
		ObservedID: src.ObservedID,
		Phase:      v1beta1.RestartPhase(src.Phase),
		InProgress: append([]string(nil), src.InProgress...),
	}
}

func betaComponent(t testing.TB, src *v1alpha1.DynamoComponentDeploymentSharedSpec) *v1beta1.DynamoComponentDeploymentSharedSpec {
	t.Helper()
	if src == nil {
		return nil
	}
	dcd := betaDCD(t, &v1alpha1.DynamoComponentDeployment{
		Spec: v1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: *src,
		},
	})
	return &dcd.Spec.DynamoComponentDeploymentSharedSpec
}

func applyAlphaMetadataToBetaComponent(component *v1beta1.DynamoComponentDeploymentSharedSpec, annotations, labels map[string]string) {
	if len(annotations) == 0 && len(labels) == 0 {
		return
	}
	if component.PodTemplate == nil {
		component.PodTemplate = &corev1.PodTemplateSpec{}
	}
	if len(annotations) > 0 {
		if component.PodTemplate.Annotations == nil {
			component.PodTemplate.Annotations = map[string]string{}
		}
		maps.Copy(component.PodTemplate.Annotations, annotations)
	}
	if len(labels) > 0 {
		if component.PodTemplate.Labels == nil {
			component.PodTemplate.Labels = map[string]string{}
		}
		maps.Copy(component.PodTemplate.Labels, labels)
	}
}
