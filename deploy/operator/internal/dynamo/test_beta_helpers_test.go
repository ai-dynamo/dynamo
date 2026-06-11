package dynamo

import (
	"encoding/json"
	"maps"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func betaComponent(t testing.TB, src *v1alpha1.DynamoComponentDeploymentSharedSpec) *v1beta1.DynamoComponentDeploymentSharedSpec {
	t.Helper()
	if src == nil {
		return nil
	}

	alpha := &v1alpha1.DynamoComponentDeployment{
		Spec: v1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: *src,
		},
	}
	beta := &v1beta1.DynamoComponentDeployment{}
	if err := alpha.ConvertTo(beta); err != nil {
		t.Fatalf("convert test DCD component to v1beta1: %v", err)
	}
	applyAlphaMetadataToBetaComponent(&beta.Spec.DynamoComponentDeploymentSharedSpec, src.Annotations, src.Labels)
	return &beta.Spec.DynamoComponentDeploymentSharedSpec
}

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
	return dst
}

func betaDCDMap(t testing.TB, src map[string]*v1alpha1.DynamoComponentDeployment) map[string]*v1beta1.DynamoComponentDeployment {
	t.Helper()
	if src == nil {
		return nil
	}
	out := make(map[string]*v1beta1.DynamoComponentDeployment, len(src))
	for name, dcd := range src {
		out[name] = betaDCD(t, dcd)
	}
	return out
}

func normalizeGeneratedDCDMap(src map[string]*v1beta1.DynamoComponentDeployment) map[string]*v1beta1.DynamoComponentDeployment {
	if src == nil {
		return nil
	}
	out := make(map[string]*v1beta1.DynamoComponentDeployment, len(src))
	for name, dcd := range src {
		out[name] = normalizeGeneratedDCD(dcd)
	}
	return out
}

func normalizeGeneratedDCD(src *v1beta1.DynamoComponentDeployment) *v1beta1.DynamoComponentDeployment {
	if src == nil {
		return nil
	}
	out := src.DeepCopy()
	maps.DeleteFunc(out.Annotations, func(key, _ string) bool {
		return v1alpha1.IsDynamoComponentDeploymentConversionAnnotation(key)
	})
	if len(out.Annotations) == 0 {
		out.Annotations = nil
	}
	if out.Spec.PodTemplate != nil {
		for _, key := range []string{
			commonconsts.KubeLabelDynamoComponent,
			commonconsts.KubeLabelDynamoNamespace,
			commonconsts.KubeLabelDynamoGraphDeploymentName,
		} {
			delete(out.Spec.PodTemplate.Labels, key)
		}
		if len(out.Spec.PodTemplate.Labels) == 0 {
			out.Spec.PodTemplate.Labels = nil
		}
		if len(out.Spec.PodTemplate.Annotations) == 0 {
			out.Spec.PodTemplate.Annotations = nil
		}
	}
	return out
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

func betaResourceRequirements(t testing.TB, src *v1alpha1.Resources) *corev1.ResourceRequirements {
	t.Helper()
	if src == nil {
		return nil
	}
	component := betaComponent(t, &v1alpha1.DynamoComponentDeploymentSharedSpec{Resources: src})
	resources := GetMainContainerResources(component)
	return &resources
}

func applyAlphaMetadataToBetaComponent(component *v1beta1.DynamoComponentDeploymentSharedSpec, annotations, labels map[string]string) {
	if len(annotations) == 0 && len(labels) == 0 {
		return
	}
	podTemplate := ensurePodTemplate(component)
	if len(annotations) > 0 {
		maps.Copy(podTemplate.Annotations, annotations)
	}
	if len(labels) > 0 {
		maps.Copy(podTemplate.Labels, labels)
	}
}

func updateDynDeploymentConfig(dcd *v1alpha1.DynamoComponentDeployment, newPort int) error {
	rawConfig := dcd.GetDynamoDeploymentConfig()
	if rawConfig == nil {
		return nil
	}

	var config map[string]map[string]any
	if err := json.Unmarshal(rawConfig, &config); err != nil {
		return err
	}
	if dcd.IsFrontendComponent() {
		if frontend, ok := config[dcd.Spec.ServiceName]; ok {
			frontend["port"] = newPort
		} else if frontend, ok := config[commonconsts.ComponentTypeFrontend]; ok {
			frontend["port"] = newPort
		}
	}
	data, err := json.Marshal(config)
	if err != nil {
		return err
	}
	dcd.SetDynamoDeploymentConfig(data)
	return nil
}

func overrideWithDynDeploymentConfig(dcd *v1alpha1.DynamoComponentDeployment) error {
	rawConfig := dcd.GetDynamoDeploymentConfig()
	if rawConfig == nil {
		return nil
	}
	config, err := ParseDynDeploymentConfig(rawConfig)
	if err != nil {
		return err
	}
	serviceConfig := config[dcd.Spec.ServiceName]
	if serviceConfig == nil || serviceConfig.ServiceArgs == nil {
		return nil
	}
	if serviceConfig.ServiceArgs.Workers != nil {
		dcd.Spec.Replicas = serviceConfig.ServiceArgs.Workers
	}
	if serviceConfig.ServiceArgs.Resources != nil {
		applyLegacyResourcesOverride(&dcd.Spec.DynamoComponentDeploymentSharedSpec, serviceConfig.ServiceArgs.Resources)
	}
	return nil
}

func applyLegacyResourcesOverride(component *v1alpha1.DynamoComponentDeploymentSharedSpec, resources *Resources) {
	if component.Resources == nil {
		component.Resources = &v1alpha1.Resources{}
	}
	if component.Resources.Requests == nil {
		component.Resources.Requests = &v1alpha1.ResourceItem{}
	}
	limits := &v1alpha1.ResourceItem{}
	if resources.CPU != nil {
		component.Resources.Requests.CPU = *resources.CPU
		limits.CPU = *resources.CPU
	}
	if resources.Memory != nil {
		component.Resources.Requests.Memory = *resources.Memory
		limits.Memory = *resources.Memory
	}
	if resources.GPU != nil {
		component.Resources.Requests.GPU = *resources.GPU
		limits.GPU = *resources.GPU
	}
	if resources.Custom != nil {
		component.Resources.Requests.Custom = maps.Clone(resources.Custom)
		limits.Custom = maps.Clone(resources.Custom)
	}
	component.Resources.Limits = limits
}
