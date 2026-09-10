// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation_test

import (
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	k8sptr "k8s.io/utils/ptr"
)

// lpxDGDAdmissionCases builds fresh LPX scenarios for the single native DGD admission table.
func lpxDGDAdmissionCases() []dgdAdmissionTestCase {
	const longLPXComponentName = "abcdefghijklmnopqrstuvwxyzabcd"

	// Keep LPX inputs and oracles together without a separate admission execution path.
	return []dgdAdmissionTestCase{
		{
			name: "valid singleton lpx deployment with implicit conductor",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.Roles = component.Roles[:1]
				component.Roles[0].PodTemplate.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{{Name: "model-storage", MountPath: "/nfs"}}
				component.Roles[0].PodTemplate.Spec.Volumes = []corev1.Volume{{Name: "model-storage", VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "model-storage"},
				}}}
				component.LPX.Settings = &apiextensionsv1.JSON{Raw: []byte(`{"stop_tokens":[1],"custom_runtime_knob":{"enabled":true}}`)}
				component.Roles[0].PodTemplate.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
						NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchExpressions: []corev1.NodeSelectorRequirement{{
							Key: corev1.LabelHostname, Operator: corev1.NodeSelectorOpIn, Values: []string{"lpu-node-a"},
						}}}},
					},
				}}
			}),
		},
		{
			name:       "valid canonical v1alpha1 lpx deployment",
			deployment: alphaLPXDGDForAdmission(nil),
		},
		{
			name: "LPX admits nine hybrid replicas with lowered conductor volumes",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.Replicas = k8sptr.To(int32(9))
				component.Roles[1].PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					// LPX lowering supplies these volume definitions after admission.
					Containers: []corev1.Container{{Name: "main", Image: "gpu-runtime", VolumeMounts: []corev1.VolumeMount{
						{Name: "config", MountPath: "/config"},
						{Name: "infiniband", MountPath: "/dev/infiniband"},
					}}},
				}}
			}),
		},
		{
			name: "LPX native template schema rejects duplicate main container names",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.Roles[1].PodTemplate = component.Roles[0].PodTemplate.DeepCopy()
				component.Roles[1].PodTemplate.Spec.Containers = []corev1.Container{
					{Name: "main", Image: "runtime"},
					{Name: "main", Image: "runtime"},
				}
			}),
			wantSchemaErr: `spec.components[0].roles[1].podTemplate.spec.containers[1]: Duplicate value: map[string]interface {}{"name":"main"}`,
		},
		{
			name: "LPX replica bounds are enforced by the schema",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Replicas = k8sptr.To(int32(10))
			}),
			wantCELErr: "spec.components[0]: Invalid value: replicas must be between 1 and 9 when type is lpx",
		},
		{
			name: "LPX autoscaling is rejected by the schema",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
			}),
			wantCELErr: "spec.components[0]: Invalid value: scalingAdapter is not supported when type is lpx",
		},
		{
			name: "LPX settings must be an object",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].LPX.Settings = &apiextensionsv1.JSON{Raw: []byte(`[]`)}
			}),
			wantSchemaErr: `spec.components[0].lpx.settings: Invalid value: spec.components[0].lpx.settings in body must be of type object: "array"`,
		},
		{
			name:        "LPX integration availability is deferred to the child controller",
			lpxDisabled: true,
			deployment:  betaLPXDGDForAdmission(nil),
		},
		{
			name:            "LPX rejects the component pathway selected when Grove is disabled",
			groveDisabled:   true,
			deployment:      betaLPXDGDForAdmission(nil),
			wantWebhookErrs: []string{`spec.components: Forbidden: requires the Grove pathway, but workload provider "component" is selected`},
		},
		{
			name: "LPX topology and checkpoint support is deferred to the child controller",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology", PackDomain: "rack",
				}
				component := &dgd.Spec.Components[0]
				component.TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
				component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}),
		},
		{
			name: "LPX admits an ordinary component checkpoint",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := *betaWorkerComponent(betaDGDForAdmission(nil))
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
				dgd.Spec.Components = append(dgd.Spec.Components, worker)
			}),
		},
		{
			name: "LPX shell allocation target validation is deferred to the child controller",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				container := &dgd.Spec.Components[0].Roles[0].PodTemplate.Spec.Containers[0]
				container.Command = []string{"/usr/bin/env"}
				container.Args = []string{"sh", "-c", "exec /opt/dynamo-lpu serve"}
			}),
		},
		{
			name:          "LPX runtime command updates are admitted while the integration is unavailable",
			lpxDisabled:   true,
			oldDeployment: betaLPXDGDForAdmission(nil),
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				container := &dgd.Spec.Components[0].Roles[0].PodTemplate.Spec.Containers[0]
				container.Command = []string{"sh", "-c"}
				container.Args = []string{"exec /opt/dynamo-lpu serve"}
			}),
		},
		{
			name: "LPX rejects an unknown Pod role",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Roles[1].Name = "unknown"
			}),
			wantWebhookErrs: []string{`spec.components[0].roles[1].name: Unsupported value: "unknown": supported values: "conductor", "agent"`},
		},
		{
			name: "v1alpha1 LPX image errors aggregate at the authored role indices",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				component := dgd.Spec.Services["lpx"]
				component.Roles[0], component.Roles[1] = component.Roles[1], component.Roles[0]
				spec := &component.Roles[1].PodTemplate.Spec
				spec.Containers = append(spec.Containers, corev1.Container{Name: "sidecar"})
				spec.InitContainers = []corev1.Container{{Name: "init"}}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].roles[1].podTemplate.spec.containers[1].image: Required value: must specify a non-empty image",
				"spec.components[0].roles[1].podTemplate.spec.initContainers[0].image: Required value: must specify a non-empty image",
			},
		},
		{
			name: "LPX rejects duplicate Pod roles",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.Roles[1] = *component.Roles[0].DeepCopy()
			}),
			wantSchemaErr: `spec.components[0].roles[1]: Duplicate value: map[string]interface {}{"name":"agent"}`,
		},
		{
			name: "LPX reserves materialized agent container names",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				role := &dgd.Spec.Components[0].Roles[0]
				role.PodTemplate.Spec.Containers = append(role.PodTemplate.Spec.Containers, corev1.Container{Name: "agent", Image: "sidecar"})
				role.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "agent", Image: "setup"}}
			}),
			wantWebhookErrs: []string{
				`spec.components[0].roles[0].podTemplate.spec.containers[1].name: Forbidden: LPX reserves "agent" for the materialized role container`,
				`spec.components[0].roles[0].podTemplate.spec.initContainers[0].name: Forbidden: LPX reserves "agent" for the materialized role container`,
			},
		},
		{
			name: "LPX defers implicit conductor name checks until build mode is known",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.Roles = component.Roles[:1]
				component.Roles[0].PodTemplate.Spec.Containers = append(
					component.Roles[0].PodTemplate.Spec.Containers,
					corev1.Container{Name: "conductor", Image: "sidecar"},
				)
			}),
		},
		{
			name: "v1alpha1 LPX defers inherited conductor init names until build mode is known",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				component := dgd.Spec.Services["lpx"]
				component.Roles[0].PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "conductor", Image: "setup"}}
			}),
		},
		{
			name: "LPX defers explicit conductor name checks until build mode is known",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.Roles[1].PodTemplate = component.Roles[0].PodTemplate.DeepCopy()
				component.Roles[1].PodTemplate.Spec.Containers = append(
					component.Roles[1].PodTemplate.Spec.Containers,
					corev1.Container{Name: "conductor", Image: "sidecar"},
				)
			}),
		},
		{
			name: "LPX allows an agent sidecar named conductor with a separate conductor template",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.Roles[1].PodTemplate = component.Roles[0].PodTemplate.DeepCopy()
				component.Roles[0].PodTemplate.Spec.Containers = append(
					component.Roles[0].PodTemplate.Spec.Containers,
					corev1.Container{Name: "conductor", Image: "sidecar"},
				)
			}),
		},
		{
			name: "LPX allows a draft init container named conductor",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaLPXSpecDec(dgd, k8sptr.To[int32](1))
				dgd.Spec.Components[0].Roles[0].PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "conductor", Image: "setup"}}
			}),
		},
		{
			name: "physical LPU rejects Grove opt out",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationEnableGrove: consts.KubeLabelValueFalse}
			}),
			wantWebhookErrs: []string{`spec.components: Forbidden: requires the Grove pathway, but workload provider "component" is selected`},
		},
		{
			name: "v1alpha1 SpecDecode draft replicas must be positive",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaLPXSpecDec(dgd, k8sptr.To(int32(0)))
			}),
			wantCELErr: "spec.services[draft]: Invalid value: replicas must be between 1 and 9 when componentType is lpx",
		},
		// Engine-local template and build validation.
		{
			name: "LPX rejects a component-level PodTemplate",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].PodTemplate = dgd.Spec.Components[0].Roles[0].PodTemplate.DeepCopy()
			}),
			wantCELErr: "spec.components[0]: Invalid value: LPX Pod templates belong to roles",
		},
		{
			name: "LPX rejects a worker without a Pod template",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Roles[0].PodTemplate = nil
			}),
			wantWebhookErrs: []string{"spec.components[0].roles[0].podTemplate: Required value: the LPX agent role requires a podTemplate"},
		},
		{
			name: "LPX requires a build for an LPU engine",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].LPX.BuildID = ""
			}),
			wantSchemaErr: "spec.components[0].lpx.buildId: Invalid value: \"\": spec.components[0].lpx.buildId in body should be at least 1 chars long",
		},
		{
			name: "v1alpha1 LPX root template remains rejected when runtime validation is deferred",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				component := dgd.Spec.Services["lpx"]
				component.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{Image: "runtime"}}
				component.LPX.BuildID = " "
			}),
			wantWebhookErrs: []string{
				"spec.components[0].podTemplate: Forbidden: LPX Pod templates belong to roles",
			},
		},
		{
			name: "v1alpha1 LPX whitespace build ID validation is deferred to the child controller",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["lpx"].LPX.BuildID = " "
			}),
		},
		// Selected LPX workload shapes.
		{
			name: "LPX roles reject provider overrides",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Roles[0].ProviderOverride = groveProviderOverride("", `{"topologyConstraint":{"pack":{"required":"rack"}}}`)
			}),
			wantWebhookErrs: []string{"spec.components[0].roles[0].providerOverride: Forbidden: LPX roles do not support provider overrides"},
		},
		{
			name:          "LPX can remove an explicit conductor and retain its implicit fallback",
			oldDeployment: betaLPXDGDForAdmission(nil),
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Roles = dgd.Spec.Components[0].Roles[:1]
			}),
		},
		{
			name:          "LPX provider override support on update is deferred to the child controller",
			oldDeployment: betaLPXDGDForAdmission(nil),
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride("", `{"spec":{"template":{"topologyConstraint":{"pack":{"required":"rack"}}}}}`)
				dgd.Spec.Components[0].ProviderOverride = groveProviderOverride("", `{"topologyConstraint":{"pack":{"required":"rack"}}}`)
			}),
		},
		{
			name: "v1alpha1 LPX provider override support is deferred after conversion",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = alphaGroveProviderOverride("", `{"spec":{"template":{"topologyConstraint":{"pack":{"required":"rack"}}}}}`)
				dgd.Spec.Services["lpx"].ProviderOverride = alphaGroveProviderOverride("", `{"topologyConstraint":{"pack":{"required":"rack"}}}`)
			}),
		},
		{
			name: "LPX admits an ordinary component provider override",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				frontend := *betaDGDForAdmission(nil).GetComponentByName("frontend")
				frontend.ProviderOverride = groveProviderOverride("", `{"topologyConstraint":{"pack":{"required":"rack"}}}`)
				dgd.Spec.Components = append(dgd.Spec.Components, frontend)
			}),
		},
		{
			name: "selected v1beta1 LPX derives SpecDecode from two models",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationLPXSchedulerBackend: "ignored-scheduler",
					consts.KubeAnnotationLPXExecutionBackend: "ignored-execution",
				}
				dgd.Spec.Scheduling = &nvidiacomv1beta1.SchedulingSpec{AttemptDeadlineSeconds: k8sptr.To(int64(30))}
				component := &dgd.Spec.Components[0]
				component.Roles[0].PodTemplate.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{{Name: "model-storage", MountPath: "/nfs"}}
				component.Roles[0].PodTemplate.Spec.Volumes = []corev1.Volume{{Name: "model-storage", VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "model-storage"},
				}}}
				setBetaLPXSpecDec(dgd, k8sptr.To(int32(2)))
				dgd.Spec.Components[0].LPX.Settings = &apiextensionsv1.JSON{Raw: []byte(`{"draft_runtime":{"enabled":true}}`)}
				dgd.Spec.Components[1].LPX.Settings = &apiextensionsv1.JSON{Raw: []byte(`null`)}
			}),
		},
		{
			name: "selected v1alpha1 LPX derives SpecDecode from two models",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Scheduling = &nvidiacomv1beta1.SchedulingSpec{AttemptDeadlineSeconds: k8sptr.To(int64(30))}
				setAlphaLPXSpecDec(dgd, nil)
			}),
		},
		{
			name: "LPX Agent placement validation is deferred to the child controller",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaLPXSpecDec(dgd, nil)
				spec := &dgd.Spec.Components[0].Roles[0].PodTemplate.Spec
				spec.SchedulerName = "custom"
				spec.Hostname = "custom-host"
				spec.Subdomain = "custom-subdomain"
				spec.NodeName = "lpu-node-a"
				spec.NodeSelector = map[string]string{corev1.LabelHostname: "lpu-node-a"}
				spec.Affinity = &corev1.Affinity{PodAffinity: &corev1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{{
						TopologyKey: corev1.LabelHostname,
					}},
				}}
				spec.TopologySpreadConstraints = []corev1.TopologySpreadConstraint{{
					MaxSkew: 1, TopologyKey: corev1.LabelHostname, WhenUnsatisfiable: corev1.DoNotSchedule,
				}}
				spec.SchedulingGates = []corev1.PodSchedulingGate{{Name: "custom"}}
				spec.ResourceClaims = []corev1.PodResourceClaim{{Name: "device", ResourceClaimName: k8sptr.To("device-claim")}}
			}),
		},
		{
			name: "LPX SpecDecode draft replica support is deferred to the child controller",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaLPXSpecDec(dgd, k8sptr.To(int32(9)))
			}),
		},
		{
			name: "lpx requires a agent role",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Roles = nil
			}),
			wantWebhookErrs: []string{`spec.components[0].roles: Required value: must contain the "agent" role`},
		},
		{
			name: "v1alpha1 lpx requires an agent role",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["lpx"].Roles = nil
			}),
			wantWebhookErrs: []string{`spec.components[0].roles: Required value: must contain the "agent" role`},
		},
		{
			name: "v1alpha1 lpx requires lpx",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["lpx"].LPX = nil
			}),
			wantCELErr: "spec.services[lpx]: Invalid value: lpx is required when componentType is lpx",
		},
		{
			name: "v1alpha1 non-lpx service rejects lpx",
			deployment: alphaLPXDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["lpx"].ComponentType = consts.ComponentTypeWorker
				dgd.Spec.Services["lpx"].Roles = nil
			}),
			wantCELErr: "spec.services[lpx]: Invalid value: lpx may only be set when componentType is lpx",
		},
		{
			name: "LPX shared leader composition is deferred to the child controller",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				second := dgd.Spec.Components[0]
				second.ComponentName = "lpx-2"
				dgd.Spec.Components = append(dgd.Spec.Components, second)
			}),
		},
		{
			name: "shared LPX draft does not consume the target Grove name budget",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaLPXSpecDec(dgd, k8sptr.To(int32(2)))
				dgd.Spec.Components[0].ComponentName = longLPXComponentName
			}),
		},
		{
			name: "LPX PCSG name overflow validation is deferred to the child controller",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = longLPXComponentName
			}),
		},
		{
			name: "implicit LPX conductor name overflow validation is deferred to the child controller",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				component := &dgd.Spec.Components[0]
				component.ComponentName = longLPXComponentName
				component.Roles = component.Roles[:1]
			}),
		},
		// Ordinary components have no LPX child to reject controller-owned scheduler selection.
		{
			name: "ordinary graph rejects manual LPX scheduler selection",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate.Spec.SchedulerName = consts.LPXSchedulerBackend
			}),
			wantWebhookErrs: []string{"spec.components[1].podTemplate.spec.schedulerName: Forbidden: LPX schedulerName is controller-owned; declare an LPX component instead"},
		},
		{
			name: "mixed graph rejects manual LPX scheduler selection on an ordinary component",
			deployment: betaLPXDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := *betaWorkerComponent(betaDGDForAdmission(nil))
				worker.PodTemplate.Spec.SchedulerName = consts.LPXSchedulerBackend
				dgd.Spec.Components = append(dgd.Spec.Components, worker)
			}),
			wantWebhookErrs: []string{"spec.components[1].podTemplate.spec.schedulerName: Forbidden: LPX schedulerName is controller-owned; declare an LPX component instead"},
		},
		// An omitted type passes CEL; the LPX topology guard must reject the update.
		{
			name: "setting a previously unset component type to LPX is rejected by the webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components = dgd.Spec.Components[:1]
				dgd.Spec.Components[0].ComponentName = "lpx"
				dgd.Spec.Components[0].ComponentType = ""
			}),
			deployment:      betaLPXDGDForAdmission(nil),
			wantWebhookErrs: []string{`spec.components[0].type: Invalid value: "lpx": cannot change node topology between LPX and non-LPX after creation`},
		},
	}
}

func betaLPXDGDForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoGraphDeployment),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	dgd := betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		dgd.Spec.Components = []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			{
				ComponentName: "lpx",
				ComponentType: nvidiacomv1beta1.ComponentTypeLPX,
				Replicas:      k8sptr.To(int32(1)),
				LPX:           &nvidiacomv1beta1.LPXConfig{BuildID: "test/build"},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "lpu-runtime"}},
					}}},
					{Name: nvidiacomv1beta1.ComponentRoleLPXConductor},
				},
			},
		}
	})
	if mutate != nil {
		mutate(dgd)
	}
	return dgd
}

func alphaLPXDGDForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoGraphDeployment),
) *nvidiacomv1alpha1.DynamoGraphDeployment {
	dgd := alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
		dgd.Spec.Services = map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"lpx": {
				ComponentType: consts.ComponentTypeLPX,
				Replicas:      k8sptr.To(int32(1)),
				LPX:           &nvidiacomv1beta1.LPXConfig{BuildID: "test/build"},
				Roles: []nvidiacomv1alpha1.ComponentRoleSpec{
					{Name: nvidiacomv1alpha1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "lpu-runtime"}},
					}}},
					{Name: nvidiacomv1alpha1.ComponentRoleLPXConductor},
				},
			},
		}
	})
	if mutate != nil {
		mutate(dgd)
	}
	return dgd
}

// setBetaLPXSpecDec splits the fixture into independently authored model components.
func setBetaLPXSpecDec(dgd *nvidiacomv1beta1.DynamoGraphDeployment, draftReplicas *int32) {
	draft := dgd.Spec.Components[0].DeepCopy()
	target := dgd.Spec.Components[0].DeepCopy()
	draft.ComponentName, target.ComponentName = "draft", "target"
	draft.LPX.BuildID, target.LPX.BuildID = "test/draft", "test/target"
	draft.Roles = draft.Roles[:1]
	draft.Replicas = draftReplicas
	dgd.Spec.Components = []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{*draft, *target}
}

// setAlphaLPXSpecDec exercises the equivalent native alpha service representation.
func setAlphaLPXSpecDec(dgd *nvidiacomv1alpha1.DynamoGraphDeployment, draftReplicas *int32) {
	draft := dgd.Spec.Services["lpx"].DeepCopy()
	target := dgd.Spec.Services["lpx"].DeepCopy()
	draft.LPX.BuildID, target.LPX.BuildID = "test/draft", "test/target"
	draft.Roles = draft.Roles[:1]
	draft.Replicas = draftReplicas
	dgd.Spec.Services = map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{"draft": draft, "target": target}
}
