// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// RenderLPXPodCliqueSet constructs the shared LPX Grove envelope without workload
// templates. Its labels include the resolved scheduler queue.
// Pointer inputs must be non-nil and are not mutated.
func RenderLPXPodCliqueSet(
	ctx context.Context,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	runtimeConfig *controller_common.RuntimeConfig,
	pcsName string,
) (*grovev1alpha1.PodCliqueSet, error) {
	// Reuse the ordinary Grove defaults once for the complete LPX graph.
	pcs, err := newGrovePodCliqueSet(dynamoDeployment, operatorConfig, runtimeConfig)
	if err != nil {
		return nil, err
	}

	pcs.Name = pcsName
	pcs.Spec.Template.StartupType = ptr.To(grovev1alpha1.CliqueStartupTypeExplicit)

	queue, err := resolveGroveSchedulerQueue(ctx, dynamoDeployment.Annotations, runtimeConfig)
	if err != nil {
		return nil, err
	}

	// Publish the validated queue for the LPX backend's KAI fallback.
	if queue != "" {
		pcs.Labels[commonconsts.KubeLabelKaiSchedulerQueue] = queue
	}

	// Explicit PCS/spec metadata takes precedence over inherited scheduler metadata.
	labels, annotations := lpxSchedulingMetadata(dynamoDeployment.Labels), lpxSchedulingMetadata(dynamoDeployment.Annotations)
	maps.Copy(labels, pcs.Labels)
	maps.Copy(annotations, pcs.Annotations)
	pcs.Labels, pcs.Annotations = labels, annotations
	return pcs, nil
}

// RenderLPXWorkloadTemplates renders one workload's roles, scaling group and resources.
// Pointer inputs must be non-nil, except secretsRetriever when no secrets are used.
// Inputs are not mutated.
func RenderLPXWorkloadTemplates(
	source *v1beta1.DynamoGraphDeployment,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	runtimeConfig *controller_common.RuntimeConfig,
	secretsRetriever SecretsRetriever,
	workload *dynamolpx.Workload,
	plan *dynamolpx.MaterializationPlan,
) (*dynamolpx.WorkloadTemplates, error) {
	// Apply Dynamo defaults independently to the selected workload's authored roles.
	component := source.GetComponentByName(workload.ServingComponentName())
	input, err := renderLPXComponents(cliqueParams{
		component: component, componentName: component.ComponentName,
		dynamoDeployment: source, operatorConfig: operatorConfig, runtimeConfig: runtimeConfig,
		secretsRetriever: secretsRetriever,
		discoveryBackend: controller_common.GetDiscoveryBackend(operatorConfig.Discovery.Backend, source.Annotations),
		discoveryContext: NewDiscoveryContext(operatorConfig.Discovery.Backend, source.Annotations),
	}, workload, plan)
	if err != nil {
		return nil, err
	}
	rendered, err := dynamolpx.RenderNodeLocal(workload, plan, *input)
	if err != nil {
		return nil, err
	}

	// Scope the rendered resources to the graph's Kubernetes namespace.
	for _, resource := range rendered.Resources {
		resource.SetNamespace(source.Namespace)
	}
	return rendered, nil
}

const (
	LPXDeploymentUIDAnnotation = dynamolpx.DeploymentUIDAnnotation
	LPXRestartAnnotation       = "lpx.nvidia.com/restart-id"
	LPXServingLabel            = "lpx.nvidia.com/serving"
)

// LPXRestartToken advances only from the DGD's persisted restart selection.
// Preserve the last delivered token while another component restarts.
func LPXRestartToken(source *v1beta1.DynamoGraphDeployment, previous string) string {
	request, observed := source.Spec.Restart, source.Status.Restart
	if request == nil || request.ID == "" || observed == nil || observed.ObservedID != request.ID {
		return previous
	}
	state := DetermineRestartState(source, observed)
	for index := range source.Spec.Components {
		component := &source.Spec.Components[index]
		if component.IsLPX() && state.ShouldAnnotateComponent(component.ComponentName) {
			return state.Timestamp
		}
	}
	return previous
}

// ErrLPXSourcePending means the source and child snapshots await a matching handoff.
var ErrLPXSourcePending = errors.New("LPXGraphDeployment is waiting for the current source input revision")

// lpxInputRevisionPayload is the normalized source intent hashed for inputRevision.
// Keep complete selection and rendering inputs so nested role templates and build
// settings cannot silently fall out of the revision.
type lpxInputRevisionPayload struct {
	Components            []v1beta1.DynamoComponentDeploymentSharedSpec
	Labels                map[string]string
	Annotations           map[string]string
	PropagatedAnnotations map[string]string
	Environment           []corev1.EnvVar `json:"Env"`
	PriorityClass         string
	BackendFramework      string
	TopologyConstraint    *v1beta1.SpecTopologyConstraint
	RestartToken          string `json:"Restart"`
	ProviderOverride      *v1beta1.ProviderOverride
	SchedulingLabels      map[string]string            `json:",omitempty"`
	EPPEnabled            bool                         `json:",omitempty"`
	AlphaLabels           map[string]map[string]string `json:",omitempty"`
	AlphaAnnotations      map[string]map[string]string `json:",omitempty"`
	AlphaSubComponentType map[string]string            `json:",omitempty"`
}

// LPXInputRevision hashes all LPX components and their shared render inputs.
// Source identity is checked separately by ValidateLPXSource. Ordinary component payloads,
// DGD bookkeeping and raw restart requests are excluded: restart is the effective
// token selected by persisted DGD restart state. Operator configuration is not a
// source revision; rejected topology inputs remain included because they change
// preflight results.
func LPXInputRevision(dgd *v1beta1.DynamoGraphDeployment, restart string) (string, error) {
	components := dynamolpx.Components(dgd)
	if len(components) == 0 {
		return "", fmt.Errorf("LPX component is required")
	}

	// Canonicalize authored lists on shallow copies; nested templates remain read-only.
	canonical := make([]v1beta1.DynamoComponentDeploymentSharedSpec, len(components))
	for index, component := range components {
		canonical[index] = *component
		canonical[index].Roles = slices.Clone(component.Roles)
		slices.SortFunc(canonical[index].Roles, func(a, b v1beta1.ComponentRoleSpec) int {
			return strings.Compare(a.Name, b.Name)
		})
	}
	slices.SortFunc(canonical, func(a, b v1beta1.DynamoComponentDeploymentSharedSpec) int {
		return strings.Compare(a.ComponentName, b.ComponentName)
	})

	annotations := lpxSchedulingMetadata(dgd.Annotations)
	for _, key := range append(slices.Clone(dgdPropagatedAnnotationKeys), commonconsts.KubeAnnotationWorkloadProvider,
		commonconsts.KubeAnnotationGroveUpdateStrategy, commonconsts.KubeAnnotationKaiSchedulerQueue, commonconsts.KubeAnnotationVolcanoQueue) {
		if value, exists := dgd.Annotations[key]; exists {
			annotations[key] = value
		}
	}
	input := lpxInputRevisionPayload{
		Components:            canonical,
		Labels:                dgd.Spec.Labels,
		Annotations:           dgd.Spec.Annotations,
		PropagatedAnnotations: annotations,
		Environment:           dgd.Spec.Env,
		PriorityClass:         dgd.Spec.PriorityClassName,
		BackendFramework:      dgd.Spec.BackendFramework,
		TopologyConstraint:    dgd.Spec.TopologyConstraint,
		RestartToken:          restart,
		ProviderOverride:      dgd.Spec.ProviderOverride,
		SchedulingLabels:      lpxSchedulingMetadata(dgd.Labels),
		EPPEnabled:            dgd.HasEPPComponent(),
		AlphaLabels:           make(map[string]map[string]string),
		AlphaAnnotations:      make(map[string]map[string]string),
		AlphaSubComponentType: make(map[string]string),
	}
	// The shared role renderer also reads preserved alpha component metadata.
	// Use the same conversion reader, excluding ordinary component payloads and
	// keeping conversion bookkeeping out of LPX's input contract.
	if alpha := getDGDAlpha(dgd); alpha != nil {
		for _, component := range components {
			if source := alpha.Spec.Services[component.ComponentName]; source != nil {
				if len(source.Labels) != 0 {
					input.AlphaLabels[component.ComponentName] = source.Labels
				}
				if len(source.Annotations) != 0 {
					input.AlphaAnnotations[component.ComponentName] = source.Annotations
				}
				if source.SubComponentType != "" {
					input.AlphaSubComponentType[component.ComponentName] = source.SubComponentType
				}
			}
		}
	}
	data, err := json.Marshal(input)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("sha256:%x", sha256.Sum256(data)), nil
}

// lpxSchedulingMetadata preserves the scheduler inputs previously inherited
// through the DGD owner. With existing scheduler RBAC, ownership traversal now
// stops at the PCS; do not include unrelated controller bookkeeping in revisions.
func lpxSchedulingMetadata(metadata map[string]string) map[string]string {
	selected := make(map[string]string)
	for key, value := range metadata {
		if strings.HasPrefix(key, "kai.scheduler/") || key == "priorityClassName" || key == "project" || key == "user" || key == commonconsts.GroveAnnotationVolcanoQueue {
			selected[key] = value
		}
	}
	return selected
}

// PCSNameForLPX keeps the owning LGD's name readable without depending on mutable components.
// The DGD handoff gives the LGD the same name as its source deployment.
// deployment must be non-nil and have a valid Kubernetes name.
func PCSNameForLPX(deployment *v1alpha1.LPXGraphDeployment) string {
	// Reserve room for Grove's group, role, replica indexes, and Pod suffixes.
	const maxPrefixLength = dynamolpx.MaxPodCliqueSetNameLength - 1 - 4
	prefix := strings.ReplaceAll(deployment.Name, ".", "-")
	if prefix[0] >= '0' && prefix[0] <= '9' {
		prefix = "lpx-" + prefix
	}
	prefix = strings.TrimRight(prefix[:min(len(prefix), maxPrefixLength)], "-")

	// Hash the original identity so truncation, dot normalization, and replacement stay distinct.
	digest := sha256.Sum256([]byte(deployment.Namespace + "/" + deployment.Name + "/" + string(deployment.UID)))
	return prefix + "-" + hex.EncodeToString(digest[:2])
}

// renderLPXComponents merges ordinary Dynamo defaults independently into every
// authored role. The full source DGD supplies discovery and shared defaults;
// LPX component's roles are returned to the same PCS renderer.
// Preflight supplies the non-nil validated workload and materialization plan.
func renderLPXComponents(p cliqueParams, workload *dynamolpx.Workload, plan *dynamolpx.MaterializationPlan) (*dynamolpx.RenderInput, error) {
	// Pass runtime inputs; deployment identity is stamped only on final resources.
	input := &dynamolpx.RenderInput{
		MinAvailable: p.component.MinAvailable,
		Stages:       make(map[string]corev1.PodTemplateSpec),
	}

	// Resolve preserved alpha metadata once for all independently rendered roles.
	var alphaComponents map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec
	if alpha := getDGDAlpha(p.dynamoDeployment); alpha != nil {
		alphaComponents = alpha.Spec.Services
	}

	// Render only members of this workload; other graph components supply shared context.
	for _, name := range workload.ComponentNames() {
		component := p.dynamoDeployment.GetComponentByName(name)
		alphaComponent := alphaComponents[component.ComponentName]
		agent := component.ComponentRole(v1beta1.ComponentRoleLPXAgent)
		lpuRole := lpxRoleComponent(component, agent.PodTemplate, p.dynamoDeployment, p.discoveryBackend)
		lpuDefaults := &podTemplateRuntimeDefaults{ComponentDefaults: &BaseComponentDefaults{}}
		lpuTemplate, err := renderSelectedLPXRole(lpuRole, p.dynamoDeployment, alphaComponent, p.operatorConfig, p.secretsRetriever,
			p.discoveryContext, lpuDefaults)
		if err != nil {
			return nil, fmt.Errorf("rendering %s.agent: rendering selected LPX base pod: %w", component.ComponentName, err)
		}
		lpuTemplate.Labels[dynamolpx.StageLabel] = component.ComponentName
		input.Stages[component.ComponentName] = *lpuTemplate
		conductor := component.ComponentRole(v1beta1.ComponentRoleLPXConductor)
		if component != p.component {
			continue
		}

		// The serving role owns its startup independently of the Agent template.
		if workload.Pipeline() != dynamolpx.PipelineLPX {
			role := lpxRoleComponent(component, conductor.PodTemplate, p.dynamoDeployment, p.discoveryBackend)
			input.Conductor, err = renderSelectedLPXRole(role, p.dynamoDeployment, alphaComponent, p.operatorConfig, p.secretsRetriever,
				p.discoveryContext, lpuDefaults)
			if err != nil {
				return nil, fmt.Errorf("rendering %s.conductor: rendering selected LPX base pod: %w", component.ComponentName, err)
			}
			input.Conductor.Labels[dynamolpx.StageLabel] = component.ComponentName
			continue
		}

		// Omitted hybrid capacity starts with one complete compiled client group.
		minimumReplicas, err := workload.MinimumCyborgReplicas()
		if err != nil {
			return nil, err
		}
		template, replicas := conductor.PodTemplate, ptr.Deref(conductor.Replicas, minimumReplicas)
		role := lpxRoleComponent(component, template, p.dynamoDeployment, p.discoveryBackend)
		role.ComponentType = v1beta1.ComponentTypeDecode
		role.Replicas = ptr.To(replicas)
		role.MinAvailable = nil
		defaults := &podTemplateRuntimeDefaults{ComponentDefaults: NewWorkerDefaults()}
		if workload.BuildFamily() == dynamolpx.BuildFamilyXT {
			input.CyborgConfigMap, err = workload.RenderCyborgConfigMap(plan)
			if err != nil {
				return nil, err
			}
		}
		gpu := p
		gpu.component = role
		gpu.componentName = component.ComponentName
		gpu.r = ServiceRole{Name: plan.CyborgTemplate, Role: RoleMain, Replicas: *role.Replicas}
		gpuTemplate, err := renderSelectedLPXRole(role, p.dynamoDeployment, alphaComponent, p.operatorConfig, p.secretsRetriever,
			p.discoveryContext, defaults)
		if err != nil {
			return nil, fmt.Errorf("rendering %s.conductor: failed to generate podSpec for role %s: %w", component.ComponentName, gpu.r.Name, err)
		}

		// Select LPX before applying Grove defaults; the PCS already owns the KAI queue.
		gpuTemplate.Spec.SchedulerName = v1alpha1.LPXSchedulerName
		clique, err := buildCliqueFromTemplate(gpu, *gpuTemplate)
		if err != nil {
			return nil, fmt.Errorf("rendering %s.conductor: %w", component.ComponentName, err)
		}

		clique.Labels[commonconsts.KubeLabelDynamoComponentType] = string(v1beta1.ComponentTypeLPX)
		clique.Labels[dynamolpx.StageLabel] = component.ComponentName
		input.Cyborg = clique
	}
	return input, nil
}

func lpxRoleComponent(source *v1beta1.DynamoComponentDeploymentSharedSpec, template *corev1.PodTemplateSpec, dgd *v1beta1.DynamoGraphDeployment, backend configv1alpha1.DiscoveryBackend) *v1beta1.DynamoComponentDeploymentSharedSpec {
	// Copy shared defaults and this role, without unrelated compiler inputs.
	seed := *source
	seed.LPX = nil
	seed.Roles = nil
	seed.PodTemplate = template
	role := seed.DeepCopy()

	// Bind source metadata on the independently owned role template.
	propagateDGDAnnotations(dgd.Annotations, role)
	role.PodTemplate.Labels[commonconsts.KubeLabelDynamoNamespace] = GetDynamoNamespace(dgd, source)
	if backend != "" {
		role.PodTemplate.Annotations[commonconsts.KubeAnnotationDynamoDiscoveryBackend] = string(backend)
	}
	return role
}

// podTemplateRuntimeDefaults leaves startup and health checks to the authored role
// while retaining shared infrastructure bindings.
type podTemplateRuntimeDefaults struct {
	ComponentDefaults
}

func (d *podTemplateRuntimeDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	container, err := d.ComponentDefaults.GetBaseContainer(context)
	if err != nil {
		return corev1.Container{}, err
	}

	// Omitted template settings must not inherit another runtime's launch or health contract.
	container.Command = nil
	container.StartupProbe = nil
	container.LivenessProbe = nil
	container.ReadinessProbe = nil

	// Bind the pod address for LPX runtimes before authored environment overrides.
	container.Env = append(container.Env, corev1.EnvVar{
		Name: commonconsts.PodIPEnvVar,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"},
		},
	})
	return container, nil
}

// renderSelectedLPXRole consumes a private component copy; other inputs are read-only.
// component and its PodTemplate must be non-nil.
// alphaComponent may be nil.
func renderSelectedLPXRole(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	dgd *v1beta1.DynamoGraphDeployment,
	alphaComponent *v1alpha1.DynamoComponentDeploymentSharedSpec,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	secretsRetriever SecretsRetriever,
	discoveryContext DiscoveryContext,
	defaults ComponentDefaults,
) (*corev1.PodTemplateSpec, error) {
	componentName := component.ComponentName
	// Capture authored precedence before PodSpec defaults fill the role's metadata.
	metadata := generatePodMetadata(component, dgd, alphaComponent, componentName, discoveryContext)
	applyDGDTemplateDefaults(component, dgd, nil)
	basePodSpec, err := generateBasePodSpecWithDefaults(
		component,
		BackendFrameworkNoop,
		secretsRetriever,
		dgd.Name,
		dgd.Namespace,
		RoleMain,
		1,
		operatorConfig,
		commonconsts.MultinodeDeploymentTypeGrove,
		componentName,
		nil,
		defaults,
		func() (int64, error) { return 0, nil },
	)
	if err != nil {
		return nil, err
	}

	// The shared renderer's filesystem-group fallback is not part of the LPX template contract.
	basePodSpec.SecurityContext = component.PodTemplate.Spec.SecurityContext.DeepCopy()

	// LPX supplies the generated config volume after merging; discard only inferred PVCs.
	explicitVolumes := component.PodTemplate.Spec.Volumes
	basePodSpec.Volumes = slices.DeleteFunc(basePodSpec.Volumes, func(volume corev1.Volume) bool {
		return volume.Name == "config" && !slices.ContainsFunc(explicitVolumes,
			func(explicit corev1.Volume) bool { return explicit.Name == volume.Name })
	})
	for _, annotationKey := range commonconsts.KubeTopologySourceAnnotationKeys() {
		delete(metadata.Annotations, annotationKey)
	}
	return &corev1.PodTemplateSpec{
		ObjectMeta: metadata,
		Spec:       *basePodSpec,
	}, nil
}
