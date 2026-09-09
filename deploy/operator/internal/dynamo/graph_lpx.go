// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"crypto/sha256"
	"encoding/json"
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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// RenderLPXBasePodCliqueSet applies shared Dynamo defaults to the Grove envelope
// and independently owned LPX role templates. The LPX controller completes the
// compiled workload and its identity. All pointer inputs except secretsRetriever
// must be non-nil; preflight supplies the validated workload and child PCS name.
func RenderLPXBasePodCliqueSet(
	ctx context.Context,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	runtimeConfig *controller_common.RuntimeConfig,
	kubeClient client.Reader,
	secretsRetriever SecretsRetriever,
	selectedWorkload *dynamolpx.SelectedWorkload,
	pcsName string,
) (*grovev1alpha1.PodCliqueSet, *dynamolpx.RenderInput, error) {
	// Reuse the Grove envelope, but keep LPX compilation out of normal rendering.
	pcs, err := newGrovePodCliqueSet(dynamoDeployment, operatorConfig, runtimeConfig)
	if err != nil {
		return nil, nil, err
	}
	pcs.Name = pcsName
	queue, err := resolveGroveSchedulerQueue(ctx, dynamoDeployment.Annotations, runtimeConfig)
	if err != nil {
		return nil, nil, err
	}

	// Cyborg uses ordinary worker KV-transfer defaults, including the binding's domains.
	var topologyDomains []v1beta1.TopologyDomain
	if dynamoDeployment.Spec.Experimental != nil {
		topologyDomains, err = resolveGroveClusterTopologyDomains(ctx, kubeClient, dynamoDeployment.Spec.Experimental.KvTransferPolicy)
		if err != nil {
			return nil, nil, err
		}
	}

	// Preserve the complete source graph as the independent role-default context.
	component := dynamolpx.ServingComponent(dynamoDeployment)

	input, gpuCliques, err := renderLPXComponents(cliqueParams{
		component: component, componentName: component.ComponentName,
		dynamoDeployment: dynamoDeployment, operatorConfig: operatorConfig, runtimeConfig: runtimeConfig,
		secretsRetriever:            secretsRetriever,
		discoveryBackend:            controller_common.GetDiscoveryBackend(operatorConfig.Discovery.Backend, dynamoDeployment.Annotations),
		discoveryContext:            NewDiscoveryContext(operatorConfig.Discovery.Backend, dynamoDeployment.Annotations),
		validatedQueueName:          queue,
		groveClusterTopologyDomains: topologyDomains,
	}, selectedWorkload)
	if err != nil {
		return nil, nil, err
	}
	pcs.Spec.Template.Cliques = gpuCliques

	// Preserve scheduler-visible source metadata below the new owner boundary.
	// Explicit PCS/spec metadata retains precedence over inherited DGD metadata.
	labels, annotations := lpxSchedulingMetadata(dynamoDeployment.Labels), lpxSchedulingMetadata(dynamoDeployment.Annotations)
	maps.Copy(labels, pcs.Labels)
	maps.Copy(annotations, pcs.Annotations)
	pcs.Labels, pcs.Annotations = labels, annotations
	return pcs, input, nil
}

const (
	LPXDeploymentUIDAnnotation        = "lpx.nvidia.com/deployment-uid"
	LPXDeploymentGenerationAnnotation = "lpx.nvidia.com/deployment-generation"
	LPXInputRevisionAnnotation        = "lpx.nvidia.com/input-revision"
	LPXRestartAnnotation              = "lpx.nvidia.com/restart-id"
	LPXServingLabel                   = "lpx.nvidia.com/serving"
	lpxGPUExecutionRole               = "gpu"
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

// ValidateLPXSource rejects foreign, stale, or fabricated child handoffs before
// the LPX controller acquires a build or publishes a workload.
// Both objects must be non-nil.
func ValidateLPXSource(deployment *v1alpha1.LPXGraphDeployment, source *v1beta1.DynamoGraphDeployment) error {
	owner := metav1.GetControllerOf(deployment)
	if deployment.Namespace != source.Namespace || deployment.Name != source.Name ||
		owner == nil || owner.APIVersion != v1beta1.GroupVersion.String() || owner.Kind != "DynamoGraphDeployment" ||
		owner.Name != source.Name || owner.UID != source.UID {
		return fmt.Errorf("LPXGraphDeployment requires its exact source DGD and controller owner")
	}
	if provider := source.Annotations[commonconsts.KubeAnnotationWorkloadProvider]; provider != "" && provider != commonconsts.WorkloadProviderGrove {
		return fmt.Errorf("LPX requires the Grove workload provider")
	}
	if !source.DeletionTimestamp.IsZero() {
		return fmt.Errorf("source DGD is being deleted")
	}
	restart := LPXRestartToken(source, deployment.Annotations[LPXRestartAnnotation])
	revision, err := LPXInputRevision(source, restart)
	if err != nil {
		return err
	}
	if deployment.Spec.InputRevision != revision || deployment.Annotations[LPXRestartAnnotation] != restart {
		return fmt.Errorf("LPXGraphDeployment is waiting for the current source input revision")
	}
	return nil
}

// lpxInputRevisionPayload is the normalized source intent hashed for inputRevision.
// Keep complete selection and rendering inputs so nested role templates and build
// settings cannot silently fall out of the revision.
type lpxInputRevisionPayload struct {
	Components            []v1beta1.DynamoComponentDeploymentSharedSpec
	Scheduling            *v1beta1.SchedulingSpec
	Labels                map[string]string
	Annotations           map[string]string
	PropagatedAnnotations map[string]string
	Environment           []corev1.EnvVar `json:"Env"`
	PriorityClass         string
	BackendFramework      string
	RestartToken          string `json:"Restart"`
	KVTransferPolicy      *v1beta1.KvTransferPolicy
	TopologyConstraint    *v1beta1.SpecTopologyConstraint
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
// token selected by persisted DGD restart state. Operator configuration and external
// dependencies are not source revisions; the LPX controller observes them separately.
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
	for _, key := range append(slices.Clone(dgdPropagatedAnnotationKeys), commonconsts.KubeAnnotationLPXSchedulerBackend, commonconsts.KubeAnnotationWorkloadProvider,
		commonconsts.KubeAnnotationGroveUpdateStrategy, commonconsts.KubeAnnotationKaiSchedulerQueue, commonconsts.KubeAnnotationVolcanoQueue) {
		if value, exists := dgd.Annotations[key]; exists {
			annotations[key] = value
		}
	}
	var policy *v1beta1.KvTransferPolicy
	if dgd.Spec.Experimental != nil {
		policy = dgd.Spec.Experimental.KvTransferPolicy
	}
	input := lpxInputRevisionPayload{
		Components:            canonical,
		Scheduling:            dgd.Spec.Scheduling,
		Labels:                dgd.Spec.Labels,
		Annotations:           dgd.Spec.Annotations,
		PropagatedAnnotations: annotations,
		Environment:           dgd.Spec.Env,
		PriorityClass:         dgd.Spec.PriorityClassName,
		BackendFramework:      dgd.Spec.BackendFramework,
		RestartToken:          restart,
		KVTransferPolicy:      policy,
		TopologyConstraint:    dgd.Spec.TopologyConstraint,
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
		if strings.HasPrefix(key, "kai.scheduler/") || key == "priorityClassName" || key == "project" || key == "user" {
			selected[key] = value
		}
	}
	return selected
}

// PCSNameForLPX hashes the source identity within the Grove name budget,
// independently of ordinary components. The LPX planner checks materialized names.
func PCSNameForLPX(dgd *v1beta1.DynamoGraphDeployment) string {
	// The serving component owns all generated LPX names, including draft cliques.
	groupBudget, cliqueBudget := 8, 0
	if component := dynamolpx.ServingComponent(dgd); component != nil {
		groupBudget = max(groupBudget, len(component.ComponentName))
		cliqueBudget = max(cliqueBudget, longestLPXCliqueNameLength(component.ComponentName), len(component.ComponentName)+len("-engine-gpu"))
	}
	budget := commonconsts.MaxCombinedGroveResourceNameLength - groupBudget - cliqueBudget
	budget = max(budget, 8)
	name := strings.ReplaceAll(dgd.Name, ".", "-")
	digest := sha256.Sum256([]byte(dgd.Namespace + "/" + dgd.Name + "/" + string(dgd.UID)))
	if budget <= 13 {
		return fmt.Sprintf("%x", digest[:4])
	}
	return strings.TrimRight(name[:min(len(name), budget-13)], "-") + fmt.Sprintf("-%x-lpx", digest[:4])
}

func longestLPXCliqueNameLength(componentName string) int {
	// At most eight drafts and one target use a single-digit model index.
	return len(strings.ToLower(componentName)) + len(commonconsts.GroveRoleSuffixWorker) + 5
}

// LPXComponentNameBudget reserves the serving group's name and its longest clique.
// Shared draft components use the serving component's generated names.
func LPXComponentNameBudget(componentName string) int {
	return len(componentName) + longestLPXCliqueNameLength(componentName)
}

// renderLPXComponents merges ordinary Dynamo defaults independently into every
// authored role. The full source DGD supplies discovery and shared defaults;
// LPX component's roles are returned to the same PCS renderer.
// Preflight has validated the runtime shape and every Agent template.
func renderLPXComponents(p cliqueParams, workload *dynamolpx.SelectedWorkload) (*dynamolpx.RenderInput, []*grovev1alpha1.PodCliqueTemplateSpec, error) {
	// Pass naming and runtime inputs; deployment identity is stamped only on final resources.
	input := &dynamolpx.RenderInput{
		DGDName:       p.dynamoDeployment.Name,
		MinAvailable:  p.component.MinAvailable,
		Stages:        make(map[string]corev1.PodTemplateSpec),
		SSHSecretName: p.operatorConfig.MPI.SSHSecretName,
	}

	// Resolve preserved alpha metadata once for all independently rendered roles.
	var alphaComponents map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec
	if alpha := getDGDAlpha(p.dynamoDeployment); alpha != nil {
		alphaComponents = alpha.Spec.Services
	}

	gpuCliques := make([]*grovev1alpha1.PodCliqueTemplateSpec, 0, 1)
	for _, component := range dynamolpx.Components(p.dynamoDeployment) {
		alphaComponent := alphaComponents[component.ComponentName]
		agent := component.ComponentRole(v1beta1.ComponentRoleWorker)
		lpuRole := lpxRoleComponent(component, agent.PodTemplate, p.dynamoDeployment, p.discoveryBackend)
		lpuDefaults := &imageEntrypointComponentDefaults{ComponentDefaults: &BaseComponentDefaults{}}
		lpuTemplate, err := renderSelectedLPXRole(lpuRole, p.dynamoDeployment, alphaComponent, p.operatorConfig, p.secretsRetriever,
			p.discoveryContext, workload, lpuDefaults, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("rendering %s.agent: rendering selected LPX base pod: %w", component.ComponentName, err)
		}
		lpuTemplate.Labels[dynamolpx.StageLabel] = component.ComponentName
		lpuTemplate.Labels[dynamolpx.ExecutionRoleLabel] = "lpu"
		input.Stages[component.ComponentName] = *lpuTemplate
		conductor := component.ComponentRole(v1beta1.ComponentRoleLeader)
		if component != p.component {
			continue
		}

		// Non-hybrid conductors use their own template, or the Agent fallback in LPX lowering.
		if workload.Pipeline() != dynamolpx.PipelineLPX {
			if conductor != nil && conductor.PodTemplate != nil {
				role := lpxRoleComponent(component, conductor.PodTemplate, p.dynamoDeployment, p.discoveryBackend)
				input.Conductor, err = renderSelectedLPXRole(role, p.dynamoDeployment, alphaComponent, p.operatorConfig, p.secretsRetriever,
					p.discoveryContext, workload, lpuDefaults, nil)
				if err != nil {
					return nil, nil, fmt.Errorf("rendering %s.conductor: rendering selected LPX base pod: %w", component.ComponentName, err)
				}
				input.Conductor.Labels[dynamolpx.StageLabel] = component.ComponentName
				input.Conductor.Labels[dynamolpx.ExecutionRoleLabel] = "lpu"
			}
			continue
		}

		// Keep the existing hybrid execution path; only its authored template location changes.
		template, replicas := agent.PodTemplate, int32(1)
		if conductor != nil {
			replicas = ptr.Deref(conductor.Replicas, 1)
			if conductor.PodTemplate != nil {
				template = conductor.PodTemplate
			}
		}
		role := lpxRoleComponent(component, template, p.dynamoDeployment, p.discoveryBackend)
		role.ComponentType = v1beta1.ComponentTypeDecode
		role.Replicas = ptr.To(replicas)
		role.MinAvailable = nil
		defaults := ComponentDefaultsFactory(string(v1beta1.ComponentTypeDecode))
		if workload.BuildFamily() == dynamolpx.BuildFamilyXT {
			defaults = &selectedCyborgComponentDefaults{
				ComponentDefaults: defaults, workload: workload, dgdName: p.dynamoDeployment.Name,
				replicas: *role.Replicas, lpxPodSpec: lpuTemplate.Spec,
			}
		} else {
			defaults = &imageEntrypointComponentDefaults{ComponentDefaults: defaults}
		}
		gpu := p
		gpu.component = role
		gpu.componentName = component.ComponentName
		gpu.r = ServiceRole{Name: workload.CyborgTemplateName(), Role: RoleMain, Replicas: *role.Replicas}
		gpuTemplate, err := renderSelectedLPXRole(role, p.dynamoDeployment, alphaComponent, p.operatorConfig, p.secretsRetriever,
			p.discoveryContext, workload, defaults, p.groveClusterTopologyDomains)
		if err != nil {
			return nil, nil, fmt.Errorf("rendering %s.conductor: failed to generate podSpec for role %s: %w", component.ComponentName, gpu.r.Name, err)
		}
		clique, err := buildCliqueFromTemplate(gpu, *gpuTemplate)
		if err != nil {
			return nil, nil, fmt.Errorf("rendering %s.conductor: %w", component.ComponentName, err)
		}
		// Every GPU worker is required by the same compiled engine replica.
		clique.Spec.MinAvailable = ptr.To(clique.Spec.Replicas)
		clique.Labels[commonconsts.KubeLabelDynamoComponentType] = string(v1beta1.ComponentTypeLPX)
		clique.Labels[dynamolpx.StageLabel] = component.ComponentName
		clique.Labels[dynamolpx.ExecutionRoleLabel] = lpxGPUExecutionRole
		gpuCliques = append(gpuCliques, clique)
	}
	return input, gpuCliques, nil
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

// imageEntrypointComponentDefaults preserves all role-specific defaults while
// leaving the executable to the workload image unless the user supplied one.
type imageEntrypointComponentDefaults struct {
	ComponentDefaults
}

func (d *imageEntrypointComponentDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	container, err := d.ComponentDefaults.GetBaseContainer(context)
	if err != nil {
		return corev1.Container{}, err
	}
	container.Command = nil
	return container, nil
}

// selectedCyborgComponentDefaults installs LPX-owned Cyborg bindings before
// the user's PodTemplate is merged, preserving the normal merge strategy.
type selectedCyborgComponentDefaults struct {
	ComponentDefaults
	workload   *dynamolpx.SelectedWorkload
	dgdName    string
	replicas   int32
	lpxPodSpec corev1.PodSpec
}

func (d *selectedCyborgComponentDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	container, err := d.ComponentDefaults.GetBaseContainer(context)
	if err != nil {
		return corev1.Container{}, err
	}
	container.Command = nil
	if err := dynamolpx.ApplySelectedCyborgContainerDefaults(
		&container,
		d.workload,
		d.dgdName,
		d.replicas,
		d.lpxPodSpec,
	); err != nil {
		return corev1.Container{}, err
	}
	return container, nil
}

func (d *selectedCyborgComponentDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	podSpec, err := d.ComponentDefaults.GetBasePodSpec(context)
	if err != nil {
		return corev1.PodSpec{}, err
	}
	dynamolpx.ApplySelectedCyborgPodDefaults(&podSpec, d.dgdName)
	return podSpec, nil
}

// renderSelectedLPXRole consumes a private component copy; other inputs are read-only.
// alphaComponent may be nil; topologyDomains may be nil when no topology is consumed.
func renderSelectedLPXRole(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	dgd *v1beta1.DynamoGraphDeployment,
	alphaComponent *v1alpha1.DynamoComponentDeploymentSharedSpec,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	secretsRetriever SecretsRetriever,
	discoveryContext DiscoveryContext,
	selectedWorkload *dynamolpx.SelectedWorkload,
	defaults ComponentDefaults,
	topologyDomains []v1beta1.TopologyDomain,
) (*corev1.PodTemplateSpec, error) {
	componentName := component.ComponentName
	// Capture authored precedence before PodSpec defaults fill the role's metadata.
	metadata := generatePodMetadata(component, dgd, alphaComponent, componentName, discoveryContext)
	applyDGDTemplateDefaults(component, dgd, topologyDomains)
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
	if component.IsLPX() && selectedWorkload.BuildFamily() == dynamolpx.BuildFamilyXT {
		explicitVolumes := component.PodTemplate.Spec.Volumes
		runtimeVolumes := []string{"config"}
		if selectedWorkload.Pipeline() != dynamolpx.PipelineLPX {
			runtimeVolumes = append(runtimeVolumes, "tmp", "hugepages", "host-dev", "host-sys")
		}
		basePodSpec.Volumes = slices.DeleteFunc(basePodSpec.Volumes, func(volume corev1.Volume) bool {
			return slices.Contains(runtimeVolumes, volume.Name) && !slices.ContainsFunc(explicitVolumes,
				func(explicit corev1.Volume) bool { return explicit.Name == volume.Name })
		})
	}
	for _, annotationKey := range commonconsts.KubeTopologySourceAnnotationKeys() {
		delete(metadata.Annotations, annotationKey)
	}
	return &corev1.PodTemplateSpec{
		ObjectMeta: metadata,
		Spec:       *basePodSpec,
	}, nil
}
