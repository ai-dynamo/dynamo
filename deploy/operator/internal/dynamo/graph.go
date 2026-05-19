/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"encoding/json"
	"fmt"
	"hash/fnv"
	"maps"
	"regexp"
	"sort"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/imdario/mergo"
	"google.golang.org/protobuf/types/known/wrapperspb"
	istioNetworking "istio.io/api/networking/v1beta1"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

// RestartState holds the restart state for DGD components.
type RestartState struct {
	// Timestamp is the restart timestamp to apply as the annotation value.
	// Format: RFC3339
	Timestamp string
	// ComponentsToAnnotate is the set of component names that should have the restart annotation.
	ComponentsToAnnotate map[string]bool
}

// ShouldAnnotateComponent returns true if the given component should have a restart annotation.
func (s *RestartState) ShouldAnnotateComponent(componentName string) bool {
	if s == nil || s.ComponentsToAnnotate == nil {
		return false
	}
	return s.ComponentsToAnnotate[componentName]
}

// DetermineRestartState computes the restart state for DGD components.
func DetermineRestartState(dgd *v1beta1.DynamoGraphDeployment, restartStatus *v1beta1.RestartStatus) *RestartState {
	if restartStatus == nil {
		return nil
	}

	if dgd.Spec.Restart == nil || dgd.Spec.Restart.ID == "" {
		// Check if there's a completed restart we need to preserve
		if restartStatus.ObservedID != "" {
			return &RestartState{
				Timestamp:            restartStatus.ObservedID,
				ComponentsToAnnotate: getAllComponentNames(dgd),
			}
		}
		return nil
	}

	specID := dgd.Spec.Restart.ID

	isNewRestart := restartStatus.ObservedID == "" ||
		dgd.Spec.Restart.ID != restartStatus.ObservedID

	if !isNewRestart && restartStatus.Phase == v1beta1.RestartPhaseSuperseded {
		// Superseded: don't push any new annotations. Existing annotations
		// are preserved via the existingRestartAnnotations fallback path.
		return nil
	}

	if !isNewRestart && restartStatus.Phase == v1beta1.RestartPhaseCompleted {
		return &RestartState{
			Timestamp:            specID,
			ComponentsToAnnotate: getAllComponentNames(dgd),
		}
	}

	if IsParallelRestart(dgd) {
		return &RestartState{
			Timestamp:            specID,
			ComponentsToAnnotate: getAllComponentNames(dgd),
		}
	}

	// Sequential restart (default or specified)
	return &RestartState{
		Timestamp:            specID,
		ComponentsToAnnotate: getComponentsToAnnotateForSequentialRestart(dgd, restartStatus),
	}
}

// getAllComponentNames returns a map of all component names in the DGD.
func getAllComponentNames(dgd *v1beta1.DynamoGraphDeployment) map[string]bool {
	components := make(map[string]bool, len(dgd.Spec.Components))
	for i := range dgd.Spec.Components {
		components[dgd.Spec.Components[i].ComponentName] = true
	}
	return components
}

// IsParallelRestart returns true if the restart strategy is parallel.
func IsParallelRestart(dgd *v1beta1.DynamoGraphDeployment) bool {
	if dgd.Spec.Restart == nil || dgd.Spec.Restart.Strategy == nil {
		return false // Default is sequential
	}
	return dgd.Spec.Restart.Strategy.Type == v1beta1.RestartStrategyTypeParallel
}

// getComponentsToAnnotateForSequentialRestart determines which components should be annotated
// for a sequential restart in progress.
func getComponentsToAnnotateForSequentialRestart(dgd *v1beta1.DynamoGraphDeployment, status *v1beta1.RestartStatus) map[string]bool {
	components := make(map[string]bool)

	order := GetRestartOrder(dgd)
	if len(order) == 0 {
		return components
	}

	// New restart or Pending phase - only first component needs to be annotated.
	if status == nil ||
		status.Phase == v1beta1.RestartPhasePending ||
		len(status.InProgress) == 0 {
		components[order[0]] = true
		return components
	}

	// Find the max index among in-progress components.
	inProgress := make(map[string]bool)
	for _, componentName := range status.InProgress {
		inProgress[componentName] = true
	}

	maxIndex := -1
	for i, componentName := range order {
		if inProgress[componentName] {
			if i > maxIndex {
				maxIndex = i
			}
		}
	}

	// Add all components up to and including maxIndex.
	// Components before the in-progress one have completed and need their annotation preserved.
	if maxIndex >= 0 {
		for i := 0; i <= maxIndex; i++ {
			components[order[i]] = true
		}
	}

	return components
}

// GetRestartOrder returns the order of components for sequential restart.
// If not specified, returns a deterministic alphabetical order.
func GetRestartOrder(dgd *v1beta1.DynamoGraphDeployment) []string {
	if dgd.Spec.Restart != nil && dgd.Spec.Restart.Strategy != nil && len(dgd.Spec.Restart.Strategy.Order) > 0 {
		return dgd.Spec.Restart.Strategy.Order
	}

	order := make([]string, 0, len(dgd.Spec.Components))
	for i := range dgd.Spec.Components {
		order = append(order, dgd.Spec.Components[i].ComponentName)
	}
	sort.Strings(order)
	return order
}

// ServiceConfig represents the YAML configuration structure for a service
type DynamoConfig struct {
	Enabled       bool   `yaml:"enabled"`
	Namespace     string `yaml:"namespace"`
	Name          string `yaml:"name"`
	ComponentType string `yaml:"component_type,omitempty"`
}

type Traffic struct {
	Timeout int `yaml:"timeout"`
}

type Autoscaling struct {
	MinReplicas int `yaml:"min_replicas"`
	MaxReplicas int `yaml:"max_replicas"`
}

type Config struct {
	Dynamo       *DynamoConfig   `yaml:"dynamo,omitempty"`
	Resources    *Resources      `yaml:"resources,omitempty"`
	Traffic      *Traffic        `yaml:"traffic,omitempty"`
	Autoscaling  *Autoscaling    `yaml:"autoscaling,omitempty"`
	HttpExposed  bool            `yaml:"http_exposed,omitempty"`
	ApiEndpoints []string        `yaml:"api_endpoints,omitempty"`
	Workers      *int32          `yaml:"workers,omitempty"`
	TotalGpus    *int32          `yaml:"total_gpus,omitempty"`
	ExtraPodSpec *corev1.PodSpec `yaml:"extraPodSpec,omitempty"`
}

type ServiceConfig struct {
	Name         string              `yaml:"name"`
	Dependencies []map[string]string `yaml:"dependencies,omitempty"`
	Config       Config              `yaml:"config"`
}

type Resources struct {
	CPU    *string           `yaml:"cpu,omitempty" json:"cpu,omitempty"`
	Memory *string           `yaml:"memory,omitempty" json:"memory,omitempty"`
	GPU    *string           `yaml:"gpu,omitempty" json:"gpu,omitempty"`
	Custom map[string]string `yaml:"custom,omitempty" json:"custom,omitempty"`
}

type DynDeploymentConfig = map[string]*DynDeploymentServiceConfig

// ServiceConfig represents the configuration for a specific service
type DynDeploymentServiceConfig struct {
	ServiceArgs *ServiceArgs `json:"ServiceArgs,omitempty"`
}

// ServiceArgs represents the arguments that can be passed to any service
type ServiceArgs struct {
	Workers   *int32     `json:"workers,omitempty"`
	Resources *Resources `json:"resources,omitempty"`
}

func (s ServiceConfig) GetNamespace() *string {
	if s.Config.Dynamo == nil || s.Config.Dynamo.Namespace == "" {
		return nil
	}
	return &s.Config.Dynamo.Namespace
}

func ParseDynDeploymentConfig(jsonContent []byte) (DynDeploymentConfig, error) {
	var config DynDeploymentConfig
	err := json.Unmarshal(jsonContent, &config)
	return config, err
}

func (r RollingUpdateContext) InProgress() bool {
	return len(r.OldWorkerReplicas) > 0
}

// RollingUpdateContext provides information about an in-progress rolling update.
type RollingUpdateContext struct {
	// NewWorkerHash is the short hash (8 chars) for the new worker spec, used for DCD naming
	NewWorkerHash string
	// OldWorkerReplicas maps service name to the desired replica count for old workers.
	// Used by the controller to patch old worker DCDs directly.
	OldWorkerReplicas map[string]int32
	// NewWorkerReplicas maps service name to the desired replica count for new workers.
	NewWorkerReplicas map[string]int32
}

// GenerateDynamoComponentsDeployments generates a map of DynamoComponentDeployments from a DynamoGraphConfig.
// The map key is the component name.
func GenerateDynamoComponentsDeployments(
	parentDGD *v1beta1.DynamoGraphDeployment,
	restartState *RestartState,
	existingRestartAnnotations map[string]string,
	rollingUpdateCtx RollingUpdateContext,
) (map[string]*v1beta1.DynamoComponentDeployment, error) {
	deployments := make(map[string]*v1beta1.DynamoComponentDeployment)
	backendFramework, err := backendFrameworkForGeneratedDCDs(parentDGD)
	if err != nil {
		return nil, err
	}

	// Generate DCDs for each component.
	for i := range parentDGD.Spec.Components {
		component := &parentDGD.Spec.Components[i]
		componentName := component.ComponentName
		dynamoNamespace := parentDGD.GetDynamoNamespaceForComponent(component)
		dcd, err := generateSingleDCD(parentDGD, componentName, component, dynamoNamespace, backendFramework, restartState, existingRestartAnnotations, rollingUpdateCtx)
		if err != nil {
			return nil, err
		}
		deployments[componentName] = dcd
	}

	return deployments, nil
}

func backendFrameworkForGeneratedDCDs(parentDGD *v1beta1.DynamoGraphDeployment) (string, error) {
	if parentDGD.Spec.BackendFramework != "" {
		return parentDGD.Spec.BackendFramework, nil
	}

	var detected BackendFramework
	for i := range parentDGD.Spec.Components {
		component := &parentDGD.Spec.Components[i]
		if !IsWorkerComponent(string(component.ComponentType)) {
			continue
		}

		backendFramework, err := getBackendFrameworkFromComponent(component, parentDGD)
		if err != nil {
			return "", fmt.Errorf("failed to determine backend framework for component %s: %w", component.ComponentName, err)
		}
		if backendFramework == "" || backendFramework == BackendFrameworkNoop {
			continue
		}
		if detected != "" && detected != backendFramework {
			return "", fmt.Errorf("multiple backend frameworks detected for generated DynamoComponentDeployments: %s and %s", detected, backendFramework)
		}
		detected = backendFramework
	}

	return string(detected), nil
}

func GetDynamoNamespace(object metav1.Object, service *v1beta1.DynamoComponentDeploymentSharedSpec) string {
	return v1beta1.ComputeDynamoNamespace(service.GlobalDynamoNamespace, object.GetNamespace(), object.GetName())
}

// generateSingleDCD creates a DynamoComponentDeployment for a single service.
func generateSingleDCD(
	parentDGD *v1beta1.DynamoGraphDeployment,
	componentName string,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	dynamoNamespace string,
	backendFramework string,
	restartState *RestartState,
	existingRestartAnnotations map[string]string,
	rollingUpdateCtx RollingUpdateContext,
) (*v1beta1.DynamoComponentDeployment, error) {
	deployment := &v1beta1.DynamoComponentDeployment{}
	deployment.Spec.DynamoComponentDeploymentSharedSpec = *component.DeepCopy()
	deployment.Name = GetDCDResourceName(parentDGD, componentName, rollingUpdateCtx.NewWorkerHash)
	deployment.Spec.BackendFramework = backendFramework
	deployment.Namespace = parentDGD.Namespace
	component = &deployment.Spec.DynamoComponentDeploymentSharedSpec

	if err := applyDGDComponentAlphaCompatibilityToDCD(parentDGD, componentName, deployment); err != nil {
		return nil, err
	}

	labels := make(map[string]string)
	maps.Copy(labels, GetPodTemplateLabels(component))
	labels[commonconsts.KubeLabelDynamoComponent] = componentName
	labels[commonconsts.KubeLabelDynamoNamespace] = dynamoNamespace
	labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = parentDGD.Name
	deployment.Labels = labels

	// only label worker DCDs with their hash for cleanup during rolling updates
	if IsWorkerComponent(string(component.ComponentType)) {
		labels[commonconsts.KubeLabelDynamoWorkerHash] = rollingUpdateCtx.NewWorkerHash
		podTemplate := ensurePodTemplate(&deployment.Spec.DynamoComponentDeploymentSharedSpec)
		podTemplate.Labels[commonconsts.KubeLabelDynamoWorkerHash] = rollingUpdateCtx.NewWorkerHash
		if parentDGD.HasEPPComponent() {
			labels[commonconsts.KubeLabelDynamoComponentClass] = commonconsts.ComponentClassWorker
			podTemplate.Labels[commonconsts.KubeLabelDynamoComponentClass] = commonconsts.ComponentClassWorker
		}
	}

	applyDGDTemplateDefaults(&deployment.Spec.DynamoComponentDeploymentSharedSpec, parentDGD)

	// Apply restart annotation if this component should be restarted.
	if restartState.ShouldAnnotateComponent(componentName) {
		podTemplate := ensurePodTemplate(&deployment.Spec.DynamoComponentDeploymentSharedSpec)
		podTemplate.Annotations[commonconsts.RestartAnnotation] = restartState.Timestamp
	} else if existingRestartAnnotations != nil {
		if existingRestartAt, ok := existingRestartAnnotations[componentName]; ok && existingRestartAt != "" {
			podTemplate := ensurePodTemplate(&deployment.Spec.DynamoComponentDeploymentSharedSpec)
			podTemplate.Annotations[commonconsts.RestartAnnotation] = existingRestartAt
		}
	}

	if component.ComponentType == commonconsts.ComponentTypePlanner {
		ensurePodTemplate(&deployment.Spec.DynamoComponentDeploymentSharedSpec).Spec.ServiceAccountName = commonconsts.PlannerServiceAccountName
	}

	if err := applyDynDeploymentConfig(deployment, commonconsts.DynamoServicePort); err != nil {
		return nil, err
	}

	// during a rolling update, the replica count is determined by the rollingUpdateCtx instead of the component spec
	if newReplicas, ok := rollingUpdateCtx.NewWorkerReplicas[componentName]; rollingUpdateCtx.InProgress() && IsWorkerComponent(string(component.ComponentType)) && ok {
		deployment.Spec.Replicas = ptr.To(newReplicas)
	} else if component.Replicas != nil {
		deployment.Spec.Replicas = component.Replicas
	}

	return deployment, nil
}

func applyDGDComponentAlphaCompatibilityToDCD(parentDGD *v1beta1.DynamoGraphDeployment, componentName string, dcd *v1beta1.DynamoComponentDeployment) error {
	component := getDGDAlphaComponent(parentDGD, componentName)
	if dcd == nil || component == nil {
		return nil
	}
	alphaDCD := &v1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dcd.Name,
			Namespace: dcd.Namespace,
		},
		Spec: v1alpha1.DynamoComponentDeploymentSpec{
			BackendFramework:                    dcd.Spec.BackendFramework,
			DynamoComponentDeploymentSharedSpec: *component.DeepCopy(),
		},
	}
	carrier := &v1beta1.DynamoComponentDeployment{}
	if err := alphaDCD.ConvertTo(carrier); err != nil {
		return err
	}
	if len(carrier.Annotations) == 0 {
		return nil
	}
	if dcd.Annotations == nil {
		dcd.Annotations = map[string]string{}
	}
	maps.Copy(dcd.Annotations, carrier.Annotations)
	return nil
}

// GetDGDComponentResourceLabels returns labels that should be applied to resources
// created directly for a DGD component.
func GetDGDComponentResourceLabels(dgd *v1beta1.DynamoGraphDeployment, componentName string, component *v1beta1.DynamoComponentDeploymentSharedSpec) map[string]string {
	labels := map[string]string{}
	if dgd != nil {
		maps.Copy(labels, dgd.Spec.Labels)
		maps.Copy(labels, getDGDComponentAlphaLabels(dgd, componentName))
	}
	maps.Copy(labels, GetPodTemplateLabels(component))
	return labels
}

// GetDGDComponentPreservedIngressSpec returns an alpha component ingress spec that
// was preserved during conversion to v1beta1.
func GetDGDComponentPreservedIngressSpec(dgd *v1beta1.DynamoGraphDeployment, componentName string) (IngressSpec, bool) {
	component := getDGDAlphaComponent(dgd, componentName)
	if component == nil || component.Ingress == nil {
		return IngressSpec{}, false
	}
	raw, err := json.Marshal(component.Ingress)
	if err != nil {
		return IngressSpec{}, false
	}
	var ingressSpec IngressSpec
	if err := json.Unmarshal(raw, &ingressSpec); err != nil {
		return IngressSpec{}, false
	}
	return ingressSpec, true
}

func applyDynDeploymentConfig(dcd *v1beta1.DynamoComponentDeployment, frontendPort int) error {
	main := GetMainContainer(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
	if main == nil {
		return nil
	}
	rawConfig := getDynamoDeploymentConfig(main)
	if rawConfig == nil {
		return nil
	}
	componentName := GetDCDComponentName(dcd)
	if dcd.IsFrontendComponent() {
		updatedConfig, err := updateDynDeploymentConfigBytes(rawConfig, componentName, frontendPort)
		if err != nil {
			return err
		}
		setDynamoDeploymentConfig(main, updatedConfig)
		rawConfig = updatedConfig
	}
	config, err := ParseDynDeploymentConfig(rawConfig)
	if err != nil {
		return err
	}
	serviceConfig := getDynDeploymentServiceConfig(config, componentName, dcd.IsFrontendComponent())
	if serviceConfig == nil || serviceConfig.ServiceArgs == nil {
		return nil
	}
	if dcd.Spec.Replicas == nil && serviceConfig.ServiceArgs.Workers != nil {
		dcd.Spec.Replicas = serviceConfig.ServiceArgs.Workers
	}
	return applyDynDeploymentResources(main, serviceConfig.ServiceArgs.Resources)
}

func getDynDeploymentServiceConfig(config DynDeploymentConfig, componentName string, isFrontend bool) *DynDeploymentServiceConfig {
	if serviceConfig := config[componentName]; serviceConfig != nil {
		return serviceConfig
	}
	if !isFrontend {
		return nil
	}
	for name, serviceConfig := range config {
		if strings.EqualFold(name, commonconsts.ComponentTypeFrontend) {
			return serviceConfig
		}
	}
	return nil
}

func getDynamoDeploymentConfig(container *corev1.Container) []byte {
	for _, env := range container.Env {
		if env.Name == commonconsts.DynamoDeploymentConfigEnvVar {
			return []byte(env.Value)
		}
	}
	return nil
}

func setDynamoDeploymentConfig(container *corev1.Container, config []byte) {
	for i := range container.Env {
		if container.Env[i].Name == commonconsts.DynamoDeploymentConfigEnvVar {
			container.Env[i].Value = string(config)
			return
		}
	}
}

func updateDynDeploymentConfigBytes(rawConfig []byte, serviceName string, newPort int) ([]byte, error) {
	var config map[string]map[string]any
	if err := json.Unmarshal(rawConfig, &config); err != nil {
		return nil, err
	}
	if frontend, ok := config[serviceName]; ok {
		frontend["port"] = newPort
	} else {
		for name, frontend := range config {
			if strings.EqualFold(name, commonconsts.ComponentTypeFrontend) {
				frontend["port"] = newPort
				break
			}
		}
	}
	return json.Marshal(config)
}

func applyDynDeploymentResources(container *corev1.Container, resources *Resources) error {
	if resources == nil {
		return nil
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = corev1.ResourceList{}
	}
	if container.Resources.Limits == nil {
		container.Resources.Limits = corev1.ResourceList{}
	}
	applyResourceQuantity := func(name corev1.ResourceName, value *string) error {
		if value == nil || *value == "" {
			return nil
		}
		quantity, err := resource.ParseQuantity(*value)
		if err != nil {
			return fmt.Errorf("parse resource %s quantity %q: %w", name, *value, err)
		}
		container.Resources.Requests[name] = quantity
		container.Resources.Limits[name] = quantity
		return nil
	}
	if err := applyResourceQuantity(corev1.ResourceCPU, resources.CPU); err != nil {
		return err
	}
	if err := applyResourceQuantity(corev1.ResourceMemory, resources.Memory); err != nil {
		return err
	}
	if err := applyResourceQuantity(corev1.ResourceName(commonconsts.KubeResourceGPUNvidia), resources.GPU); err != nil {
		return err
	}
	for name, value := range resources.Custom {
		resourceName := corev1.ResourceName(name)
		quantity, err := resource.ParseQuantity(value)
		if err != nil {
			return fmt.Errorf("parse resource %s quantity %q: %w", name, value, err)
		}
		container.Resources.Requests[resourceName] = quantity
		container.Resources.Limits[resourceName] = quantity
	}
	return nil
}

func MergeEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
	envMap := make(map[string]corev1.EnvVar)

	// Add all common environment variables.
	for _, env := range common {
		envMap[env.Name] = env
	}

	// Override or add with service-specific environment variables.
	for _, env := range specific {
		envMap[env.Name] = env
	}

	// Convert the map back to a slice.
	merged := make([]corev1.EnvVar, 0, len(envMap))
	for _, env := range envMap {
		merged = append(merged, env)
	}
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Name < merged[j].Name
	})
	return merged
}

// GetDCDResourceName returns the Kubernetes resource name for a DynamoComponentDeployment.
// If using for a non DCD resource (i.e. Ingress or VirtualService), use the empty string for the workerSuffix.
// For DCD Resources, Worker components include the workerSuffix; for non-workers, workerSuffix is ignored
func GetDCDResourceName(dgd *v1beta1.DynamoGraphDeployment, componentName string, workerSuffix string) string {
	baseName := fmt.Sprintf("%s-%s", dgd.Name, strings.ToLower(componentName))
	if spec := dgd.GetComponentByName(componentName); spec != nil && IsWorkerComponent(string(spec.ComponentType)) && workerSuffix != "" {
		return baseName + "-" + workerSuffix
	}
	return baseName
}

// NormalizeKubeResourceName preserves the existing Dynamo resource naming
// contract while making names acceptable for Kubernetes resources that reject
// dots, such as Services.
func NormalizeKubeResourceName(name string) string {
	return strings.ToLower(strings.ReplaceAll(name, ".", "-"))
}

type SecretsRetriever interface {
	GetSecrets(namespace, registry string) ([]string, error)
}

func resolveImagePullSecrets(retriever SecretsRetriever, namespace, image string) []corev1.LocalObjectReference {
	names, err := retriever.GetSecrets(namespace, image)
	if err != nil {
		return nil
	}
	refs := make([]corev1.LocalObjectReference, 0, len(names))
	for _, name := range names {
		refs = append(refs, corev1.LocalObjectReference{Name: name})
	}
	return refs
}

// applyCliqueStartupDependencies configures StartsAfter dependencies for cliques in a PodCliqueSet
// based on the backend framework, multinode deployment patterns, and the
// inter-pod GMS layout.
//
// Rules:
//   - For TRTLLM multinode: leader clique starts after worker cliques
//   - For inter-pod GMS: engine PCLQs start after their corresponding GMS PCLQ
//     (per rank). This applies both to the standalone inter-pod layout and to
//     the inter-pod layout with failover; the ordering reflects that engines
//     load weights from the weight-server pod regardless of whether shadows are
//     present.
//   - Sets the PodCliqueSet StartupType to Explicit if any dependencies are configured
func applyCliqueStartupDependencies(
	gangSet *grovev1alpha1.PodCliqueSet,
	roles []ServiceRole,
	backendFramework BackendFramework,
	numberOfNodes int32,
	isInterPodGMS bool,
) {
	enabledMultinode := backendFramework == BackendFrameworkTRTLLM && numberOfNodes > 1
	if !enabledMultinode && !isInterPodGMS {
		return
	}

	var leaderCliqueName string
	var workerCliqueNames []string
	// For GMS: map rank -> GMS clique name
	gmsCliqueByRank := map[int32]string{}

	for _, r := range roles {
		cliqueName := strings.ToLower(r.Name)
		switch r.Role {
		case RoleLeader:
			leaderCliqueName = cliqueName
		case RoleWorker:
			workerCliqueNames = append(workerCliqueNames, cliqueName)
		case RoleGMS:
			gmsCliqueByRank[r.Rank] = cliqueName
		}
	}

	hasDependencies := false
	for _, clique := range gangSet.Spec.Template.Cliques {
		var cliqueRole Role
		var cliqueRank int32
		found := false
		for _, r := range roles {
			if strings.ToLower(r.Name) == clique.Name {
				cliqueRole = r.Role
				cliqueRank = r.Rank
				found = true
				break
			}
		}
		if !found {
			continue
		}

		var startsAfter []string

		// GMS dependencies: engine PCLQs start after their rank's GMS PCLQ
		if isInterPodGMS && cliqueRole != RoleGMS {
			if gmsName, ok := gmsCliqueByRank[cliqueRank]; ok {
				startsAfter = append(startsAfter, gmsName)
			}
		}

		// Existing multinode dependencies
		if enabledMultinode {
			multiDeps := getCliqueStartupDependencies(cliqueRole, backendFramework, leaderCliqueName, workerCliqueNames)
			startsAfter = append(startsAfter, multiDeps...)
		}

		if len(startsAfter) > 0 {
			clique.Spec.StartsAfter = startsAfter
			hasDependencies = true
		}
	}

	if hasDependencies {
		explicitStartupType := grovev1alpha1.CliqueStartupTypeExplicit
		gangSet.Spec.Template.StartupType = &explicitStartupType
	}
}

// getCliqueStartupDependencies determines the StartsAfter dependencies for a clique
// based on its role, backend framework, and available leader/worker clique names.
//
// Rules:
// - For VLLM and SGLang: worker cliques start after leader clique
// - For TRTLLM: leader clique starts after worker cliques
// - For other backends or single-node deployments: no dependencies
func getCliqueStartupDependencies(
	role Role,
	backendFramework BackendFramework,
	leaderCliqueName string,
	workerCliqueNames []string,
) []string {
	switch backendFramework {
	case BackendFrameworkVLLM, BackendFrameworkSGLang:
		// For vllm and sglang: worker cliques start after leader clique
		if role == RoleWorker && leaderCliqueName != "" {
			return []string{leaderCliqueName}
		}
	case BackendFrameworkTRTLLM:
		// For trtllm: leader clique starts after worker cliques
		if role == RoleLeader && len(workerCliqueNames) > 0 {
			return workerCliqueNames
		}
	}

	// No dependencies for other cases
	return nil
}

// ComponentServiceParams contains all the fields needed to generate a Kubernetes
// Service for a Dynamo component, independent of whether the caller is the DGD
// (Grove) or DCD controller.
type ComponentServiceParams struct {
	ServiceName     string
	Namespace       string
	ComponentType   string
	DynamoNamespace string
	ComponentName   string // original user-provided name, used in selector
	Labels          map[string]string
	Annotations     map[string]string
	IsK8sDiscovery  bool
}

func GenerateComponentService(params ComponentServiceParams) (*corev1.Service, error) {
	var servicePort corev1.ServicePort
	switch params.ComponentType {
	case commonconsts.ComponentTypeFrontend:
		servicePort = corev1.ServicePort{
			Name:       commonconsts.DynamoServicePortName,
			Port:       commonconsts.DynamoServicePort,
			TargetPort: intstr.FromString(commonconsts.DynamoContainerPortName),
			Protocol:   corev1.ProtocolTCP,
		}
	case commonconsts.ComponentTypeEPP:
		servicePort = corev1.ServicePort{
			Name:        commonconsts.EPPGRPCPortName,
			Port:        commonconsts.EPPGRPCPort,
			TargetPort:  intstr.FromInt(commonconsts.EPPGRPCPort),
			Protocol:    corev1.ProtocolTCP,
			AppProtocol: ptr.To("http2"),
		}
	default:
		servicePort = corev1.ServicePort{
			Name:       commonconsts.DynamoSystemPortName,
			Port:       commonconsts.DynamoSystemPort,
			TargetPort: intstr.FromString(commonconsts.DynamoSystemPortName),
			Protocol:   corev1.ProtocolTCP,
		}
	}

	labels := make(map[string]string)
	for k, v := range params.Labels {
		labels[k] = v
	}
	if params.IsK8sDiscovery {
		labels[commonconsts.KubeLabelDynamoDiscoveryBackend] = commonconsts.DiscoveryBackendKubernetes
		labels[commonconsts.KubeLabelDynamoDiscoveryEnabled] = commonconsts.KubeLabelValueTrue
	}

	selector := map[string]string{
		commonconsts.KubeLabelDynamoComponentType: params.ComponentType,
		commonconsts.KubeLabelDynamoNamespace:     params.DynamoNamespace,
		commonconsts.KubeLabelDynamoComponent:     params.ComponentName,
	}

	annotations := make(map[string]string)
	for k, v := range params.Annotations {
		annotations[k] = v
	}

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			// Service names must be DNS-1035 labels (no dots). Replace dots with
			// hyphens so model names like "Qwen3-0.6B" don't cause rejections.
			Name:        NormalizeKubeResourceName(params.ServiceName),
			Namespace:   params.Namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: corev1.ServiceSpec{
			Selector: selector,
			Ports:    []corev1.ServicePort{servicePort},
		},
	}
	return service, nil
}

func GenerateComponentIngress(ctx context.Context, componentName, componentNamespace string, ingressSpec IngressSpec) *networkingv1.Ingress {
	resourceName := NormalizeKubeResourceName(componentName)
	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      resourceName,
			Namespace: componentNamespace,
		},
	}
	host := getIngressHost(ingressSpec)
	ingress.Spec = networkingv1.IngressSpec{
		IngressClassName: ingressSpec.IngressControllerClassName,
		Rules: []networkingv1.IngressRule{
			{
				Host: host,
				IngressRuleValue: networkingv1.IngressRuleValue{
					HTTP: &networkingv1.HTTPIngressRuleValue{
						Paths: []networkingv1.HTTPIngressPath{
							{
								Path:     "/",
								PathType: &[]networkingv1.PathType{networkingv1.PathTypePrefix}[0],
								Backend: networkingv1.IngressBackend{
									Service: &networkingv1.IngressServiceBackend{
										Name: resourceName,
										Port: networkingv1.ServiceBackendPort{
											Number: commonconsts.DynamoServicePort,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if ingressSpec.TLS != nil {
		ingress.Spec.TLS = []networkingv1.IngressTLS{
			{
				Hosts:      []string{host},
				SecretName: ingressSpec.TLS.SecretName,
			},
		}
	}
	return ingress
}

func getIngressHost(ingressSpec IngressSpec) string {
	host := ingressSpec.Host
	if ingressSpec.HostPrefix != nil {
		host = *ingressSpec.HostPrefix + host
	}
	ingressSuffix := commonconsts.DefaultIngressSuffix
	if ingressSpec.HostSuffix != nil {
		ingressSuffix = *ingressSpec.HostSuffix
	}
	return fmt.Sprintf("%s.%s", host, ingressSuffix)
}

func GenerateComponentVirtualService(ctx context.Context, componentName, componentNamespace string, ingressSpec IngressSpec) *networkingv1beta1.VirtualService {
	resourceName := NormalizeKubeResourceName(componentName)
	vs := &networkingv1beta1.VirtualService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      resourceName,
			Namespace: componentNamespace,
		},
	}
	if ingressSpec.IsVirtualServiceEnabled() {
		vs.Spec = istioNetworking.VirtualService{
			Hosts: []string{
				getIngressHost(ingressSpec),
			},
			Gateways: []string{*ingressSpec.VirtualServiceGateway},
			Http: []*istioNetworking.HTTPRoute{
				{
					Match: []*istioNetworking.HTTPMatchRequest{
						{
							Uri: &istioNetworking.StringMatch{
								MatchType: &istioNetworking.StringMatch_Prefix{Prefix: "/"},
							},
						},
					},
					Route: []*istioNetworking.HTTPRouteDestination{
						{
							Destination: &istioNetworking.Destination{
								Host: resourceName,
								Port: &istioNetworking.PortSelector{
									Number: commonconsts.DynamoServicePort,
								},
							},
						},
					},
				},
			},
		}
	}
	return vs
}

// GenerateEPPDestinationRule builds an Istio DestinationRule for an EPP service.
// This tells the mesh sidecar how to connect to the EPP's gRPC endpoint,
// avoiding double-TLS issues when the EPP serves TLS (SecureServing=true).
func GenerateEPPDestinationRule(serviceName, namespace string, meshConfig configv1alpha1.ServiceMeshConfiguration) *networkingv1beta1.DestinationRule {
	// Normalize the service name the same way GenerateComponentService does
	// so the DestinationRule host matches the actual Service DNS name.
	normalizedName := NormalizeKubeResourceName(serviceName)

	dr := &networkingv1beta1.DestinationRule{
		ObjectMeta: metav1.ObjectMeta{
			Name:      normalizedName,
			Namespace: namespace,
		},
	}

	if !meshConfig.IsEnabled() || meshConfig.Istio == nil {
		return dr
	}

	tlsMode := istioNetworking.ClientTLSSettings_SIMPLE
	switch meshConfig.Istio.TLSMode {
	case "DISABLE":
		tlsMode = istioNetworking.ClientTLSSettings_DISABLE
	case "ISTIO_MUTUAL":
		tlsMode = istioNetworking.ClientTLSSettings_ISTIO_MUTUAL
	case "MUTUAL":
		tlsMode = istioNetworking.ClientTLSSettings_MUTUAL
	}

	skipVerify := true
	if meshConfig.Istio.InsecureSkipVerify != nil {
		skipVerify = *meshConfig.Istio.InsecureSkipVerify
	}

	dr.Spec = istioNetworking.DestinationRule{
		Host: fmt.Sprintf("%s.%s.svc.cluster.local", normalizedName, namespace),
		TrafficPolicy: &istioNetworking.TrafficPolicy{
			Tls: &istioNetworking.ClientTLSSettings{
				Mode:               tlsMode,
				InsecureSkipVerify: wrapperspb.Bool(skipVerify),
			},
		},
	}
	return dr
}

func GenerateDefaultIngressSpec(dynamoDeployment *v1beta1.DynamoGraphDeployment, ingressConfig configv1alpha1.IngressConfiguration) IngressSpec {
	res := IngressSpec{
		Enabled:           ingressConfig.VirtualServiceGateway != "" || ingressConfig.ControllerClassName != "",
		Host:              dynamoDeployment.Name,
		UseVirtualService: ingressConfig.VirtualServiceGateway != "",
	}
	if ingressConfig.ControllerClassName != "" {
		res.IngressControllerClassName = &ingressConfig.ControllerClassName
	}
	if ingressConfig.ControllerTLSSecretName != "" {
		res.TLS = &IngressTLSSpec{
			SecretName: ingressConfig.ControllerTLSSecretName,
		}
	}
	if ingressConfig.HostSuffix != "" {
		res.HostSuffix = &ingressConfig.HostSuffix
	}
	if ingressConfig.VirtualServiceGateway != "" {
		res.VirtualServiceGateway = &ingressConfig.VirtualServiceGateway
	}
	return res
}

// Define Role enum for leader/worker/main
// Use this type everywhere instead of string for role

type Role string

const (
	RoleLeader     Role = "leader"
	RoleWorker     Role = "worker"
	RoleMain       Role = "main"
	RoleCheckpoint Role = "checkpoint"
	RoleGMS        Role = "gms"
)

// ServiceRole describes one PodClique (PCLQ) to be materialised for a
// service. A single DynamoComponentDeploymentSharedSpec can expand into
// multiple ServiceRoles depending on the deployment topology:
//
//   - single-node, no GMS: 1 role (RoleMain)
//   - multinode, no GMS:    2 roles (RoleLeader + RoleWorker)
//   - single-node, inter-pod GMS: 1 engine PCLQ (replicated) + 1 RoleGMS
//     weight-server PCLQ
//   - multinode, inter-pod GMS: N engine PCLQs (one per rank, replicated)
//   - 1 RoleGMS weight-server PCLQ
//
// The fields carry the information buildCliqueForRole needs to produce a
// concrete PodCliqueTemplateSpec:
//
//   - Name: PCLQ name suffix used for Grove resource naming and hostname
//     derivation.
//   - Role:     the pod's semantic role (main/leader/worker/gms). Drives
//     backend-specific wiring (e.g. --load-format, --node-rank, discovery
//     labels).
//   - Replicas: the PCLQ replica count. For GMS this is the number of
//     engine pods per rank (primary + NumShadows shadows); for non-GMS
//     roles it is typically 1 (the PCSG-level serviceReplicas controls
//     horizontal scaling).
//   - Rank:     static node rank (0 = leader/main, 1..N-1 = workers).
//     Non-trivial for inter-pod GMS because each rank becomes its own
//     PCLQ and shares a pod index across shadows; for non-GMS multinode
//     pods the rank is derived dynamically from GROVE_PCLQ_POD_INDEX.
type ServiceRole struct {
	Name     string
	Role     Role
	Replicas int32
	Rank     int32 // node rank: 0 = leader/main, 1..N-1 = workers
}

// expandRolesForComponent turns a component's (numberOfNodes,
// gpuMemoryService.mode, failover.mode, replicas) tuple into the concrete
// list of ServiceRole entries the rest of the Grove rendering pipeline
// iterates over. It is the single place that decides how many PodCliques a
// component produces and what each PCLQ looks like (name, role, replicas,
// static rank).
//
// The inter-pod GMS branch is selected by IsInterPodGMSEnabled() (layout)
// rather than IsInterPodFailoverEnabled() (hot-spares): both the standalone
// inter-pod layout (1 engine pod + 1 weight-server pod per rank) and the
// inter-pod layout with failover (primary + N shadows + 1 weight-server pod
// per rank) use the same PCLQ topology, differing only in the per-rank engine
// clique's Replicas (derived from GetTotalEnginePods).
//
// Callers that iterate "engine roles" must still gate on
// IsInterPodGMSEnabled() — this function emits the GMS weight-server PCLQ
// as a regular ServiceRole, not as a separate concept.
func expandRolesForComponent(componentName string, componentReplicas *int32, numberOfNodes int32, component *v1beta1.DynamoComponentDeploymentSharedSpec) []ServiceRole {
	isInterPodGMS := component.IsInterPodGMSEnabled()
	isMultinode := numberOfNodes > 1

	switch {
	case isMultinode && isInterPodGMS:
		return expandMultinodeGMSRoles(componentName, numberOfNodes, component.GetTotalEnginePods())
	case isMultinode:
		return expandMultinodeRoles(componentName, numberOfNodes)
	case isInterPodGMS:
		return expandSingleNodeGMSRoles(componentName, component.GetTotalEnginePods())
	default:
		return expandSingleNodeRoles(componentName, componentReplicas)
	}
}

func expandSingleNodeRoles(componentName string, componentReplicas *int32) []ServiceRole {
	replicas := int32(1)
	if componentReplicas != nil {
		replicas = *componentReplicas
	}
	return []ServiceRole{
		{Name: componentName, Role: RoleMain, Replicas: replicas},
	}
}

func expandMultinodeRoles(componentName string, numberOfNodes int32) []ServiceRole {
	return []ServiceRole{
		{Name: componentName + "-" + commonconsts.GroveRoleSuffixLeader, Role: RoleLeader, Replicas: 1},
		{Name: componentName + "-" + commonconsts.GroveRoleSuffixWorker, Role: RoleWorker, Replicas: numberOfNodes - 1},
	}
}

func expandSingleNodeGMSRoles(componentName string, totalEnginePods int32) []ServiceRole {
	return []ServiceRole{
		{Name: fmt.Sprintf("%s-%s-0", componentName, commonconsts.GroveRoleSuffixGMS), Role: RoleGMS, Replicas: 1, Rank: 0},
		{Name: componentName, Role: RoleMain, Replicas: totalEnginePods, Rank: 0},
	}
}

func expandMultinodeGMSRoles(componentName string, numberOfNodes int32, totalEnginePods int32) []ServiceRole {
	roles := make([]ServiceRole, 0, numberOfNodes*2)
	for rank := int32(0); rank < numberOfNodes; rank++ {
		gmsName := fmt.Sprintf("%s-%s-%d", componentName, commonconsts.GroveRoleSuffixGMS, rank)
		roles = append(roles, ServiceRole{Name: gmsName, Role: RoleGMS, Replicas: 1, Rank: rank})

		var engineName string
		var engineRole Role
		if rank == 0 {
			engineName = componentName + "-" + commonconsts.GroveRoleSuffixLeader
			engineRole = RoleLeader
		} else {
			engineName = fmt.Sprintf("%s-%s-%d", componentName, commonconsts.GroveRoleSuffixWorker, rank)
			engineRole = RoleWorker
		}
		roles = append(roles, ServiceRole{Name: engineName, Role: engineRole, Replicas: totalEnginePods, Rank: rank})
	}
	return roles
}

// PCSNameForDGD computes the PodCliqueSet name for a DGD, auto-truncating if
// the DGD name is too long to fit within Grove's combined resource name limit.
//
// For short DGD names the PCS name equals the DGD name (backwards compatible).
// For long names, the PCS name is truncated with a deterministic 4-char hash
// suffix to guarantee uniqueness and reconcile-loop stability.
func PCSNameForDGD(dgdName string, components []v1beta1.DynamoComponentDeploymentSharedSpec) string {
	maxComponentBudget := 0
	for i := range components {
		component := &components[i]
		componentName := component.ComponentName
		lowerName := strings.ToLower(componentName)
		var budget int
		if component.GetNumberOfNodes() > 1 || component.IsInterPodGMSEnabled() {
			maxCliqueNameLen := 0
			for _, role := range expandRolesForComponent(componentName, component.Replicas, component.GetNumberOfNodes(), component) {
				if cliqueNameLen := len(strings.ToLower(role.Name)); cliqueNameLen > maxCliqueNameLen {
					maxCliqueNameLen = cliqueNameLen
				}
			}
			// PCSG = lowerName, PCLQ = longest rendered role name.
			budget = len(lowerName) + maxCliqueNameLen
		} else {
			// Single-node: PCLQ = lowerName (no PCSG).
			budget = len(lowerName)
		}
		if budget > maxComponentBudget {
			maxComponentBudget = budget
		}
	}

	pcsBudget := commonconsts.MaxCombinedGroveResourceNameLength - maxComponentBudget
	// Clamp to a minimum so we always produce a usable name
	const minPCSNameLength = 8
	if pcsBudget < minPCSNameLength {
		pcsBudget = minPCSNameLength
	}

	if len(dgdName) <= pcsBudget {
		return dgdName
	}

	// Truncate with a deterministic hash suffix for uniqueness
	hash := fnv.New32a()
	hash.Write([]byte(dgdName))
	suffix := fmt.Sprintf("%04x", hash.Sum32()&0xFFFF)
	return dgdName[:pcsBudget-5] + "-" + suffix
}

func PCSNameForAlphaDGDServices(dgdName string, services map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec) string {
	componentNames := make([]string, 0, len(services))
	for componentName := range services {
		componentNames = append(componentNames, componentName)
	}
	sort.Strings(componentNames)

	components := make([]v1beta1.DynamoComponentDeploymentSharedSpec, 0, len(componentNames))
	for _, componentName := range componentNames {
		service := services[componentName]
		component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: componentName}
		if service != nil {
			if service.Multinode != nil {
				component.Multinode = &v1beta1.MultinodeSpec{}
				v1alpha1.ConvertFromMultinodeSpec(service.Multinode, component.Multinode)
			}
			if service.Replicas != nil {
				component.Replicas = ptr.To(*service.Replicas)
			}
			if service.GPUMemoryService != nil && service.GPUMemoryService.Enabled {
				if component.Experimental == nil {
					component.Experimental = &v1beta1.ExperimentalSpec{}
				}
				component.Experimental.GPUMemoryService = &v1beta1.GPUMemoryServiceSpec{}
				v1alpha1.ConvertFromGPUMemoryServiceSpec(service.GPUMemoryService, component.Experimental.GPUMemoryService)
			}
			if service.Failover != nil && service.Failover.Enabled {
				if component.Experimental == nil {
					component.Experimental = &v1beta1.ExperimentalSpec{}
				}
				component.Experimental.Failover = &v1beta1.FailoverSpec{}
				v1alpha1.ConvertFromFailoverSpec(service.Failover, component.Experimental.Failover)
			}
		}
		components = append(components, component)
	}
	return PCSNameForDGD(dgdName, components)
}

// Define BackendFramework enum for sglang, vllm, trtllm

type BackendFramework string

const (
	BackendFrameworkSGLang BackendFramework = "sglang"
	BackendFrameworkVLLM   BackendFramework = "vllm"
	BackendFrameworkTRTLLM BackendFramework = "trtllm"
	BackendFrameworkNoop   BackendFramework = "noop"
)

// ParseBackendFramework converts a string to BackendFramework type.
// Returns an error if the framework string is not recognized.
func ParseBackendFramework(framework string) (BackendFramework, error) {
	bf := BackendFramework(framework)
	switch bf {
	case BackendFrameworkVLLM, BackendFrameworkSGLang, BackendFrameworkTRTLLM, BackendFrameworkNoop:
		return bf, nil
	default:
		return "", fmt.Errorf("unsupported backend framework: %s (valid values: vllm, sglang, trtllm)", framework)
	}
}

// Backend interface for modular backend logic
// Each backend (SGLang, VLLM, etc.) implements this interface
type Backend interface {
	UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer)
	UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer)
}

// NoopBackend does no processing - used for non-worker components like frontend, planner, router
type NoopBackend struct{}

func (b *NoopBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	// No-op: frontend, planner, router, etc. don't need backend-specific processing
}

func (b *NoopBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1beta1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	// No-op: frontend, planner, router, etc. don't need backend-specific processing
}

type MultinodeDeployer interface {
	GetLeaderHostname(serviceName string) string
	GetHostNames(serviceName string, numberOfNodes int32) []string
	GetNodeRank() (string, bool) // returns (rank, needsShellInterpretation)
	NeedsDNSWait() bool          // returns true if DNS wait is needed to launch multinode components
}

// BackendFactory creates backend instances based on the framework type
func BackendFactory(backendFramework BackendFramework, operatorConfig *configv1alpha1.OperatorConfiguration, parentGraphDeploymentName string) Backend {
	switch backendFramework {
	case BackendFrameworkSGLang:
		return &SGLangBackend{}
	case BackendFrameworkVLLM:
		return &VLLMBackend{ParentGraphDeploymentName: parentGraphDeploymentName}
	case BackendFrameworkTRTLLM:
		return &TRTLLMBackend{
			MpiRunSecretName: operatorConfig.MPI.SSHSecretName,
		}
	case BackendFrameworkNoop:
		return &NoopBackend{}
	default:
		return nil
	}
}

func MultinodeDeployerFactory(multinodeDeploymentType commonconsts.MultinodeDeploymentType) MultinodeDeployer {
	switch multinodeDeploymentType {
	case commonconsts.MultinodeDeploymentTypeGrove:
		return &GroveMultinodeDeployer{}
	case commonconsts.MultinodeDeploymentTypeLWS:
		return &LWSMultinodeDeployer{}
	default:
		return nil
	}
}

// IsWorkerComponent checks if a component is a worker that needs backend framework detection
func IsWorkerComponent(componentType string) bool {
	return componentType == commonconsts.ComponentTypeWorker ||
		componentType == commonconsts.ComponentTypePrefill ||
		componentType == commonconsts.ComponentTypeDecode
}

// AddStandardEnvVars adds the standard environment variables that are common to
// both checkpoint jobs and generated worker pods.
func AddStandardEnvVars(container *corev1.Container, operatorConfig *configv1alpha1.OperatorConfiguration) {
	standardEnvVars := []corev1.EnvVar{}
	if operatorConfig.Infrastructure.NATSAddress != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "NATS_SERVER",
			Value: operatorConfig.Infrastructure.NATSAddress,
		})
	}

	if operatorConfig.Infrastructure.ETCDAddress != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "ETCD_ENDPOINTS",
			Value: operatorConfig.Infrastructure.ETCDAddress,
		})
	}

	if operatorConfig.Infrastructure.ModelExpressURL != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "MODEL_EXPRESS_URL",
			Value: operatorConfig.Infrastructure.ModelExpressURL,
		})
	}
	if operatorConfig.Infrastructure.PrometheusEndpoint != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "PROMETHEUS_ENDPOINT",
			Value: operatorConfig.Infrastructure.PrometheusEndpoint,
		})
	}
	// merge the env vars to allow users to override the standard env vars
	container.Env = MergeEnvs(standardEnvVars, container.Env)
}

// applyDefaultSecurityContext sets secure defaults for pod security context.
// Currently only sets fsGroup to solve volume permission issues.
// Does NOT set runAsUser/runAsGroup/runAsNonRoot to maintain backward compatibility
// with images that may expect to run as root.
// User-provided security context values (via extraPodSpec) will override these defaults.
//
// Note: OpenShift's restricted-v2 SCC rejects pods with fsGroup outside the
// namespace's allocated UID range. To run on OpenShift, bind a permissive SCC
// (anyuid / anyuid-v2) to the workload service account, or have the user
// supply an extraPodSpec.securityContext with an in-range value.
func applyDefaultSecurityContext(podSpec *corev1.PodSpec) {
	// Initialize SecurityContext if not present
	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}

	// Only set fsGroup by default
	// This fixes volume permission issues without forcing a specific UID/GID
	// which maintains compatibility with both root and non-root images
	if podSpec.SecurityContext.FSGroup == nil {
		podSpec.SecurityContext.FSGroup = ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup))
	}
}

// GenerateBasePodSpec creates a basic PodSpec with common logic shared between controller and grove
// Includes standard environment variables (DYNAMO_PORT, NATS_SERVER, ETCD_ENDPOINTS)
// Deployment-specific environment merging should be handled by the caller
//
//nolint:gocyclo
func GenerateBasePodSpec(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	parentGraphDeploymentName string,
	namespace string,
	role Role,
	numberOfNodes int32,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
	serviceName string,
	checkpointInfo *checkpoint.CheckpointInfo, // Optional checkpoint info (resolved by ResolveCheckpointForService)
	deployerOverride MultinodeDeployer, // Optional: overrides factory-created deployer when non-nil
) (*corev1.PodSpec, error) {
	// Start with base container generated per component type
	annotations := GetPodTemplateAnnotations(component)
	componentContext := generateComponentContext(component, parentGraphDeploymentName, namespace, numberOfNodes, NewDiscoveryContext(operatorConfig.Discovery.Backend, annotations))
	componentDefaults := ComponentDefaultsFactory(string(component.ComponentType))
	container, err := componentDefaults.GetBaseContainer(componentContext)
	if err != nil {
		return nil, fmt.Errorf("failed to get base container: %w", err)
	}

	if main := GetMainContainer(component); main != nil {
		if err := mergeContainerByName(&container, main); err != nil {
			return nil, fmt.Errorf("failed to merge podTemplate main container: %w", err)
		}
	}
	if err := validateContainerVolumeMounts(container.VolumeMounts); err != nil {
		return nil, err
	}

	if err := applyCompilationCache(&container, component, backendFramework); err != nil {
		return nil, err
	}

	AddStandardEnvVars(&container, operatorConfig)
	frontendSidecarMounts := append([]corev1.VolumeMount(nil), container.VolumeMounts...)

	// Apply backend-specific container modifications
	multinodeDeployer := deployerOverride
	if multinodeDeployer == nil {
		multinodeDeployer = MultinodeDeployerFactory(multinodeDeploymentType)
		if multinodeDeployer == nil {
			return nil, fmt.Errorf("unsupported multinode deployment type: %s", multinodeDeploymentType)
		}
	}
	backend := BackendFactory(backendFramework, operatorConfig, parentGraphDeploymentName)
	if backend == nil {
		return nil, fmt.Errorf("unsupported backend framework: %s", backendFramework)
	}
	backend.UpdateContainer(&container, numberOfNodes, role, component, serviceName, multinodeDeployer)

	// get base podspec from component
	podSpec, err := componentDefaults.GetBasePodSpec(componentContext)
	if err != nil {
		return nil, fmt.Errorf("failed to get base podspec: %w", err)
	}

	// Check if user provided their own security context before merging
	userProvidedSecurityContext := component.PodTemplate != nil && component.PodTemplate.Spec.SecurityContext != nil
	sidecars := make([]corev1.Container, 0)

	if component.PodTemplate != nil {
		podSpecOverride := component.PodTemplate.Spec.DeepCopy()
		for _, userContainer := range podSpecOverride.Containers {
			if userContainer.Name != commonconsts.MainContainerName {
				sidecars = append(sidecars, userContainer)
			}
		}

		podSpecOverride.Containers = nil
		if err := mergo.Merge(&podSpec, podSpecOverride, mergo.WithOverride); err != nil {
			return nil, fmt.Errorf("failed to merge podTemplate spec: %w", err)
		}
	}

	// Apply default security context ONLY if user didn't provide any security context
	// If user provides ANY securityContext (even partial), they get full control with no defaults injected
	// This allows users to intentionally set fields to nil (e.g., to run as root)
	if !userProvidedSecurityContext {
		applyDefaultSecurityContext(&podSpec)
	}

	if controller_common.IsK8sDiscoveryEnabled(operatorConfig.Discovery.Backend, annotations) {
		if podSpec.ServiceAccountName == "" {
			podSpec.ServiceAccountName = discovery.GetK8sDiscoveryServiceAccountName(parentGraphDeploymentName)
		}
	}

	if component.CompilationCache != nil {
		podSpec.Volumes = appendUniqueVolume(podSpec.Volumes, corev1.Volume{
			Name: component.CompilationCache.PVCName,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: component.CompilationCache.PVCName,
				},
			},
		})
	}

	ApplySharedMemoryVolumeAndMount(&podSpec, &container, component.SharedMemorySize)
	podSpec.Containers = append([]corev1.Container{container}, sidecars...)

	if component.FrontendSidecar != nil {
		if err := mergeFrontendSidecarDefaults(&podSpec, *component.FrontendSidecar, componentContext, operatorConfig, frontendSidecarMounts); err != nil {
			return nil, err
		}
	}

	backend.UpdatePodSpec(&podSpec, numberOfNodes, role, component, serviceName, multinodeDeployer)
	podSpec.Volumes = appendMissingPVCVolumesForMounts(podSpec.Volumes, podSpec.Containers[0].VolumeMounts)

	shouldDisableImagePullSecret := annotations[commonconsts.KubeAnnotationDisableImagePullSecretDiscovery] == commonconsts.KubeLabelValueTrue
	if !shouldDisableImagePullSecret && secretsRetriever != nil {
		imagePullSecrets := []corev1.LocalObjectReference{}
		for _, ctr := range podSpec.Containers {
			if ctr.Image != "" {
				imagePullSecrets = controller_common.AppendUniqueImagePullSecrets(imagePullSecrets, resolveImagePullSecrets(secretsRetriever, namespace, ctr.Image))
			}
		}
		podSpec.ImagePullSecrets = controller_common.AppendUniqueImagePullSecrets(podSpec.ImagePullSecrets, imagePullSecrets)
	}

	// Intra-pod GMS: replace nvidia.com/gpu with a shared DRA claim and add the server
	// sidecar directly into this pod.
	//
	// Inter-pod GMS (gpuMemoryService.mode=interPod, with or without failover)
	// must be skipped here — that layout wires DRA claims and the GMS server
	// on a dedicated weight-server pod at the PCSG level (see
	// generateGrovePodCliqueSet → gmsWeightServerPodSpec); re-applying the
	// claim and injecting a sidecar here would produce a double-wired engine
	// pod (stray GMS sidecar, conflicting claim).
	if GetGPUMemoryService(component) != nil && !component.IsInterPodGMSEnabled() {
		claimTemplateName := dra.ResourceClaimTemplateName(parentGraphDeploymentName, serviceName)
		if err := dra.ApplyClaim(&podSpec, claimTemplateName); err != nil {
			return nil, fmt.Errorf("failed to apply DRA claim for GMS: %w", err)
		}
		gms.EnsureServerSidecar(&podSpec, &podSpec.Containers[0])
	}

	// Clone main container into two engine containers (active + standby) for failover.
	// Runs after GMS so the main container already has DRA claims and shared volume.
	if IsIntraPodFailoverEnabled(component) {
		if err := buildFailoverPod(&podSpec, numberOfNodes, backendFramework); err != nil {
			return nil, fmt.Errorf("failed to build failover pod: %w", err)
		}
	}

	return &podSpec, nil
}

func validateContainerVolumeMounts(volumeMounts []corev1.VolumeMount) error {
	for _, mount := range volumeMounts {
		if mount.Name == "" {
			return fmt.Errorf("volumeMount.name is required")
		}
		if mount.MountPath == "" {
			return fmt.Errorf("volumeMount.mountPath is required for %s", mount.Name)
		}
	}
	return nil
}

func mergeContainerByName(base *corev1.Container, override *corev1.Container) error {
	if override == nil {
		return nil
	}
	user := override.DeepCopy()
	user.Name = commonconsts.MainContainerName
	baseEnv := base.Env
	if err := mergo.Merge(base, *user, mergo.WithOverride); err != nil {
		return err
	}
	base.Env = MergeEnvs(baseEnv, user.Env)
	if user.LivenessProbe != nil {
		base.LivenessProbe = user.LivenessProbe.DeepCopy()
	}
	if user.ReadinessProbe != nil {
		base.ReadinessProbe = user.ReadinessProbe.DeepCopy()
	}
	if user.StartupProbe != nil {
		base.StartupProbe = user.StartupProbe
	}
	base.Name = commonconsts.MainContainerName
	return nil
}

func applyCompilationCache(container *corev1.Container, component *v1beta1.DynamoComponentDeploymentSharedSpec, backendFramework BackendFramework) error {
	if component.CompilationCache == nil {
		return nil
	}
	if component.CompilationCache.PVCName == "" {
		return fmt.Errorf("compilationCache.pvcName is required when compilationCache is set")
	}
	mountPath := component.CompilationCache.MountPath
	if mountPath == "" {
		mountPath = getDefaultCompilationCacheMountPoint(backendFramework)
		if mountPath == "" {
			return fmt.Errorf("compilationCache.mountPath is required for backend framework %s (no default available)", backendFramework)
		}
		component.CompilationCache.MountPath = mountPath
	}
	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      component.CompilationCache.PVCName,
		MountPath: mountPath,
	})
	return nil
}

func appendUniqueVolume(volumes []corev1.Volume, volume corev1.Volume) []corev1.Volume {
	for i := range volumes {
		if volumes[i].Name == volume.Name {
			return volumes
		}
	}
	return append(volumes, volume)
}

func appendMissingPVCVolumesForMounts(volumes []corev1.Volume, mounts []corev1.VolumeMount) []corev1.Volume {
	volumesByName := make(map[string]corev1.Volume, len(volumes))
	for _, volume := range volumes {
		volumesByName[volume.Name] = volume
	}

	ordered := make([]corev1.Volume, 0, len(volumes)+len(mounts))
	seen := make(map[string]struct{}, len(volumes)+len(mounts))
	for _, mount := range mounts {
		if mount.Name == "" {
			continue
		}
		if volume, ok := volumesByName[mount.Name]; ok {
			ordered = append(ordered, volume)
			seen[mount.Name] = struct{}{}
			continue
		}
		ordered = append(ordered, corev1.Volume{
			Name: mount.Name,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: mount.Name,
				},
			},
		})
		seen[mount.Name] = struct{}{}
	}
	for _, volume := range volumes {
		if _, ok := seen[volume.Name]; ok {
			continue
		}
		ordered = append(ordered, volume)
	}
	return ordered
}

func mergeFrontendSidecarDefaults(podSpec *corev1.PodSpec, sidecarName string, parentContext ComponentContext, operatorConfig *configv1alpha1.OperatorConfiguration, parentMounts []corev1.VolumeMount) error {
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name != sidecarName {
			continue
		}
		frontendContext := ComponentContext{
			numberOfNodes:                  1,
			ComponentType:                  commonconsts.ComponentTypeFrontend,
			ParentGraphDeploymentName:      parentContext.ParentGraphDeploymentName,
			ParentGraphDeploymentNamespace: parentContext.ParentGraphDeploymentNamespace,
			Discovery:                      parentContext.Discovery,
			DynamoNamespace:                parentContext.DynamoNamespace,
		}
		frontendDefaults := NewFrontendDefaults()
		base, err := frontendDefaults.GetBaseContainer(frontendContext)
		if err != nil {
			return fmt.Errorf("failed to get frontend sidecar defaults: %w", err)
		}
		base.Name = sidecarName
		baseEnv := base.Env
		user := podSpec.Containers[i].DeepCopy()
		if err := mergo.Merge(&base, *user, mergo.WithOverride); err != nil {
			return fmt.Errorf("failed to merge frontend sidecar %q: %w", sidecarName, err)
		}
		base.Env = MergeEnvs(baseEnv, user.Env)
		AddStandardEnvVars(&base, operatorConfig)
		base.VolumeMounts = appendMissingVolumeMounts(base.VolumeMounts, parentMounts)
		podSpec.Containers[i] = base
		return nil
	}
	return fmt.Errorf("frontendSidecar %q does not match any podTemplate container", sidecarName)
}

func appendMissingVolumeMounts(dst, mounts []corev1.VolumeMount) []corev1.VolumeMount {
	if len(mounts) == 0 {
		return dst
	}
	seen := make(map[string]struct{}, len(dst))
	for _, mount := range dst {
		seen[mount.Name] = struct{}{}
	}
	for _, mount := range mounts {
		if _, ok := seen[mount.Name]; ok {
			continue
		}
		dst = append(dst, mount)
		seen[mount.Name] = struct{}{}
	}
	return dst
}

func setMetricsLabels(labels map[string]string, dynamoGraphDeployment *v1beta1.DynamoGraphDeployment) {
	// Convert user-provided metrics annotation into controller-managed label
	// By default (no annotation), metrics are enabled
	if metricsAnnotationValue, ok := dynamoGraphDeployment.Annotations[commonconsts.KubeAnnotationEnableMetrics]; ok && metricsAnnotationValue == commonconsts.KubeLabelValueFalse {
		// Explicitly disabled, don't add the label
		return
	}
	// Any other value (including empty) enables metrics
	labels[commonconsts.KubeLabelMetricsEnabled] = commonconsts.KubeLabelValueTrue
}

func generateComponentContext(component *v1beta1.DynamoComponentDeploymentSharedSpec, parentGraphDeploymentName string, namespace string, numberOfNodes int32, discovery DiscoveryContext) ComponentContext {
	dynamoNamespace := v1beta1.ComputeDynamoNamespace(component.GlobalDynamoNamespace, namespace, parentGraphDeploymentName)
	var workerHashSuffix string
	labels := GetPodTemplateLabels(component)
	if IsWorkerComponent(string(component.ComponentType)) && labels[commonconsts.KubeLabelDynamoWorkerHash] != "" {
		workerHashSuffix = labels[commonconsts.KubeLabelDynamoWorkerHash]
	}

	componentContext := ComponentContext{
		numberOfNodes:                  numberOfNodes,
		ComponentType:                  string(component.ComponentType),
		ParentGraphDeploymentName:      parentGraphDeploymentName,
		ParentGraphDeploymentNamespace: namespace,
		Discovery:                      discovery,
		DynamoNamespace:                dynamoNamespace,
		EPPConfig:                      component.EPPConfig,
		WorkerHashSuffix:               workerHashSuffix,
	}
	return componentContext
}

// GeneratePodSpecForComponent creates a PodSpec for Grove deployments (simplified wrapper)
// deployerOverride, when non-nil, overrides the default MultinodeDeployer from the factory.
func GeneratePodSpecForComponent(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
	role Role,
	numberOfNodes int32,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
	serviceName string,
	checkpointInfo *checkpoint.CheckpointInfo,
	deployerOverride MultinodeDeployer,
) (*corev1.PodSpec, error) {
	if component == nil {
		return nil, fmt.Errorf("component is nil")
	}
	if dynamoDeployment == nil {
		return nil, fmt.Errorf("dynamoDeployment is nil")
	}
	component = component.DeepCopy()
	applyDGDTemplateDefaults(component, dynamoDeployment)
	if operatorConfig == nil {
		operatorConfig = &configv1alpha1.OperatorConfiguration{}
	}

	podSpec, err := GenerateBasePodSpec(component, backendFramework, secretsRetriever, dynamoDeployment.Name, dynamoDeployment.Namespace, role, numberOfNodes, operatorConfig, multinodeDeploymentType, serviceName, checkpointInfo, deployerOverride)
	if err != nil {
		return nil, err
	}
	return podSpec, nil
}

func applyDGDTemplateDefaults(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
) {
	if component == nil || dynamoDeployment == nil {
		return
	}
	if len(dynamoDeployment.Spec.Env) > 0 {
		podTemplate := ensurePodTemplate(component)
		main := ensureMainContainer(podTemplate)
		main.Env = MergeEnvs(dynamoDeployment.Spec.Env, main.Env)
	}

	propagateDGDAnnotations(dynamoDeployment.GetAnnotations(), component)
	propagateDGDSpecMetadata(dynamoDeployment.Spec.Annotations, dynamoDeployment.Spec.Labels, component)
}

// dgdPropagatedAnnotationKeys lists DGD metadata annotations that are propagated
// to component-level annotations (for both the DCD/controller and Grove paths).
// Service-level annotations take precedence (are never overwritten).
var dgdPropagatedAnnotationKeys = []string{
	commonconsts.KubeAnnotationEnableMetrics,
	commonconsts.KubeAnnotationDynamoDiscoveryBackend,
	commonconsts.KubeAnnotationDynamoKubeDiscoveryMode,
	commonconsts.KubeAnnotationDynamoOperatorOriginVersion,
	commonconsts.KubeAnnotationVLLMDistributedExecutorBackend,
}

// propagateDGDAnnotations copies DGD-level annotations into the component
// annotations so that downstream logic can read them uniformly.
// Service-level annotations take precedence (are never overwritten).
func propagateDGDAnnotations(dgdAnnotations map[string]string, component *v1beta1.DynamoComponentDeploymentSharedSpec) {
	podTemplate := ensurePodTemplate(component)
	for _, key := range dgdPropagatedAnnotationKeys {
		if val, exists := dgdAnnotations[key]; exists {
			if _, serviceHas := podTemplate.Annotations[key]; !serviceHas {
				podTemplate.Annotations[key] = val
			}
		}
	}
}

// propagateDGDSpecMetadata merges DGD spec-level annotations and labels into
// the component as a low-priority base. Service-level values take precedence.
func propagateDGDSpecMetadata(annotations, labels map[string]string, component *v1beta1.DynamoComponentDeploymentSharedSpec) {
	podTemplate := ensurePodTemplate(component)
	podTemplate.Annotations = mergeLowPriorityMetadata(podTemplate.Annotations, annotations)
	podTemplate.Labels = mergeLowPriorityMetadata(podTemplate.Labels, labels)
}

// GenerateGrovePodCliqueSet generates a Grove PodCliqueSet for the given deployment, supporting both single-node and multinode cases.
// cliqueParams groups the context needed to build a single PodClique template
// from a ServiceRole. All fields come from the enclosing GenerateGrovePodCliqueSet
// loop iteration and are read-only.
type cliqueParams struct {
	r                          ServiceRole
	component                  *v1beta1.DynamoComponentDeploymentSharedSpec
	backendFramework           BackendFramework
	secretsRetriever           SecretsRetriever
	dynamoDeployment           *v1beta1.DynamoGraphDeployment
	numberOfNodes              int32
	operatorConfig             *configv1alpha1.OperatorConfiguration
	runtimeConfig              *controller_common.RuntimeConfig
	componentName              string
	checkpointInfo             *checkpoint.CheckpointInfo
	isMultinode                bool
	usesPCSG                   bool
	isInterPodGMS              bool
	isInterPodFailover         bool
	discoveryBackend           configv1alpha1.DiscoveryBackend
	discoveryContext           DiscoveryContext
	restartState               *RestartState
	existingRestartAnnotations map[string]string
	validatedQueueName         string
	kubeClient                 ctrlclient.Client
	ctx                        context.Context
}

// buildCliqueForRole generates a single PodCliqueTemplateSpec for the given role,
// injecting labels, annotations, checkpoint config, and scheduler settings.
func buildCliqueForRole(p cliqueParams) (*grovev1alpha1.PodCliqueTemplateSpec, error) {
	podSpec, err := generatePodSpecForRole(
		p.r, p.component, p.backendFramework, p.secretsRetriever,
		p.dynamoDeployment, p.numberOfNodes, p.operatorConfig, p.componentName, p.checkpointInfo,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate podSpec for role %s: %w", p.r.Name, err)
	}

	// GMS weight servers load weights fresh from disk and are not CRIU targets.
	if p.operatorConfig.Checkpoint.Enabled && p.r.Role != RoleGMS {
		if err := checkpoint.InjectCheckpointIntoPodSpecWithStorageConfig(
			p.ctx, p.kubeClient, p.dynamoDeployment.Namespace, podSpec, p.checkpointInfo,
			p.operatorConfig.Checkpoint.Storage,
			p.operatorConfig.Checkpoint.EffectiveSeccompProfile(),
		); err != nil {
			return nil, fmt.Errorf("failed to inject checkpoint config for role %s: %w", p.r.Name, err)
		}
	}

	// minAvailable controls Grove gang-scheduling: the clique is only
	// considered available when at least this many replicas are Ready.
	//
	// The invariant we want is "minAvailable = Replicas unless the clique
	// has redundant replicas". Concretely:
	//
	//   - Plain multinode (no inter-pod GMS failover): the worker clique
	//     collapses non-leader ranks into a single clique with
	//     Replicas = numberOfNodes - 1 and those pods are NCCL peers of each
	//     other — losing any one breaks the collective, so all replicas
	//     must be Ready. Standalone inter-pod GMS on multinode also lands
	//     here but has Replicas = 1 per PCLQ (primary only, no shadows), so
	//     the same rule evaluates to minAvailable = 1 without a special case.
	//
	//   - Inter-pod GMS failover (single- or multinode): within each rank
	//     Replicas = primary + shadows and shadows ARE redundant hot spares
	//     — requiring every shadow to be Ready would defeat failover, so
	//     the clique stays at minAvailable = 1.
	//
	//   - Single-node clique (no multinode, with or without intra-pod
	//     failover or standalone inter-pod GMS): Replicas is at most 1 or a
	//     small DP fanout under the outer PCSG where the replicas are
	//     independent of each other; minAvailable = 1 is correct.
	//
	// The two-line rule below captures all of the above: take the baseline
	// of 1, then lift it to Replicas only on plain multinode without
	// inter-pod failover (the only layout that combines >1 replicas per
	// clique with no redundancy between them).
	minAvailable := int32(1)
	if p.isMultinode && !p.isInterPodFailover {
		minAvailable = p.r.Replicas
	}

	clique := &grovev1alpha1.PodCliqueTemplateSpec{
		Name: strings.ToLower(p.r.Name),
		Spec: grovev1alpha1.PodCliqueSpec{
			RoleName:     strings.ToLower(p.r.Name),
			Replicas:     p.r.Replicas,
			MinAvailable: ptr.To(minAvailable),
			PodSpec:      *podSpec,
		},
	}

	if !p.usesPCSG {
		clique.TopologyConstraint = toGroveTopologyConstraint(p.component.TopologyConstraint)
	}

	labels, err := generateLabels(p.component, p.dynamoDeployment, p.componentName, p.discoveryContext)
	if err != nil {
		return nil, fmt.Errorf("failed to generate labels: %w", err)
	}
	clique.Labels = labels
	if p.isInterPodFailover && p.r.Role != RoleGMS {
		clique.Labels[commonconsts.KubeLabelDynamoFailoverEngineGroupMember] = commonconsts.KubeLabelValueTrue
	}
	// Strip discovery labels from RoleGMS pods. generateLabels applies them
	// unconditionally to every role for container-mode Pod reflector filtering
	// (see #8067), but GMS weight-server pods run gpu_memory_service.cli.server
	// — not the dynamo runtime — and never register a DynamoWorkerMetadata CR.
	// Leaving the labels on them would make the Rust discovery daemon include
	// them in its reflector store for no purpose and wake its debounce loop on
	// every GMS restart/fast-kill event.
	if p.r.Role == RoleGMS {
		delete(clique.Labels, commonconsts.KubeLabelDynamoDiscoveryBackend)
		delete(clique.Labels, commonconsts.KubeLabelDynamoDiscoveryEnabled)
	}

	annotations, err := generateAnnotations(p.component, p.dynamoDeployment, p.componentName)
	if err != nil {
		return nil, fmt.Errorf("failed to generate annotations: %w", err)
	}
	if p.r.Role != RoleGMS {
		if err := checkpoint.ApplyRestorePodMetadataWithStorageConfig(labels, annotations, p.checkpointInfo, p.operatorConfig.Checkpoint.Storage); err != nil {
			return nil, fmt.Errorf("failed to apply checkpoint metadata for role %s: %w", p.r.Name, err)
		}
	}
	annotations = applyRestartAnnotation(annotations, p.componentName, p.restartState, p.existingRestartAnnotations)
	clique.Annotations = annotations

	injectKaiSchedulerIfEnabled(clique, p.runtimeConfig, p.validatedQueueName)
	return clique, nil
}

// applyRestartAnnotation adds the restart annotation to the map if needed,
// creating the map when it is nil.
func applyRestartAnnotation(annotations map[string]string, componentName string, restartState *RestartState, existingRestartAnnotations map[string]string) map[string]string {
	if restartState.ShouldAnnotateComponent(componentName) {
		if annotations == nil {
			annotations = make(map[string]string)
		}
		annotations[commonconsts.RestartAnnotation] = restartState.Timestamp
	} else if existingRestartAnnotations != nil {
		if existingTimestamp, ok := existingRestartAnnotations[componentName]; ok {
			if annotations == nil {
				annotations = make(map[string]string)
			}
			annotations[commonconsts.RestartAnnotation] = existingTimestamp
		}
	}
	return annotations
}

func GenerateGrovePodCliqueSet(
	ctx context.Context,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	runtimeConfig *controller_common.RuntimeConfig,
	kubeClient ctrlclient.Client,
	secretsRetriever SecretsRetriever,
	restartState *RestartState,
	existingRestartAnnotations map[string]string,
	checkpointInfoByComponent map[string]*checkpoint.CheckpointInfo,
) (*grovev1alpha1.PodCliqueSet, error) {
	gangSet := &grovev1alpha1.PodCliqueSet{}
	gangSet.Name = PCSNameForDGD(dynamoDeployment.Name, dynamoDeployment.Spec.Components)
	gangSet.Namespace = dynamoDeployment.Namespace
	gangSet.Labels = maps.Clone(dynamoDeployment.Spec.Labels)
	if gangSet.Labels == nil {
		gangSet.Labels = make(map[string]string)
	}
	gangSet.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = dynamoDeployment.Name
	gangSet.Annotations = maps.Clone(dynamoDeployment.Spec.Annotations)
	gangSet.Spec.Replicas = 1
	gangSet.Spec.Template.HeadlessServiceConfig = &grovev1alpha1.HeadlessServiceConfig{
		PublishNotReadyAddresses: true,
	}
	gangSet.Spec.Template.StartupType = ptr.To(grovev1alpha1.CliqueStartupTypeAnyOrder)
	if operatorConfig.Orchestrators.Grove.TerminationDelay.Duration > 0 {
		gangSet.Spec.Template.TerminationDelay = &operatorConfig.Orchestrators.Grove.TerminationDelay
	}

	// Inject deployment-level topology constraint (PCS template).
	// specToGroveTopologyConstraint returns nil when input is nil, so this is a no-op without TAS.
	gangSet.Spec.Template.TopologyConstraint = specToGroveTopologyConstraint(dynamoDeployment.Spec.TopologyConstraint)

	// Validate kai-scheduler queue once if kai-scheduler is enabled
	var validatedQueueName string
	if runtimeConfig.GroveEnabled && runtimeConfig.KaiSchedulerEnabled {
		var err error
		validatedQueueName, err = DetermineKaiSchedulerQueue(ctx, dynamoDeployment.Annotations)
		if err != nil {
			return nil, fmt.Errorf("failed to determine kai-scheduler queue: %w", err)
		}
	}

	discoveryBackend := controller_common.GetDiscoveryBackend(operatorConfig.Discovery.Backend, dynamoDeployment.Annotations)
	discoveryContext := NewDiscoveryContext(operatorConfig.Discovery.Backend, dynamoDeployment.Annotations)

	var scalingGroups []grovev1alpha1.PodCliqueScalingGroupConfig
	var resourceClaimTemplates []grovev1alpha1.ResourceClaimTemplateConfig

	for i := range dynamoDeployment.Spec.Components {
		component := dynamoDeployment.Spec.Components[i].DeepCopy()
		componentName := component.ComponentName
		dynamoNamespace := GetDynamoNamespace(dynamoDeployment, component)

		propagateDGDAnnotations(dynamoDeployment.GetAnnotations(), component)
		podTemplate := ensurePodTemplate(component)
		podTemplate.Labels[commonconsts.KubeLabelDynamoNamespace] = dynamoNamespace
		// Determine backend framework using hybrid approach
		backendFramework, err := getBackendFrameworkFromComponent(component, dynamoDeployment)
		if err != nil {
			return nil, fmt.Errorf("failed to determine backend framework for component %s: %w", componentName, err)
		}

		if discoveryBackend != "" {
			podTemplate.Annotations[commonconsts.KubeAnnotationDynamoDiscoveryBackend] = string(discoveryBackend)
		}

		// Get checkpoint info for this component if available.
		var checkpointInfo *checkpoint.CheckpointInfo
		if checkpointInfoByComponent != nil {
			checkpointInfo = checkpointInfoByComponent[componentName]
		}

		numberOfNodes := component.GetNumberOfNodes()
		isMultinode := numberOfNodes > 1
		isInterPodGMS := component.IsInterPodGMSEnabled()
		isInterPodFailover := component.IsInterPodFailoverEnabled()
		usesPCSG := isMultinode || isInterPodGMS
		roles := expandRolesForComponent(componentName, component.Replicas, numberOfNodes, component)
		var cliqueNames []string

		for _, r := range roles {
			clique, err := buildCliqueForRole(cliqueParams{
				r:                          r,
				component:                  component,
				backendFramework:           backendFramework,
				secretsRetriever:           secretsRetriever,
				dynamoDeployment:           dynamoDeployment,
				numberOfNodes:              numberOfNodes,
				operatorConfig:             operatorConfig,
				runtimeConfig:              runtimeConfig,
				componentName:              componentName,
				checkpointInfo:             checkpointInfo,
				isMultinode:                isMultinode,
				usesPCSG:                   usesPCSG,
				isInterPodGMS:              isInterPodGMS,
				isInterPodFailover:         isInterPodFailover,
				discoveryBackend:           discoveryBackend,
				discoveryContext:           discoveryContext,
				restartState:               restartState,
				existingRestartAnnotations: existingRestartAnnotations,
				validatedQueueName:         validatedQueueName,
				kubeClient:                 kubeClient,
				ctx:                        ctx,
			})
			if err != nil {
				return nil, err
			}
			gangSet.Spec.Template.Cliques = append(gangSet.Spec.Template.Cliques, clique)
			cliqueNames = append(cliqueNames, strings.ToLower(r.Name))
		}

		applyCliqueStartupDependencies(gangSet, roles, backendFramework, numberOfNodes, isInterPodGMS)

		if isInterPodGMS {
			configs, err := gmsResourceClaimTemplateConfigs(componentName, GetGPUMemoryService(component), GetMainContainerResources(component), roles)
			if err != nil {
				return nil, fmt.Errorf("failed to build GMS ResourceClaimTemplate configs for component %s: %w", componentName, err)
			}
			resourceClaimTemplates = append(resourceClaimTemplates, configs...)
		}

		if usesPCSG {
			pcsg := grovev1alpha1.PodCliqueScalingGroupConfig{
				Name:               strings.ToLower(componentName),
				CliqueNames:        cliqueNames,
				Replicas:           component.Replicas,
				MinAvailable:       ptr.To(int32(1)),
				TopologyConstraint: toGroveTopologyConstraint(component.TopologyConstraint),
			}
			if isInterPodGMS {
				pcsg.ResourceSharing = gmsResourceSharingEntries(componentName, roles)
			}
			scalingGroups = append(scalingGroups, pcsg)
		}
	}
	if len(scalingGroups) > 0 {
		gangSet.Spec.Template.PodCliqueScalingGroupConfigs = scalingGroups
	}
	if len(resourceClaimTemplates) > 0 {
		gangSet.Spec.Template.ResourceClaimTemplates = resourceClaimTemplates
	}

	return gangSet, nil
}

// generatePodSpecForRole builds the pod spec for a single role, handling GMS
// weight server pods and GMS engine pods differently from regular pods.
func generatePodSpecForRole(
	r ServiceRole,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
	numberOfNodes int32,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	serviceName string,
	checkpointInfo *checkpoint.CheckpointInfo,
) (*corev1.PodSpec, error) {
	isInterPodGMS := component.IsInterPodGMSEnabled()

	if r.Role == RoleGMS {
		// GMS weight server: generate a base engine spec then transform it
		basePodSpec, err := GeneratePodSpecForComponent(
			component, backendFramework, secretsRetriever, dynamoDeployment,
			RoleMain, 1, operatorConfig,
			commonconsts.MultinodeDeploymentTypeGrove, serviceName, checkpointInfo, nil,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to generate base podSpec for GMS: %w", err)
		}
		gpuCount, err := getGPUCount(GetMainContainerResources(component))
		if err != nil {
			return nil, fmt.Errorf("failed to get GPU count for GMS weight server: %w", err)
		}
		return gmsWeightServerPodSpec(basePodSpec, r.Rank, int(gpuCount)), nil
	}

	// Engine pod (or non-GMS pod): optionally use a rank-aware deployer for multinode inter-pod GMS
	var deployer MultinodeDeployer
	if isInterPodGMS && numberOfNodes > 1 {
		deployer = &GroveMultinodeDeployer{IsInterPodGMS: true, Rank: r.Rank}
	}

	podSpec, err := GeneratePodSpecForComponent(
		component, backendFramework, secretsRetriever, dynamoDeployment,
		r.Role, numberOfNodes, operatorConfig,
		commonconsts.MultinodeDeploymentTypeGrove, serviceName, checkpointInfo, deployer,
	)
	if err != nil {
		return nil, err
	}

	if isInterPodGMS {
		augmentEngineForGMS(podSpec, r.Rank, component.IsInterPodFailoverEnabled())
	}

	return podSpec, nil
}

func generateLabels(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
	componentName string,
	discovery DiscoveryContext,
) (map[string]string, error) {
	labels := make(map[string]string)
	labels[commonconsts.KubeLabelDynamoSelector] = GetDCDResourceName(dynamoDeployment, componentName, "")
	labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = dynamoDeployment.Name
	labels[commonconsts.KubeLabelDynamoComponent] = componentName
	if component.ComponentType != "" {
		labels[commonconsts.KubeLabelDynamoComponentType] = string(component.ComponentType)
	}
	if dynamoDeployment.HasEPPComponent() && IsWorkerComponent(string(component.ComponentType)) {
		labels[commonconsts.KubeLabelDynamoComponentClass] = commonconsts.ComponentClassWorker
	}
	if subComponentType := getDGDComponentAlphaSubComponentType(dynamoDeployment, componentName); subComponentType != "" {
		labels[commonconsts.KubeLabelDynamoSubComponentType] = subComponentType
	}
	labels[commonconsts.KubeLabelDynamoNamespace] = GetDynamoNamespace(dynamoDeployment, component)
	// Add base model label if modelRef is specified
	AddBaseModelLabel(labels, component.ModelRef)
	// Merge user-supplied labels first so they cannot overwrite checkpoint labels.
	setMetricsLabels(labels, dynamoDeployment)
	if dynamoDeployment.Spec.Labels != nil {
		if err := mergo.Merge(&labels, dynamoDeployment.Spec.Labels, mergo.WithOverride); err != nil {
			return nil, fmt.Errorf("failed to merge labels: %w", err)
		}
	}
	if componentLabels := getDGDComponentAlphaLabels(dynamoDeployment, componentName); componentLabels != nil {
		if err := mergo.Merge(&labels, componentLabels, mergo.WithOverride); err != nil {
			return nil, fmt.Errorf("failed to merge preserved component labels: %w", err)
		}
	}
	if podTemplateLabels := GetPodTemplateLabels(component); podTemplateLabels != nil {
		if err := mergo.Merge(&labels, podTemplateLabels, mergo.WithOverride); err != nil {
			return nil, fmt.Errorf("failed to merge podTemplate labels: %w", err)
		}
	}
	// Re-apply system labels after user merge to prevent override
	labels[commonconsts.KubeLabelDynamoSelector] = GetDCDResourceName(dynamoDeployment, componentName, "")
	labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = dynamoDeployment.Name
	labels[commonconsts.KubeLabelDynamoComponent] = componentName
	if component.ComponentType != "" {
		labels[commonconsts.KubeLabelDynamoComponentType] = string(component.ComponentType)
	}
	if dynamoDeployment.HasEPPComponent() && IsWorkerComponent(string(component.ComponentType)) {
		labels[commonconsts.KubeLabelDynamoComponentClass] = commonconsts.ComponentClassWorker
	}
	if subComponentType := getDGDComponentAlphaSubComponentType(dynamoDeployment, componentName); subComponentType != "" {
		labels[commonconsts.KubeLabelDynamoSubComponentType] = subComponentType
	}
	labels[commonconsts.KubeLabelDynamoNamespace] = GetDynamoNamespace(dynamoDeployment, component)
	if workerHash := GetPodTemplateLabels(component)[commonconsts.KubeLabelDynamoWorkerHash]; workerHash != "" {
		labels[commonconsts.KubeLabelDynamoWorkerHash] = workerHash
	}
	// Discovery labels on pod template — needed for Pod reflector filtering in
	// container mode (see lib/runtime/src/discovery/kube/daemon.rs). Applied to
	// every role by default because any role may host the dynamo runtime — for
	// example, multinode vLLM workers in data-parallel hybrid-lb mode run their
	// own API server (see RoleWorker branch in injectDataParallelLaunchFlags).
	// Callers that render non-dynamo pods (specifically the RoleGMS weight
	// server, which runs gpu_memory_service.cli.server and never registers a
	// DynamoWorkerMetadata CR) are responsible for stripping these labels after
	// the fact — see buildCliqueForRole.
	if discovery.Backend == configv1alpha1.DiscoveryBackendKubernetes {
		labels[commonconsts.KubeLabelDynamoDiscoveryBackend] = commonconsts.DiscoveryBackendKubernetes
		labels[commonconsts.KubeLabelDynamoDiscoveryEnabled] = commonconsts.KubeLabelValueTrue
	}
	return labels, nil
}

func getDGDComponentAlphaSubComponentType(dgd *v1beta1.DynamoGraphDeployment, componentName string) string {
	component := getDGDAlphaComponent(dgd, componentName)
	if component == nil {
		return ""
	}
	return component.SubComponentType
}

func getDGDComponentAlphaLabels(dgd *v1beta1.DynamoGraphDeployment, componentName string) map[string]string {
	component := getDGDAlphaComponent(dgd, componentName)
	if component == nil {
		return nil
	}
	return component.Labels
}

func getDGDComponentAlphaAnnotations(dgd *v1beta1.DynamoGraphDeployment, componentName string) map[string]string {
	component := getDGDAlphaComponent(dgd, componentName)
	if component == nil {
		return nil
	}
	return component.Annotations
}

func getDGDAlphaComponent(dgd *v1beta1.DynamoGraphDeployment, componentName string) *v1alpha1.DynamoComponentDeploymentSharedSpec {
	alpha := getDGDAlpha(dgd)
	if alpha == nil || alpha.Spec.Services == nil {
		return nil
	}
	return alpha.Spec.Services[componentName]
}

func getDGDAlpha(dgd *v1beta1.DynamoGraphDeployment) *v1alpha1.DynamoGraphDeployment {
	if dgd == nil {
		return nil
	}
	alpha := &v1alpha1.DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(dgd.DeepCopy()); err != nil {
		return nil
	}
	return alpha
}

// GetDGDPreservedAlphaPVCs returns legacy v1alpha1 top-level PVC declarations
// preserved on a converted v1beta1 DynamoGraphDeployment.
func GetDGDPreservedAlphaPVCs(dgd *v1beta1.DynamoGraphDeployment) []v1alpha1.PVC {
	alpha := getDGDAlpha(dgd)
	if alpha == nil || len(alpha.Spec.PVCs) == 0 {
		return nil
	}
	return append([]v1alpha1.PVC(nil), alpha.Spec.PVCs...)
}

func generateAnnotations(component *v1beta1.DynamoComponentDeploymentSharedSpec, dynamoDeployment *v1beta1.DynamoGraphDeployment, componentName string) (map[string]string, error) {
	annotations := make(map[string]string)
	if dynamoDeployment.Spec.Annotations != nil {
		if err := mergo.Merge(&annotations, dynamoDeployment.Spec.Annotations, mergo.WithOverride); err != nil {
			return nil, fmt.Errorf("failed to merge DGD annotations: %w", err)
		}
	}
	if componentAnnotations := getDGDComponentAlphaAnnotations(dynamoDeployment, componentName); componentAnnotations != nil {
		if err := mergo.Merge(&annotations, componentAnnotations, mergo.WithOverride); err != nil {
			return nil, fmt.Errorf("failed to merge preserved component annotations: %w", err)
		}
	}
	if podTemplateAnnotations := GetPodTemplateAnnotations(component); podTemplateAnnotations != nil {
		err := mergo.Merge(&annotations, podTemplateAnnotations, mergo.WithOverride)
		if err != nil {
			return nil, fmt.Errorf("failed to merge annotations: %w", err)
		}
	}
	return annotations, nil
}

// detectBackendFrameworkFromArgs detects the backend framework from command/args
func detectBackendFrameworkFromArgs(command []string, args []string) (BackendFramework, error) {
	// Combine command and args to search through all parts
	allParts := append(command, args...)
	fullCommand := strings.Join(allParts, " ")

	// Pattern to match python -m dynamo.{backend}.something
	patterns := map[BackendFramework]*regexp.Regexp{
		BackendFrameworkVLLM:   regexp.MustCompile(`python[0-9.]*\s+[^|&;]*-m\s+[^|&;]*dynamo\.vllm[^|&;]*`),
		BackendFrameworkSGLang: regexp.MustCompile(`python[0-9.]*\s+[^|&;]*-m\s+[^|&;]*dynamo\.sglang[^|&;]*`),
		BackendFrameworkTRTLLM: regexp.MustCompile(`python[0-9.]*\s+[^|&;]*-m\s+[^|&;]*dynamo\.trtllm[^|&;]*`),
	}

	var detected []BackendFramework
	for framework, pattern := range patterns {
		if pattern.MatchString(fullCommand) {
			detected = append(detected, framework)
		}
	}

	if len(detected) == 0 {
		return BackendFrameworkNoop, nil
	}

	if len(detected) > 1 {
		return "", fmt.Errorf("multiple backend frameworks detected from command: %v in %q", detected, fullCommand)
	}

	return detected[0], nil
}

// determineBackendFramework is the core logic for hybrid backend framework detection
// Takes extracted parameters and applies the detection logic
func determineBackendFramework(
	componentType string,
	command []string,
	args []string,
	explicitBackendFramework string,
) (BackendFramework, error) {
	// Check if this is a worker component - if not, use noop backend
	if !IsWorkerComponent(componentType) {
		return BackendFrameworkNoop, nil
	}

	// Worker component - apply backend framework detection
	var detectedFramework BackendFramework
	var detectionError error

	// Try to detect from command/args
	if len(command) > 0 || len(args) > 0 {
		detected, err := detectBackendFrameworkFromArgs(command, args)
		if err == nil {
			detectedFramework = detected
		} else {
			detectionError = err
		}
	}

	// Get explicit framework
	var explicitFramework BackendFramework
	if explicitBackendFramework != "" {
		explicitFramework = BackendFramework(explicitBackendFramework)
	}

	// Validate consistency if both detected and explicit exist
	if detectedFramework != "" && detectedFramework != BackendFrameworkNoop && explicitFramework != "" && detectedFramework != explicitFramework {
		return "", fmt.Errorf("backend framework mismatch: detected %q from command but explicitly configured as %q",
			detectedFramework, explicitFramework)
	}

	// Return in order of preference: detected > explicit > error
	if detectedFramework != "" && detectedFramework != BackendFrameworkNoop {
		return detectedFramework, nil
	}

	if explicitFramework != "" {
		return explicitFramework, nil
	}

	// If we couldn't detect and no explicit config, return error
	if detectionError != nil {
		return "", fmt.Errorf("could not determine backend framework: %w", detectionError)
	}

	// No command/args to detect from and no explicit config
	return BackendFrameworkNoop, nil
}

// getBackendFrameworkFromComponent attempts to determine backend framework using hybrid approach:
// 1. Check if component is a worker - if not, return noop
// 2. For workers: try to detect from command/args, fall back to explicit config
// 3. Return error if worker has neither detection nor explicit config
// Also validates consistency between detected and explicit if both exist
func getBackendFrameworkFromComponent(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
) (BackendFramework, error) {
	// Extract command/args from component
	var command, args []string
	if main := GetMainContainer(component); main != nil {
		command = main.Command
		args = main.Args
	}

	// Extract explicit backend framework from deployment
	explicitBackendFramework := dynamoDeployment.Spec.BackendFramework

	return determineBackendFramework(
		string(component.ComponentType),
		command,
		args,
		explicitBackendFramework,
	)
}

// ConvertDynamoComponentDeploymentToSpec converts a DynamoComponentDeployment to our component spec interface
// This is a helper for the controller to use our backend logic
func ConvertDynamoComponentDeploymentToSpec(dynComponent *v1beta1.DynamoComponentDeployment) *v1beta1.DynamoComponentDeploymentSharedSpec {
	return dynComponent.Spec.DynamoComponentDeploymentSharedSpec.DeepCopy()
}

// GetBackendFrameworkFromDynamoComponent determines backend framework for a DynamoComponentDeployment
func GetBackendFrameworkFromDynamoComponent(dynComponent *v1beta1.DynamoComponentDeployment) (BackendFramework, error) {
	// Extract command/args from component
	var command, args []string
	if main := GetMainContainer(&dynComponent.Spec.DynamoComponentDeploymentSharedSpec); main != nil {
		command = main.Command
		args = main.Args
	}

	// Extract explicit backend framework
	explicitBackendFramework := dynComponent.Spec.BackendFramework

	return determineBackendFramework(
		string(dynComponent.Spec.ComponentType),
		command,
		args,
		explicitBackendFramework,
	)
}

type GenerateBasePodSpecForControllerOptions struct {
	// WorkloadComponentType overrides the component type used for pod workload
	// defaults and env vars. Empty means use dynComponent.Spec.ComponentType.
	WorkloadComponentType v1beta1.ComponentType
}

// GenerateBasePodSpecForController generates a PodSpec using backend logic for controller usage
// This preserves the base pod generation while allowing controller-specific enhancements
func GenerateBasePodSpecForController(
	dynComponent *v1beta1.DynamoComponentDeployment,
	secretsRetriever SecretsRetriever,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	role Role,
	multinodeDeploymentType commonconsts.MultinodeDeploymentType,
	checkpointInfo *checkpoint.CheckpointInfo, // Optional checkpoint info (resolved by caller)
	options GenerateBasePodSpecForControllerOptions,
) (*corev1.PodSpec, error) {
	// Convert to our interface
	componentSpec := ConvertDynamoComponentDeploymentToSpec(dynComponent)
	if options.WorkloadComponentType != "" {
		componentSpec.ComponentType = options.WorkloadComponentType
	}
	if workerHash := dynComponent.GetLabels()[commonconsts.KubeLabelDynamoWorkerHash]; workerHash != "" && IsWorkerComponent(string(componentSpec.ComponentType)) {
		ensurePodTemplate(componentSpec).Labels[commonconsts.KubeLabelDynamoWorkerHash] = workerHash
	}

	numberOfNodes := componentSpec.GetNumberOfNodes()

	// Determine backend framework using hybrid approach
	backendFramework, err := GetBackendFrameworkFromDynamoComponent(dynComponent)
	if err != nil {
		return nil, fmt.Errorf("failed to determine backend framework: %w", err)
	}

	parentGraphDeploymentName := dynComponent.GetParentGraphDeploymentName()
	if parentGraphDeploymentName == "" {
		parentGraphDeploymentName = dynComponent.Name
	}

	// Generate base PodSpec with standard env vars using merged component envs
	componentName := GetDCDComponentName(dynComponent)
	podSpec, err := GenerateBasePodSpec(
		componentSpec,
		backendFramework,
		secretsRetriever,
		parentGraphDeploymentName,
		dynComponent.Namespace,
		role,
		numberOfNodes,
		operatorConfig,
		multinodeDeploymentType,
		componentName,
		checkpointInfo,
		nil, // use default deployer
	)
	if err != nil {
		return nil, err
	}

	return podSpec, nil
}

// getDefaultCompilationCacheMountPoint returns the default mount point for compilation cache based on backend framework
func getDefaultCompilationCacheMountPoint(backendFramework BackendFramework) string {
	switch backendFramework {
	case BackendFrameworkVLLM:
		return commonconsts.DefaultVLLMCacheMountPoint
	case BackendFrameworkSGLang, BackendFrameworkTRTLLM:
		// SGLang and TensorRT-LLM don't currently support compilation caches
		// Return empty string as these should not be used
		return ""
	default:
		// For unknown backends, don't assume compilation cache support
		return ""
	}
}
