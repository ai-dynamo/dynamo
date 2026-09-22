// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	consts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	deploymentUIDAnnotation = dynamo.LPXDeploymentUIDAnnotation
	deploymentUIDLabel      = deploymentUIDAnnotation
)

// resolveWorkloads resolves every component group and finalizes its resource names.
// deployment and dgd must be non-nil; dgd must have passed admission.
// Returned maps share conductor component keys. Inputs are not mutated.
func (r *graphReconciler) resolveWorkloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
) (map[string]*lpx.Workload, map[string]*lpx.MaterializationPlan, error) {
	var (
		workloads = make(map[string]*lpx.Workload)
		plans     = make(map[string]*lpx.MaterializationPlan)
		groups    = lpx.ComponentGroups(dgd)
	)

	for _, groupName := range slices.Sorted(maps.Keys(groups)) {
		workload, err := lpx.ResolveWorkload(ctx, dgd, groups[groupName], r.modelRegistry)
		if err != nil {
			return nil, nil, err
		}
		workloads[groupName] = workload

		plan, err := workload.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(deployment))
		if err != nil {
			return nil, nil, err
		}

		// Only independent workloads need additional names within the shared PCS.
		if len(groups) > 1 {
			plan, err = plan.WithGroup(groupName)
			if err != nil {
				return nil, nil, err
			}
		}

		// Compare finalized identities only against previously resolved groups.
		for previousGroup, previous := range plans {
			if previous.ScalingGroupTemplate == plan.ScalingGroupTemplate || previous.ResourcePrefix == plan.ResourcePrefix {
				return nil, nil, fmt.Errorf("LPX components %q and %q resolve to conflicting resource names; rename one component", previousGroup, groupName)
			}
		}
		plans[groupName] = plan
	}

	return workloads, plans, nil
}

// renderPodCliqueSet composes resolved workloads into one Grove envelope with
// runtime ConfigMaps and, for Kubernetes discovery, serving Services.
// Inputs must be non-nil and workloads must contain every component group.
// Plans must have finalized names and use the same keys as workloads.
// Inputs remain read-only.
func (r *graphReconciler) renderPodCliqueSet(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	workloads map[string]*lpx.Workload,
	plans map[string]*lpx.MaterializationPlan,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	// Shared defaults and queue resolution belong to the single PCS envelope.
	pcs, err := dynamo.RenderLPXPodCliqueSet(ctx, dgd, r.config, r.runtimeConfig, dynamo.PCSNameForLPX(deployment))
	if err != nil {
		return nil, nil, err
	}

	var (
		resources []client.Object
		digest    = sha256.New()
	)

	for _, groupName := range slices.Sorted(maps.Keys(plans)) {
		workload, plan := workloads[groupName], plans[groupName]

		// Render this workload's roles using the full graph for shared defaults.
		rendered, err := dynamo.RenderLPXWorkloadTemplates(dgd, r.config, r.runtimeConfig, r.dockerSecretRetriever, workload, plan)
		if err != nil {
			return nil, nil, err
		}

		// Discovery selects only this workload's conductor; all auxiliary roles stay private.
		serving := plan.ConductorTemplate
		if serving == "" {
			serving = plan.CyborgTemplate
		}
		for _, clique := range rendered.Cliques {
			if token := deployment.Annotations[dynamo.LPXRestartAnnotation]; token != "" {
				clique.Annotations[consts.RestartAnnotation] = token
			}
			delete(clique.Labels, dynamo.LPXServingLabel)
			if clique.Name == serving {
				clique.Labels[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
			} else {
				delete(clique.Labels, consts.KubeLabelDynamoDiscoveryEnabled)
				delete(clique.Labels, consts.KubeLabelDynamoDiscoveryBackend)
				delete(clique.Labels, consts.KubeLabelDynamoBaseModelHash)
			}
		}

		// Component order fixes both rendering and the graph's immutable workload digest.
		pcs.Spec.Template.Cliques = append(pcs.Spec.Template.Cliques, rendered.Cliques...)
		pcs.Spec.Template.PodCliqueScalingGroupConfigs = append(pcs.Spec.Template.PodCliqueScalingGroupConfigs, rendered.ScalingGroup)
		resources = append(resources, rendered.Resources...)

		// Kubernetes discovery exposes only this workload's serving role within its PCS.
		if commoncontroller.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, dgd.Annotations) {
			component := dgd.GetComponentByName(workload.ServingComponentName())
			service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
				ServiceName: plan.ResourcePrefix + "-serve", Namespace: deployment.Namespace,
				ComponentType: string(component.ComponentType), ComponentName: component.ComponentName,
				DynamoNamespace: dgd.GetDynamoNamespaceForComponent(component), IsK8sDiscovery: true,
				Labels:      dynamo.GetDGDComponentResourceLabels(dgd, component.ComponentName, component),
				Annotations: dynamo.GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component),
			})
			if err != nil {
				return nil, nil, err
			}
			service.Spec.Selector[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
			service.Spec.Selector[grovecommon.LabelPartOfKey] = pcs.Name
			resources = append(resources, service)
		}

		writeIdentityHashField(digest, groupName, workload.Digest().String())
	}

	pcs.Annotations[lpx.WorkloadDigestAnnotation] = fmt.Sprintf("sha256:%x", digest.Sum(nil))
	stampDeploymentIdentity(deployment, pcs, resources)

	// Enforce the aggregate size budget after identity and discovery metadata are final.
	serialized, err := json.Marshal(pcs)
	if err != nil {
		return nil, nil, fmt.Errorf("serializing selected LPX PodCliqueSet: %w", err)
	}
	if len(serialized) > lpx.MaxRenderedPodCliqueSetBytes {
		return nil, nil, fmt.Errorf("rendered LPX PodCliqueSet is %d bytes; maximum is %d", len(serialized), lpx.MaxRenderedPodCliqueSetBytes)
	}
	return pcs, resources, nil
}

// stampDeploymentIdentity propagates stable ownership labels and annotations, never DGD revision.
func stampDeploymentIdentity(deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, resources []client.Object) {
	// The initial DGD lookup validated this controller owner; rendering trusts that observation.
	dgdOwner := metav1.GetControllerOf(deployment)
	identity := map[string]string{
		lpx.DeploymentNameAnnotation: deployment.Name,
		deploymentUIDAnnotation:      string(deployment.UID),
		lpx.DGDUIDAnnotation:         string(dgdOwner.UID),
	}
	stamp := func(annotations *map[string]string) {
		if *annotations == nil {
			*annotations = make(map[string]string)
		}
		delete(*annotations, lpx.DGDGenerationAnnotation)
		maps.Copy(*annotations, identity)
	}
	stampOwnerLabel := func(object client.Object) {
		labels := object.GetLabels()
		if labels == nil {
			labels = make(map[string]string)
		}
		labels[deploymentUIDLabel] = string(deployment.UID)
		object.SetLabels(labels)
	}
	stamp(&pcs.Annotations)
	stampOwnerLabel(pcs)
	for _, clique := range pcs.Spec.Template.Cliques {
		stamp(&clique.Annotations)
	}
	for i := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		stamp(&pcs.Spec.Template.PodCliqueScalingGroupConfigs[i].Annotations)
	}
	for _, resource := range resources {
		annotations := resource.GetAnnotations()
		stamp(&annotations)
		resource.SetAnnotations(annotations)
		stampOwnerLabel(resource)
	}
}
