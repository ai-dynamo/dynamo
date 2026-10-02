/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	"k8s.io/apimachinery/pkg/util/validation"
)

const (
	lpxScalingGroupTemplateName = "lpx"
	conductorTemplateName       = "cond"
	maxWorkloadReplicas         = 2496
	minGroupNameLength          = 6
)

// MaxPodCliqueSetNameLength preserves the established PCS identity budget.
// nameResources separately validates the extra names needed by multiple workloads.
const MaxPodCliqueSetNameLength = commonconsts.MaxCombinedGroveResourceNameLength -
	len(lpxScalingGroupTemplateName) - len(conductorTemplateName)

// Workload is the immutable result of resolving one LPX component group: its
// models in canonical runtime order and the Grove and runtime resource names that
// materialize them inside one PodCliqueSet scaling group.
// Its methods require a successfully resolved, non-nil workload.
type Workload struct {
	// name is the conductor component, which names the group within a shared PCS.
	name   string
	models []*Model
	digest WorkloadDigest
	// scalingGroupReplicas is the authored capacity, or its seed when capacity is external.
	scalingGroupReplicas int32
	// minAvailable seeds the scaling-group template and is its gang availability threshold.
	minAvailable int32

	// scalingGroupTemplate and scalingGroup are the PCSG template and materialized names.
	scalingGroupTemplate string
	scalingGroup         string
	// resourcePrefix scopes runtime ConfigMaps and discovery Services to this workload.
	resourcePrefix string
	// conductorTemplate is the Nova conductor or, for hybrid pipelines, Cyborg clique template.
	conductorTemplate string
	// conductorReplicas is the conductor clique's Pod count: one Nova conductor, or
	// the authored Cyborg width, defaulting to one complete client group.
	conductorReplicas int32
}

// Name returns the conductor component that names this workload.
func (w *Workload) Name() string {
	return w.name
}

// ScalingGroup returns the materialized PodCliqueScalingGroup name.
func (w *Workload) ScalingGroup() string {
	return w.scalingGroup
}

// ResourcePrefix returns the prefix of this workload's ConfigMaps and Services.
func (w *Workload) ResourcePrefix() string {
	return w.resourcePrefix
}

// ConductorTemplate returns the conductor clique template: the Nova conductor for
// LPU-only pipelines and the Cyborg workers for hybrid pipelines.
func (w *Workload) ConductorTemplate() string {
	return w.conductorTemplate
}

// Models returns a read-only view of the workload's models in canonical runtime order.
// Callers must not modify the slice or its models.
func (w *Workload) Models() []*Model {
	return w.models
}

// Digest returns the workload's digest.
func (w *Workload) Digest() WorkloadDigest {
	return w.digest
}

// Pipeline returns the workload's runtime pipeline.
func (w *Workload) Pipeline() Pipeline {
	return w.models[0].component.pipeline
}

// ComponentNames returns this workload's members in canonical runtime order.
func (w *Workload) ComponentNames() []string {
	// Draft fanout can contribute several consecutive models from one component.
	var names []string
	for _, model := range w.models {
		if len(names) == 0 || names[len(names)-1] != model.component.name {
			names = append(names, model.component.name)
		}
	}
	return names
}

// nameResources assigns the workload's Grove and runtime resource names inside
// PodCliqueSet pcsName. group names the workload within a PCS that holds several
// workloads; it is empty for a sole workload, which keeps the unscoped names.
// The resolver calls it once, before publishing the workload.
func (w *Workload) nameResources(pcsName, group string) error {
	// Leave room for the fixed scaling group and every LPX role in Grove's name budget.
	if strings.TrimSpace(pcsName) == "" {
		return fmt.Errorf("PodCliqueSet name is required")
	}
	if len(pcsName) > MaxPodCliqueSetNameLength {
		return fmt.Errorf("PodCliqueSet name %q exceeds the LPX maximum of %d characters", pcsName, MaxPodCliqueSetNameLength)
	}

	// Each model has its own Agent clique, numbered only when the workload has several.
	agentTemplates := make([]string, len(w.models))
	for index := range agentTemplates {
		agentTemplates[index] = "agt"
		if len(w.models) > 1 {
			agentTemplates[index] = fmt.Sprintf("agt%d", index)
		}
	}

	// Grouped workloads reserve the group name in both the scaling group and its longest role.
	scalingGroupTemplate, resourcePrefix, rolePrefix := lpxScalingGroupTemplateName, pcsName, ""
	if group != "" {
		roleLength := len(conductorTemplateName)
		for _, template := range agentTemplates {
			roleLength = max(roleLength, len(template))
		}
		name, err := boundedGroupName(group, (commonconsts.MaxCombinedGroveResourceNameLength-len(pcsName)-roleLength-1)/2)
		if err != nil {
			return fmt.Errorf("naming group %q in PCS %q: %w", group, pcsName, err)
		}

		// ConfigMaps and Services retain more of the group name than Grove allows.
		const maxResourcePrefixLength = validation.DNS1123LabelMaxLength - len("-serve")
		resourceName, err := boundedGroupName(group, maxResourcePrefixLength-len(pcsName)-1)
		if err != nil {
			return err
		}
		scalingGroupTemplate, resourcePrefix, rolePrefix = name, pcsName+"-"+resourceName, name+"-"
	}

	// Materialize every name beneath Grove's replica-zero scaling-group identity.
	w.scalingGroupTemplate = scalingGroupTemplate
	w.scalingGroup = grovecommon.GeneratePodCliqueScalingGroupName(
		grovecommon.ResourceNameReplica{Name: pcsName, Replica: 0}, scalingGroupTemplate,
	)
	w.resourcePrefix = resourcePrefix
	w.conductorTemplate = rolePrefix + conductorTemplateName
	for index, model := range w.models {
		model.agentTemplate = rolePrefix + agentTemplates[index]
	}
	if err := w.ValidateReplicas(w.scalingGroupReplicas); err != nil {
		return err
	}

	// Bound Cyborg hostnames at the authored capacity or initial external seed.
	if w.Pipeline() == PipelineHybrid {
		return w.validatePodHostname("Cyborg", w.conductorTemplate, w.scalingGroupReplicas, int(w.conductorReplicas)-1)
	}
	return nil
}

// ValidateReplicas bounds the workload's scaling-group replicas and checks that
// the last replica's Nova conductor and Agent Pod hostnames are valid DNS labels.
func (w *Workload) ValidateReplicas(replicas int32) error {
	if replicas < 0 || replicas > maxWorkloadReplicas {
		return fmt.Errorf("LPX replica count must be between 0 and %d", maxWorkloadReplicas)
	}
	if w.Pipeline() != PipelineHybrid {
		if err := w.validatePodHostname("conductor", w.conductorTemplate, replicas, 0); err != nil {
			return err
		}
	}
	for _, model := range w.models {
		if err := w.validatePodHostname("Agent", model.agentTemplate, replicas, model.component.agentReplicas-1); err != nil {
			return err
		}
	}
	return nil
}

// ConductorCliqueName returns the materialized conductor PodClique name in one
// scaling-group replica: the Nova conductor, or the Cyborg workers for hybrid pipelines.
func (w *Workload) ConductorCliqueName(replica int32) string {
	return w.cliqueName(w.conductorTemplate, replica)
}

// cliqueName returns the materialized PodClique name of template in one scaling-group replica.
func (w *Workload) cliqueName(template string, replica int32) string {
	return grovecommon.GeneratePodCliqueName(
		grovecommon.ResourceNameReplica{Name: w.scalingGroup, Replica: int(replica)},
		template,
	)
}

// validatePodHostname checks the hostname of podIndex in the last of replicas.
func (w *Workload) validatePodHostname(role, template string, replicas int32, podIndex int) error {
	cliqueName := w.cliqueName(template, max(0, replicas-1))
	hostname := podHostname(cliqueName, podIndex)
	if problems := validation.IsDNS1123Label(hostname); len(problems) != 0 {
		return fmt.Errorf(
			"materialized %s Pod hostname %q is invalid: %s",
			role,
			hostname,
			strings.Join(problems, "; "),
		)
	}
	return nil
}

func podHostname(cliqueName string, podIndex int) string {
	return fmt.Sprintf("%s-%d", cliqueName, podIndex)
}

// boundedGroupName preserves admitted component names when they fit. Hashes retain
// the original identity when lowercasing or shortening; at least one readable
// character and four hash characters must fit alongside the separator.
func boundedGroupName(component string, maxLength int) (string, error) {
	name := strings.ToLower(component)
	if name == component && len(name) <= maxLength {
		return name, nil
	}
	if maxLength < minGroupNameLength {
		return "", fmt.Errorf("component name needs more than %d characters; shorten the deployment name", maxLength)
	}

	// Prefer eight hash characters, reducing only for Grove's tighter name budget.
	digest := sha256.Sum256([]byte(component))
	suffix := fmt.Sprintf("%x", digest[:4])[:min(8, maxLength-2)]
	prefixLength := min(len(name), maxLength-len(suffix)-1)
	return strings.TrimRight(name[:prefixLength], "-") + "-" + suffix, nil
}
