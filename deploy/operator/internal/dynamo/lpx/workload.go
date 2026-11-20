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
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
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
const MaxPodCliqueSetNameLength = commonconsts.MaxCombinedGroveResourceNameLength -
	len(lpxScalingGroupTemplateName) - len(conductorTemplateName)

// Workload is the immutable result of resolving one LPX component group
// for scheduling and runtime materialization.
// Its methods require a successfully resolved, non-nil workload with projections.
type Workload struct {
	modelProjections []*ModelProjection
	digest           WorkloadDigest
	// group and rolePrefix are empty for the sole workload in a PCS, which keeps unscoped names.
	group                string
	scalingGroupTemplate string
	scalingGroup         string
	resourcePrefix       string
	rolePrefix           string
	// cyborgReplicas is the Cyborg width of a hybrid workload, and zero for any other workload.
	cyborgReplicas int32
}

// newWorkload names one projected component group in its PodCliqueSet and bounds its Pod hostnames.
// shared reports that other workloads use the same PCS. projections is nonempty and in canonical
// runtime order. cyborgReplicas is positive for a hybrid workload, and zero for any other workload.
// Inputs are not mutated.
func newWorkload(
	pcsName string,
	shared bool,
	projections []*ModelProjection,
	scalingGroupReplicas, cyborgReplicas int32,
) (*Workload, error) {
	// Leave room for the fixed scaling group and every LPX role in Grove's name budget.
	if strings.TrimSpace(pcsName) == "" {
		return nil, fmt.Errorf("PodCliqueSet name is required")
	}
	if len(pcsName) > MaxPodCliqueSetNameLength {
		return nil, fmt.Errorf("PodCliqueSet name %q exceeds the LPX maximum of %d characters", pcsName, MaxPodCliqueSetNameLength)
	}

	// The sole workload in a PCS keeps unscoped names; the others add their serving component.
	var group, scalingGroupTemplate, resourcePrefix, rolePrefix string
	if !shared {
		scalingGroupTemplate, resourcePrefix = lpxScalingGroupTemplateName, pcsName
	} else {
		// The group name scopes the scaling group and, a second time, its longest role.
		roleLength := len(agentRole(len(projections)-1, len(projections)))
		if cyborgReplicas > 0 {
			roleLength = max(roleLength, len(conductorTemplateName))
		}
		serving := projections[len(projections)-1].stage
		name, err := boundedGroupName(serving, (commonconsts.MaxCombinedGroveResourceNameLength-len(pcsName)-roleLength-1)/2)
		if err != nil {
			return nil, fmt.Errorf("naming group %q in PCS %q: %w", serving, pcsName, err)
		}

		// Services retain more of the group name than Grove allows.
		const maxResourcePrefixLength = validation.DNS1123LabelMaxLength - len("-serve")
		resourceName, err := boundedGroupName(serving, maxResourcePrefixLength-len(pcsName)-1)
		if err != nil {
			return nil, err
		}
		group, scalingGroupTemplate, rolePrefix, resourcePrefix = serving, name, name+"-", pcsName+"-"+resourceName
	}

	// Canonical roles expand into default or draft0..draft7 followed by target.
	digest, err := workloadSetDigest(projections)
	if err != nil {
		return nil, err
	}
	w := &Workload{
		modelProjections: projections, digest: digest, cyborgReplicas: cyborgReplicas,
		group: group, scalingGroupTemplate: scalingGroupTemplate, resourcePrefix: resourcePrefix, rolePrefix: rolePrefix,
		scalingGroup: grovecommon.GeneratePodCliqueScalingGroupName(
			grovecommon.ResourceNameReplica{Name: pcsName, Replica: 0}, scalingGroupTemplate,
		),
	}

	// Bound the authored capacity, then each role's longest Pod name at any supported capacity.
	if err := w.ValidateReplicas(scalingGroupReplicas); err != nil {
		return nil, err
	}
	if cyborgReplicas > 0 {
		if err := w.validatePodHostname("Cyborg", w.CyborgTemplate(), int(cyborgReplicas)); err != nil {
			return nil, err
		}
	}
	for model, projection := range projections {
		if projection.agentReplicas == 0 {
			continue
		}
		if err := w.validatePodHostname("Agent", w.agentTemplate(model), projection.agentReplicas); err != nil {
			return nil, err
		}
	}
	return w, nil
}

// Group returns the component group that scopes this workload's identities in a shared PCS.
// It is empty for a sole workload, which keeps unscoped identities.
func (w *Workload) Group() string {
	return w.group
}

// ScalingGroup returns the materialized Grove scaling-group name.
func (w *Workload) ScalingGroup() string {
	return w.scalingGroup
}

// ResourcePrefix returns the prefix for this workload's discovery Service.
func (w *Workload) ResourcePrefix() string {
	return w.resourcePrefix
}

// CyborgTemplate returns the Cyborg template of a hybrid workload, or empty for any other workload.
func (w *Workload) CyborgTemplate() string {
	if w.cyborgReplicas == 0 {
		return ""
	}
	return w.rolePrefix + conductorTemplateName
}

// CyborgReplicas returns the resolved Cyborg width of a hybrid workload, or zero for any other workload.
func (w *Workload) CyborgReplicas() int32 {
	return w.cyborgReplicas
}

// ServingTemplate returns the Cyborg template or the serving model's Agent template.
func (w *Workload) ServingTemplate() string {
	if w.cyborgReplicas > 0 {
		return w.CyborgTemplate()
	}
	return w.agentTemplate(len(w.modelProjections) - 1)
}

// ModelProjections returns a read-only view of the workload's model projections.
// Callers must not modify the slice or its projections.
func (w *Workload) ModelProjections() []*ModelProjection {
	return w.modelProjections
}

// Digest returns the workload's digest.
func (w *Workload) Digest() WorkloadDigest {
	return w.digest
}

// Pipeline returns the workload's runtime pipeline.
func (w *Workload) Pipeline() Pipeline {
	return w.modelProjections[0].pipeline
}

// ComponentNames returns this workload's members in canonical runtime order.
func (w *Workload) ComponentNames() []string {
	// Draft fanout can contribute several consecutive models from one component.
	var names []string
	for _, projection := range w.modelProjections {
		if len(names) == 0 || names[len(names)-1] != projection.stage {
			names = append(names, projection.stage)
		}
	}
	return names
}

// ServingComponentName returns the component that serves the endpoint, placed last by ResolveWorkloads.
func (w *Workload) ServingComponentName() string {
	return w.modelProjections[len(w.modelProjections)-1].stage
}

// ValidateCyborgReplicas checks an observed Cyborg clique's width and its longest Pod hostname.
// The receiver must describe a hybrid workload.
func (w *Workload) ValidateCyborgReplicas(replicas int32) error {
	if err := validateCyborgReplicas(w.modelProjections[0].configuredBuild, replicas); err != nil {
		return err
	}
	return w.validatePodHostname("Cyborg", w.CyborgTemplate(), int(replicas))
}

// ValidateReplicas bounds a scaling-group replica count.
// Resolution has checked the Pod hostnames for every count in this range.
func (w *Workload) ValidateReplicas(replicas int32) error {
	if replicas < 0 || replicas > maxWorkloadReplicas {
		return fmt.Errorf("LPX replica count must be between 0 and %d", maxWorkloadReplicas)
	}
	return nil
}

// CyborgCliqueName returns the Cyborg clique in one workload replica.
// The receiver must describe a hybrid workload.
func (w *Workload) CyborgCliqueName(replica int32) string {
	return w.cliqueName(w.CyborgTemplate(), replica)
}

// RequestSpec returns a fresh request for one model in one workload replica.
// model indexes ModelProjections. The workload is not mutated.
func (w *Workload) RequestSpec(model int, replica int32) lpxv1alpha1.LPUPipelineRequestSpec {
	spec := w.modelProjections[model].requestSpec()
	spec.MaterializationTarget = lpxv1alpha1.MaterializationTarget{
		PodCliqueSetReplicaIndex: 0,
		PodCliqueScalingGroupRef: &lpxv1alpha1.PodCliqueScalingGroupReference{
			Name: w.scalingGroup, ReplicaIndex: int64(replica),
		},
	}
	spec.NodeLocal.AgentPodCliqueRef = lpxv1alpha1.PodCliqueReference{Name: w.cliqueName(w.agentTemplate(model), replica)}
	if w.Pipeline() == PipelineHybrid {
		spec.CyborgPodCliqueRef = &lpxv1alpha1.PodCliqueReference{Name: w.CyborgCliqueName(replica)}
	}
	return spec
}

// agentTemplate returns the Agent template of one model. model indexes ModelProjections.
func (w *Workload) agentTemplate(model int) string {
	return w.rolePrefix + agentRole(model, len(w.modelProjections))
}

// cliqueName returns the materialized template name in one workload replica.
func (w *Workload) cliqueName(template string, replica int32) string {
	return grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: w.scalingGroup, Replica: int(replica)}, template)
}

// validatePodHostname checks a role's last Pod name in the highest workload replica.
// No supported replica count or lower Pod index gives a longer name.
func (w *Workload) validatePodHostname(role, template string, replicas int) error {
	hostname := fmt.Sprintf("%s-%d", w.cliqueName(template, maxWorkloadReplicas-1), replicas-1)
	if problems := validation.IsDNS1123Label(hostname); len(problems) != 0 {
		return fmt.Errorf("materialized %s Pod hostname %q is invalid: %s", role, hostname, strings.Join(problems, "; "))
	}
	return nil
}

// agentRole returns the Agent role of one model, numbered only in a multi-model workload.
func agentRole(model, models int) string {
	if models == 1 {
		return "agt"
	}
	return fmt.Sprintf("agt%d", model)
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
