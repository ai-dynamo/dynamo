/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	"k8s.io/apimachinery/pkg/util/validation"
)

const minimumHashedCliqueNameLength = 8

const maximumCyborgPodIndex = math.MaxInt32

// ExpectedAgent binds one projected model to the exact Grove
// PodClique identity that must materialize it. The controller uses this plan
// to observe objects; it never reconstructs a row from a generated Pod name.
type ExpectedAgent struct {
	// CliqueName is the materialized Grove PodClique name for the selected replica.
	CliqueName string
	// TemplateName is the model's Grove PodClique template name.
	TemplateName string
	// Replicas is the number of Agent pods in the clique.
	Replicas int
}

// MaterializationPlan is the deterministic identity projection shared by the
// renderer and the controller's API-object observer.
type MaterializationPlan struct {
	// PodCliqueSetName is the actual PCS used for every generated child address.
	PodCliqueSetName string
	// ConductorClique is the materialized conductor PodClique name, if present.
	ConductorClique string
	// ConductorTemplate is the conductor PodClique template name, if present.
	ConductorTemplate string
	// Agents contains the expected materialized Agent identities.
	Agents []ExpectedAgent
	// CyborgClique is the materialized Cyborg PodClique name, if present.
	CyborgClique string
	// CyborgTemplate is the Cyborg PodClique template name, if present.
	CyborgTemplate string
	// LPXScalingGroup is the materialized Grove scaling-group name.
	LPXScalingGroup string
	// LPXScalingGroupTemplate is the Grove scaling-group template name.
	LPXScalingGroupTemplate string
	// ReplicaIndex is the scaling-group replica represented by this projection.
	ReplicaIndex int32
	// Replicas is the total scaling-group replica count.
	Replicas int32
}

// PlanNodeLocalMaterialization derives the exact identities shared by graph
// rendering and lifecycle observation without mutating the workload.
func (w *SelectedWorkload) PlanNodeLocalMaterialization(pcsName string) (*MaterializationPlan, error) {
	componentName := w.LPXComponentName()
	cyborgTemplateName := w.CyborgTemplateName()
	if strings.TrimSpace(pcsName) == "" || strings.TrimSpace(componentName) == "" {
		return nil, fmt.Errorf("PodCliqueSet and selected LPX component names are required")
	}

	conductorTemplate := ""
	if w.Pipeline() != PipelineLPX {
		conductorTemplate = strings.ToLower(fmt.Sprintf("%s-%s", componentName, commonconsts.GroveRoleSuffixLeader))
		if problems := validation.IsDNS1123Label(conductorTemplate); len(problems) != 0 {
			return nil, fmt.Errorf("conductor template name %q is invalid: %s", conductorTemplate, strings.Join(problems, "; "))
		}
	}
	agents := make([]ExpectedAgent, 0, len(w.modelProjections))
	agentScalingGroupTemplateBudget := validation.DNS1123LabelMaxLength
	for index, projection := range w.modelProjections {
		name := strings.ToLower(fmt.Sprintf("%s-%s-m-%d", componentName, commonconsts.GroveRoleSuffixWorker, index))
		if problems := validation.IsDNS1123Label(name); len(problems) != 0 {
			return nil, fmt.Errorf("Agent template name %q is invalid: %s", name, strings.Join(problems, "; "))
		}
		agents = append(agents, ExpectedAgent{TemplateName: name, Replicas: projection.agentReplicas})
		agentScalingGroupTemplateBudget = min(
			agentScalingGroupTemplateBudget,
			materializedScalingGroupTemplateNameBudget(pcsName, name, projection.agentReplicas-1),
		)
	}
	maxScalingGroupTemplateLength := validation.DNS1123LabelMaxLength
	if conductorTemplate != "" {
		maxScalingGroupTemplateLength = materializedScalingGroupTemplateNameBudget(pcsName, conductorTemplate, 0)
	}
	maxScalingGroupTemplateLength = min(maxScalingGroupTemplateLength, agentScalingGroupTemplateBudget)
	if cyborgTemplateName != "" {
		maxScalingGroupTemplateLength = min(
			maxScalingGroupTemplateLength,
			materializedScalingGroupTemplateNameBudget(
				pcsName,
				cyborgTemplateName,
				maximumCyborgPodIndex,
			),
		)
	}
	lpxScalingGroupTemplate, err := boundedNameTo(strings.ToLower(componentName), maxScalingGroupTemplateLength)
	if err != nil {
		return nil, fmt.Errorf("derive LPU scaling-group template name: %w", err)
	}

	// Construct the plan with Grove's canonical replica-zero scaling-group identity.
	plan := &MaterializationPlan{
		PodCliqueSetName:  pcsName,
		ConductorTemplate: conductorTemplate,
		CyborgTemplate:    cyborgTemplateName,
		Agents:            agents,
		LPXScalingGroup: grovecommon.GeneratePodCliqueScalingGroupName(
			grovecommon.ResourceNameReplica{Name: pcsName, Replica: 0}, lpxScalingGroupTemplate,
		),
		LPXScalingGroupTemplate: lpxScalingGroupTemplate,
		Replicas:                w.scalingGroupReplicas,
	}
	plan = plan.ForReplica(0)

	// All admitted replica indices are 0..8, so their names have the same length.
	// Validate final hostnames once, retaining conductor/Agent/Cyborg error order.
	if plan.ConductorClique != "" {
		if err := validateMaterializedPodHostnameAtIndex("conductor", plan.ConductorClique, 0); err != nil {
			return nil, err
		}
	}
	for _, agent := range plan.Agents {
		if err := validateMaterializedPodHostnameAtIndex("Agent", agent.CliqueName, agent.Replicas-1); err != nil {
			return nil, err
		}
	}
	if plan.CyborgClique != "" {
		if err := validateMaterializedPodHostnameAtIndex("Cyborg", plan.CyborgClique, maximumCyborgPodIndex); err != nil {
			return nil, err
		}
	}
	return plan, nil
}

// ForReplica returns the child clique identities for one scaling-group replica.
// The receiver is a successfully constructed plan and 0 <= index < Replicas.
// Neither the receiver nor its Agents slice is mutated.
func (p *MaterializationPlan) ForReplica(index int32) *MaterializationPlan {
	out := *p
	out.ReplicaIndex = index
	out.Agents = slices.Clone(p.Agents)

	// Materialize this replica beneath the same Grove scaling-group identity.
	if p.ConductorTemplate != "" {
		out.ConductorClique = materializedCliqueNameForReplica(p.LPXScalingGroup, p.ConductorTemplate, index)
	}
	for agentIndex, agent := range out.Agents {
		out.Agents[agentIndex].CliqueName = materializedCliqueNameForReplica(
			p.LPXScalingGroup,
			agent.TemplateName,
			index,
		)
	}
	if p.CyborgTemplate != "" {
		out.CyborgClique = materializedCliqueNameForReplica(p.LPXScalingGroup, p.CyborgTemplate, index)
	}
	return &out
}

func materializedCliqueNameForReplica(pcsName, templateName string, replica int32) string {
	return grovecommon.GeneratePodCliqueName(
		grovecommon.ResourceNameReplica{Name: pcsName, Replica: int(replica)},
		templateName,
	)
}

func validateMaterializedPodHostnameAtIndex(role, cliqueName string, podIndex int) error {
	hostname := materializedPodHostname(cliqueName, podIndex)
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

func materializedPodHostname(cliqueName string, podIndex int) string {
	return fmt.Sprintf("%s-%d", cliqueName, podIndex)
}

func materializedScalingGroupTemplateNameBudget(pcsName, childTemplateName string, podIndex int) int {
	// Grove materializes "<pcs>-0-<group>-0-<child>-<index>".
	return validation.DNS1123LabelMaxLength -
		len(pcsName) -
		len(childTemplateName) -
		7 -
		len(strconv.Itoa(podIndex))
}

func boundedNameTo(candidate string, maxLength int) (string, error) {
	if maxLength > validation.DNS1123LabelMaxLength {
		maxLength = validation.DNS1123LabelMaxLength
	}
	if maxLength < minimumHashedCliqueNameLength {
		return "", fmt.Errorf(
			"available DNS label budget %d is smaller than the %d-character collision-resistant minimum",
			maxLength,
			minimumHashedCliqueNameLength,
		)
	}
	if len(candidate) <= maxLength && len(validation.IsDNS1123Label(candidate)) == 0 {
		return candidate, nil
	}
	digest := sha256.Sum256([]byte(candidate))
	hash := fmt.Sprintf("%x", digest[:4])
	prefixLength := maxLength - len(hash) - 1
	if prefixLength <= 0 {
		return hash, nil
	}
	prefix := strings.TrimRight(candidate[:min(len(candidate), prefixLength)], "-.")
	if prefix == "" {
		return hash, nil
	}
	return prefix + "-" + hash, nil
}
