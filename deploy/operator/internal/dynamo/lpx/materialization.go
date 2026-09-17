/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"slices"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	"k8s.io/apimachinery/pkg/util/validation"
)

const (
	lpxScalingGroupTemplateName = "lpx"
	conductorTemplateName       = "cond"
)

// MaxPodCliqueSetNameLength reserves Grove's combined name budget for the fixed
// scaling group and longest role name, independently of the selected components.
const MaxPodCliqueSetNameLength = commonconsts.MaxCombinedGroveResourceNameLength -
	len(lpxScalingGroupTemplateName) - len(conductorTemplateName)

// LPXAttemptRequestStatus stores a name (lpxRequestName uses at most 60 bytes),
// sha256 digest, and server-issued UUID. Reserve a full DNS label plus JSON syntax
// and an array separator; all generated values are ASCII.
const maximumSchedulingAttemptRequestBytes = len(`{"name":"","attemptDigest":"","uid":""},`) +
	validation.DNS1123LabelMaxLength + len("sha256:") + sha256.Size*2 + 36

// Leave half the object allowance for the enclosing status, spec, and metadata.
const schedulingAttemptRequestBytesBudget = MaxRenderedPodCliqueSetBytes / 2

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
	// ReplicaIndex is the scaling-group replica represented by this projection.
	ReplicaIndex int32
	// Replicas is the total scaling-group replica count.
	Replicas int32
}

// PlanNodeLocalMaterialization derives the exact identities shared by graph
// rendering and lifecycle observation without mutating the workload.
func (w *SelectedWorkload) PlanNodeLocalMaterialization(pcsName string) (*MaterializationPlan, error) {
	// Leave room for the fixed scaling group and every LPX role in Grove's name budget.
	if strings.TrimSpace(pcsName) == "" {
		return nil, fmt.Errorf("PodCliqueSet name is required")
	}
	if len(pcsName) > MaxPodCliqueSetNameLength {
		return nil, fmt.Errorf("PodCliqueSet name %q exceeds the LPX maximum of %d characters", pcsName, MaxPodCliqueSetNameLength)
	}

	// Nova and Cyborg both implement the authored conductor role.
	conductorTemplate, cyborgTemplate := conductorTemplateName, ""
	if w.Pipeline() == PipelineLPX {
		conductorTemplate, cyborgTemplate = "", conductorTemplateName
	}
	agents := make([]ExpectedAgent, 0, len(w.modelProjections))
	for index, projection := range w.modelProjections {
		name := "agt"
		if len(w.modelProjections) > 1 {
			name = fmt.Sprintf("agt%d", index)
		}
		agents = append(agents, ExpectedAgent{TemplateName: name, Replicas: projection.agentReplicas})
	}

	// Construct the plan with Grove's canonical replica-zero scaling-group identity.
	plan := &MaterializationPlan{
		PodCliqueSetName:  pcsName,
		ConductorTemplate: conductorTemplate,
		CyborgTemplate:    cyborgTemplate,
		Agents:            agents,
		LPXScalingGroup: grovecommon.GeneratePodCliqueScalingGroupName(
			grovecommon.ResourceNameReplica{Name: pcsName, Replica: 0}, lpxScalingGroupTemplateName,
		),
		Replicas: w.scalingGroupReplicas,
	}
	plan = plan.ForReplica(0)

	return plan, plan.ValidateReplicaCount()
}

// ValidateReplicaCount checks conductor/Agent hostnames and bounds request/status allocation.
func (p *MaterializationPlan) ValidateReplicaCount() error {
	// Charge every model/replica for its complete persisted request identity.
	if p.Replicas < 0 || int64(p.Replicas)*int64(len(p.Agents)) > int64(schedulingAttemptRequestBytesBudget/maximumSchedulingAttemptRequestBytes) {
		return fmt.Errorf("LPX replica count exceeds the scheduling status size budget")
	}
	if p.ConductorTemplate != "" {
		if err := p.validatePodHostname("conductor", p.ConductorTemplate, 0); err != nil {
			return err
		}
	}
	for _, agent := range p.Agents {
		if err := p.validatePodHostname("Agent", agent.TemplateName, agent.Replicas-1); err != nil {
			return err
		}
	}
	return nil
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

func (p *MaterializationPlan) validatePodHostname(role, templateName string, podIndex int) error {
	// Validate the longest replica name without copying the full materialization plan.
	cliqueName := materializedCliqueNameForReplica(p.LPXScalingGroup, templateName, max(0, p.Replicas-1))
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
