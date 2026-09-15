/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1beta1

import (
	"encoding/json"
	"os"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/yaml"
)

func TestDynamoGraphDeploymentEngineGroupSchemeRegistration(t *testing.T) {
	t.Log("Register the v1beta1 API types in a new scheme")
	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	t.Log("Resolve the Engine Group root kind from the registered Go type")
	kinds, unversioned, err := scheme.ObjectKinds(&DynamoGraphDeploymentEngineGroup{})
	if err != nil {
		t.Fatal(err)
	}
	if unversioned {
		t.Fatal("Engine Group unexpectedly registered as unversioned")
	}
	if len(kinds) != 1 || kinds[0] != DynamoGraphDeploymentEngineGroupGVK {
		t.Fatalf("registered kinds = %v, want [%s]", kinds, DynamoGraphDeploymentEngineGroupGVK)
	}
}

func TestDynamoGraphDeploymentEngineGroupPreservesDistinctReplicaStates(t *testing.T) {
	t.Log("Build one Engine Group with different allocated, available, and active counts")
	group := &DynamoGraphDeploymentEngineGroup{
		Spec: DynamoGraphDeploymentEngineGroupSpec{Replicas: 8},
		Status: DynamoGraphDeploymentEngineGroupStatus{
			Replicas:          9,
			AvailableReplicas: 8,
			ActiveReplicas:    7,
			Selector:          "nvidia.com/dynamo-scale-representative=true",
		},
	}

	t.Log("Round-trip the public JSON representation")
	raw, err := json.Marshal(group)
	if err != nil {
		t.Fatal(err)
	}
	got := &DynamoGraphDeploymentEngineGroup{}
	if err := json.Unmarshal(raw, got); err != nil {
		t.Fatal(err)
	}

	t.Log("Verify desired, allocated, available, active, and selector state remain independent")
	if got.Spec.Replicas != 8 || got.Status.Replicas != 9 || got.Status.AvailableReplicas != 8 || got.Status.ActiveReplicas != 7 {
		t.Fatalf("replica state = desired %d, allocated %d, available %d, active %d", got.Spec.Replicas, got.Status.Replicas, got.Status.AvailableReplicas, got.Status.ActiveReplicas)
	}
	if got.Status.Selector != group.Status.Selector {
		t.Fatalf("selector = %q, want %q", got.Status.Selector, group.Status.Selector)
	}
}

func TestDynamoGraphDeploymentEngineGroupDeepCopyPreservesIdentityIsolation(t *testing.T) {
	t.Log("Build status containing concrete Pod, runtime, and native-member identities")
	group := &DynamoGraphDeploymentEngineGroup{
		Status: DynamoGraphDeploymentEngineGroupStatus{
			ReplicaStates: []EngineGroupReplicaStatus{{
				ReplicaID: "replica-0",
				SlotID:    "slot-0",
				Current: &EngineGroupReplicaIncarnation{
					RuntimeIncarnation: "runtime-0",
					CapacityRefs: []EngineGroupCapacityRef{{
						Name: "worker-0",
						UID:  types.UID("pod-uid-0"),
					}},
				},
				NativeMembers: []string{"dp-0"},
			}},
		},
	}

	t.Log("Deep-copy and mutate the copied identity slices")
	copy := group.DeepCopy()
	copy.Status.ReplicaStates[0].Current.CapacityRefs[0].UID = types.UID("replacement-uid")
	copy.Status.ReplicaStates[0].NativeMembers[0] = "dp-1"

	t.Log("Verify the source retains its original concrete identities")
	got := group.Status.ReplicaStates[0]
	if got.Current.CapacityRefs[0].UID != types.UID("pod-uid-0") {
		t.Fatalf("source Pod UID = %q, want pod-uid-0", got.Current.CapacityRefs[0].UID)
	}
	if got.NativeMembers[0] != "dp-0" {
		t.Fatalf("source native member = %q, want dp-0", got.NativeMembers[0])
	}
}

func TestDynamoGraphDeploymentEngineGroupGeneratedScaleContract(t *testing.T) {
	t.Log("Read the generated Engine Group CRD")
	raw, err := os.ReadFile("../../config/crd/bases/nvidia.com_dynamographdeploymentenginegroups.yaml")
	if err != nil {
		t.Fatal(err)
	}
	jsonRaw, err := yaml.YAMLToJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := json.Unmarshal(jsonRaw, crd); err != nil {
		t.Fatal(err)
	}

	t.Log("Locate the sole served and stored v1beta1 version")
	if len(crd.Spec.Versions) != 1 {
		t.Fatalf("versions = %d, want 1", len(crd.Spec.Versions))
	}
	version := crd.Spec.Versions[0]
	if version.Name != "v1beta1" || !version.Served || !version.Storage {
		t.Fatalf("version = %#v, want served storage v1beta1", version)
	}

	t.Log("Verify the scale subresource maps logical replica fields and the representative selector")
	if version.Subresources == nil || version.Subresources.Scale == nil {
		t.Fatal("scale subresource is missing")
	}
	scale := version.Subresources.Scale
	if scale.SpecReplicasPath != ".spec.replicas" || scale.StatusReplicasPath != ".status.replicas" || scale.LabelSelectorPath == nil || *scale.LabelSelectorPath != ".status.selector" {
		t.Fatalf("scale contract = %#v", scale)
	}

	t.Log("Verify the printer columns name the logical scale unit")
	for _, column := range version.AdditionalPrinterColumns {
		if column.Name == "UNITS" {
			if column.Type != "string" || column.JSONPath != ".status.scaleUnit" {
				t.Fatalf("UNITS printer column = %#v", column)
			}
			return
		}
	}
	t.Fatal("UNITS printer column is missing")
}
