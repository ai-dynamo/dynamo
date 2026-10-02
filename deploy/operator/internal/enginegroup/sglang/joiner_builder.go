/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"fmt"
	"net"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const mainContainerName = "main"

// JoinerPodBuilder derives SGLang's operation-specific joining process from a
// profile-resolved primary Pod. The class is deliberately replaceable when the
// backend exposes a mode-independent bootstrap contract.
type JoinerPodBuilder struct{}

// Build creates a Pod template for one width-one DP/EP joining rank.
func (JoinerPodBuilder) Build(
	primary *corev1.Pod,
	target enginegroup.CapacityReplicaTarget,
) (*corev1.Pod, error) {
	if primary == nil {
		return nil, fmt.Errorf("primary Pod is required")
	}
	if target.Bootstrap == nil || len(target.Bootstrap.NativeMembers) != 1 {
		return nil, fmt.Errorf("SGLang width-one join requires exactly one native member")
	}
	rank, err := nativeRank(target.Bootstrap.NativeMembers[0])
	if err != nil {
		return nil, err
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: primary.Name + "-joiner-",
			Labels:       make(map[string]string),
			Annotations:  copyStrings(primary.Annotations),
		},
		Spec: *primary.Spec.DeepCopy(),
	}
	// Let the scheduler place new capacity. Every other scheduling constraint,
	// volume, service account, image, and security setting remains profile-owned.
	pod.Spec.NodeName = ""
	pod.Spec.Hostname = ""
	pod.Spec.Subdomain = ""
	pod.Spec.ReadinessGates = nil
	pod.Spec.SchedulingGates = nil

	main, err := findContainer(pod.Spec.Containers, mainContainerName)
	if err != nil {
		return nil, err
	}
	// A joiner is an engine process, not another Dynamo serving worker. Keeping
	// it out of runtime discovery prevents one logical Engine Group from being
	// advertised as two independent endpoints after commit.
	main.Command = []string{"sglang", "serve"}
	args := append([]string(nil), main.Args...)
	args = setOption(args, []string{"--tp-size", "--tensor-parallel-size", "--tp"}, "1")
	args = setOption(args, []string{"--dp-size", "--data-parallel-size", "--dp"}, "1")
	args = setOption(args, []string{"--elastic-ep-join-mode"}, "scale")
	args = setOption(args, []string{"--elastic-ep-join-rank-offset"}, strconv.Itoa(rank))
	args, err = pinJoinerRendezvous(args, primary.Status.PodIP)
	if err != nil {
		return nil, err
	}
	main.Args = args
	return pod, nil
}

func pinJoinerRendezvous(args []string, primaryIP string) ([]string, error) {
	if primaryIP == "" {
		return nil, fmt.Errorf("primary Pod IP is required for SGLang joining bootstrap")
	}
	address := optionValue(args, "--dist-init-addr")
	if address == "" {
		return nil, fmt.Errorf("primary SGLang declaration must include --dist-init-addr")
	}
	_, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, fmt.Errorf("parse primary SGLang --dist-init-addr %q: %w", address, err)
	}
	return setOption(args, []string{"--dist-init-addr"}, net.JoinHostPort(primaryIP, port)), nil
}

func optionValue(args []string, name string) string {
	for index := len(args) - 1; index >= 0; index-- {
		if strings.HasPrefix(args[index], name+"=") {
			return strings.TrimPrefix(args[index], name+"=")
		}
		if args[index] == name && index+1 < len(args) {
			return args[index+1]
		}
	}
	return ""
}

func nativeRank(member enginegroup.NativeMemberID) (int, error) {
	literal := strings.TrimPrefix(string(member), "dp-")
	rank, err := strconv.Atoi(literal)
	if err != nil || rank < 0 || "dp-"+strconv.Itoa(rank) != string(member) {
		return 0, fmt.Errorf("SGLang native member %q must have canonical dp-N form", member)
	}
	return rank, nil
}

func findContainer(containers []corev1.Container, name string) (*corev1.Container, error) {
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i], nil
		}
	}
	return nil, fmt.Errorf("primary Pod has no %q container", name)
}

func setOption(args []string, aliases []string, value string) []string {
	filtered := make([]string, 0, len(args)+2)
	for index := 0; index < len(args); index++ {
		matched := false
		for _, alias := range aliases {
			if args[index] == alias {
				matched = true
				index++
				break
			}
			if strings.HasPrefix(args[index], alias+"=") {
				matched = true
				break
			}
		}
		if !matched {
			filtered = append(filtered, args[index])
		}
	}
	return append(filtered, aliases[0], value)
}

func copyStrings(source map[string]string) map[string]string {
	result := make(map[string]string, len(source)+5)
	for key, value := range source {
		result[key] = value
	}
	return result
}
