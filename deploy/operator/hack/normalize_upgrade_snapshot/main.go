// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Export API observations through the shared historical snapshot writer.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func main() {
	// Keep raw observations untouched; normalize only the published snapshot.
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: normalize_upgrade_snapshot input.json output.yaml")
		os.Exit(1)
	}
	if err := export(os.Args[1], os.Args[2]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func export(input, output string) error {
	// Decode JSON directly so historical string and integer types survive export.
	data, err := os.ReadFile(input)
	if err != nil {
		return err
	}
	var objects []unstructured.Unstructured
	if err := json.Unmarshal(data, &objects); err != nil {
		return err
	}
	return golden.WriteManifests(output, objects)
}
