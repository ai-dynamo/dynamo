/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package aisimulate

import (
	"bytes"
	_ "embed"
	"fmt"
	"sync"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// Only the active snapshot is embedded; versioned snapshots remain repository history.
//
//go:embed schemas/latest.json
var inputSchemaJSON []byte

// LoadInputSchema compiles the bundled schema once and caches the schema or error.
// Webhook setup loads it before registration; admission reuses the same compiled schema.
var LoadInputSchema = sync.OnceValues(func() (*jsonschema.Schema, error) {
	document, err := jsonschema.UnmarshalJSON(bytes.NewReader(inputSchemaJSON))
	if err != nil {
		return nil, fmt.Errorf("decode AISimulate %s schema: %w", SchemaVersion, err)
	}

	// Resolve local definitions against an in-memory resource; no Python runs in admission.
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	const location = "https://aisimulate.invalid/recommendation.json"
	if err := compiler.AddResource(location, document); err != nil {
		return nil, fmt.Errorf("register AISimulate %s schema: %w", SchemaVersion, err)
	}
	return compiler.Compile(location)
})

// ValidateInput checks the complete lowered input against the schema generated from
// the bundled AISimulate release. It does not apply upstream defaults or Python-only
// model validators. Adapter-owned sections have object shape only; their detailed
// validation remains with the registered Python adapters.
func ValidateInput(input map[string]any) error {
	schema, err := LoadInputSchema()
	if err != nil {
		return err
	}
	return schema.Validate(input)
}
