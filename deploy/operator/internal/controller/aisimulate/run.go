/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package aisimulate

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
)

// InputForRun returns the AISimulate input a Run executes. run.spec.aisimulate.parameters is the
// complete resolved native input, frozen when the Run was created. Run execution must use this
// object directly and must not pass the Run through Request lowering (Lower): the typed fields of
// the Run are copies of the Request and would bring back keys that spec.aisimulate.parameters
// deleted with a JSON merge patch null, and the nulls themselves are no longer in the stored result.
func InputForRun(run *v1beta2.DynamoGraphDeploymentRun) (map[string]any, error) {
	if run.Spec.AISimulate == nil || run.Spec.AISimulate.Parameters == nil {
		return nil, errors.New("run has no spec.aisimulate.parameters")
	}
	raw := run.Spec.AISimulate.Parameters.Raw
	if len(raw) == 0 && run.Spec.AISimulate.Parameters.Object != nil {
		var err error
		if raw, err = json.Marshal(run.Spec.AISimulate.Parameters.Object); err != nil {
			return nil, fmt.Errorf("run spec.aisimulate.parameters: %w", err)
		}
	}
	if len(raw) == 0 {
		return nil, errors.New("run spec.aisimulate.parameters is empty")
	}
	var input map[string]any
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("run spec.aisimulate.parameters must be a JSON object: %w", err)
	}
	if len(input) == 0 {
		return nil, errors.New("run spec.aisimulate.parameters is empty")
	}
	return input, nil
}
