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

package gpu

import (
	"context"
	"testing"
	"time"
)

func TestDCGMPhaseContextReservesTimeForTheFallback(t *testing.T) {
	outer, cancelOuter := context.WithTimeout(t.Context(), 4*time.Second)
	defer cancelOuter()
	outerDeadline, _ := outer.Deadline()

	t.Log("With the fallback enabled the DCGM phase ends well before the outer deadline")
	phase, cancel := dcgmPhaseContext(outer, func() bool { return true })
	defer cancel()
	phaseDeadline, ok := phase.Deadline()
	if !ok {
		t.Fatal("phase context has no deadline")
	}
	if remaining := outerDeadline.Sub(phaseDeadline); remaining < time.Second {
		t.Fatalf("only %v is left for the fallback, want at least 1s", remaining)
	}

	t.Log("Without the fallback the whole context is used")
	same, cancelSame := dcgmPhaseContext(outer, func() bool { return false })
	defer cancelSame()
	if same != outer {
		t.Fatal("context was narrowed although there is no fallback to reserve time for")
	}

	t.Log("A context without a deadline is left alone")
	bare, cancelBare := dcgmPhaseContext(t.Context(), func() bool {
		t.Fatal("the fallback callback was asked although the context has no deadline")
		return true
	})
	defer cancelBare()
	if _, hasDeadline := bare.Deadline(); hasDeadline {
		t.Fatal("a deadline was invented for a context that had none")
	}
}
