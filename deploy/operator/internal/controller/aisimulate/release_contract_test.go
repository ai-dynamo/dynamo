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
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// The repository root, relative to this package: deploy/operator/internal/controller/aisimulate.
const repositoryRoot = "../../../../.."

var (
	pinPattern     = regexp.MustCompile(`aisimulate==(\d+\.\d+\.\d+)`)
	releasePattern = regexp.MustCompile(`(?m)^version\s*=\s*"(\d+\.\d+)\.`)
)

func readRepositoryFile(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repositoryRoot, path))
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(raw)
}

// pinnedAISimulate returns the released core (major.minor.patch) of the aisimulate pin in a file,
// so 0.13.0.dev202610160000000073 normalizes to 0.13.0.
func pinnedAISimulate(t *testing.T, path string) string {
	t.Helper()
	match := pinPattern.FindStringSubmatch(readRepositoryFile(t, path))
	if match == nil {
		t.Fatalf("%s pins no aisimulate==<version>", path)
	}
	return match[1]
}

// TestBundledAISimulateVersionMatchesTheDynamoRelease ties the release contract to the repository:
// the AISimulate version the current Dynamo minor maps to must be the one the repository pins. It
// fails when a Dynamo minor bump has no mapping entry, or when the pin moves without the entry.
func TestBundledAISimulateVersionMatchesTheDynamoRelease(t *testing.T) {
	canonical := pinnedAISimulate(t, "container/deps/requirements.aisimulate.txt")
	for _, path := range []string{"pyproject.toml", "benchmarks/pyproject.toml"} {
		if got := pinnedAISimulate(t, path); got != canonical {
			t.Errorf("%s pins aisimulate %s, but container/deps/requirements.aisimulate.txt pins %s", path, got, canonical)
		}
	}

	release := releasePattern.FindStringSubmatch(readRepositoryFile(t, "pyproject.toml"))
	if release == nil {
		t.Fatal("pyproject.toml has no project version")
	}
	minor := release[1]
	mapped, ok := aisimulateVersionByDynamoMinor[minor]
	if !ok {
		t.Fatalf("Dynamo %s has no entry in aisimulateVersionByDynamoMinor; add %q: %q with the release", minor, minor, canonical)
	}
	if mapped != canonical {
		t.Fatalf("aisimulateVersionByDynamoMinor[%q] = %q, but the repository pins aisimulate %s", minor, mapped, canonical)
	}
}
