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

package dgdrrunpublisher

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

// DefaultTerminationLog is where the kubelet reads a container's termination message
// from unless the pod spec says otherwise. The message ends up in the container's
// terminated state, which is how the run controller learns why the publisher failed a
// run without parsing logs.
const DefaultTerminationLog = "/dev/termination-log"

// maxDiagnosticBytes bounds each free-text field carried out of the publisher. The kubelet
// keeps at most 4096 bytes of a termination message, and the whole message is JSON, so
// two fields of this size (every byte possibly escaped) plus the phase stay inside it.
const maxDiagnosticBytes = 768

// TerminalOutcome is the producer's account of how the run ended: the terminal phase and
// the bounded free-text diagnostics from the snapshot. It is the payload of the handoff to
// the run controller, so a Completed condition can say why, not only that the Job failed.
type TerminalOutcome struct {
	Phase   string `json:"phase,omitempty"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

// RunFailedError is the publisher's outcome when the run ended without a usable result.
// It wraps ErrRunFailed, so errors.Is(err, ErrRunFailed) and ExitCode keep working, and
// carries the bounded diagnostics for the handoff.
type RunFailedError struct {
	Outcome TerminalOutcome
}

func (e *RunFailedError) Error() string {
	parts := []string{ErrRunFailed.Error()}
	for _, detail := range []string{e.Outcome.Message, e.Outcome.Error} {
		if detail != "" {
			parts = append(parts, detail)
		}
	}
	return strings.Join(parts, ": ")
}

// Unwrap keeps errors.Is(err, ErrRunFailed) true.
func (e *RunFailedError) Unwrap() error { return ErrRunFailed }

// boundDiagnostic makes producer-supplied text safe to carry: valid UTF-8, no control
// characters (a multi-line trace becomes one line), at most maxDiagnosticBytes, cut on a
// character boundary.
func boundDiagnostic(text string) string {
	text = strings.ToValidUTF8(strings.TrimSpace(text), "?")
	text = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return ' '
		}
		return r
	}, text)
	if len(text) <= maxDiagnosticBytes {
		return text
	}
	const ellipsis = "…"
	cut := maxDiagnosticBytes - len(ellipsis)
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut] + ellipsis
}

// TerminationOutcome is what the publisher reports for err: the run's own outcome for a
// run failure, otherwise the bounded error text.
func TerminationOutcome(err error) TerminalOutcome {
	if failed, ok := errors.AsType[*RunFailedError](err); ok {
		return failed.Outcome
	}
	return TerminalOutcome{Error: boundDiagnostic(err.Error())}
}

// WriteTerminationMessage records the outcome of a failed publisher as JSON at path (the
// container termination log). It writes nothing for a nil error.
func WriteTerminationMessage(path string, err error) error {
	if err == nil {
		return nil
	}
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if encodeErr := encoder.Encode(TerminationOutcome(err)); encodeErr != nil {
		return encodeErr
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}
