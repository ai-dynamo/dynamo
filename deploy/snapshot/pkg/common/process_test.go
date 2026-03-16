package common

import (
	"testing"
)

func TestParseProcExitCode(t *testing.T) {
	tests := []struct {
		name     string
		statLine string
		wantCode int
		wantErr  bool
	}{
		{
			// Real /proc/<pid>/stat line (simplified). Fields after ")" start with state.
			// The last field (field 52) is exit_code.
			name:     "normal exit code 0",
			statLine: "123 (python3) S 1 123 123 0 -1 4194304 1000 0 0 0 100 50 0 0 20 0 1 0 1000 10000000 500 18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
			wantCode: 0,
		},
		{
			name:     "non-zero exit code",
			statLine: "456 (bash) Z 1 456 456 0 -1 4194304 100 0 0 0 10 5 0 0 20 0 1 0 500 0 0 18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 256",
			wantCode: 256, // signal 1 encoded as WaitStatus
		},
		{
			// Process names can contain spaces and parentheses.
			// The parser must use LastIndex(")") to handle this correctly.
			name:     "process name with spaces and parens",
			statLine: "789 (python3 -m vllm.entrypoints.openai.api_server (worker)) S 1 789 789 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 100 0 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 42",
			wantCode: 42,
		},
		{
			name:     "malformed line no closing paren",
			statLine: "123 (python3 S 1 123",
			wantErr:  true,
		},
		{
			name:     "empty string",
			statLine: "",
			wantErr:  true,
		},
		{
			name:     "only pid and comm, nothing after paren",
			statLine: "1 (init)",
			wantErr:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ws, err := ParseProcExitCode(tc.statLine)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error, got WaitStatus=%d", ws)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if int(ws) != tc.wantCode {
				t.Errorf("exit code = %d, want %d", int(ws), tc.wantCode)
			}
		})
	}
}
