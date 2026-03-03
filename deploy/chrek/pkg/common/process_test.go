package common

import (
	"testing"
)

func TestParseInnermostNSpid(t *testing.T) {
	tests := []struct {
		name    string
		status  string
		want    int
		wantErr bool
	}{
		{
			name:   "single PID namespace (host process)",
			status: "Name:\tbash\nNSpid:\t12345\nPPid:\t1\n",
			want:   12345,
		},
		{
			name:   "two PID namespaces (container process)",
			status: "Name:\tpython3\nNSpid:\t98765\t42\nPPid:\t1\n",
			want:   42,
		},
		{
			name:   "three nested PID namespaces",
			status: "Name:\tworker\nNSpid:\t98765\t500\t1\nPPid:\t1\n",
			want:   1,
		},
		{
			name:    "missing NSpid line",
			status:  "Name:\tbash\nPPid:\t1\n",
			wantErr: true,
		},
		{
			name:    "malformed NSpid line (no values)",
			status:  "NSpid:\n",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseInnermostNSpid(tc.status, 999)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error, got %d", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}

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
