package common

import (
	"os"
	"path/filepath"
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

func TestParseNSPIDs(t *testing.T) {
	tests := []struct {
		name    string
		status  string
		hostPID int
		want    []int
		wantErr bool
	}{
		{
			name:    "happy path",
			status:  "Name:\tpython3\nNSpid:\t2402711 1018\n",
			hostPID: 2402711,
			want:    []int{2402711, 1018},
		},
		{
			name:    "missing nspid line",
			status:  "Name:\tpython3\nState:\tS (sleeping)\n",
			hostPID: 2402711,
			wantErr: true,
		},
		{
			name:    "non integer nspid",
			status:  "Name:\tpython3\nNSpid:\t2402711 abc\n",
			hostPID: 2402711,
			wantErr: true,
		},
		{
			name:    "host pid mismatch",
			status:  "Name:\tpython3\nNSpid:\t2402712 1018\n",
			hostPID: 2402711,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseNSPIDs(tc.status, tc.hostPID)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseNSPIDs(%q, %d) unexpectedly succeeded with %v", tc.status, tc.hostPID, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseNSPIDs(%q, %d): %v", tc.status, tc.hostPID, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("parseNSPIDs(%q, %d) len = %d, want %d", tc.status, tc.hostPID, len(got), len(tc.want))
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("parseNSPIDs(%q, %d)[%d] = %d, want %d", tc.status, tc.hostPID, i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestReadProcessDetails(t *testing.T) {
	procRoot := t.TempDir()
	pid := 1018
	procDir := filepath.Join(procRoot, "1018")
	if err := os.MkdirAll(procDir, 0755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", procDir, err)
	}
	if err := os.WriteFile(filepath.Join(procDir, "status"), []byte("Name:\tpython3\nNSpid:\t2402711 1018\n"), 0644); err != nil {
		t.Fatalf("WriteFile(status): %v", err)
	}
	if err := os.WriteFile(filepath.Join(procDir, "cmdline"), []byte("python3\x00-m\x00dynamo.vllm\x00"), 0644); err != nil {
		t.Fatalf("WriteFile(cmdline): %v", err)
	}

	details := ReadProcessDetails(procRoot, pid)
	if details.HostPID != 2402711 {
		t.Fatalf("HostPID = %d, want 2402711", details.HostPID)
	}
	if details.NamespacePID != 1018 {
		t.Fatalf("NamespacePID = %d, want 1018", details.NamespacePID)
	}
	if details.Cmdline != "python3 -m dynamo.vllm" {
		t.Fatalf("Cmdline = %q, want %q", details.Cmdline, "python3 -m dynamo.vllm")
	}
}
