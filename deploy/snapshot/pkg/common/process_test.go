package common

import (
	"os"
	"path/filepath"
	"strconv"
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

func TestReadProcessDetails(t *testing.T) {
	procRoot := t.TempDir()
	pid := 1018
	procDir := filepath.Join(procRoot, "1018")
	if err := os.MkdirAll(procDir, 0755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", procDir, err)
	}
	if err := os.WriteFile(filepath.Join(procDir, "status"), []byte("Name:\tpython3\nPPid:\t0\nNSpid:\t2402711 1018\n"), 0644); err != nil {
		t.Fatalf("WriteFile(status): %v", err)
	}
	if err := os.WriteFile(filepath.Join(procDir, "cmdline"), []byte("python3\x00-m\x00dynamo.vllm\x00"), 0644); err != nil {
		t.Fatalf("WriteFile(cmdline): %v", err)
	}

	details, err := ReadProcessDetails(procRoot, pid)
	if err != nil {
		t.Fatalf("ReadProcessDetails(%q, %d): %v", procRoot, pid, err)
	}
	if details.ObservedPID != 1018 {
		t.Fatalf("ObservedPID = %d, want 1018", details.ObservedPID)
	}
	if details.ParentPID != 0 {
		t.Fatalf("ParentPID = %d, want 0", details.ParentPID)
	}
	if details.OutermostPID != 2402711 {
		t.Fatalf("OutermostPID = %d, want 2402711", details.OutermostPID)
	}
	if details.InnermostPID != 1018 {
		t.Fatalf("InnermostPID = %d, want 1018", details.InnermostPID)
	}
	if len(details.NamespacePIDs) != 2 || details.NamespacePIDs[0] != 2402711 || details.NamespacePIDs[1] != 1018 {
		t.Fatalf("NamespacePIDs = %v, want [2402711 1018]", details.NamespacePIDs)
	}
	if details.Cmdline != "python3 -m dynamo.vllm" {
		t.Fatalf("Cmdline = %q, want %q", details.Cmdline, "python3 -m dynamo.vllm")
	}
}

func TestReadProcessTable(t *testing.T) {
	procRoot := t.TempDir()
	writeEntry := func(pid int, status string, cmdline string) {
		t.Helper()
		procDir := filepath.Join(procRoot, strconv.Itoa(pid))
		if err := os.MkdirAll(procDir, 0755); err != nil {
			t.Fatalf("MkdirAll(%q): %v", procDir, err)
		}
		if err := os.WriteFile(filepath.Join(procDir, "status"), []byte(status), 0644); err != nil {
			t.Fatalf("WriteFile(status): %v", err)
		}
		if err := os.WriteFile(filepath.Join(procDir, "cmdline"), []byte(cmdline), 0644); err != nil {
			t.Fatalf("WriteFile(cmdline): %v", err)
		}
	}

	writeEntry(768, "Name:\tworker\nPPid:\t1\nNSpid:\t2444000 768\n", "VLLM::Worker_TP0\x00")
	writeEntry(1, "Name:\tpython3\nPPid:\t0\nNSpid:\t2443990 1\n", "python3\x00-m\x00dynamo.vllm\x00")

	processes, err := ReadProcessTable(procRoot)
	if err != nil {
		t.Fatalf("ReadProcessTable(%q): %v", procRoot, err)
	}
	if len(processes) != 2 {
		t.Fatalf("len(ReadProcessTable(%q)) = %d, want 2", procRoot, len(processes))
	}
	if processes[0].InnermostPID != 1 || processes[1].InnermostPID != 768 {
		t.Fatalf("process order innermost PIDs = [%d %d], want [1 768]", processes[0].InnermostPID, processes[1].InnermostPID)
	}
}

func TestResolveManifestPIDsToObservedPIDs(t *testing.T) {
	processes := []ProcessDetails{
		{ObservedPID: 1, ParentPID: 0, OutermostPID: 1, InnermostPID: 1, NamespacePIDs: []int{1}, Cmdline: "sleep infinity"},
		{ObservedPID: 50, ParentPID: 0, OutermostPID: 50, InnermostPID: 50, NamespacePIDs: []int{50}, Cmdline: "nsrestore"},
		{ObservedPID: 74, ParentPID: 50, OutermostPID: 74, InnermostPID: 1, NamespacePIDs: []int{74, 1}, Cmdline: "python3 -m dynamo.vllm"},
		{ObservedPID: 80, ParentPID: 74, OutermostPID: 80, InnermostPID: 750, NamespacePIDs: []int{80, 750}, Cmdline: "VLLM::EngineCore"},
		{ObservedPID: 81, ParentPID: 74, OutermostPID: 81, InnermostPID: 749, NamespacePIDs: []int{81, 749}, Cmdline: "resource_tracker"},
	}

	resolved, err := ResolveManifestPIDsToObservedPIDs(processes, 74, []int{1, 750})
	if err != nil {
		t.Fatalf("ResolveManifestPIDsToObservedPIDs(...) returned error: %v", err)
	}
	if len(resolved) != 2 {
		t.Fatalf("len(resolved) = %d, want 2", len(resolved))
	}
	if resolved[0] != 74 || resolved[1] != 80 {
		t.Fatalf("resolved PIDs = %v, want [74 80]", resolved)
	}
}

func TestResolveManifestPIDsToObservedPIDsFailsWhenManifestPIDMissingFromRestoredSubtree(t *testing.T) {
	processes := []ProcessDetails{
		{ObservedPID: 1, ParentPID: 0, OutermostPID: 1, InnermostPID: 1, NamespacePIDs: []int{1}, Cmdline: "sleep infinity"},
		{ObservedPID: 50, ParentPID: 0, OutermostPID: 50, InnermostPID: 50, NamespacePIDs: []int{50}, Cmdline: "nsrestore"},
		{ObservedPID: 74, ParentPID: 50, OutermostPID: 74, InnermostPID: 1, NamespacePIDs: []int{74, 1}, Cmdline: "python3 -m dynamo.vllm"},
	}

	_, err := ResolveManifestPIDsToObservedPIDs(processes, 74, []int{1, 750})
	if err == nil {
		t.Fatal("ResolveManifestPIDsToObservedPIDs(...) unexpectedly succeeded")
	}
}

func TestResolveManifestPIDsToObservedPIDsFailsWhenNamespaceDepthIsNotTwo(t *testing.T) {
	processes := []ProcessDetails{
		{ObservedPID: 50, ParentPID: 0, OutermostPID: 50, InnermostPID: 50, NamespacePIDs: []int{50}, Cmdline: "nsrestore"},
		{ObservedPID: 74, ParentPID: 50, OutermostPID: 74, InnermostPID: 1, NamespacePIDs: []int{900, 74, 1}, Cmdline: "python3 -m dynamo.vllm"},
		{ObservedPID: 80, ParentPID: 74, OutermostPID: 80, InnermostPID: 750, NamespacePIDs: []int{900, 80, 750}, Cmdline: "VLLM::EngineCore"},
	}

	_, err := ResolveManifestPIDsToObservedPIDs(processes, 74, []int{1, 750})
	if err == nil {
		t.Fatal("ResolveManifestPIDsToObservedPIDs(...) unexpectedly succeeded")
	}
}
