package types

import "testing"

func TestResolvedSocketPolicy(t *testing.T) {
	t.Run("legacy tcpClose maps to close-all-connected", func(t *testing.T) {
		policy, err := (CRIUSettings{TcpClose: true}).ResolvedSocketPolicy()
		if err != nil {
			t.Fatalf("ResolvedSocketPolicy: %v", err)
		}
		if policy != SocketPolicyCloseAllConnected {
			t.Fatalf("policy = %q, want %q", policy, SocketPolicyCloseAllConnected)
		}
	})

	t.Run("legacy tcpEstablished maps to preserve-all-connected", func(t *testing.T) {
		policy, err := (CRIUSettings{TcpEstablished: true}).ResolvedSocketPolicy()
		if err != nil {
			t.Fatalf("ResolvedSocketPolicy: %v", err)
		}
		if policy != SocketPolicyPreserveAllConnected {
			t.Fatalf("policy = %q, want %q", policy, SocketPolicyPreserveAllConnected)
		}
	})

	t.Run("empty settings default to preserve-loopback-only", func(t *testing.T) {
		policy, err := (CRIUSettings{}).ResolvedSocketPolicy()
		if err != nil {
			t.Fatalf("ResolvedSocketPolicy: %v", err)
		}
		if policy != SocketPolicyPreserveLoopbackOnly {
			t.Fatalf("policy = %q, want %q", policy, SocketPolicyPreserveLoopbackOnly)
		}
	})

	t.Run("explicit policy rejects legacy tcp flags", func(t *testing.T) {
		_, err := (CRIUSettings{
			SocketPolicy:   string(SocketPolicyPreserveLoopbackOnly),
			TcpEstablished: true,
		}).ResolvedSocketPolicy()
		if err == nil {
			t.Fatal("expected error when socketPolicy and legacy tcp flags are combined")
		}
	})
}
