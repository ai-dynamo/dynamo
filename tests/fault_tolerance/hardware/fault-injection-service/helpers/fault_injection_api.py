"""
Fault injection API client for GPU XID error injection.

Provides high-level interface for injecting GPU faults via the
fault injection service API.
"""

from typing import Dict, Optional

import requests


class FaultInjectionClient:
    """Client for fault injection API."""

    def __init__(self, base_url: str, timeout: int = 60):
        """
        Initialize fault injection client.

        Args:
            base_url: Base URL of fault injection API (e.g., "http://localhost:8080")
            timeout: Request timeout in seconds (default: 60)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        """
        Check if API is healthy.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def inject_xid(
        self,
        node_name: str,
        xid_type: int,
        gpu_id: int = 0,
        duration: Optional[int] = None,
    ) -> Dict:
        """
        Inject GPU XID error on a specific node.

        Args:
            node_name: Kubernetes node name
            xid_type: XID error type (e.g., 79 for "GPU fell off bus")
            gpu_id: GPU device ID (default: 0)
            duration: Fault duration in seconds (None = permanent until cleared)

        Returns:
            Dict with keys: success, fault_id, message
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/faults/gpu/inject/xid-{xid_type}",
                json={
                    "node_name": node_name,
                    "xid_type": xid_type,
                    "gpu_id": gpu_id,
                    "duration": duration,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "fault_id": data.get("fault_id"),
                    "message": "XID injected successfully",
                }
            else:
                return {
                    "success": False,
                    "fault_id": None,
                    "message": f"Injection failed ({response.status_code}): {response.text}",
                }

        except Exception as e:
            return {
                "success": False,
                "fault_id": None,
                "message": f"Exception during injection: {str(e)}",
            }

    def clear_fault(self, fault_id: str) -> bool:
        """
        Clear a previously injected fault.

        Args:
            fault_id: Fault ID returned from inject_xid()

        Returns:
            True if fault cleared successfully
        """
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/faults/{fault_id}", timeout=self.timeout
            )
            return response.status_code in [200, 204]
        except Exception:
            return False

    def get_fault_status(self, fault_id: str) -> Optional[Dict]:
        """
        Get status of an injected fault.

        Args:
            fault_id: Fault ID returned from inject_xid()

        Returns:
            Dict with fault information, or None if not found
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/faults/{fault_id}", timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception:
            return None


class XIDTestHelper:
    """Helper class for XID-based fault tolerance tests."""

    def __init__(self, api_client: FaultInjectionClient):
        """
        Initialize XID test helper.

        Args:
            api_client: FaultInjectionClient instance
        """
        self.api = api_client
        self.active_faults = []

    def inject_xid_79(self, node_name: str, gpu_id: int = 0) -> Optional[str]:
        """
        Inject XID 79 (GPU fell off the bus).

        Args:
            node_name: Kubernetes node name
            gpu_id: GPU device ID (default: 0)

        Returns:
            Fault ID if successful, None otherwise
        """
        print(f"\n[→] Injecting XID 79 on node: {node_name} (GPU {gpu_id})")

        result = self.api.inject_xid(node_name, xid_type=79, gpu_id=gpu_id)

        if result["success"]:
            fault_id = result["fault_id"]
            self.active_faults.append(fault_id)
            print("[✓] XID 79 injected successfully")
            print(f"    Fault ID: {fault_id}")
            return fault_id
        else:
            print(f"[✗] XID 79 injection failed: {result['message']}")
            return None

    def inject_xid_43(self, node_name: str, gpu_id: int = 0) -> Optional[str]:
        """
        Inject XID 43 (GPU stopped responding).

        Args:
            node_name: Kubernetes node name
            gpu_id: GPU device ID (default: 0)

        Returns:
            Fault ID if successful, None otherwise
        """
        print(f"\n[→] Injecting XID 43 on node: {node_name} (GPU {gpu_id})")

        result = self.api.inject_xid(node_name, xid_type=43, gpu_id=gpu_id)

        if result["success"]:
            fault_id = result["fault_id"]
            self.active_faults.append(fault_id)
            print("[✓] XID 43 injected successfully")
            print(f"    Fault ID: {fault_id}")
            return fault_id
        else:
            print(f"[✗] XID 43 injection failed: {result['message']}")
            return None

    def cleanup_all_faults(self):
        """Clear all faults injected by this helper."""
        print(f"\n[→] Cleaning up {len(self.active_faults)} fault(s)...")

        for fault_id in self.active_faults:
            try:
                if self.api.clear_fault(fault_id):
                    print(f"    ✓ Cleared: {fault_id}")
                else:
                    print(f"    ⚠ Failed to clear: {fault_id}")
            except Exception as e:
                print(f"    ✗ Error clearing {fault_id}: {e}")

        self.active_faults = []
