# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
GPU Fault Injection Pytest Helpers.

Easy-to-use functions for injecting specific GPU XID errors in tests.
"""

import logging
from contextlib import contextmanager
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class XIDError(str, Enum):
    """GPU XID Error Types"""
    XID_43 = "xid_43_kernel_assert"
    XID_48 = "xid_48_dbe_ecc"
    XID_74 = "xid_74_nvlink_error"
    XID_79 = "xid_79_gpu_fell_off_bus"
    XID_94 = "xid_94_contained_ecc"
    XID_95 = "xid_95_uncontained_error"
    XID_119 = "xid_119_gsp_error"
    XID_120 = "xid_120_gsp_rm_error"


class GPUFaultHelper:
    """Helper class for GPU fault injection in tests"""
    
    def __init__(self, api_url: str = "http://localhost:8080", timeout: int = 60):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.http_client = httpx.Client(timeout=timeout)
    
    # ========================================================================
    # Individual XID Error Injection Functions
    # ========================================================================
    
    def inject_xid_43_kernel_assert(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 43: GPU Kernel Assert Error
        
        Simulates CUDA kernel assertion failure.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 43 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-43",
            json={
                "node_name": node_name,
                "xid_type": 43,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    def inject_xid_48_dbe_ecc(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 48: Double Bit ECC Error
        
        Simulates uncorrectable memory error.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 48 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-48",
            json={
                "node_name": node_name,
                "xid_type": 48,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    def inject_xid_74_nvlink_error(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 74: NVLink Error
        
        Simulates NVLink communication failure.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 74 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-74",
            json={
                "node_name": node_name,
                "xid_type": 74,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    def inject_xid_79_gpu_fell_off_bus(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 79: GPU Fell Off Bus (CRITICAL)
        
        Simulates complete GPU hardware failure.
        This is the most severe GPU error.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 79 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-79",
            json={
                "node_name": node_name,
                "xid_type": 79,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    def inject_xid_94_contained_ecc(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 94: Contained ECC Error
        
        Simulates correctable memory error.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 94 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-94",
            json={
                "node_name": node_name,
                "xid_type": 94,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    def inject_xid_95_uncontained_error(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 95: Uncontained Error (CRITICAL)
        
        Simulates severe uncontained memory corruption.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 95 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-95",
            json={
                "node_name": node_name,
                "xid_type": 95,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    def inject_xid_119_gsp_error(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 119: GSP Error
        
        Simulates GPU firmware/GSP failure.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 119 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-119",
            json={
                "node_name": node_name,
                "xid_type": 119,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    def inject_xid_120_gsp_rm_error(
        self,
        node_name: str,
        gpu_id: int = 0,
        duration: Optional[int] = None
    ) -> dict:
        """
        Inject XID 120: GSP Resource Manager Error
        
        Simulates GPU resource exhaustion.
        
        Args:
            node_name: Target node name
            gpu_id: GPU device ID (default: 0)
            duration: Duration in seconds (None = permanent)
        
        Returns:
            Fault injection response
        """
        logger.info(f"Injecting XID 120 on {node_name}, GPU {gpu_id}")
        
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject/xid-120",
            json={
                "node_name": node_name,
                "xid_type": 120,
                "gpu_id": gpu_id,
                "duration": duration,
            }
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # Context Managers for Easy Testing
    # ========================================================================
    
    @contextmanager
    def xid_43(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 43 injection with automatic recovery"""
        fault = self.inject_xid_43_kernel_assert(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    @contextmanager
    def xid_48(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 48 injection with automatic recovery"""
        fault = self.inject_xid_48_dbe_ecc(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    @contextmanager
    def xid_74(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 74 injection with automatic recovery"""
        fault = self.inject_xid_74_nvlink_error(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    @contextmanager
    def xid_79(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 79 injection with automatic recovery"""
        fault = self.inject_xid_79_gpu_fell_off_bus(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    @contextmanager
    def xid_94(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 94 injection with automatic recovery"""
        fault = self.inject_xid_94_contained_ecc(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    @contextmanager
    def xid_95(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 95 injection with automatic recovery"""
        fault = self.inject_xid_95_uncontained_error(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    @contextmanager
    def xid_119(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 119 injection with automatic recovery"""
        fault = self.inject_xid_119_gsp_error(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    @contextmanager
    def xid_120(self, node_name: str, gpu_id: int = 0, duration: Optional[int] = None):
        """Context manager for XID 120 injection with automatic recovery"""
        fault = self.inject_xid_120_gsp_rm_error(node_name, gpu_id, duration)
        try:
            yield fault
        finally:
            self.recover_fault(fault["fault_id"])
    
    # ========================================================================
    # Utility Functions
    # ========================================================================
    
    def recover_fault(self, fault_id: str):
        """Recover from a specific fault"""
        try:
            response = self.http_client.post(
                f"{self.api_url}/api/v1/faults/{fault_id}/recover"
            )
            response.raise_for_status()
            logger.info(f"Recovered from fault: {fault_id}")
        except Exception as e:
            logger.error(f"Failed to recover fault {fault_id}: {e}")
    
    def list_xid_types(self) -> dict:
        """List all available XID error types"""
        response = self.http_client.get(f"{self.api_url}/api/v1/faults/gpu/xid-types")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close HTTP client"""
        self.http_client.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================================================
# Convenience Functions for Quick Testing
# ============================================================================


def inject_xid_43(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 43"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_43_kernel_assert(node_name, gpu_id)


def inject_xid_48(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 48"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_48_dbe_ecc(node_name, gpu_id)


def inject_xid_74(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 74"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_74_nvlink_error(node_name, gpu_id)


def inject_xid_79(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 79 (CRITICAL)"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_79_gpu_fell_off_bus(node_name, gpu_id)


def inject_xid_94(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 94"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_94_contained_ecc(node_name, gpu_id)


def inject_xid_95(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 95 (CRITICAL)"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_95_uncontained_error(node_name, gpu_id)


def inject_xid_119(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 119"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_119_gsp_error(node_name, gpu_id)


def inject_xid_120(node_name: str, gpu_id: int = 0, api_url: str = "http://localhost:8080"):
    """Quick function to inject XID 120"""
    with GPUFaultHelper(api_url) as helper:
        return helper.inject_xid_120_gsp_rm_error(node_name, gpu_id)

