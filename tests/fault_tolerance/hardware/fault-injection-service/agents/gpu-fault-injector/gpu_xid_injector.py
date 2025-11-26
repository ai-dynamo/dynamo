# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
GPU XID Error Injector via nsenter+kmsg.

Injects fake XID messages to host's /dev/kmsg to trigger NVSentinel detection.
Uses nsenter to enter host namespaces and write kernel messages that NVSentinel
syslog-health-monitor can detect naturally.

Method: nsenter --target 1 (all namespaces) → echo to /dev/kmsg → NVSentinel detection

Supported XIDs:
===============
This injector accepts ANY XID error code (1-255+) for maximum testing flexibility.

Pre-defined Messages for All DCGM/NVSentinel Monitored XIDs:
-------------------------------------------------------------
Based on DCGM health monitoring subsystems and NVSentinel detection rules.

DEVASTATING XIDs (DCGM_HEALTH_RESULT_FAIL - always monitored):
- 79:  GPU fell off bus (most critical - node-level action)
- 74:  NVLink uncorrectable error (multi-GPU communication failure)
- 48:  Double-bit ECC error (severe memory error)
- 94:  Contained ECC error (less severe memory error)
- 95:  Uncontained error (very severe, GPU reset required)
- 119: GSP RPC Timeout (GPU Service Processor communication)
- 120: GSP Error (GPU Service Processor internal error)
- 140: ECC unrecovered error (persistent memory issue)

SUBSYSTEM XIDs (DCGM_HEALTH_RESULT_WARN - may escalate):

Memory Subsystem (DCGM_HEALTH_WATCH_MEM):
- 31:  MMU Error
- 32:  PBDMA Error
- 43:  Reset Channel Verification Error
- 63:  Pending Page Retirements
- 64:  Row Remap Failure

PCIe Subsystem (DCGM_HEALTH_WATCH_PCIE):
- 38:  PCIe Bus Error
- 39:  PCIe Fabric Error
- 42:  PCIe Replay Rate exceeded

Thermal Subsystem (DCGM_HEALTH_WATCH_THERMAL):
- 60:  Clocks Event: Thermal limit exceeded
- 61:  EDPP Power Brake: Thermal limit
- 62:  Thermal Violations detected

Power Subsystem (DCGM_HEALTH_WATCH_POWER):
- 54:  Power state change event
- 56:  Clock change event
- 57:  Clocks Event: Power limit exceeded

Graphics/Common XIDs:
- 13:  Graphics Engine Exception
- 45:  Preemptive Cleanup (due to previous errors)
- 69:  Graphics Exception: Class Error

Unknown XIDs:
-------------
Any XID not in XID_MESSAGES dict will use a generic error message format.
NVSentinel will parse and handle based on its own XID database and rules.

Note: XIDs 43, 48, 74, 94, 95 are already supported via CUDA interception
(cuda_intercept.c LD_PRELOAD). kmsg injection adds complementary syslog-based
detection path for NVSentinel's syslog-health-monitor.
"""

import logging
import os
import subprocess
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# XID error code to descriptive message mapping
# Based on DCGM XID database and NVSentinel monitoring rules
# Source: DCGM/modules/health/DcgmHealthWatch.cpp BuildXidMappings()
XID_MESSAGES: Dict[int, str] = {
    # Devastating XIDs (DCGM_HEALTH_RESULT_FAIL - always monitored)
    79: "GPU has fallen off the bus",
    48: "DBE (Double Bit Error) ECC Error",
    74: "NVLink: Uncorrectable error",
    94: "Contained ECC error",
    95: "Uncontained error - GPU requires reset",
    119: "GSP RPC Timeout",
    120: "GSP Error",
    140: "ECC unrecovered error",
    # Memory Subsystem XIDs (DCGM_HEALTH_WATCH_MEM)
    31: "MMU Error",
    32: "PBDMA Error",
    43: "Reset Channel Verification Error",
    63: "Pending Page Retirements",
    64: "Row Remap Failure",
    # PCIe Subsystem XIDs (DCGM_HEALTH_WATCH_PCIE)
    38: "PCIe Bus Error",
    39: "PCIe Fabric Error",
    42: "PCIe Replay Rate exceeded",
    # 74 already defined above (can be PCIe or NVLink context)
    # Thermal Subsystem XIDs (DCGM_HEALTH_WATCH_THERMAL)
    60: "Clocks Event: Thermal limit exceeded",
    61: "EDPP Power Brake: Thermal limit",
    62: "Thermal Violations detected",
    # 63 can be thermal or memory context ("Thermal diode detects short")
    # Power Subsystem XIDs (DCGM_HEALTH_WATCH_POWER)
    54: "Power state change event",
    56: "Clock change event",
    57: "Clocks Event: Power limit exceeded",
    # Common Graphics XIDs (often seen in test environments)
    13: "Graphics Engine Exception",
    31: "GPU stopped responding",  # Can be both MMU or timeout context
    45: "Preemptive Cleanup, due to previous errors",
    69: "Graphics Exception: Class Error",
}


class GPUXIDInjectorKernel:
    """
    XID injector via nsenter+kmsg (triggers NVSentinel detection).

    Accepts ANY XID error code for maximum flexibility in testing.
    Pre-defined messages exist for common critical XIDs, but any XID value
    can be injected - NVSentinel will parse and handle based on its own rules.

    Pre-defined messages for all DCGM/NVSentinel monitored XIDs:

    Devastating XIDs (always trigger FAIL):
    - 79: GPU fell off bus, 74: NVLink error, 48: ECC DBE, 94/95: ECC errors
    - 119/120: GSP errors, 140: ECC unrecovered

    Subsystem XIDs (trigger WARN, may escalate):
    - Memory (31, 32, 43, 63, 64): MMU, PBDMA, page retirement errors
    - PCIe (38, 39, 42): Bus, fabric, replay rate errors
    - Thermal (60, 61, 62, 63): Temperature limit violations
    - Power (54, 56, 57): Power/clock state changes
    - Graphics (13, 45, 69): SM exceptions, preemptive cleanup

    Unknown XIDs use a generic error message format.
    """

    def __init__(self):
        self.node_name = os.getenv("NODE_NAME", "unknown")
        self.privileged = self._check_privileged()

        logger.info(f"XID Injector initialized on {self.node_name}")
        logger.info(f"Privileged: {self.privileged}")
        logger.info(f"Known XIDs with specific messages: {sorted(XID_MESSAGES.keys())}")
        logger.info("Method: nsenter+kmsg → NVSentinel detection → Full FT workflow")
        logger.info("Note: Accepts ANY XID value - unknown XIDs use generic message")

    def _check_privileged(self) -> bool:
        """Check if we have privileged access (required for nsenter)"""
        return os.geteuid() == 0

    def _normalize_pci_address(self, pci_addr: str) -> str:
        """
        Normalize PCI address from nvidia-smi format to kernel sysfs format.

        nvidia-smi returns: 00000001:00:00.0 (8-digit domain)
        kernel expects:     0001:00:00.0     (4-digit domain)

        Azure VMs use extended PCI addresses, but the kernel shortens them.
        """
        parts = pci_addr.split(":")
        if len(parts) >= 3:
            # Keep only last 4 digits of domain
            domain = parts[0][-4:] if len(parts[0]) > 4 else parts[0]
            normalized = f"{domain}:{parts[1]}:{parts[2]}"
            logger.debug(f"Normalized PCI address: {pci_addr} -> {normalized}")
            return normalized
        return pci_addr

    def inject_xid(self, xid_type: int, gpu_id: int = 0) -> Tuple[bool, str]:
        """
        Inject ANY XID error code via nsenter+kmsg.

        This method accepts any integer XID value for maximum testing flexibility.
        Pre-defined messages exist for well-known XIDs (79, 74, 48, etc.), but
        any XID can be injected. Unknown XIDs use a generic error message.

        Args:
            xid_type: XID error code (any integer, commonly 1-255)
            gpu_id: GPU device ID (default: 0)

        Returns:
            Tuple of (success: bool, message: str)
        """
        logger.info(f"Injecting XID {xid_type} for GPU {gpu_id}")

        if not self.privileged:
            return (
                False,
                f"XID {xid_type} injection requires privileged mode (nsenter needs root)",
            )

        success, msg = self._inject_fake_xid_to_kmsg(gpu_id, xid_type)

        if success:
            logger.info(f"XID {xid_type} injected successfully: {msg}")
            return True, msg
        else:
            logger.error(f"XID {xid_type} injection failed: {msg}")
            return False, msg

    # Convenience methods for specific XIDs (backward compatibility)
    def inject_xid_79_gpu_fell_off_bus(self, gpu_id: int = 0) -> Tuple[bool, str]:
        """Inject XID 79 (GPU Fell Off Bus) - most critical hardware failure."""
        return self.inject_xid(79, gpu_id)

    def inject_xid_74_nvlink_error(self, gpu_id: int = 0) -> Tuple[bool, str]:
        """Inject XID 74 (NVLink error) - multi-GPU communication failure."""
        return self.inject_xid(74, gpu_id)

    def inject_xid_48_ecc_dbe(self, gpu_id: int = 0) -> Tuple[bool, str]:
        """Inject XID 48 (Double-bit ECC error) - severe memory error."""
        return self.inject_xid(48, gpu_id)

    def inject_xid_94_ecc_contained(self, gpu_id: int = 0) -> Tuple[bool, str]:
        """Inject XID 94 (Contained ECC error) - less severe memory error."""
        return self.inject_xid(94, gpu_id)

    def inject_xid_95_uncontained(self, gpu_id: int = 0) -> Tuple[bool, str]:
        """Inject XID 95 (Uncontained error) - very severe, GPU reset required."""
        return self.inject_xid(95, gpu_id)

    def _inject_fake_xid_to_kmsg(self, gpu_id: int, xid: int) -> Tuple[bool, str]:
        """
        Inject fake XID message to host's /dev/kmsg via nsenter.

        Uses nsenter to enter all host namespaces (PID 1) and write to /dev/kmsg.
        Creates real kernel messages with proper metadata that NVSentinel can detect.

        Message format: "NVRM: NVRM: Xid (PCI:address): xid, message"
        Duplicate "NVRM:" needed because /dev/kmsg splits on first colon.

        Args:
            gpu_id: GPU device ID (from nvidia-smi)
            xid: XID error code (currently only 79 is used by public API)

        Returns:
            Tuple of (success: bool, message: str)

        Note: This method accepts any XID code as a parameter for extensibility.
        To add support for other XIDs (74, 48, 95, etc.), create corresponding
        public methods like inject_xid_74_nvlink_error() and update the error
        message template for each XID type.
        """
        try:
            # Get PCI address for the GPU
            pci_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader",
                    "-i",
                    str(gpu_id),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if pci_result.returncode != 0:
                return (
                    False,
                    f"Failed to get PCI address for GPU {gpu_id}: {pci_result.stderr}",
                )

            pci_addr_full = pci_result.stdout.strip()
            pci_addr = self._normalize_pci_address(pci_addr_full)

            # Get appropriate error message for this XID type
            # If XID is known, use specific message; otherwise use generic format
            error_msg = XID_MESSAGES.get(
                xid, f"Graphics Exception: XID {xid} occurred on GPU"
            )

            # Format XID message (duplicate "NVRM:" for /dev/kmsg parsing)
            # Format matches NVSentinel pattern: NVRM: Xid (PCI:addr): code, description
            xid_message = f"NVRM: NVRM: Xid (PCI:{pci_addr}): {xid}, {error_msg}"
            logger.debug(f"Formatted XID message: {xid_message}")

            # Write to host's /dev/kmsg via nsenter
            kmsg_message = f"<3>{xid_message}"  # <3> = kernel error priority
            nsenter_cmd = [
                "nsenter",
                "--target",
                "1",  # Target host PID 1 (init)
                "--mount",  # Enter mount namespace (for /dev/kmsg access)
                "--uts",  # Enter UTS namespace (hostname)
                "--ipc",  # Enter IPC namespace
                "--pid",  # Enter PID namespace (appear as host process)
                "--",
                "sh",
                "-c",
                f"echo '{kmsg_message}' > /dev/kmsg",
            ]

            nsenter_result = subprocess.run(
                nsenter_cmd, capture_output=True, text=True, timeout=5
            )

            if nsenter_result.returncode != 0:
                return (
                    False,
                    f"Failed to write to host /dev/kmsg: {nsenter_result.stderr}",
                )

            return (
                True,
                f"XID {xid} injected for GPU {gpu_id} (PCI: {pci_addr}) → NVSentinel",
            )

        except Exception as e:
            logger.error(f"XID injection failed: {type(e).__name__}: {e}")
            return False, f"Failed to inject XID: {e}"
