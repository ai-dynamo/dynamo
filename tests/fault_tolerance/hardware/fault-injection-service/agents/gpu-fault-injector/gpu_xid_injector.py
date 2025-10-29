# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
GPU XID 79 Error Injector via nsenter+kmsg.

Injects fake XID 79 messages to host's /dev/kmsg to trigger NVSentinel detection.
Uses nsenter to enter host namespaces and write kernel messages that NVSentinel
syslog-health-monitor can detect naturally.

Method: nsenter --target 1 (all namespaces) → echo to /dev/kmsg → NVSentinel detection
"""

import logging
import os
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)


class GPUXIDInjectorKernel:
    """XID 79 injector via nsenter+kmsg (triggers NVSentinel detection)"""

    def __init__(self):
        self.node_name = os.getenv("NODE_NAME", "unknown")
        self.privileged = self._check_privileged()
        
        logger.info(f"XID 79 Injector initialized on {self.node_name}")
        logger.info(f"Privileged: {self.privileged}")
        logger.info(f"Method: nsenter+kmsg → NVSentinel detection → Full FT workflow")

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
        parts = pci_addr.split(':')
        if len(parts) >= 3:
            # Keep only last 4 digits of domain
            domain = parts[0][-4:] if len(parts[0]) > 4 else parts[0]
            normalized = f"{domain}:{parts[1]}:{parts[2]}"
            logger.debug(f"Normalized PCI address: {pci_addr} -> {normalized}")
            return normalized
        return pci_addr

    def inject_xid_79_gpu_fell_off_bus(self, gpu_id: int = 0) -> Tuple[bool, str]:
        """
        Inject XID 79 (GPU Fell Off Bus) via nsenter+kmsg.
        
        Writes XID message to host's /dev/kmsg → NVSentinel detects → Full FT workflow
        
        Returns: (success, message)
        """
        logger.info(f"Injecting XID 79 for GPU {gpu_id}")
        
        if not self.privileged:
            return False, "XID 79 injection requires privileged mode (nsenter needs root)"
        
        success, msg = self._inject_fake_xid_to_kmsg(gpu_id, 79)
        
        if success:
            logger.info(f"XID 79 injected successfully: {msg}")
            return True, msg
        else:
            logger.error(f"XID 79 injection failed: {msg}")
            return False, msg
    
    def _inject_fake_xid_to_kmsg(self, gpu_id: int, xid: int) -> Tuple[bool, str]:
        """
        Inject fake XID message to host's /dev/kmsg via nsenter.
        
        Uses nsenter to enter all host namespaces (PID 1) and write to /dev/kmsg.
        Creates real kernel messages with proper metadata that NVSentinel can detect.
        
        Message format: "NVRM: NVRM: Xid (PCI:address): xid, message"
        Duplicate "NVRM:" needed because /dev/kmsg splits on first colon.
        """
        try:
            # Get PCI address for the GPU
            pci_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(gpu_id)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if pci_result.returncode != 0:
                return False, f"Failed to get PCI address for GPU {gpu_id}: {pci_result.stderr}"
            
            pci_addr_full = pci_result.stdout.strip()
            pci_addr = self._normalize_pci_address(pci_addr_full)
            
            # Format XID message (duplicate "NVRM:" for /dev/kmsg parsing)
            xid_message = f"NVRM: NVRM: Xid (PCI:{pci_addr}): {xid}, GPU has fallen off the bus."
            
            # Write to host's /dev/kmsg via nsenter
            kmsg_message = f"<3>{xid_message}"  # <3> = kernel error priority
            nsenter_cmd = [
                "nsenter",
                "--target", "1",     # Target host PID 1 (init)
                "--mount",           # Enter mount namespace (for /dev/kmsg access)
                "--uts",             # Enter UTS namespace (hostname)
                "--ipc",             # Enter IPC namespace
                "--pid",             # Enter PID namespace (appear as host process)
                "--",
                "sh", "-c",
                f"echo '{kmsg_message}' > /dev/kmsg"
            ]
            
            nsenter_result = subprocess.run(
                nsenter_cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if nsenter_result.returncode != 0:
                return False, f"Failed to write to host /dev/kmsg: {nsenter_result.stderr}"
            
            return True, f"XID {xid} injected for GPU {gpu_id} (PCI: {pci_addr}) → NVSentinel"
                
        except Exception as e:
            logger.error(f"XID injection failed: {type(e).__name__}: {e}")
            return False, f"Failed to inject XID: {e}"

