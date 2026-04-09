// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone test for XPU (Level-Zero) backend
//!
//! This test verifies basic XPU functionality without requiring NIXL or disk storage.

use anyhow::Result;
use dynamo_llm::block_manager::v2::device::{DeviceBackend, DeviceContext};

fn main() -> Result<()> {
    eprintln!("=== XPU (Level-Zero) Backend Standalone Test ===\n");

    // Check availability
    let backend = DeviceBackend::Ze;
    eprintln!("Backend: {}", backend.name());
    eprintln!("Available: {}\n", backend.is_available());

    if !backend.is_available() {
        eprintln!("XPU backend not available on this system");
        eprintln!("Make sure Level-Zero drivers are installed and devices are accessible");
        return Ok(());
    }

    // Test 1: Create device context
    eprintln!("Test 1: Creating device context for device 0...");
    let ctx = DeviceContext::new(backend, 0)?;
    eprintln!("✓ Device context created successfully");
    eprintln!("  Device ID: {}", ctx.device_id());
    eprintln!("  Backend: {:?}\n", ctx.backend());

    // Test 2: Allocate device memory
    eprintln!("Test 2: Allocating device memory (1 MB)...");
    let size = 1024 * 1024; // 1 MB
    let dev_ptr = ctx.allocate_device(size)?;
    eprintln!("✓ Device memory allocated");
    eprintln!("  Size: {} bytes", size);
    eprintln!("  Pointer: 0x{:x}\n", dev_ptr);

    // Test 3: Allocate pinned host memory
    eprintln!("Test 3: Allocating pinned host memory (1 MB)...");
    let pinned_ptr = ctx.allocate_pinned(size)?;
    eprintln!("✓ Pinned memory allocated");
    eprintln!("  Size: {} bytes", size);
    eprintln!("  Pointer: 0x{:x}\n", pinned_ptr);

    // Test 4: Create stream
    eprintln!("Test 4: Creating stream...");
    let stream = ctx.create_stream()?;
    eprintln!("✓ Stream created successfully\n");

    // Test 5: H2D transfer
    eprintln!("Test 5: Host-to-Device transfer...");
    let test_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    // Write test data to pinned memory
    unsafe {
        let pinned_slice = std::slice::from_raw_parts_mut(pinned_ptr as *mut u8, size);
        pinned_slice.copy_from_slice(&test_data);
    }

    stream.copy_h2d(dev_ptr, &test_data)?;
    eprintln!("✓ H2D transfer completed\n");

    // Test 6: D2H transfer
    eprintln!("Test 6: Device-to-Host transfer...");
    let mut result_data = vec![0u8; size];
    stream.copy_d2h(&mut result_data, dev_ptr)?;
    eprintln!("✓ D2H transfer completed\n");

    // Test 7: Stream synchronization
    eprintln!("Test 7: Stream synchronization...");
    stream.synchronize()?;
    eprintln!("✓ Stream synchronized\n");

    // Test 8: Verify data integrity
    eprintln!("Test 8: Verifying data integrity...");
    let matches = test_data == result_data;
    if matches {
        eprintln!("✓ Data verification PASSED");
        eprintln!("  All {} bytes match after H2D->D2H roundtrip\n", size);
    } else {
        eprintln!("✗ Data verification FAILED");
        let mismatches: usize = test_data
            .iter()
            .zip(result_data.iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("  Mismatches: {} / {} bytes\n", mismatches, size);
        anyhow::bail!("Data integrity check failed");
    }

    // Test 9: Event recording and synchronization
    eprintln!("Test 9: Event recording and synchronization...");
    let event = stream.record_event()?;
    eprintln!("  Event recorded");

    event.synchronize()?;
    eprintln!("  Event synchronized");

    let complete = event.is_complete()?;
    eprintln!("  Event complete: {}", complete);
    if complete {
        eprintln!("✓ Event operations successful\n");
    } else {
        eprintln!("✗ Event not marked as complete\n");
        anyhow::bail!("Event completion check failed");
    }

    // Test 10: Device-to-Device copy
    eprintln!("Test 10: Device-to-Device copy...");
    let dev_ptr2 = ctx.allocate_device(size)?;
    eprintln!("  Allocated second device buffer: 0x{:x}", dev_ptr2);

    stream.copy_d2d(dev_ptr2, dev_ptr, size)?;
    eprintln!("  D2D copy issued");

    let mut result_data2 = vec![0u8; size];
    stream.copy_d2h(&mut result_data2, dev_ptr2)?;
    stream.synchronize()?;
    eprintln!("  Copied back to host for verification");

    let matches = test_data == result_data2;
    if matches {
        eprintln!("✓ D2D copy verification PASSED\n");
    } else {
        eprintln!("✗ D2D copy verification FAILED\n");
        anyhow::bail!("D2D copy integrity check failed");
    }

    // Cleanup
    eprintln!("Cleanup: Freeing allocated memory...");
    ctx.free_device(dev_ptr)?;
    ctx.free_device(dev_ptr2)?;
    ctx.free_pinned(pinned_ptr)?;
    eprintln!("✓ Memory freed successfully\n");

    eprintln!("=== All tests PASSED ===");
    Ok(())
}
