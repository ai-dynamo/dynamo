// Minimal test to isolate HPU allocation issue
// Run with: cargo test --package dynamo-llm --lib --test test_hpu_minimal --features hpu

#[cfg(all(test, feature = "hpu"))]
mod hpu_minimal_test {
    use synapse::{Context, Device};
    use synapse::synapse_sys;

    #[test]
    fn test_minimal_hpu_allocation() {
        println!("\n=== Testing HPU Minimal Allocation ===");

        // Step 1: Initialize Synapse runtime
        println!("1. Initializing Synapse runtime...");
        let _ctx = Context::new().expect("Failed to initialize Synapse runtime");
        println!("   ✓ Synapse runtime initialized");

        // Step 2: Acquire device using acquire_first()
        println!("2. Acquiring first device...");
        let device = Device::acquire_first().expect("Failed to acquire first device");
        println!("   ✓ Device acquired: ID = {}", device.id());

        // Step 3: Allocate device memory
        println!("3. Allocating 8192 bytes on device...");
        let mut addr: u64 = 0;
        let size = 8192u64;
        let status = unsafe {
            synapse_sys::runtime::synDeviceMalloc(
                device.id(),
                size,
                0,  // reqAddr
                0,  // flags
                &mut addr as *mut _,
            )
        };

        assert_eq!(status, synapse_sys::synStatus_synSuccess,
            "Device allocation failed with status: {:?} (decimal: {})", status, status);

        println!("   ✓ Successfully allocated {} bytes at 0x{:x}", size, addr);

        // Free it
        let free_status = unsafe {
            synapse_sys::runtime::synDeviceFree(device.id(), addr, 0)
        };

        assert_eq!(free_status, synapse_sys::synStatus_synSuccess,
            "Device free failed with status: {:?}", free_status);

        println!("   ✓ Successfully freed memory");
        println!("=== ✅ All tests passed! ===\n");
    }

    #[test]
    fn test_unowned_device_allocation() {
        println!("\n=== Testing Unowned Device Allocation ===");

        // Initialize runtime
        let _ctx = Context::new().expect("Failed to initialize Synapse runtime");

        // Try using unowned device reference
        let device = Device::from_id_unowned(0);
        println!("Created unowned device reference: ID = {}", device.id());

        // Try to allocate with unowned device
        let mut addr: u64 = 0;
        let size = 4096u64;
        let status = unsafe {
            synapse_sys::runtime::synDeviceMalloc(
                device.id(),
                size,
                0,
                0,
                &mut addr as *mut _,
            )
        };

        if status == synapse_sys::synStatus_synSuccess {
            println!("✓ Unowned device CAN allocate memory!");
            unsafe {
                synapse_sys::runtime::synDeviceFree(device.id(), addr, 0);
            }
        } else {
            println!("✗ Unowned device CANNOT allocate memory (status: {})", status);
            println!("This explains the benchmark failure!");
        }
    }
}
