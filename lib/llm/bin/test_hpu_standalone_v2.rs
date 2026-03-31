// Standalone HPU test - mimics the unit test but as a binary

use dynamo_llm::block_manager::v2::device::{DeviceBackend, DeviceContext};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Standalone HPU Test ===\n");

    // Create DeviceContext (same as unit tests)
    println!("Creating HPU DeviceContext for device 0...");
    let ctx = Arc::new(DeviceContext::new(DeviceBackend::Hpu, 0)?);
    println!("✓ DeviceContext created\n");

    // Try allocation (same as unit tests)
    println!("Allocating 1024 bytes...");
    let dev_ptr = ctx.allocate_device(1024)?;
    println!("✓ Allocated at 0x{:x}\n", dev_ptr);

    // Free it
    println!("Freeing memory...");
    ctx.free_device(dev_ptr)?;
    println!("✓ Memory freed\n");

    println!("=== ✅ Success! ===");
    Ok(())
}
