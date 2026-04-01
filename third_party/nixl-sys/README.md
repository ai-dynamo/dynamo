# Rust Bindings for NIXL

Rust bindings for the NVIDIA Inference Xfer Library (NIXL). These bindings provide a safe and idiomatic Rust interface to the NIXL C++ library.

## Prerequisites

### Install Rust and Cargo using [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Install the NIXL library on your system

Refer to the [NIXL README](https://github.com/ai-dynamo/nixl/blob/main/README.md) for instructions on how to install the NIXL library on your system.



## Building

The bindings can be built using Cargo, Rust's package manager:

```bash
# From the src/bindings/rust directory
cargo build
```

### Building with Stubs (No NIXL Library Required)

You can compile the bindings using stub implementations that don't require the actual NIXL library to be installed. This is useful for:
- Development environments where NIXL isn't available
- CI/CD pipelines
- Building documentation

```bash
# Build with stub API
cargo build --features stub-api
```

**Important**: When using stubs, any attempt to actually call NIXL functions at runtime will print an error message and abort the program.
- The stubs are only meant for compilation, not execution.

### Environment Variables

- `NIXL_PREFIX`: Path to the NIXL installation (default: `/opt/nvidia/nvda_nixl`)

## Documentation

The crate is documented using Rust's standard documentation system. You can generate and view the documentation with:

```bash
cargo doc
```


## Testing

The bindings include a comprehensive test suite that can be run with:
Note that multithreading is disabled because NIXL might deadlock.

```bash
cargo test -- --test-threads=1
```

**Note**: Tests cannot be run with the `stub-api` feature as they require actual NIXL functionality.