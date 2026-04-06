// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SYCL C++ port of the OpenCL vectorized_copy kernel.
//
// Each work-group handles one (src, dst) pointer pair.  Work-items within the
// group cooperatively copy `copy_size_in_bytes` bytes using the widest
// naturally-aligned load/store width (16 B -> 8 B -> 4 B), then handle any
// remaining tail bytes one at a time.
//
// Compile & run the self-test:
//   icpx -fsycl -o vectorized_copy vectorized_copy.cpp && ./vectorized_copy
//
// Target a specific device (e.g., BMG):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg -o vectorized_copy \
//        vectorized_copy.cpp
// Or, if your icpx doesn't recognize the shorthand:
//   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" \
//        -o vectorized_copy vectorized_copy.cpp
//
// Emit device-only SPIR-V (no host binary):
//   icpx -fsycl -fsycl-device-only -o vectorized_copy.spv vectorized_copy.cpp
// Note: the resulting SPIR-V has mangled entry points and SYCL runtime
// metadata, so it cannot be loaded via ZeModule::from_spirv().
// Use the OpenCL .cl -> ocloc path for raw Level Zero loading.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>


/// Work-group size — must match WORK_GROUP_SIZE in ze_vectorized_copy.rs.
constexpr int WORK_GROUP_SIZE = 128;

/// Maximum number of work-groups (grid dimension X).
constexpr int MAX_GROUPS = 65535;

/// SYCL kernel functor for vectorized scatter-gather copy.
///
/// @param src_addrs   Device pointer to array of source addresses (num_pairs).
/// @param dst_addrs   Device pointer to array of destination addresses (num_pairs).
/// @param copy_size   Bytes to copy per (src, dst) pair.
/// @param num_pairs   Number of (src, dst) pairs.
class VectorizedCopyKernel {
public:
    VectorizedCopyKernel(uint64_t* src_addrs, uint64_t* dst_addrs,
                         uint64_t copy_size, int num_pairs)
        : src_addrs_(src_addrs), dst_addrs_(dst_addrs),
          copy_size_(copy_size), num_pairs_(num_pairs) {}

    void operator()(sycl::nd_item<1> item) const {
        int pair_id      = static_cast<int>(item.get_group(0));
        int group_stride = static_cast<int>(item.get_group_range(0));
        int tid          = static_cast<int>(item.get_local_id(0));
        int local_size   = static_cast<int>(item.get_local_range(0));

        for (; pair_id < num_pairs_; pair_id += group_stride) {
            auto* src = reinterpret_cast<uint8_t*>(src_addrs_[pair_id]);
            auto* dst = reinterpret_cast<uint8_t*>(dst_addrs_[pair_id]);

            auto src_addr = reinterpret_cast<uintptr_t>(src);
            auto dst_addr = reinterpret_cast<uintptr_t>(dst);
            uint64_t vectorized_bytes = 0;

            // --- 16-byte (sycl::ulong2) vectorized path ---
            if (((src_addr & 0xF) == 0) &&
                ((dst_addr & 0xF) == 0) &&
                (copy_size_ >= 16))
            {
                uint64_t n = copy_size_ >> 4;
                auto* sv = reinterpret_cast<sycl::ulong2*>(src);
                auto* dv = reinterpret_cast<sycl::ulong2*>(dst);
                for (uint64_t i = tid; i < n; i += local_size)
                    dv[i] = sv[i];
                vectorized_bytes = n << 4;
            }
            // --- 8-byte (uint64_t) vectorized path ---
            else if (((src_addr & 0x7) == 0) &&
                     ((dst_addr & 0x7) == 0) &&
                     (copy_size_ >= 8))
            {
                uint64_t n = copy_size_ >> 3;
                auto* sv = reinterpret_cast<uint64_t*>(src);
                auto* dv = reinterpret_cast<uint64_t*>(dst);
                for (uint64_t i = tid; i < n; i += local_size)
                    dv[i] = sv[i];
                vectorized_bytes = n << 3;
            }
            // --- 4-byte (uint32_t) vectorized path ---
            else if (((src_addr & 0x3) == 0) &&
                     ((dst_addr & 0x3) == 0) &&
                     (copy_size_ >= 4))
            {
                uint64_t n = copy_size_ >> 2;
                auto* sv = reinterpret_cast<uint32_t*>(src);
                auto* dv = reinterpret_cast<uint32_t*>(dst);
                for (uint64_t i = tid; i < n; i += local_size)
                    dv[i] = sv[i];
                vectorized_bytes = n << 2;
            }

            // --- Tail bytes (1-byte) ---
            uint64_t remaining = copy_size_ - vectorized_bytes;
            for (uint64_t i = tid; i < remaining; i += local_size)
                dst[vectorized_bytes + i] = src[vectorized_bytes + i];
        }
    }

private:
    uint64_t* src_addrs_;
    uint64_t* dst_addrs_;
    uint64_t  copy_size_;
    int       num_pairs_;
};

/// Submit the vectorized_copy kernel to a SYCL queue.
///
/// @param q           SYCL queue targeting the desired device.
/// @param src_addrs   Device-accessible array of source addresses.
/// @param dst_addrs   Device-accessible array of destination addresses.
/// @param copy_size   Bytes to copy per pair.
/// @param num_pairs   Number of (src, dst) pairs.
/// @return            SYCL event representing kernel completion.
sycl::event vectorized_copy(sycl::queue& q,
                            uint64_t* src_addrs,
                            uint64_t* dst_addrs,
                            uint64_t copy_size,
                            int num_pairs) {
    int num_groups = std::min(num_pairs, MAX_GROUPS);
    sycl::range<1> global(num_groups * WORK_GROUP_SIZE);
    sycl::range<1> local(WORK_GROUP_SIZE);

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<VectorizedCopyKernel>(
            sycl::nd_range<1>(global, local),
            VectorizedCopyKernel(src_addrs, dst_addrs, copy_size, num_pairs));
    });
}

// ---------------------------------------------------------------------------
// Self-contained test: allocate N pairs, fill src with a pattern, copy via
// the kernel, verify dst matches.
// ---------------------------------------------------------------------------

/// Run a single test case.  Returns true on success.
static bool test_copy(sycl::queue& q, int num_pairs, size_t copy_size,
                       const char* label) {
    // Host reference buffers.
    std::vector<std::vector<uint8_t>> src_host(num_pairs);
    std::vector<std::vector<uint8_t>> dst_host(num_pairs);
    for (int i = 0; i < num_pairs; ++i) {
        src_host[i].resize(copy_size);
        dst_host[i].resize(copy_size, 0);
        // Fill with a per-pair pattern so we can distinguish pairs.
        std::iota(src_host[i].begin(), src_host[i].end(),
                  static_cast<uint8_t>(i * 7));
    }

    // Device buffers - one allocation per pair.
    std::vector<uint8_t*> src_dev(num_pairs);
    std::vector<uint8_t*> dst_dev(num_pairs);
    for (int i = 0; i < num_pairs; ++i) {
        src_dev[i] = sycl::malloc_device<uint8_t>(copy_size, q);
        dst_dev[i] = sycl::malloc_device<uint8_t>(copy_size, q);
        q.memcpy(src_dev[i], src_host[i].data(), copy_size);
        q.memset(dst_dev[i], 0, copy_size);
    }
    q.wait();

    // Build address arrays on device.
    auto* src_addrs = sycl::malloc_device<uint64_t>(num_pairs, q);
    auto* dst_addrs = sycl::malloc_device<uint64_t>(num_pairs, q);
    {
        std::vector<uint64_t> sa(num_pairs), da(num_pairs);
        for (int i = 0; i < num_pairs; ++i) {
            sa[i] = reinterpret_cast<uint64_t>(src_dev[i]);
            da[i] = reinterpret_cast<uint64_t>(dst_dev[i]);
        }
        q.memcpy(src_addrs, sa.data(), num_pairs * sizeof(uint64_t));
        q.memcpy(dst_addrs, da.data(), num_pairs * sizeof(uint64_t));
        q.wait();
    }

    // Launch kernel.
    vectorized_copy(q, src_addrs, dst_addrs,
                    static_cast<uint64_t>(copy_size), num_pairs)
        .wait();

    // Read back.
    for (int i = 0; i < num_pairs; ++i)
        q.memcpy(dst_host[i].data(), dst_dev[i], copy_size);
    q.wait();

    // Verify.
    bool ok = true;
    for (int i = 0; i < num_pairs && ok; ++i) {
        if (std::memcmp(src_host[i].data(), dst_host[i].data(), copy_size) != 0) {
            std::fprintf(stderr, "  FAIL %s: pair %d mismatch\n", label, i);
            ok = false;
        }
    }

    // Cleanup.
    sycl::free(src_addrs, q);
    sycl::free(dst_addrs, q);
    for (int i = 0; i < num_pairs; ++i) {
        sycl::free(src_dev[i], q);
        sycl::free(dst_dev[i], q);
    }

    if (ok)
        std::printf("  PASS %s  (pairs=%d, copy_size=%zu)\n",
                    label, num_pairs, copy_size);
    return ok;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property::queue::in_order()};

    auto dev = q.get_device();
    std::printf("Device: %s\n",
                dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Running vectorized_copy self-test...\n\n");

    int passed = 0, failed = 0;

    auto run = [&](int pairs, size_t size, const char* label) {
        if (test_copy(q, pairs, size, label)) ++passed; else ++failed;
    };

    // 16-byte aligned paths (ulong2).
    run(1,    16,       "16B-align,  1 pair,  16 B");
    run(4,   256,       "16B-align,  4 pairs, 256 B");
    run(8,  4096,       "16B-align,  8 pairs, 4 KB");
    run(16, 1048576,    "16B-align, 16 pairs, 1 MB");

    // Odd sizes that exercise the tail-byte path.
    run(1,    1,        "tail-only,  1 pair,  1 B");
    run(4,    7,        "tail-only,  4 pairs, 7 B");
    run(2,   33,        "mixed,      2 pairs, 33 B");
    run(3,  1025,       "mixed,      3 pairs, 1025 B");

    // Larger realistic sizes (KV cache chunk-like).
    run(80, 262144,     "KV-like,   80 pairs, 256 KB");
    run(160, 131072,    "KV-like,  160 pairs, 128 KB");

    // Many pairs to exercise group-stride loop.
    run(100000, 64,     "many-pairs, 100K pairs, 64 B");

    std::printf("\n%d passed, %d failed.\n", passed, failed);
    return failed ? 1 : 0;
}
