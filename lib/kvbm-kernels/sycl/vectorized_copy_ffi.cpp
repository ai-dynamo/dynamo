// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// C FFI implementation for the SYCL vectorized_copy kernel.
//
// Uses SYCL Level Zero interop (ext::oneapi::level_zero) to create a
// sycl::queue that shares the same L0 context as the caller's Rust code.
// This means device memory allocated by ZeMemPool / zeMemAllocDevice is
// directly accessible by the SYCL kernel — no copies needed.
//
// Build:
//   cd lib/kvbm-kernels/sycl
//   icpx -fsycl -shared -fPIC -o libvectorized_copy_sycl.so \
//        vectorized_copy_ffi.cpp

#include "vectorized_copy_ffi.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// Constants — must match ze_vectorized_copy.rs and vectorized_copy.cl
// ---------------------------------------------------------------------------
static constexpr int WORK_GROUP_SIZE = 128;
static constexpr int MAX_GROUPS      = 65535;

// ---------------------------------------------------------------------------
// Kernel functor (identical to vectorized_copy.cpp)
// ---------------------------------------------------------------------------
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

            // 16-byte path
            if (((src_addr & 0xF) == 0) && ((dst_addr & 0xF) == 0) &&
                (copy_size_ >= 16)) {
                uint64_t n = copy_size_ >> 4;
                auto* sv = reinterpret_cast<sycl::ulong2*>(src);
                auto* dv = reinterpret_cast<sycl::ulong2*>(dst);
                for (uint64_t i = tid; i < n; i += local_size)
                    dv[i] = sv[i];
                vectorized_bytes = n << 4;
            }
            // 8-byte path
            else if (((src_addr & 0x7) == 0) && ((dst_addr & 0x7) == 0) &&
                     (copy_size_ >= 8)) {
                uint64_t n = copy_size_ >> 3;
                auto* sv = reinterpret_cast<uint64_t*>(src);
                auto* dv = reinterpret_cast<uint64_t*>(dst);
                for (uint64_t i = tid; i < n; i += local_size)
                    dv[i] = sv[i];
                vectorized_bytes = n << 3;
            }
            // 4-byte path
            else if (((src_addr & 0x3) == 0) && ((dst_addr & 0x3) == 0) &&
                     (copy_size_ >= 4)) {
                uint64_t n = copy_size_ >> 2;
                auto* sv = reinterpret_cast<uint32_t*>(src);
                auto* dv = reinterpret_cast<uint32_t*>(dst);
                for (uint64_t i = tid; i < n; i += local_size)
                    dv[i] = sv[i];
                vectorized_bytes = n << 2;
            }

            // Tail bytes
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

// ---------------------------------------------------------------------------
// Opaque state
// ---------------------------------------------------------------------------
struct SyclVcState {
    sycl::queue q;
};

// ---------------------------------------------------------------------------
// C FFI entry points
// ---------------------------------------------------------------------------

extern "C" SyclVcState* sycl_vc_init(void* ze_context, void* ze_device) {
    using namespace sycl;
    namespace l0 = ext::oneapi::level_zero;

    try {
        // --- Find the matching SYCL device by L0 handle ---
        device match_dev;
        bool found = false;
        for (auto& plat : platform::get_platforms()) {
            for (auto& dev : plat.get_devices(info::device_type::gpu)) {
                try {
                    auto native = get_native<backend::ext_oneapi_level_zero>(dev);
                    if (native == static_cast<ze_device_handle_t>(ze_device)) {
                        match_dev = dev;
                        found = true;
                        break;
                    }
                } catch (...) {
                    continue;
                }
            }
            if (found) break;
        }

        if (!found) {
            std::fprintf(stderr,
                "sycl_vc_init: no SYCL device matches L0 handle %p\n",
                ze_device);
            return nullptr;
        }

        // Create SYCL context from the L0 context handle.
        // ownership::keep: SYCL will NOT destroy the L0 context — Rust owns it.
        backend_input_t<backend::ext_oneapi_level_zero, context> ctx_input = {
            static_cast<ze_context_handle_t>(ze_context),
            {match_dev},
            l0::ownership::keep
        };
        auto sycl_ctx = make_context<backend::ext_oneapi_level_zero>(ctx_input);

        // Create an in-order queue on that device + context.
        auto sycl_q = queue(sycl_ctx, match_dev,
                            property::queue::in_order());

        return new SyclVcState{std::move(sycl_q)};
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sycl_vc_init failed: %s\n", e.what());
        return nullptr;
    }
}

extern "C" int sycl_vc_run(SyclVcState* state,
                            uint64_t src_addrs_dev,
                            uint64_t dst_addrs_dev,
                            uint64_t copy_size,
                            int      num_pairs) {
    if (!state || num_pairs <= 0) return -1;

    try {
        auto* src_ptrs = reinterpret_cast<uint64_t*>(src_addrs_dev);
        auto* dst_ptrs = reinterpret_cast<uint64_t*>(dst_addrs_dev);

        int num_groups = std::min(num_pairs, MAX_GROUPS);
        sycl::range<1> global(num_groups * WORK_GROUP_SIZE);
        sycl::range<1> local(WORK_GROUP_SIZE);

        state->q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<VectorizedCopyKernel>(
                sycl::nd_range<1>(global, local),
                VectorizedCopyKernel(src_ptrs, dst_ptrs, copy_size, num_pairs));
        }).wait();

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sycl_vc_run failed: %s\n", e.what());
        return -1;
    }
}

extern "C" void sycl_vc_destroy(SyclVcState* state) {
    delete state;
}
