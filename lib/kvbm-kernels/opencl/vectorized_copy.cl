// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Level Zero SPIR-V port of the CUDA vectorized_copy kernel.
//
// Each work-group handles one (src, dst) pointer pair.  Work-items within the
// group cooperatively copy `copy_size_in_bytes` bytes using the widest
// naturally-aligned load/store width (16 B -> 8 B -> 4 B), then handle any
// remaining tail bytes one at a time.
//
// Two entry points:
//
//   vectorized_copy          — 4 kernel arguments (src_addrs, dst_addrs,
//                              copy_size_in_bytes, num_pairs).
//
//   vectorized_copy_indirect — 1 kernel argument: a device-side buffer
//                              holding the same 4 values as packed ulongs.
//                              Eliminates per-dispatch zeKernelSetArgumentValue
//                              calls; args are uploaded via BCS memcpy instead.
//
// Compile to SPIR-V:
//   ocloc compile -file vectorized_copy.cl -device bmg \
//                 -out_dir . -options "-cl-std=CL2.0"
//
// OR
//
//   ocloc compile -file vectorized_copy.cl -spv_only \
//                 -options "-cl-std=CL2.0"
// The generated SPIR-V binary is loaded at runtime via
// ZeModule::from_spirv().

// ---- shared copy logic (called by both entry points) -----------------------

inline void do_vectorized_copy(
    __global const ulong* src_addrs,
    __global const ulong* dst_addrs,
    ulong copy_size_in_bytes,
    int num_pairs)
{
    int pair_id      = get_group_id(0);
    int group_stride = get_num_groups(0);
    int tid          = get_local_id(0);
    int local_size   = get_local_size(0);

    for (; pair_id < num_pairs; pair_id += group_stride) {
        __global const uchar* src = (__global const uchar*)src_addrs[pair_id];
        __global uchar*       dst = (__global uchar*)      dst_addrs[pair_id];

        ulong src_addr = (ulong)src;
        ulong dst_addr = (ulong)dst;
        ulong vectorized_bytes = 0;

        // --- 16-byte (ulong2) vectorized path ---
        if (((src_addr & 0xF) == 0) &&
            ((dst_addr & 0xF) == 0) &&
            (copy_size_in_bytes >= 16))
        {
            ulong n = copy_size_in_bytes >> 4;
            __global const ulong2* sv = (__global const ulong2*)src;
            __global ulong2*       dv = (__global ulong2*)dst;
            for (ulong i = tid; i < n; i += local_size)
                dv[i] = sv[i];
            vectorized_bytes = n << 4;
        }
        // --- 8-byte (ulong) vectorized path ---
        else if (((src_addr & 0x7) == 0) &&
                 ((dst_addr & 0x7) == 0) &&
                 (copy_size_in_bytes >= 8))
        {
            ulong n = copy_size_in_bytes >> 3;
            __global const ulong* sv = (__global const ulong*)src;
            __global ulong*       dv = (__global ulong*)dst;
            for (ulong i = tid; i < n; i += local_size)
                dv[i] = sv[i];
            vectorized_bytes = n << 3;
        }
        // --- 4-byte (uint) vectorized path ---
        else if (((src_addr & 0x3) == 0) &&
                 ((dst_addr & 0x3) == 0) &&
                 (copy_size_in_bytes >= 4))
        {
            ulong n = copy_size_in_bytes >> 2;
            __global const uint* sv = (__global const uint*)src;
            __global uint*       dv = (__global uint*)dst;
            for (ulong i = tid; i < n; i += local_size)
                dv[i] = sv[i];
            vectorized_bytes = n << 2;
        }

        // --- Tail bytes (1-byte) ---
        ulong remaining = copy_size_in_bytes - vectorized_bytes;
        for (ulong i = tid; i < remaining; i += local_size)
            dst[vectorized_bytes + i] = src[vectorized_bytes + i];
    }
}

// ---- direct-args entry point (original 4-parameter signature) --------------

__kernel void vectorized_copy(
    __global ulong* src_addrs,          // device address array (num_pairs)
    __global ulong* dst_addrs,          // device address array (num_pairs)
    ulong           copy_size_in_bytes, // bytes to copy per pair
    int             num_pairs)          // number of (src, dst) pairs
{
    do_vectorized_copy(src_addrs, dst_addrs, copy_size_in_bytes, num_pairs);
}

// ---- indirect-args entry point ---------------------------------------------
//
// Reads all parameters from a device-side buffer, eliminating
// per-dispatch zeKernelSetArgumentValue calls.
//
// args layout (4 x ulong = 32 bytes):
//   args[0] = device pointer to src_addrs array  (__global ulong*)
//   args[1] = device pointer to dst_addrs array  (__global ulong*)
//   args[2] = copy_size_in_bytes                  (ulong)
//   args[3] = num_pairs                           (cast to int)

__kernel void vectorized_copy_indirect(
    __global const ulong* args)
{
    do_vectorized_copy(
        (__global const ulong*) args[0],
        (__global const ulong*) args[1],
        args[2],
        (int) args[3]);
}
