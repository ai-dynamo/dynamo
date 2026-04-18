// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Level Zero SPIR-V port of the CUDA block/universal permute kernels.
//
// Mirrors the CUDA kernels in cuda/tensor_kernels.cu:
//   - universal_from_block  (block stacks -> universal tensor)
//   - block_from_universal  (universal tensor -> block stacks)
//
// Tensor views:
//   Universal: contiguous [nh, nl, no, nt, hd] per block.
//   Block:     nl*no pointers per block, each chunk is either
//              NHD=[nt, nh, hd] or HND=[nh, nt, hd].
//
// Pointer tables (device-accessible):
//   universal_ptrs : [num_blocks]         pointers to universal buffers
//   block_ptrs     : [num_blocks*nl*no]   pointers to block chunks
//
// Since the permute operation only rearranges elements without arithmetic,
// we treat all data types as raw bytes.  The `elem_size` argument (2, 4,
// or 8) tells the kernel how many bytes to copy per element.
//
// Layout encoding:
//   layout == 0  ->  NHD  (chunk inner axes: [nt, nh, hd])
//   layout == 1  ->  HND  (chunk inner axes: [nh, nt, hd])
//
// Compile to SPIR-V:
//   ocloc compile -file tensor_permute.cl -spv_only \
//                 -options "-cl-std=CL2.0"

// ---- Index helpers ----------------------------------------------------------

/// Compute the within-chunk flat byte offset for a given (nt_idx, nh_idx, hd_idx).
inline ulong block_inner_offset(
    int layout,
    ulong nt_idx, ulong nh_idx, ulong hd_idx,
    ulong nt,     ulong nh,     ulong hd,
    ulong elem_size)
{
    ulong elem_offset;
    if (layout == 0) {
        // NHD: [nt, nh, hd]
        elem_offset = ((nt_idx * nh) + nh_idx) * hd + hd_idx;
    } else {
        // HND: [nh, nt, hd]
        elem_offset = ((nh_idx * nt) + nt_idx) * hd + hd_idx;
    }
    return elem_offset * elem_size;
}

/// Copy `elem_size` bytes from src to dst.  Handles 2, 4, and 8 byte elements
/// with naturally-aligned loads/stores; falls back to byte-by-byte otherwise.
inline void copy_element(
    __global const uchar* src,
    __global uchar*       dst,
    ulong elem_size)
{
    if (elem_size == 8) {
        *((__global ulong*)dst) = *((__global const ulong*)src);
    } else if (elem_size == 4) {
        *((__global uint*)dst) = *((__global const uint*)src);
    } else if (elem_size == 2) {
        *((__global ushort*)dst) = *((__global const ushort*)src);
    } else {
        for (ulong i = 0; i < elem_size; ++i)
            dst[i] = src[i];
    }
}

// ---- universal_from_block ---------------------------------------------------
//
// Copy block stacks -> universal tensors.
//
// Each work-item handles one element identified by a global linear index
// across all blocks.  The linear index is decomposed into
// (block_idx, nh_idx, nl_idx, no_idx, nt_idx, hd_idx) matching the
// universal [nh, nl, no, nt, hd] layout.
//
// Parameters (all ulong except layout which is int):
//   universal_ptrs  - device pointer to array of universal buffer pointers
//   block_ptrs      - device pointer to array of block chunk pointers
//   num_blocks      - number of independent blocks
//   nh, nl, no, nt, hd - tensor dimensions
//   elem_size       - bytes per element (2=f16/bf16, 4=f32, 8=f64)
//   layout          - 0=NHD, 1=HND

__kernel void universal_from_block(
    __global const ulong* universal_ptrs,
    __global const ulong* block_ptrs,
    ulong num_blocks,
    ulong nh,
    ulong nl,
    ulong no,
    ulong nt,
    ulong hd,
    ulong elem_size,
    int   layout)
{
    ulong total_per_block = nh * nl * no * nt * hd;
    ulong total = total_per_block * num_blocks;

    ulong gid    = get_global_id(0);
    ulong stride = get_global_size(0);

    for (ulong thread_id = gid; thread_id < total; thread_id += stride) {
        ulong block_idx = thread_id / total_per_block;
        ulong residual  = thread_id % total_per_block;

        // Decompose residual -> (nh_idx, nl_idx, no_idx, nt_idx, hd_idx)
        ulong tmp = residual;
        ulong hd_idx = tmp % hd;  tmp /= hd;
        ulong nt_idx = tmp % nt;  tmp /= nt;
        ulong no_idx = tmp % no;  tmp /= no;
        ulong nl_idx = tmp % nl;  tmp /= nl;
        ulong nh_idx = tmp;

        // Source: block chunk pointer
        ulong block_stride = nl * no;
        ulong chunk_ptr_idx = block_idx * block_stride + nl_idx * no + no_idx;
        __global const uchar* chunk_base = (__global const uchar*) block_ptrs[chunk_ptr_idx];
        ulong chunk_byte_offset = block_inner_offset(layout, nt_idx, nh_idx, hd_idx, nt, nh, hd, elem_size);

        // Destination: universal buffer
        __global uchar* univ_base = (__global uchar*) universal_ptrs[block_idx];
        ulong univ_byte_offset = residual * elem_size;

        copy_element(chunk_base + chunk_byte_offset, univ_base + univ_byte_offset, elem_size);
    }
}

// ---- block_from_universal ---------------------------------------------------
//
// Copy universal tensors -> block stacks  (inverse of universal_from_block).

__kernel void block_from_universal(
    __global const ulong* universal_ptrs,
    __global const ulong* block_ptrs,
    ulong num_blocks,
    ulong nh,
    ulong nl,
    ulong no,
    ulong nt,
    ulong hd,
    ulong elem_size,
    int   layout)
{
    ulong total_per_block = nh * nl * no * nt * hd;
    ulong total = total_per_block * num_blocks;

    ulong gid    = get_global_id(0);
    ulong stride = get_global_size(0);

    for (ulong thread_id = gid; thread_id < total; thread_id += stride) {
        ulong block_idx = thread_id / total_per_block;
        ulong residual  = thread_id % total_per_block;

        // Decompose residual -> (nh_idx, nl_idx, no_idx, nt_idx, hd_idx)
        ulong tmp = residual;
        ulong hd_idx = tmp % hd;  tmp /= hd;
        ulong nt_idx = tmp % nt;  tmp /= nt;
        ulong no_idx = tmp % no;  tmp /= no;
        ulong nl_idx = tmp % nl;  tmp /= nl;
        ulong nh_idx = tmp;

        // Source: universal buffer
        __global const uchar* univ_base = (__global const uchar*) universal_ptrs[block_idx];
        ulong univ_byte_offset = residual * elem_size;

        // Destination: block chunk pointer
        ulong block_stride = nl * no;
        ulong chunk_ptr_idx = block_idx * block_stride + nl_idx * no + no_idx;
        __global uchar* chunk_base = (__global uchar*) block_ptrs[chunk_ptr_idx];
        ulong chunk_byte_offset = block_inner_offset(layout, nt_idx, nh_idx, hd_idx, nt, nh, hd, elem_size);

        copy_element(univ_base + univ_byte_offset, chunk_base + chunk_byte_offset, elem_size);
    }
}
