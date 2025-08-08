#!/bin/bash

kind='dynamo_agg'

# tep4
max_batch=1024
tp_size=4
ep_size=${tp_size}
enable_attention_dp=false
mtp=0

concurrency_list="1 2 4 8 16 32 64 128 256 512"

max_num_tokens=$(( ((mtp+1)*max_batch+isl+128+63)/64*64 ))
sbatch --nodes=1 --ntasks=${tp_size} --ntasks-per-node=${tp_size} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} ${concurrency_list} ${mtp} ${kind}

# dep4
max_batch=1024
tp_size=4
ep_size=${tp_size}
enable_attention_dp=true
mtp=0

concurrency_list="32 64 128 256 512 1024"
max_num_tokens=$(( ((mtp+1)*max_batch+isl+128+63)/64*64 ))
sbatch --nodes=1 --ntasks=${tp_size} --ntasks-per-node=${tp_size} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} ${concurrency_list} ${mtp} ${kind}

concurrency_list="2048 4096"
max_num_tokens=$(( ((mtp+1)*max_batch+isl+128+63)/64*64 ))
sbatch --nodes=1 --ntasks=${tp_size} --ntasks-per-node=${tp_size} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} ${concurrency_list} ${mtp} ${kind}


# tep8
max_batch=1024
tp_size=8
ep_size=${tp_size}
enable_attention_dp=false
mtp=0

concurrency_list="1 2 4 8 16 32 64 128 256 512"
max_num_tokens=$(( ((mtp+1)*max_batch+isl+128+63)/64*64 ))
sbatch --nodes=1 --ntasks=${tp_size} --ntasks-per-node=${tp_size} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} ${concurrency_list} ${mtp} ${kind}

# dep8
max_batch=1024
tp_size=8
ep_size=${tp_size}
enable_attention_dp=true
mtp=0

concurrency_list="32 64 128 256 512 1024"
max_num_tokens=$(( ((mtp+1)*max_batch+isl+128+63)/64*64 ))
sbatch --nodes=1 --ntasks=${tp_size} --ntasks-per-node=${tp_size} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} ${concurrency_list} ${mtp} ${kind}

concurrency_list="2048 4096"
max_num_tokens=$(( ((mtp+1)*max_batch+isl+128+63)/64*64 ))
sbatch --nodes=1 --ntasks=${tp_size} --ntasks-per-node=${tp_size} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} ${concurrency_list} ${mtp} ${kind}

# New: dep8 concurrency greater than 4096 as a separate group
concurrency_list="6144 8192"
max_num_tokens=$(( ((mtp+1)*max_batch+isl+128+63)/64*64 ))
sbatch --nodes=1 --ntasks=${tp_size} --ntasks-per-node=${tp_size} benchmark_agg.slurm ${tp_size} ${ep_size} ${max_batch} ${max_num_tokens} ${enable_attention_dp} ${concurrency_list} ${mtp} ${kind}
