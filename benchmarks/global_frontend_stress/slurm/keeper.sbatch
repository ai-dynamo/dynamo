#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Holds four nodes for the stress campaign until $CAMPAIGN/control/RELEASE exists.
# Every step runs on them as an `srun --overlap` step from the login node while this
# job lives. Submit from $CAMPAIGN/control (the output path below is relative to it),
# naming your partition and account: sbatch -p PARTITION -A ACCOUNT keeper.sbatch
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J gfstress.keeper
#SBATCH -o keeper-%j.out

CONTROL=${SLURM_SUBMIT_DIR:?}
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$CONTROL/nodes.txt"
echo "$SLURM_JOB_ID" > "$CONTROL/job"
date -Is > "$CONTROL/started"
while [[ ! -e "$CONTROL/RELEASE" ]]; do sleep 20; done
