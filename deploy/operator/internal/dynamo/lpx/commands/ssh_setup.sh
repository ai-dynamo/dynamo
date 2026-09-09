# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

mkdir -p $HOME/.ssh $HOME/.ssh/host_keys
ls -la /ssh-pk/
cp /ssh-pk/private.key $HOME/.ssh/id_rsa
cp /ssh-pk/private.key.pub $HOME/.ssh/id_rsa.pub
cp /ssh-pk/private.key.pub $HOME/.ssh/authorized_keys
chmod 600 $HOME/.ssh/id_rsa $HOME/.ssh/authorized_keys
chmod 644 $HOME/.ssh/id_rsa.pub
printf 'Host *\nIdentityFile '$HOME'/.ssh/id_rsa\nStrictHostKeyChecking no\nPort @@LPX_SSH_PORT@@\n' > $HOME/.ssh/config

ssh-keygen -t rsa -f $HOME/.ssh/host_keys/ssh_host_rsa_key -N ''
ssh-keygen -t ecdsa -f $HOME/.ssh/host_keys/ssh_host_ecdsa_key -N ''
ssh-keygen -t ed25519 -f $HOME/.ssh/host_keys/ssh_host_ed25519_key -N ''
