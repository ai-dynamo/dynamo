# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(slots=True)
class BetaBandit:
    """Simple Beta-Bernoulli bandit used by the Thompson router."""

    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def observe(self, reward: float) -> None:
        reward = max(0.0, min(1.0, float(reward)))
        self.alpha += reward
        self.beta += 1.0 - reward
