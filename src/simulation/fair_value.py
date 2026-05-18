"""
src/simulation/fair_value.py
-----------------------------
Latent Fair Value Process

The "fair value" (also called the true price or fundamental value) is the
price the asset SHOULD trade at based on all available information.

In real markets this is unobservable. In our simulation it is the ground
truth against which we measure market efficiency and measure how quickly
the midprice converges to it.

Process
-------
The fair value follows a random walk with:
    1. Gaussian diffusion (continuous small movements)
    2. Jump component (sudden news shocks, regime shifts)

This mirrors the microstructure literature (Glosten-Milgrom, Kyle model).

    V(t+1) = V(t) + drift + σ * ε + J * Bernoulli(λ)

Where:
    drift : small systematic drift (can be zero)
    σ     : per-step volatility
    ε     ~ N(0,1)
    J     ~ N(0, jump_std)
    λ     : jump probability per step

The jump component creates the "news events" that informed traders exploit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class FairValueConfig:
    """Configuration for the fair value process."""
    initial_price: float = 100.0
    drift: float = 0.0
    volatility: float = 0.05        # per-step std of continuous component
    jump_prob: float = 0.02         # probability of a jump each step
    jump_std: float = 0.50          # std of jump size (mean zero)
    min_price: float = 1.0          # floor (prevents negative prices)


class FairValueProcess:
    """
    Generates the sequence of latent fair values for the simulation.

    Parameters
    ----------
    config : FairValueConfig
    random_seed : int, optional
    """

    def __init__(
        self,
        config: Optional[FairValueConfig] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.config = config or FairValueConfig()
        self._rng = random.Random(random_seed)
        self._value = self.config.initial_price
        self._step = 0
        self._history: list[float] = [self._value]
        self._jumps: list[int] = []  # timesteps where jumps occurred

    @property
    def value(self) -> float:
        """Current fair value."""
        return self._value

    @property
    def history(self) -> list[float]:
        """Full history of fair values."""
        return list(self._history)

    @property
    def jump_steps(self) -> list[int]:
        """Timesteps at which jumps occurred."""
        return list(self._jumps)

    def step(self) -> float:
        """
        Advance the fair value by one timestep.

        Returns
        -------
        float : new fair value
        """
        cfg = self.config
        self._step += 1

        # Continuous component
        diffusion = cfg.drift + cfg.volatility * self._rng.gauss(0, 1)

        # Jump component
        jump = 0.0
        if self._rng.random() < cfg.jump_prob:
            jump = self._rng.gauss(0, cfg.jump_std)
            self._jumps.append(self._step)

        self._value = max(cfg.min_price, self._value + diffusion + jump)
        self._history.append(self._value)
        return self._value

    def reset(self) -> None:
        """Reset to initial conditions."""
        self._value = self.config.initial_price
        self._step = 0
        self._history = [self._value]
        self._jumps = []
