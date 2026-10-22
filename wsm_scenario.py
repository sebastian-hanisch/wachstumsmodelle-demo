"""Szenario: ein Wachstumsmodell (Exponent, Attraktivität, Alterung, Fitness) mit festem Zufalls-Seed; die lineare Anbindung läuft über den schnellen Sampler."""

from dataclasses import dataclass

import numpy as np

import wsm_algorithm as W


@dataclass(frozen=True)
class Model:
    n: int
    m: int
    a: float = 1.0
    a0: float = 0.0
    nu: float = 0.0
    fitness: bool = False

    @property
    def is_linear(self) -> bool:
        """Reines Barabási–Albert-Modell (Exponent 1, keine Anfangsattraktivität, keine Alterung, keine Fitness)."""
        return self.a == 1.0 and self.a0 == 0.0 and self.nu == 0.0 and not self.fitness


def run_model(model: Model, seed: int) -> W.Growth:
    """Wachstum nach dem Modell, deterministisch aus (Modell, Zufalls-Seed); die reine lineare Anbindung nutzt den schnellen Sampler (gleiche Verteilung)."""
    rng = np.random.default_rng(seed)
    if model.is_linear:
        return W.grow_linear(model.n, model.m, rng)
    return W.grow(model.n, model.m, model.a, model.a0, model.nu, model.fitness, rng)


def theory_kind(model: Model) -> str | None:
    """Welche Theorie der Gradverteilung gilt: „linear“ (2m(m+1)/(k(k+1)(k+2))), „uniform“ (exponentiell, Zufallsanbindung) oder None."""
    if model.is_linear:
        return "linear"
    if model.a == 0.0 and model.nu == 0.0 and not model.fitness:
        return "uniform"
    return None
