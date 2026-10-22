"""Kern: wachsende Netze. Jeder neue Knoten verbindet sich mit m bestehenden nach einer Gewichtsregel (bevorzugte Anbindung und Varianten),
dazu Gradverteilung, Exponent-Schätzer und Theorie. Jede Regel ist eine eigene kleine Funktion (DEMO-PLAYBOOK §1b); Zufall nur über einen übergebenen Generator."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class Growth:
    deg: np.ndarray             # Grad jedes Knotens am Ende
    birth: np.ndarray           # Ankunftszeit (Knotennummer)
    fitness: np.ndarray | None  # Fitness je Knoten oder None
    edges: np.ndarray           # (Zahl der Kanten, 2): (neuer Knoten, Ziel)
    kmax_trace: np.ndarray      # größter Grad nach der Ankunft jedes Knotens (Länge n)


def attachment_weights(deg: np.ndarray, birth: np.ndarray, fitness: np.ndarray | None, t: int, a: float, A: float, nu: float) -> np.ndarray:
    """Gewicht jedes bestehenden Knotens (0..t−1) beim Eintreffen von Knoten t: (k + A)^a · (t − Ankunft + 1)^(−nu) · Fitness."""
    w = (deg[:t] + A) ** a
    if nu:
        w = w * (t - birth[:t] + 1.0) ** (-nu)
    if fitness is not None:
        w = w * fitness[:t]
    return w


def draw_targets(w: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """m verschiedene Ziele, nacheinander ohne Zurücklegen mit Wahrscheinlichkeit proportional zum Gewicht (bei weniger als m Knoten mit positivem Gewicht: alle davon)."""
    w = np.array(w, dtype=float)
    out = []
    for _ in range(m):
        total = w.sum()
        if total <= 0:
            break
        cdf = np.cumsum(w)
        i = int(np.searchsorted(cdf, rng.random() * total, side="right"))
        i = min(i, len(w) - 1)
        out.append(i)
        w[i] = 0.0
    return np.array(out, dtype=np.int64)


def make_fitness(n: int, rng: np.random.Generator) -> np.ndarray:
    """Gleichverteilte Fitness in (0, 1]."""
    return 1.0 - rng.random(n)


def grow(n: int, m: int, a: float, A: float, nu: float, use_fitness: bool, rng: np.random.Generator) -> Growth:
    """Wachstum von m + 1 vollständig verbundenen Startknoten auf n Knoten; jeder neue Knoten wählt m verschiedene Ziele nach `attachment_weights`. Quadratische Laufzeit in n."""
    n = max(n, m + 1)
    deg = np.zeros(n, dtype=np.int64)
    birth = np.arange(n, dtype=float)
    fitness = make_fitness(n, rng) if use_fitness else None
    deg[: m + 1] = m
    edges = [(u, v) for u in range(m + 1) for v in range(u)]
    trace = np.empty(n, dtype=np.int64)
    trace[: m + 1] = m
    kmax = m
    for t in range(m + 1, n):
        targets = draw_targets(attachment_weights(deg, birth, fitness, t, a, A, nu), m, rng)
        deg[targets] += 1
        deg[t] = len(targets)
        edges.extend((t, int(v)) for v in targets)
        kmax = max(kmax, int(deg[targets].max()) if len(targets) else 0, int(deg[t]))
        trace[t] = kmax
    return Growth(deg, birth, fitness, np.array(edges, dtype=np.int64).reshape(-1, 2), trace)


def grow_linear(n: int, m: int, rng: np.random.Generator) -> Growth:
    """Lineare bevorzugte Anbindung (Barabási–Albert) in fast linearer Zeit: jeder Knoten steht so oft in der Liste der Kantenenden, wie sein Grad ist; m verschiedene Enden zufällig ziehen
    ist dieselbe Verteilung wie die Gewichte (k + 0)^1 mit Ziehen ohne Zurücklegen."""
    n = max(n, m + 1)
    deg = np.zeros(n, dtype=np.int64)
    ends: list[int] = []
    edges = []
    for u in range(m + 1):
        for v in range(u):
            ends += [u, v]
            edges.append((u, v))
    deg[: m + 1] = m
    trace = np.empty(n, dtype=np.int64)
    trace[: m + 1] = m
    kmax = m
    u = rng.random(size=n * m * 4 + 64)
    pos = 0
    for t in range(m + 1, n):
        chosen: list[int] = []
        while len(chosen) < m:
            if pos >= len(u):
                u = rng.random(size=len(u))
                pos = 0
            v = ends[int(u[pos] * len(ends))]
            pos += 1
            if v not in chosen:
                chosen.append(v)
        for v in chosen:
            deg[v] += 1
            ends.append(v)
            ends.append(t)
            edges.append((t, v))
            if deg[v] > kmax:
                kmax = int(deg[v])
        deg[t] = m
        trace[t] = kmax
    return Growth(deg, np.arange(n, dtype=float), None, np.array(edges, dtype=np.int64).reshape(-1, 2), trace)


# --- Kennzahlen ----------------------------------------------------------------------------------------------------------------------------


def degree_counts(deg) -> dict[int, int]:
    """Zahl der Knoten je Grad (nur Grade über null)."""
    vals, counts = np.unique(np.asarray(deg), return_counts=True)
    return {int(k): int(c) for k, c in zip(vals, counts) if k > 0}


def mle_exponent(deg, kmin: int) -> float:
    """Exponent des Potenzgesetzes P(k) ~ k^(−α) für die Grade ab `kmin` (Schätzer für diskrete Daten: 1 + N / Σ ln(k / (kmin − 0,5))); nan bei weniger als zwei Knoten oder kmin unter 1."""
    x = np.asarray([k for k in deg if k >= kmin], dtype=float)
    if len(x) < 2 or kmin < 1:
        return float("nan")
    s = float(np.sum(np.log(x / (kmin - 0.5))))
    return float("nan") if s <= 0 else 1.0 + len(x) / s


def max_degree(deg) -> int:
    return int(np.max(deg)) if len(deg) else 0


def hub_share(deg) -> float:
    """Anteil des größten Grades an der Summe aller Grade (der größte Hub hält so viele Kantenenden)."""
    total = float(np.sum(deg))
    return 0.0 if total == 0 else float(np.max(deg)) / total


# --- Theorie -------------------------------------------------------------------------------------------------------------------------------


def linear_degree_pmf(k: int, m: int) -> float:
    """Gradverteilung der linearen bevorzugten Anbindung im Grenzfall n → ∞: 2m(m+1) / (k(k+1)(k+2)) für k ≥ m, sonst 0."""
    return 0.0 if k < m else 2.0 * m * (m + 1) / (k * (k + 1) * (k + 2))


def uniform_degree_pmf(k: int, m: int) -> float:
    """Gradverteilung der Zufallsanbindung (jedes Ziel gleich wahrscheinlich) im Grenzfall n → ∞: (1 / (m+1)) · (m / (m+1))^(k − m) für k ≥ m, sonst 0."""
    return 0.0 if k < m else (1.0 / (m + 1)) * (m / (m + 1.0)) ** (k - m)


def theory_exponent(m: int, A: float) -> float:
    """Exponent der linearen Anbindung mit anfänglicher Attraktivität A (Gewicht k + A): 3 + A / m."""
    return 3.0 + A / m


def edge_count(n: int, m: int) -> int:
    """Zahl der Kanten des Wachstums: Startklique m(m+1)/2 plus m je weiterem Knoten."""
    n = max(n, m + 1)
    return m * (m + 1) // 2 + (n - m - 1) * m


def expected_max_degree_linear(n: int, m: int) -> float:
    """Größenordnung des größten Grades der linearen Anbindung: m · √n (Mean-Field-Näherung, nur eine Größenordnung)."""
    return m * math.sqrt(n)
