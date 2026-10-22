"""Kennzahlen, Messreihen und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text)."""

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import wsm_algorithm as W
import wsm_constants as C
from wsm_scenario import Model, run_model, theory_kind


def num(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    a: float = C.DEFAULT_A
    a0: float = C.DEFAULT_A0
    nu: float = C.DEFAULT_NU
    fitness: str = C.DEFAULT_FITNESS
    seed: int = C.DEFAULT_SEED

    @property
    def model(self) -> Model:
        return Model(self.n, self.m, self.a, self.a0, self.nu, self.fitness == "uniform")


@dataclass(eq=False)
class Analysis:
    settings: Settings
    growth: W.Growth
    kmin: int          # Untergrenze des Exponent-Schätzers
    alpha: float       # geschätzter Exponent des Potenzgesetzes (nan, wenn zu wenige Knoten)
    kmax: int
    share: float       # Anteil des größten Grades an allen Kantenenden
    reference: float   # m · √n, Größenordnung des größten Grades der linearen Anbindung
    theory: str | None


def analyse(settings: Settings) -> Analysis:
    model = settings.model
    g = run_model(model, settings.seed)
    kmin = C.MLE_KMIN_FACTOR * settings.m
    return Analysis(settings, g, kmin, W.mle_exponent(g.deg, kmin), W.max_degree(g.deg), W.hub_share(g.deg), W.expected_max_degree_linear(settings.n, settings.m), theory_kind(model))


# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------


def series_exponent(n: int = C.EXP_N, m: int = C.EXP_M, kmins=C.EXP_KMINS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Geschätzter Exponent der linearen Anbindung je Untergrenze kmin (Median über die festen Pläne) neben der Theorie 3."""
    degs = [run_model(Model(n, m), s).deg for s in seeds]
    return [{"kmin": k, "alpha": float(median(W.mle_exponent(d, k) for d in degs)), "tail": float(median(int((d >= k).sum()) for d in degs))} for k in kmins]


def series_hub_growth(ns=C.HUB_NS, m: int = C.HUB_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Größter Grad der linearen Anbindung über n und geteilt durch √n (Median über die Pläne)."""
    rows = []
    for n in ns:
        kmaxs = [W.max_degree(run_model(Model(n, m), s).deg) for s in seeds]
        k = float(median(kmaxs))
        rows.append({"n": n, "kmax": k, "ratio": k / math.sqrt(n), "low": min(kmaxs), "high": max(kmaxs)})
    return rows


def series_models(n: int = C.MODEL_N, m: int = C.MODEL_M, models=C.MODELS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Modelle im Vergleich: größter Grad, Anteil des größten Hubs und Exponent-Schätzer (Median über die Pläne)."""
    rows = []
    for name, a, a0, nu, fit in models:
        kmaxs, shares, alphas = [], [], []
        for s in seeds:
            g = run_model(Model(n, m, a, a0, nu, fit), s)
            kmaxs.append(W.max_degree(g.deg))
            shares.append(W.hub_share(g.deg))
            alphas.append(W.mle_exponent(g.deg, C.MLE_KMIN_FACTOR * m))
        rows.append({"name": name, "a": a, "a0": a0, "nu": nu, "fitness": fit, "kmax": float(median(kmaxs)), "share": float(median(shares)), "alpha": float(median(alphas))})
    return rows


def series_kernel(exponents=C.KERNEL_EXPONENTS, n: int = C.KERNEL_N, m: int = C.KERNEL_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Anteil des größten Hubs an allen Kantenenden über dem Exponenten der Anbindung (Median über die Pläne)."""
    rows = []
    for a in exponents:
        shares, kmaxs = [], []
        for s in seeds:
            g = run_model(Model(n, m, a), s)
            shares.append(W.hub_share(g.deg))
            kmaxs.append(W.max_degree(g.deg))
        rows.append({"a": a, "share": float(median(shares)), "kmax": float(median(kmaxs)), "low": min(shares), "high": max(shares)})
    return rows


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------


def verdict_model(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: was entsteht bei diesem Modell, ein Potenzgesetz, ein einziger Hub oder gar keine Hubs?"""
    s = a.settings
    base = f"Größter Grad {a.kmax} bei n = {s.n} und m = {s.m} (die lineare Anbindung liegt etwa bei m·√n = {num(a.reference, 0)}); der größte Hub hält {num(a.share * 100, 1)} % aller Kantenenden"
    if a.share >= 0.1:
        return "warning", base + ". Ein einziger Hub zieht einen großen Teil aller Kanten an sich: das Netz ist nicht mehr skalenfrei, sondern von einem Zentrum beherrscht."
    if a.theory == "linear":
        return "success", base + f". Der Exponent des Potenzgesetzes ist geschätzt α̂ = {num(a.alpha, 2)} ab Grad {a.kmin} (Theorie für n → ∞: 3,0)."
    if a.theory == "uniform":
        return "info", base + ". Ohne Bevorzugung fällt die Gradverteilung exponentiell, es gibt kein Potenzgesetz; der Exponent-Schätzer liefert trotzdem eine Zahl, sie ist bedeutungslos."
    return "info", base + f". Der Exponent-Schätzer ab Grad {a.kmin} ergibt α̂ = {num(a.alpha, 2)}; eine Theorie der Gradverteilung gibt es für diese Einstellung hier nicht."


def verdict_exponent(rows: list[dict], hub: list[dict]) -> tuple[str, str]:
    """Experiment Exponent: Schätzer bei kleinster und größter Untergrenze, und das Verhältnis größter Grad zu √n über die Größen."""
    lo, hi = rows[0], rows[-1]
    ratios = [r["ratio"] for r in hub]
    return "info", (f"Der geschätzte Exponent hängt von der Untergrenze ab: α̂ = {num(lo['alpha'], 2)} ab Grad {lo['kmin']} und {num(hi['alpha'], 2)} ab Grad {hi['kmin']} (Theorie für n → ∞: 3,0). "
                    f"Der größte Grad geteilt durch √n liegt bei n = {hub[0]['n']} bis {hub[-1]['n']} zwischen {num(min(ratios), 1)} und {num(max(ratios), 1)}: er wächst wie √n.")


def verdict_models(rows: list[dict]) -> tuple[str, str]:
    """Experiment Modelle: größter Grad der linearen Anbindung gegen Zufall, überlinear, Alterung und Fitness."""
    by = {r["name"]: r for r in rows}
    lin, rnd, sup = by["linear (Barabási–Albert)"], by["Zufallsanbindung (a = 0)"], by["überlinear (a = 1,5)"]
    old, fit = by["linear mit Alterung (nu = 1)"], by["linear mit Fitness"]
    return "info", (f"Bei n = {C.MODEL_N} und m = {C.MODEL_M} hat die lineare Anbindung im Median den größten Grad {num(lin['kmax'], 0)}, die Zufallsanbindung {num(rnd['kmax'], 0)}, mit Alterung {num(old['kmax'], 0)}, mit Fitness {num(fit['kmax'], 0)}; "
                    f"bei überlinearer Anbindung hält ein einziger Hub {num(sup['share'] * 100, 0)} % aller Kantenenden (größter Grad {num(sup['kmax'], 0)}).")


def verdict_kernel(rows: list[dict]) -> tuple[str, str]:
    """Experiment überlineare Anbindung: Anteil des größten Hubs bei Exponent 1 und bei den größeren Exponenten."""
    by = {r["a"]: r for r in rows}
    one, big, top = by[1.0], by[1.5], rows[-1]
    return "info", (f"Bei Exponent 1 hält der größte Hub {num(one['share'] * 100, 1)} % aller Kantenenden, bei {num(big['a'], 1)} sind es {num(big['share'] * 100, 0)} % und bei {num(top['a'], 1)} {num(top['share'] * 100, 0)} %: "
                    "oberhalb von 1 steigt der Anteil des größten Hubs steil an.")
