"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App, mit den festen Plänen.

Die Zufallszahlen eines numpy-Generators sind über Versionen nicht stabil: die Messwerte gelten deshalb als Bänder um die in der README genannten Zahlen.
Die Theoriewerte sind exakt."""

import math

import pytest

import wsm_algorithm as W
import wsm_constants as C
import wsm_evaluation as ev


@pytest.fixture(scope="module")
def presets():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


@pytest.fixture(scope="module")
def exponent():
    return ev.series_exponent()


@pytest.fixture(scope="module")
def hub():
    return ev.series_hub_growth()


@pytest.fixture(scope="module")
def models():
    return {r["name"]: r for r in ev.series_models()}


@pytest.fixture(scope="module")
def kernel():
    return {r["a"]: r for r in ev.series_kernel()}


def test_the_series_use_the_documented_fixed_plans():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100005)) and (C.EXP_N, C.EXP_M) == (16000, 3) and C.EXP_KMINS == (3, 4, 6, 10, 20, 40)
    assert C.HUB_NS == (1000, 4000, 16000, 64000) and C.HUB_M == 3 and (C.MODEL_N, C.MODEL_M) == (4000, 2) and (C.KERNEL_N, C.KERNEL_M) == (3000, 2)
    assert (C.N_MIN, C.N_MAX, C.DEFAULT_N, C.M_MIN, C.M_MAX, C.DEFAULT_M) == (500, 10000, 3000, 1, 5, 2) and C.MLE_KMIN_FACTOR == 3


def test_theory_values():
    assert W.linear_degree_pmf(2, 2) == pytest.approx(2 * 2 * 3 / (2 * 3 * 4)) == pytest.approx(0.5)
    assert W.linear_degree_pmf(3, 2) == pytest.approx(0.2) and W.uniform_degree_pmf(2, 2) == pytest.approx(1 / 3)
    assert W.theory_exponent(2, 0.0) == 3.0 and W.theory_exponent(2, 3.0) == pytest.approx(4.5)
    assert W.edge_count(3000, 2) == 5997 and W.expected_max_degree_linear(3000, 2) == pytest.approx(2 * math.sqrt(3000))


def test_presets_hold_what_their_help_texts_say(presets):
    std, rnd, sup, old, fit = (presets[name] for name in C.PRESETS)
    assert std.kmax == pytest.approx(103, abs=45) and std.share == pytest.approx(0.0086, abs=0.006) and std.alpha == pytest.approx(2.80, abs=0.25)
    assert rnd.kmax == pytest.approx(22, abs=8) and rnd.share < 0.006 and rnd.alpha == pytest.approx(3.98, abs=0.6)
    assert sup.kmax == pytest.approx(2037, abs=500) and sup.share == pytest.approx(0.204, abs=0.05)
    assert old.kmax == pytest.approx(18, abs=7) and old.share < 0.006 and old.alpha == pytest.approx(4.34, abs=0.7)
    assert fit.kmax == pytest.approx(233, abs=110) and fit.kmax > std.kmax and fit.alpha < std.alpha


def test_the_help_texts_claims_hold_in_the_order_of_the_four_comparisons(presets):
    std, rnd, sup, old, fit = (presets[name] for name in C.PRESETS)
    assert rnd.kmax < std.kmax / 2 and old.kmax < std.kmax / 2   # ohne Bevorzugung und mit Alterung keine Hubs
    assert sup.share > 0.1 > std.share and sup.share > 10 * std.share
    assert fit.kmax > std.kmax


def test_the_estimated_exponent_depends_on_the_lower_bound(exponent):
    by = {r["kmin"]: r for r in exponent}
    assert by[3]["alpha"] == pytest.approx(2.58, abs=0.1) and by[6]["alpha"] == pytest.approx(2.78, abs=0.1) and by[40]["alpha"] == pytest.approx(2.95, abs=0.1)
    alphas = [r["alpha"] for r in exponent]
    assert alphas == sorted(alphas) and alphas[0] < 2.7 and alphas[-1] > 2.85 and all(a < 3.05 for a in alphas)
    assert by[3]["tail"] == 16000 and 4300 < by[6]["tail"] < 4900 and 90 < by[40]["tail"] < 160


def test_the_largest_degree_grows_like_the_root_of_n(hub):
    ratios = [r["ratio"] for r in hub]
    assert min(ratios) > 3.2 and max(ratios) < 4.4 and max(ratios) / min(ratios) < 1.25
    assert [r["kmax"] for r in hub] == sorted(r["kmax"] for r in hub)
    assert hub[-1]["kmax"] == pytest.approx(972, rel=0.25) and hub[0]["kmax"] == pytest.approx(117, rel=0.3)
    by_n = {r["n"]: r for r in hub}
    assert by_n[4000]["low"] == pytest.approx(169, rel=0.2) and by_n[4000]["high"] == pytest.approx(294, rel=0.2)  # Streuung der Pläne (Ehrliche Grenzen)
    # m·√n ist nur die Größenordnung: in den fünf festen Plänen etwa das 1,2- bis 1,3-fache (mit mehreren hundert Zufalls-Seeds etwa das 1,0-fache, siehe README)
    scale = [r["kmax"] / (C.HUB_M * math.sqrt(r["n"])) for r in hub]
    assert 1.05 < min(scale) and max(scale) < 1.5


def test_the_model_comparison(models):
    kmax = {name: r["kmax"] for name, r in models.items()}
    assert kmax["Zufallsanbindung (a = 0)"] == pytest.approx(20, abs=7) and kmax["unterlinear (a = 0,5)"] == pytest.approx(36, abs=12)
    assert kmax["linear (Barabási–Albert)"] == pytest.approx(145, abs=45) and kmax["überlinear (a = 1,5)"] == pytest.approx(3327, rel=0.2)
    assert kmax["linear mit Alterung (nu = 1)"] == pytest.approx(21, abs=8) and kmax["linear mit Fitness"] == pytest.approx(746, rel=0.4)
    assert kmax["linear mit Anfangsattraktivität 3"] == pytest.approx(54, abs=20)
    # Reihenfolge: ohne Bevorzugung < unterlinear < Anfangsattraktivität < linear < Fitness < überlinear
    order = ["Zufallsanbindung (a = 0)", "unterlinear (a = 0,5)", "linear mit Anfangsattraktivität 3", "linear (Barabási–Albert)", "linear mit Fitness", "überlinear (a = 1,5)"]
    assert [kmax[k] for k in order] == sorted(kmax[k] for k in order)
    assert models["überlinear (a = 1,5)"]["share"] == pytest.approx(0.208, abs=0.04) and models["linear (Barabási–Albert)"]["share"] == pytest.approx(0.009, abs=0.006)
    assert models["linear mit Fitness"]["share"] == pytest.approx(0.047, abs=0.03)
    alphas = {name: r["alpha"] for name, r in models.items()}
    assert alphas["linear (Barabási–Albert)"] == pytest.approx(2.78, abs=0.2) and alphas["linear mit Alterung (nu = 1)"] > 4.0
    assert alphas["linear mit Anfangsattraktivität 3"] == pytest.approx(3.2, abs=0.35)  # Theorie 3 + A0/m = 4,5 gilt für n → ∞, bei 4 000 Knoten liegt die Schätzung deutlich darunter


def test_the_kernel_series(kernel):
    share = {a: r["share"] for a, r in kernel.items()}
    assert share[0.0] == pytest.approx(0.0017, abs=0.0015) and share[1.0] == pytest.approx(0.0098, abs=0.005) and share[1.25] == pytest.approx(0.068, abs=0.03)
    assert share[1.5] == pytest.approx(0.204, abs=0.04) and share[2.0] == pytest.approx(0.247, abs=0.02)
    assert [share[a] for a in sorted(share)] == sorted(share.values())
    assert share[2.0] == pytest.approx(1 / (2 * C.KERNEL_M), abs=0.015)  # Grenzfall: ein Hub nimmt eine von zwei Kanten jedes neuen Knotens
    assert share[1.0] < 0.02 and share[1.5] > 0.1


def test_the_verdict_texts_match_the_series(exponent, hub, models, kernel):
    kind, text = ev.verdict_exponent(exponent, hub)
    assert kind == "info" and "α̂ = 2,58 ab Grad 3" in text and "ab Grad 40" in text
    kind, text = ev.verdict_models(list(models.values()))
    assert "21 %" in text or "20 %" in text or "22 %" in text
    kind, text = ev.verdict_kernel(list(kernel.values()))
    assert "bei 2,0 25 %" in text and "bei 1,5 sind es 20 %" in text
