"""Auswertung: Analyse, Messreihen (Struktur und Richtung)."""

import math

import numpy as np
import pytest

import wsm_algorithm as W
import wsm_constants as C
import wsm_evaluation as ev
from wsm_scenario import Model


def test_num_uses_a_decimal_comma():
    assert ev.num(3.14159, 2) == "3,14" and ev.num(10, 0) == "10" and ev.num(2.5, 2) == "2,50"


def test_settings_model_mapping():
    s = ev.Settings(n=500, m=3, a=1.5, a0=2.0, nu=0.5, fitness="uniform", seed=1)
    assert s.model == Model(500, 3, 1.5, 2.0, 0.5, True) and ev.Settings().model == Model(C.DEFAULT_N, C.DEFAULT_M)


def test_analyse_by_hand_for_a_tiny_network():
    a = ev.analyse(ev.Settings(n=3, m=2, seed=1))  # nur die Startklique aus drei Knoten
    assert a.kmax == 2 and a.share == pytest.approx(2 / 6) and a.kmin == 6 and math.isnan(a.alpha) and a.theory == "linear"
    assert a.reference == pytest.approx(2 * math.sqrt(3))


def test_analyse_is_consistent_with_the_kernel():
    s = ev.Settings(n=800, m=2, a=1.5, seed=3)
    a = ev.analyse(s)
    assert a.kmax == W.max_degree(a.growth.deg) and a.share == W.hub_share(a.growth.deg) and a.theory is None and a.kmin == C.MLE_KMIN_FACTOR * 2
    assert ev.analyse(s).kmax == a.kmax  # deterministisch


def test_series_exponent_structure_and_direction():
    rows = ev.series_exponent(n=2000, m=3, kmins=(3, 6, 20), seeds=(1, 2, 3))
    assert [r["kmin"] for r in rows] == [3, 6, 20]
    assert rows[0]["alpha"] < rows[-1]["alpha"] and all(r["alpha"] > 2 for r in rows) and rows[0]["tail"] > rows[-1]["tail"]


def test_series_hub_growth_structure():
    rows = ev.series_hub_growth(ns=(500, 2000), m=3, seeds=(1, 2, 3))
    assert [r["n"] for r in rows] == [500, 2000]
    for r in rows:
        assert r["low"] <= r["kmax"] <= r["high"] and r["ratio"] == pytest.approx(r["kmax"] / math.sqrt(r["n"]))
    assert rows[1]["kmax"] > rows[0]["kmax"]


def test_series_models_structure_and_the_documented_order():
    rows = {r["name"]: r for r in ev.series_models(n=1500, m=2, seeds=(1, 2, 3))}
    assert set(rows) == {name for name, *_ in C.MODELS}
    assert rows["linear (Barabási–Albert)"]["kmax"] > rows["Zufallsanbindung (a = 0)"]["kmax"] > rows["linear mit Alterung (nu = 1)"]["kmax"] * 0.5
    assert rows["überlinear (a = 1,5)"]["share"] > 0.1 > rows["linear (Barabási–Albert)"]["share"]
    assert rows["linear mit Fitness"]["kmax"] > rows["linear (Barabási–Albert)"]["kmax"]


def test_series_kernel_structure_and_direction():
    rows = ev.series_kernel(exponents=(0.0, 1.0, 2.0), n=1000, m=2, seeds=(1, 2, 3))
    assert [r["a"] for r in rows] == [0.0, 1.0, 2.0]
    assert rows[0]["share"] < rows[1]["share"] < rows[2]["share"] and all(r["low"] <= r["share"] <= r["high"] for r in rows)


def test_the_fixed_plans_are_the_documented_ones():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100005)) and (C.EXP_N, C.EXP_M) == (16000, 3) and C.EXP_KMINS == (3, 4, 6, 10, 20, 40)
    assert C.HUB_NS == (1000, 4000, 16000, 64000) and (C.MODEL_N, C.MODEL_M, C.KERNEL_N, C.KERNEL_M) == (4000, 2, 3000, 2)
    assert C.KERNEL_EXPONENTS == (0.0, 0.5, 1.0, 1.25, 1.5, 2.0) and len(C.MODELS) == 7
    assert isinstance(np.float64(1.0), float)
