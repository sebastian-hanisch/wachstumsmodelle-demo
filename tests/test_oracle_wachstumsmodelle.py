"""Orakel-Tests: exakte Gradverteilung kleiner Netze durch Rekursion über alle Zustände, die schnelle gegen die allgemeine lineare Anbindung (KS-Test),
die Theorie der Gradverteilung gegen große Netze, `networkx.barabasi_albert_graph`, der Exponent-Schätzer an bekannten Potenzgesetzen."""

import math

import numpy as np
import pytest

import wsm_algorithm as W


def exact_degree_of_node0(n, m, a, A, nu):
    """Verteilung des Endgrades von Knoten 0 für m = 1, exakt über alle Zustände (Gradfolge) mit eigener Gewichtsformel; Start: zwei verbundene Knoten.
    Rückgabe: {Grad: Wahrscheinlichkeit}. Unabhängig vom Kern gerechnet (Gewichte direkt aus der Definition)."""
    assert m == 1
    states = {(1, 1): 1.0}
    for t in range(2, n):
        nxt = {}
        for deg, p in states.items():
            w = [((d + A) ** a) * ((t - i + 1.0) ** (-nu)) for i, d in enumerate(deg)]
            total = sum(w)
            for i, d in enumerate(deg):
                new = list(deg)
                new[i] += 1
                new.append(1)
                key = tuple(new)
                nxt[key] = nxt.get(key, 0.0) + p * w[i] / total
        states = nxt
    out = {}
    for deg, p in states.items():
        out[deg[0]] = out.get(deg[0], 0.0) + p
    return out


@pytest.mark.parametrize("a, A, nu", [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.5, 0.0, 0.0), (1.0, 2.0, 0.5)])
def test_degree_of_the_oldest_node_matches_the_exact_distribution(a, A, nu):
    n = 7
    exact = exact_degree_of_node0(n, 1, a, A, nu)
    assert sum(exact.values()) == pytest.approx(1.0)
    reps = 12000
    rng = np.random.default_rng(10)
    counts = {}
    for _ in range(reps):
        g = W.grow(n, 1, a, A, nu, False, rng)
        counts[int(g.deg[0])] = counts.get(int(g.deg[0]), 0) + 1
    assert set(counts) <= set(exact)
    for k, p in exact.items():
        se = math.sqrt(max(p * (1 - p), 1e-12) / reps)
        assert abs(counts.get(k, 0) / reps - p) < 5 * se + 1e-3, (k, p, counts.get(k, 0) / reps)


def test_exact_distribution_of_the_oldest_node_with_two_nodes_added_by_hand():
    # Start: beide Knoten Grad 1. Dritter Knoten: Knoten 0 wächst mit Wahrscheinlichkeit 1/2 (Zustand (2, 1, 1)), sonst (1, 2, 1).
    # Vierter Knoten: Knoten 0 wächst im ersten Zustand mit 2/4, im zweiten mit 1/4. Endgrad 3: 1/2 · 1/2; Endgrad 2: 1/2 · 1/2 + 1/2 · 1/4; Endgrad 1: 1/2 · 3/4.
    exact = exact_degree_of_node0(4, 1, 1.0, 0.0, 0.0)
    assert exact[3] == pytest.approx(0.25) and exact[2] == pytest.approx(0.375) and exact[1] == pytest.approx(0.375)


def test_fast_linear_growth_has_the_same_distribution_as_the_general_one():
    stats = pytest.importorskip("scipy.stats")
    n, m, reps = 60, 2, 2500
    rng = np.random.default_rng(11)
    fast = [int(W.grow_linear(n, m, rng).deg[0]) for _ in range(reps)]
    slow = [int(W.grow(n, m, 1.0, 0.0, 0.0, False, rng).deg[0]) for _ in range(reps)]
    assert stats.ks_2samp(fast, slow).pvalue > 0.001
    fast_max = [W.max_degree(W.grow_linear(n, m, rng).deg) for _ in range(reps // 2)]
    slow_max = [W.max_degree(W.grow(n, m, 1.0, 0.0, 0.0, False, rng).deg) for _ in range(reps // 2)]
    assert stats.ks_2samp(fast_max, slow_max).pvalue > 0.001


def test_degree_distribution_of_a_large_linear_network_matches_the_theory():
    n, m = 64000, 3
    g = W.grow_linear(n, m, np.random.default_rng(12))
    c = W.degree_counts(g.deg)
    for k in (3, 4, 5, 6, 8, 10):
        p, q = c.get(k, 0) / n, W.linear_degree_pmf(k, m)
        assert abs(p - q) / q < 0.08, (k, p, q)


def test_degree_distribution_of_random_attachment_matches_the_theory():
    m = 2
    rng = np.random.default_rng(13)
    g = W.grow(8000, m, 0.0, 0.0, 0.0, False, rng)
    c = W.degree_counts(g.deg)
    for k in (2, 3, 4, 6, 8):
        p, q = c.get(k, 0) / 8000, W.uniform_degree_pmf(k, m)
        assert abs(p - q) / q < 0.15, (k, p, q)


def test_networkx_barabasi_albert_has_the_same_size_of_hubs_and_exponent():
    nx = pytest.importorskip("networkx")
    n, m, reps = 4000, 3, 12
    ours_max, nx_max, ours_a, nx_a = [], [], [], []
    for r in range(reps):
        g = W.grow_linear(n, m, np.random.default_rng(100 + r))
        ours_max.append(W.max_degree(g.deg))
        ours_a.append(W.mle_exponent(g.deg, 6))
        h = nx.barabasi_albert_graph(n, m, seed=200 + r)
        d = [x for _, x in h.degree()]
        nx_max.append(max(d))
        nx_a.append(W.mle_exponent(d, 6))
    assert abs(np.mean(ours_max) - np.mean(nx_max)) / np.mean(nx_max) < 0.2
    assert abs(np.mean(ours_a) - np.mean(nx_a)) < 0.08


def test_mle_exponent_recovers_a_known_discrete_power_law():
    rng = np.random.default_rng(14)
    ks = np.arange(5, 100000)
    for alpha in (2.5, 3.0, 3.5):
        p = ks.astype(float) ** (-alpha)
        p /= p.sum()
        sample = rng.choice(ks, size=40000, p=p)
        assert W.mle_exponent(sample, 5) == pytest.approx(alpha, abs=0.07)


def test_maximum_degree_of_linear_growth_scales_with_the_root_of_n():
    ratios = []
    for n in (1000, 4000, 16000):
        ratios.append(np.mean([W.max_degree(W.grow_linear(n, 3, np.random.default_rng(300 + r)).deg) for r in range(8)]) / math.sqrt(n))
    assert all(2.0 < r < 5.0 for r in ratios) and max(ratios) / min(ratios) < 1.5
