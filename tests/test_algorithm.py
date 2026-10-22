"""Kern von Hand an der Mini-Instanz: Gewichte, Ziehen, Wachstum, Kennzahlen, Theorie."""

import math

import numpy as np
import pytest

import wsm_algorithm as W


def test_attachment_weights_by_hand(mini_state):
    deg, birth, fit, t = mini_state
    assert W.attachment_weights(deg, birth, None, t, 1.0, 0.0, 0.0).tolist() == [2, 3, 1]
    assert W.attachment_weights(deg, birth, None, t, 2.0, 1.0, 0.0).tolist() == [9, 16, 4]  # (k + 1)^2
    assert W.attachment_weights(deg, birth, None, t, 0.0, 0.0, 0.0).tolist() == [1, 1, 1]  # Zufallsanbindung
    assert np.allclose(W.attachment_weights(deg, birth, None, t, 1.0, 0.0, 1.0), [0.5, 1.0, 0.5])
    assert np.allclose(W.attachment_weights(deg, birth, fit, t, 1.0, 0.0, 1.0), [0.5, 0.5, 0.125])
    assert np.allclose(W.attachment_weights(deg, birth, fit, t, 1.0, 0.0, 0.0), [2, 1.5, 0.25])
    assert len(W.attachment_weights(deg, birth, None, 2, 1.0, 0.0, 0.0)) == 2  # nur die Knoten vor t


def test_draw_targets_by_hand():
    rng = np.random.default_rng(1)
    assert W.draw_targets(np.array([0.0, 0.0, 5.0]), 1, rng).tolist() == [2]
    assert W.draw_targets(np.array([1.0, 0.0, 0.0]), 2, rng).tolist() == [0]  # weniger Knoten mit Gewicht als m: nur dieser
    assert sorted(W.draw_targets(np.array([1.0, 1.0, 1.0]), 3, rng).tolist()) == [0, 1, 2]
    assert len(W.draw_targets(np.array([0.0, 0.0]), 1, rng)) == 0


def test_draw_targets_follow_the_weights():
    rng = np.random.default_rng(2)
    hits = sum(int(W.draw_targets(np.array([1.0, 3.0]), 1, rng)[0]) for _ in range(4000))
    assert abs(hits / 4000 - 0.75) < 4 * math.sqrt(0.75 * 0.25 / 4000)


def test_draw_targets_do_not_change_the_input_weights():
    w = np.array([1.0, 2.0, 3.0])
    W.draw_targets(w, 2, np.random.default_rng(3))
    assert w.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("a, A, nu, fit", [(1.0, 0.0, 0.0, False), (0.0, 0.0, 0.0, False), (1.5, 0.0, 0.0, False), (1.0, 3.0, 1.0, True)])
def test_growth_invariants(a, A, nu, fit):
    g = W.grow(300, 3, a, A, nu, fit, np.random.default_rng(4))
    assert len(g.edges) == W.edge_count(300, 3) and g.deg.sum() == 2 * len(g.edges) and g.deg.min() >= 3
    assert len({tuple(sorted(e)) for e in g.edges.tolist()}) == len(g.edges)  # keine Mehrfachkanten
    assert (g.edges[:, 0] != g.edges[:, 1]).all()
    assert g.kmax_trace[-1] == g.deg.max() and (np.diff(g.kmax_trace) >= 0).all() and len(g.kmax_trace) == 300
    assert (g.fitness is not None) == fit


def test_growth_is_deterministic_and_the_start_is_a_clique():
    a = W.grow(100, 2, 1.0, 0.0, 0.0, False, np.random.default_rng(5))
    b = W.grow(100, 2, 1.0, 0.0, 0.0, False, np.random.default_rng(5))
    assert (a.deg == b.deg).all() and (a.edges == b.edges).all()
    assert sorted(map(tuple, a.edges[:3].tolist())) == [(1, 0), (2, 0), (2, 1)]  # Startklique aus 3 Knoten
    only = W.grow(3, 2, 1.0, 0.0, 0.0, False, np.random.default_rng(1))
    assert only.deg.tolist() == [2, 2, 2] and len(only.edges) == 3
    tiny = W.grow(1, 2, 1.0, 0.0, 0.0, False, np.random.default_rng(1))
    assert len(tiny.deg) == 3  # n wird auf m + 1 angehoben


def test_the_first_new_node_connects_to_m_distinct_start_nodes():
    g = W.grow(4, 2, 1.0, 0.0, 0.0, False, np.random.default_rng(6))
    new = [tuple(e) for e in g.edges.tolist() if e[0] == 3]
    assert len(new) == 2 and len({v for _, v in new}) == 2 and all(0 <= v <= 2 for _, v in new)


def test_linear_growth_invariants_and_determinism():
    g = W.grow_linear(500, 3, np.random.default_rng(7))
    assert len(g.edges) == W.edge_count(500, 3) and g.deg.sum() == 2 * len(g.edges) and g.deg.min() >= 3
    assert len({tuple(sorted(e)) for e in g.edges.tolist()}) == len(g.edges)
    assert g.kmax_trace[-1] == g.deg.max() and (np.diff(g.kmax_trace) >= 0).all()
    h = W.grow_linear(500, 3, np.random.default_rng(7))
    assert (g.deg == h.deg).all()
    assert W.grow_linear(2, 3, np.random.default_rng(1)).deg.tolist() == [3, 3, 3, 3]  # n auf m + 1 = 4 angehoben


def test_edge_count_by_hand():
    assert W.edge_count(10, 2) == 3 + 7 * 2 and W.edge_count(3, 2) == 3 and W.edge_count(1, 2) == 3 and W.edge_count(5, 1) == 1 + 3


def test_degree_counts_and_hub_share_by_hand():
    assert W.degree_counts([1, 2, 2, 0, 5]) == {1: 1, 2: 2, 5: 1}
    assert W.hub_share(np.array([1, 2, 3, 4])) == pytest.approx(0.4) and W.hub_share(np.array([])) == 0.0 and W.hub_share(np.array([0, 0])) == 0.0
    assert W.max_degree(np.array([3, 9, 2])) == 9 and W.max_degree(np.array([])) == 0


def test_mle_exponent_by_hand():
    assert W.mle_exponent([2, 2], 2) == pytest.approx(1 + 2 / (2 * math.log(2 / 1.5)))
    assert W.mle_exponent([1, 2, 3, 4], 3) == pytest.approx(1 + 2 / (math.log(3 / 2.5) + math.log(4 / 2.5)))
    assert math.isnan(W.mle_exponent([5], 2)) and math.isnan(W.mle_exponent([1, 1], 3)) and math.isnan(W.mle_exponent([2, 2], 0))


def test_theory_pmfs_by_hand_and_normalisation():
    assert W.linear_degree_pmf(2, 2) == pytest.approx(2 * 2 * 3 / (2 * 3 * 4)) and W.linear_degree_pmf(1, 2) == 0.0
    assert W.uniform_degree_pmf(2, 2) == pytest.approx(1 / 3) and W.uniform_degree_pmf(3, 2) == pytest.approx(2 / 9) and W.uniform_degree_pmf(1, 2) == 0.0
    for m in (1, 2, 5):
        assert sum(W.linear_degree_pmf(k, m) for k in range(m, 200000)) == pytest.approx(1.0, abs=2e-4)  # Teleskopsumme: 1 − m(m+1)/((K+1)(K+2))
        assert sum(W.uniform_degree_pmf(k, m) for k in range(m, 3000)) == pytest.approx(1.0, abs=1e-9)


def test_theory_exponent_and_mean_field_size():
    assert W.theory_exponent(2, 0.0) == 3.0 and W.theory_exponent(2, 3.0) == pytest.approx(4.5) and W.theory_exponent(3, 6.0) == pytest.approx(5.0)
    assert W.expected_max_degree_linear(10000, 3) == pytest.approx(300.0)
