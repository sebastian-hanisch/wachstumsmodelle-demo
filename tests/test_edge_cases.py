"""Randfälle: Mindestgröße, m = 1, alle Gewichte null, extreme Exponenten, Alterung ohne Überlauf."""

import math

import numpy as np
import pytest

import wsm_algorithm as W
import wsm_evaluation as ev
import wsm_visualization as V


def test_growth_below_the_start_clique_returns_the_clique():
    g = W.grow(1, 2, 1.0, 0.0, 0.0, False, np.random.default_rng(1))
    assert len(g.deg) == 3 and list(g.deg) == [2, 2, 2] and len(g.edges) == 3


def test_m_equal_one_gives_a_tree_with_n_minus_one_edges_plus_the_start_edge():
    g = W.grow(200, 1, 1.0, 0.0, 0.0, False, np.random.default_rng(2))
    assert len(g.edges) == W.edge_count(200, 1) == 199 and g.deg.sum() == 2 * 199 and g.deg.min() == 1


def test_zero_weights_fall_back_to_all_available_targets():
    # Exponent 2 mit Anfangsattraktivität 0 hat nie Gewicht null; mit a = 0 und A = 0 ist jedes Gewicht eins — hier der Randfall für draw_targets
    picks = W.draw_targets(np.zeros(3), 2, np.random.default_rng(1))
    assert len(set(picks.tolist())) == len(picks) <= 2


def test_extreme_exponent_two_does_not_overflow():
    g = W.grow(1500, 3, 2.0, 0.0, 0.0, False, np.random.default_rng(1))
    assert g.deg.sum() == 2 * W.edge_count(1500, 3) and np.isfinite(W.hub_share(g.deg))


def test_strong_aging_does_not_overflow():
    g = W.grow(1500, 2, 1.0, 0.0, 2.0, False, np.random.default_rng(1))
    assert g.deg.sum() == 2 * W.edge_count(1500, 2) and W.max_degree(g.deg) < 200


def test_analyse_at_the_smallest_allowed_network_runs_for_every_model_variant():
    for a in (0.0, 1.0, 2.0):
        for fitness in ("none", "uniform"):
            r = ev.analyse(ev.Settings(n=500, m=5, a=a, a0=10.0, nu=2.0, fitness=fitness, seed=1))
            assert r.kmax >= 5 and 0 < r.share < 1


def test_mle_is_nan_for_too_few_nodes_and_a_bad_kmin():
    assert math.isnan(W.mle_exponent(np.array([5]), 3)) and math.isnan(W.mle_exponent(np.array([5, 6, 7]), 0))
    assert math.isnan(W.mle_exponent(np.array([1, 2, 2]), 10))


def test_figures_survive_a_network_without_theory_and_with_one_node_count():
    fig = V.build_degrees(np.array([2, 2, 2]), 2, None)
    assert len(fig.data) == 1
    fig = V.build_hub_growth(np.array([2, 2, 3, 3]), 2)
    assert len(fig.data) == 2


class _EdgeRng:
    """Zufallsquelle, die an der oberen Kante der Rundung zieht (random() == 1,0 ergibt ein Ziel hinter dem letzten Gewicht)."""

    def random(self):
        return 1.0


def test_draw_targets_clamps_a_draw_at_the_upper_edge_to_the_last_node():
    assert W.draw_targets(np.array([1.0, 2.0, 3.0]), 1, _EdgeRng()).tolist() == [2]


class _ShortBufferRng:
    """Liefert höchstens fünf Zufallszahlen je Aufruf: erzwingt den Wechsel des Puffers in `grow_linear`."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def random(self, size):
        return self.rng.random(min(size, 5))


def test_linear_growth_refills_its_random_buffer():
    g = W.grow_linear(60, 2, _ShortBufferRng(1))
    assert len(g.edges) == W.edge_count(60, 2) and g.deg.sum() == 2 * len(g.edges) and g.deg.min() >= 2
    assert len({tuple(sorted(e)) for e in g.edges.tolist()}) == len(g.edges)


def test_mle_exponent_is_defined_for_kmin_one():
    deg = np.array([1, 1, 2, 3, 5, 8])
    expected = 1.0 + 6 / sum(math.log(k / 0.5) for k in deg)
    assert W.mle_exponent(deg, 1) == pytest.approx(expected)
