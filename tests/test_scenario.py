"""Szenario: Modell, Wahl des Samplers, Theorie-Art, Determinismus."""

import numpy as np

import wsm_algorithm as W
import wsm_scenario as S


def test_is_linear_only_for_the_plain_barabasi_albert_rule():
    assert S.Model(100, 2).is_linear
    assert not S.Model(100, 2, a=1.1).is_linear and not S.Model(100, 2, a0=1.0).is_linear
    assert not S.Model(100, 2, nu=0.5).is_linear and not S.Model(100, 2, fitness=True).is_linear


def test_theory_kind_by_hand():
    assert S.theory_kind(S.Model(100, 2)) == "linear"
    assert S.theory_kind(S.Model(100, 2, a=0.0)) == "uniform" and S.theory_kind(S.Model(100, 2, a=0.0, a0=5.0)) == "uniform"
    assert S.theory_kind(S.Model(100, 2, a=0.0, nu=1.0)) is None and S.theory_kind(S.Model(100, 2, a=0.0, fitness=True)) is None
    assert S.theory_kind(S.Model(100, 2, a=1.5)) is None and S.theory_kind(S.Model(100, 2, a0=3.0)) is None


def test_run_model_is_deterministic_and_seed_dependent():
    a, b, c = S.run_model(S.Model(300, 2), 7), S.run_model(S.Model(300, 2), 7), S.run_model(S.Model(300, 2), 8)
    assert (a.deg == b.deg).all() and not (a.deg == c.deg).all()
    x, y = S.run_model(S.Model(300, 2, a=1.5), 7), S.run_model(S.Model(300, 2, a=1.5), 7)
    assert (x.deg == y.deg).all() and len(x.edges) == W.edge_count(300, 2)


def test_run_model_uses_the_fast_sampler_exactly_for_the_linear_model():
    fast = S.run_model(S.Model(400, 2), 3)
    direct = W.grow_linear(400, 2, np.random.default_rng(3))
    assert (fast.deg == direct.deg).all() and fast.fitness is None
    general = S.run_model(S.Model(400, 2, a=1.0, fitness=True), 3)
    assert general.fitness is not None and len(general.fitness) == 400
