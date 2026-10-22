"""Abbildungen: Spuren je Figur, Theorielinien, Farben, gesperrte Achsen."""

import numpy as np

import wsm_algorithm as W
import wsm_constants as C
import wsm_evaluation as ev
import wsm_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_degree_figure_has_theory_only_when_a_theory_exists():
    g = W.grow_linear(800, 2, np.random.default_rng(1))
    with_theory = V.build_degrees(g.deg, 2, "linear")
    assert [t.name for t in with_theory.data] == ["gemessen", "Theorie (lineare Anbindung)"] and _locked(with_theory)
    assert [t.name for t in V.build_degrees(g.deg, 2, "uniform").data][1] == "Theorie (Zufallsanbindung)"
    assert [t.name for t in V.build_degrees(g.deg, 2, None).data] == ["gemessen"]


def test_degree_points_are_the_shares_of_nodes():
    fig = V.build_degrees(np.array([2, 2, 3, 5]), 2, None)
    pts = dict(zip(fig.data[0].x, fig.data[0].y))
    assert pts == {2: 0.5, 3: 0.25, 5: 0.25}
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"


def test_hub_growth_figure_has_the_root_curve():
    g = W.grow_linear(600, 2, np.random.default_rng(1))
    fig = V.build_hub_growth(g.kmax_trace, 2)
    assert [t.name for t in fig.data] == ["größter Grad", "m · √t"] and _locked(fig)
    assert fig.data[1].y[-1] == 2 * np.sqrt(fig.data[1].x[-1])


def test_exponent_figure_draws_the_theory_line_at_three():
    rows = ev.series_exponent(n=1500, m=3, kmins=(3, 10), seeds=(1, 2))
    fig = V.build_exponent(rows)
    assert list(fig.data[0].y) == [3.0, 3.0] and list(fig.data[0].x) == [3, 10] and len(fig.data[1].y) == 2 and _locked(fig)


def test_hub_scaling_figure_uses_m_times_root_n():
    rows = ev.series_hub_growth(ns=(500, 2000), m=C.HUB_M, seeds=(1, 2))
    fig = V.build_hub_scaling(rows)
    assert [t.name for t in fig.data] == ["Minimum bis Maximum", "m · √n", "größter Grad (Median)"]
    assert list(fig.data[1].y) == [C.HUB_M * np.sqrt(500), C.HUB_M * np.sqrt(2000)] and _locked(fig)


def test_model_figure_colours_dominating_hubs_red():
    rows = [{"name": "A", "kmax": 20.0, "share": 0.001}, {"name": "B", "kmax": 3000.0, "share": 0.2}, {"name": "C", "kmax": 100.0, "share": 0.1}]
    fig = V.build_models(rows)
    assert list(fig.data[0].marker.color) == [C.COLOR_MEASURED, C.COLOR_ALERT, C.COLOR_ALERT] and _locked(fig)


def test_kernel_figure_shows_percent_and_marks_exponent_one():
    rows = [{"a": 0.0, "share": 0.002, "kmax": 20.0, "low": 0.001, "high": 0.003}, {"a": 1.0, "share": 0.011, "kmax": 170.0, "low": 0.009, "high": 0.013}]
    fig = V.build_kernel(rows)
    assert list(fig.data[1].y) == [0.2, 1.1] or np.allclose(fig.data[1].y, [0.2, 1.1])
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == 1.0 and _locked(fig)
