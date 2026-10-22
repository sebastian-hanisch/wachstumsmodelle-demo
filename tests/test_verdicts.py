"""Verdikte (reine Funktionen): Art und jede Zahl im Text an handgesetzten Werten, einschließlich der Schwellen und Zweige."""

import numpy as np

import wsm_evaluation as ev
import wsm_algorithm as W


def make(kmax=173, share=0.011, alpha=2.81, theory="linear", n=3000, m=2, kmin=6):
    s = ev.Settings(n=n, m=m)
    g = W.Growth(np.array([m, m]), np.array([0.0, 1.0]), None, np.zeros((0, 2), dtype=int), np.array([m, m]))
    return ev.Analysis(settings=s, growth=g, kmin=kmin, alpha=alpha, kmax=kmax, share=share, reference=W.expected_max_degree_linear(n, m), theory=theory)


def test_model_verdict_for_the_linear_rule_names_the_exponent_and_the_reference():
    kind, text = ev.verdict_model(make())
    assert kind == "success" and "Größter Grad 173 bei n = 3000 und m = 2" in text and "m·√n = 110" in text and "1,1 %" in text and "α̂ = 2,81 ab Grad 6" in text and "3,0" in text


def test_model_verdict_for_a_dominating_hub():
    kind, text = ev.verdict_model(make(kmax=3531, share=0.221, theory=None))
    assert kind == "warning" and "Ein einziger Hub" in text and "22,1 %" in text and "3531" in text


def test_model_verdict_threshold_for_the_dominating_hub_is_ten_percent():
    assert ev.verdict_model(make(share=0.1, theory=None))[0] == "warning" and ev.verdict_model(make(share=0.0999, theory=None))[0] == "info"
    assert ev.verdict_model(make(share=0.1))[0] == "warning"  # auch bei linearer Theorie hat der Hub-Zweig Vorrang


def test_model_verdict_for_random_attachment_and_unknown_theory():
    kind, text = ev.verdict_model(make(kmax=24, share=0.002, alpha=3.9, theory="uniform"))
    assert kind == "info" and "exponentiell" in text and "kein Potenzgesetz" in text and "24" in text
    kind, text = ev.verdict_model(make(kmax=580, share=0.036, alpha=2.47, theory=None))
    assert kind == "info" and "α̂ = 2,47" in text and "eine Theorie der Gradverteilung gibt es für diese Einstellung hier nicht" in text and "3,6 %" in text


def test_exponent_verdict_uses_the_first_and_last_rows():
    rows = [{"kmin": 3, "alpha": 2.58, "tail": 9000.0}, {"kmin": 6, "alpha": 2.79, "tail": 3000.0}, {"kmin": 40, "alpha": 3.02, "tail": 100.0}]
    hub = [{"n": 1000, "ratio": 2.89, "kmax": 91, "low": 80, "high": 100}, {"n": 64000, "ratio": 3.10, "kmax": 784, "low": 700, "high": 800}, {"n": 16000, "ratio": 3.42, "kmax": 433, "low": 400, "high": 450}]
    kind, text = ev.verdict_exponent(rows, hub)
    assert kind == "info" and "α̂ = 2,58 ab Grad 3 und 3,02 ab Grad 40" in text and "zwischen 2,9 und 3,4" in text


def test_models_verdict_names_each_model():
    rows = [
        {"name": "Zufallsanbindung (a = 0)", "kmax": 21.5, "share": 0.001, "alpha": 3.9}, {"name": "linear (Barabási–Albert)", "kmax": 168.6, "share": 0.011, "alpha": 2.8},
        {"name": "überlinear (a = 1,5)", "kmax": 3358.0, "share": 0.21, "alpha": 2.2}, {"name": "linear mit Alterung (nu = 1)", "kmax": 22.9, "share": 0.001, "alpha": 4.5},
        {"name": "linear mit Fitness", "kmax": 435.5, "share": 0.027, "alpha": 2.5},
    ]
    kind, text = ev.verdict_models(rows)
    assert kind == "info" and "größten Grad 169" in text and "Zufallsanbindung 22" in text and "mit Alterung 23" in text and "mit Fitness 436" in text and "21 % aller Kantenenden" in text and "3358" in text


def test_kernel_verdict_by_hand():
    rows = [{"a": 0.0, "share": 0.002, "kmax": 20.0, "low": 0.001, "high": 0.003}, {"a": 1.0, "share": 0.011, "kmax": 170.0, "low": 0.009, "high": 0.013},
            {"a": 1.5, "share": 0.21, "kmax": 3300.0, "low": 0.2, "high": 0.23}, {"a": 2.0, "share": 0.5, "kmax": 5000.0, "low": 0.4, "high": 0.55}]
    kind, text = ev.verdict_kernel(rows)
    assert kind == "info" and "Bei Exponent 1 hält der größte Hub 1,1 %" in text and "bei 1,5 sind es 21 %" in text and "bei 2,0 50 %" in text and "oberhalb von 1 steigt der Anteil des größten Hubs steil an" in text
