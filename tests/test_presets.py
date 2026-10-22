"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keiner auf einem Reglerrand; Hilfe nur mit Aussagen, die test_claims prüft."""

import wsm_constants as C
import wsm_evaluation as ev
import wsm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and C.A_MIN <= p["a"] <= C.A_MAX, name
        assert C.A0_MIN <= p["a0"] <= C.A0_MAX and C.NU_MIN <= p["nu"] <= C.NU_MAX and p["fitness"] in C.FITNESS and 0 <= p["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall: lineare Anbindung"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_preset_changes_more_than_n_and_one_model_knob_against_the_default():
    default = C.PRESETS["Standardfall: lineare Anbindung"]
    for name, p in C.PRESETS.items():
        changed = {k for k in p if p[k] != default[k]} - {"n"}
        assert len(changed) <= 1, (name, changed)


def test_presets_are_pairwise_different():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_each_preset_uses_a_documented_model():
    """Jedes Preset ist (bis auf n) eines der Modelle der Vergleichsreihe, damit seine Aussage dort wiederkehrt."""
    models = {(a, a0, nu, f) for _, a, a0, nu, f in C.MODELS}
    for name, p in C.PRESETS.items():
        assert (p["a"], p["a0"], p["nu"], p["fitness"] == "uniform") in models, name
