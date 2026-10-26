"""Wachstumsmodelle – was erzeugt Hubs? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Linie „Netzmodelle“: Wie entstehen Hubs, wenn ein Netz wächst? Bevorzugte Anbindung und ihre Varianten (Exponent, Anfangsattraktivität, Alterung, Fitness);
Gradverteilung, größter Grad und Exponent gegen die Theorie. Der Kern liegt in wsm_algorithm.py.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import wsm_constants as C
import wsm_evaluation as ev
from wsm_evaluation import Settings, analyse, num, verdict_exponent, verdict_kernel, verdict_model, verdict_models
from wsm_layout import dunkles_design, kopf_css
from wsm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from wsm_visualization import build_degrees, build_exponent, build_hub_growth, build_hub_scaling, build_kernel, build_models

st.set_page_config(page_title="Wachstumsmodelle – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _series():
    """Die Messreihen (festes Programm, keine Eingabe): Exponent, Wachstum des größten Grades, Modellvergleich, überlineare Anbindung."""
    return ev.series_exponent(), ev.series_hub_growth(), ev.series_models(), ev.series_kernel()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wsm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌱 Wachstumsmodelle – was erzeugt Hubs?")
st.markdown(
    """
Reale Netze haben **Hubs**: wenige Knoten mit sehr vielen Kanten. Wie entstehen sie? Die einfachste Erklärung ist die **bevorzugte Anbindung**: Ein Netz wächst Knoten für Knoten, und jeder neue Knoten verbindet sich bevorzugt
mit Knoten, die schon viele Kanten haben (Barabási und Albert). Die Demo lässt das Netz wachsen und verändert die Regel: Zufallsanbindung ganz ohne Bevorzugung, **unter- und überlineare** Bevorzugung,
**Alterung** (alte Knoten verlieren ihren Vorsprung) und **Fitness** (Neulinge mit hoher Eignung holen auf). Sie zeigt, was daraus für die Gradverteilung, den **größten Grad** und den **Exponenten** des Potenzgesetzes folgt,
und warum sich der Exponent aus Daten nicht einfach ablesen lässt. Drittes Stück der Linie „Netzmodelle“; es erklärt, woher die Gradfolgen des Konfigurationsmodells kommen können.

Weiter unten: 🔬 **Messreihen** (Exponent und Untergrenze, Wachstum des größten Grades, Modellvergleich, überlineare Anbindung), 🚧 **Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wächst ein Netz", expanded=True):
    st.markdown(
        """
1. **Start:** m + 1 Knoten, alle miteinander verbunden.
2. **Ankunft:** Ein neuer Knoten verbindet sich mit **m verschiedenen** bestehenden Knoten.
3. **Wahl der Ziele:** Jeder bestehende Knoten bekommt ein **Gewicht**: (Grad + A₀)ᵃ, bei Alterung mal (Alter + 1)^(−ν), bei Fitness mal seine Fitness. Das Ziel wird mit Wahrscheinlichkeit proportional zum Gewicht gewählt.
4. **Messen:** Gradverteilung, größter Grad, Exponent des Potenzgesetzes. Bei a = 1 ohne Zusätze ist die **Theorie** bekannt: P(k) = 2m(m+1) / (k(k+1)(k+2)), also ein Potenzgesetz mit Exponent 3.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", step=100, help="Wie viele Knoten das Netz am Ende hat. Das allgemeine Wachstum rechnet quadratisch in n.")
    m = st.slider("Kanten je neuem Knoten m", *bounds("m_slider"), key="m_slider", help="Mit wie vielen bestehenden Knoten sich jeder neue Knoten verbindet.")
    st.markdown("**Anbindungsregel**")
    a = st.slider("Exponent a", *bounds("a_slider"), key="a_slider", step=0.1, help="Gewicht (Grad + A₀) hoch a: 0 = Zufall, 1 = linear (Barabási–Albert), über 1 = überlinear.")
    a0 = st.slider("Anfangsattraktivität A₀", *bounds("a0_slider"), key="a0_slider", step=0.5, help="Wird zum Grad addiert: auch Knoten mit wenigen Kanten haben eine Chance (Exponent 3 + A₀/m bei a = 1).")
    nu = st.slider("Alterung ν", *bounds("nu_slider"), key="nu_slider", step=0.25, help="Das Gewicht sinkt mit dem Alter eines Knotens wie (Alter + 1)^(−ν); 0 = keine Alterung.")
    fitness = st.selectbox("Fitness", list(C.FITNESS), format_func=C.FITNESS.get, key="fitness_select", help="Eine zufällige Fitness je Knoten als Faktor im Gewicht.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "a_slider": round(float(a), 1), "a0_slider": round(float(a0), 1), "nu_slider": round(float(nu), 2), "fitness_select": fitness, "seed_input": int(seed)})

settings = Settings(n=int(n), m=int(m), a=round(float(a), 1), a0=round(float(a0), 1), nu=round(float(nu), 2), fitness=fitness, seed=int(seed))
with st.spinner("Lasse das Netz wachsen …"):
    r = _analysis(settings)
g = r.growth

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Knoten / Kanten", f"{len(g.deg)} / {len(g.edges)}", delta=f"mittlerer Grad {num(2 * len(g.edges) / len(g.deg), 2)}", delta_color="off")
m2.metric("Größter Grad", f"{r.kmax}", delta=f"lineare Anbindung etwa {num(r.reference, 0)} (m·√n)", delta_color="off")
m3.metric("Größter Hub hält", f"{num(r.share * 100, 1)} %", delta="aller Kantenenden", delta_color="off")
m4.metric("Exponent geschätzt", "–" if r.alpha != r.alpha else num(r.alpha, 2), delta=f"ab Grad {r.kmin}; Theorie bei a = 1: {num(3 + settings.a0 / settings.m, 1) if settings.a == 1.0 and settings.nu == 0 and settings.fitness == 'none' else 'keine'}", delta_color="off")
c1, c2 = st.columns(2)
c1.plotly_chart(build_degrees(g.deg, settings.m, r.theory), width="stretch", key="main_degrees")
c2.plotly_chart(build_hub_growth(g.kmax_trace, settings.m), width="stretch", key="main_growth")
st.caption(
    "Links die Gradverteilung (Anteil der Knoten je Grad, logarithmische Achsen), bei Zufalls- und linearer Anbindung mit der Theorie; "
    "rechts der größte Grad während des Wachsens neben der Wurzel-Kurve m·√t."
)

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was erzeugt bei dieser Regel Hubs?")
kind, text = verdict_model(r)
getattr(st, kind)(text)

st.markdown("---")

# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Messreihen: Exponent, Größe des größten Hubs, Modelle und überlineare Anbindung")
st.caption(
    f"Vier feste Programme, unabhängig von den Reglern: lineare Anbindung mit {C.EXP_N} Knoten und {C.HUB_NS[0]} bis {C.HUB_NS[-1]} Knoten, sieben Modelle mit {C.MODEL_N} Knoten, überlineare Anbindung "
    f"mit {C.KERNEL_N} Knoten; Mediane über fünf feste Zufalls-Seeds. Das Rechnen dauert einige Sekunden."
)
if st.button("Messreihen rechnen", key="series_button"):
    t0 = time.perf_counter()
    with st.spinner("Rechne die Messreihen (einige Sekunden) …"):
        _series()
    st.session_state["series_done"] = time.perf_counter() - t0
if "series_done" in st.session_state:
    exponent, hub, models, kernel = _series()
    tab1, tab2, tab3 = st.tabs(["Exponent und Größe des Hubs", "Modelle im Vergleich", "Überlineare Anbindung"])
    with tab1:
        st.markdown("**Der Exponent ist nicht 3.** Die Theorie gilt für n → ∞; der Schätzer aus einem endlichen Netz hängt davon ab, ab welchem Grad man das Potenzgesetz anpasst.")
        t1, t2 = st.columns(2)
        t1.plotly_chart(build_exponent(exponent), width="stretch", key="exponent_chart")
        t2.plotly_chart(build_hub_scaling(hub), width="stretch", key="hub_chart")
        kind, text = verdict_exponent(exponent, hub)
        getattr(st, kind)(text)
    with tab2:
        st.markdown("**Dieselbe Regel in Varianten:** Zufall, unter- und überlinear, Alterung, Fitness und Anfangsattraktivität; rot: der größte Hub hält mehr als ein Zehntel aller Kantenenden.")
        st.plotly_chart(build_models(models), width="stretch", key="models_chart")
        kind, text = verdict_models(models)
        getattr(st, kind)(text)
    with tab3:
        st.markdown("**Oberhalb von Exponent 1 wächst der Anteil des größten Hubs steil:** wer schon viele hat, bekommt überproportional mehr, und der größte Knoten hält schließlich einen großen Teil aller Kanten (bei m = 1 ein einziger Knoten, bei m ≥ 2 halten auch weitere Hubs viel).")
        st.plotly_chart(build_kernel(kernel), width="stretch", key="kernel_chart")
        kind, text = verdict_kernel(kernel)
        getattr(st, kind)(text)
    st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['series_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Regel ist bekannt** | Aus einem Netz allein lässt sich die Wachstumsregel nicht ablesen: verschiedene Regeln können ähnliche Gradverteilungen erzeugen (in dieser Demo nicht gemessen). | Modellwahl (Folgestück) |
| **Der Exponent ist eine Zahl** | Aus endlichen Daten hängt der geschätzte Exponent von der Untergrenze ab; ob die Verteilung überhaupt ein Potenzgesetz ist, prüft ein Test. | Skalenfrei? Potenzgesetz-Test (Folgestück) |
| **Nur Wachstum der Kanten an neue Knoten** | Reale Netze bekommen auch Kanten zwischen alten Knoten und verlieren Knoten; das Modell kennt weder Gemeinschaften noch Dreiecke noch Raum. | Netze im Raum, Motive (Folgestücke) |
| **Die Gradfolge entsteht aus der Regel** | Das Konfigurationsmodell würfelt Netze zu gegebenen Graden, ohne zu fragen, woher sie kommen. | Konfigurationsmodell (Vorgänger) |
| **Größe n bis 10 000** | Der größte Grad folgt der Theorie als Größenordnung; bei kleinen Netzen streut er stark von Netz zu Netz. | diese Demo |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die Wachstumsregel; gesucht ist die Gradverteilung des Netzes mit $n$ Knoten.

| Symbol | Bedeutung |
|---|---|
| $m$ | Zahl der Kanten, mit denen sich jeder neue Knoten verbindet |
| $k_i$ | Grad von Knoten $i$ |
| $t_i$ | Ankunftszeit von Knoten $i$ |
| $a, A_0, \nu$ | Exponent, Anfangsattraktivität, Alterung |
| $\eta_i$ | Fitness von Knoten $i$ (1 ohne Fitness) |

**Regel.** Ein neuer Knoten, der zur Zeit $t$ eintrifft, wählt einen bestehenden Knoten $i$ mit Wahrscheinlichkeit proportional zu

$$w_i = (k_i + A_0)^{a} \, (t - t_i + 1)^{-\nu} \, \eta_i.$$

**Lineare Anbindung** ($a = 1$, $A_0 = \nu = 0$, $\eta_i = 1$). Im Grenzfall $n \to \infty$ hat die Gradverteilung für $k \ge m$ den Wert

$$P(k) = \frac{2m(m+1)}{k(k+1)(k+2)} \sim k^{-3}.$$

Im Mean-Field-Bild wächst der Grad eines Knotens, der zur Zeit $t_i$ ankam, wie $k_i(t) = m \sqrt{t / t_i}$; der älteste Knoten hat deshalb den größten Grad $k_{\max} \sim m \sqrt{n}$.

**Zufallsanbindung** ($a = 0$). Jedes Ziel gleich wahrscheinlich: die Gradverteilung ist exponentiell,

$$P(k) = \frac{1}{m+1} \left( \frac{m}{m+1} \right)^{k-m}.$$

**Anfangsattraktivität.** Bei $a = 1$ und $A_0 > 0$ ist der Exponent $3 + A_0 / m$.

**Überlinear** ($a > 1$). Ein einziger Knoten (bei $m = 1$) bzw. wenige Knoten (bei $m \ge 2$) ziehen einen großen Anteil aller Kanten an sich (Krapivsky, Redner, Leyvraz); bei $m = 2$ halten die zwei größten Hubs zusammen 29 bis 39 % der Kantenenden (n = 4 000 bis 10 000, mehrere Pläne). Ob der Anteil für $n \to \infty$ fest bleibt, prüft die Demo nicht: er wächst mit n.

**Exponent-Schätzer.** Für die Grade $k \ge k_{\min}$ ist $\hat\alpha = 1 + N \big/ \sum_i \ln\!\big(k_i / (k_{\min} - 1/2)\big)$.

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Barabási und Albert, *Emergence of scaling in random networks*, Science 286 (1999); Bianconi und Barabási, *Competition and multiscaling in evolving networks*, Europhysics Letters 54 (2001); Krapivsky, Redner und Leyvraz, *Connectivity of growing random networks*, Physical Review Letters 85 (2000); Dorogovtsev, Mendes und Samukhin, *Structure of growing networks with preferential linking*, Physical Review Letters 85 (2000).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
