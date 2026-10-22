"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import wsm_algorithm as W
import wsm_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_degrees(deg, m: int, theory: str | None):
    """Gradverteilung (Anteil der Knoten je Grad, beide Achsen logarithmisch); bei Zufalls- und linearer Anbindung die Theorie als Linie."""
    n = len(deg)
    counts = sorted(W.degree_counts(deg).items())
    ks = [k for k, _ in counts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[c / n for _, c in counts], mode="markers", marker=dict(size=8, color=C.COLOR_MEASURED), name="gemessen",
                             hovertemplate="Grad %{x}: %{y:.4f}<extra></extra>"))
    if theory is not None and ks:
        pmf = W.linear_degree_pmf if theory == "linear" else W.uniform_degree_pmf
        grid = sorted({int(round(x)) for x in np.geomspace(max(m, 1), max(ks), 60)})
        label = "Theorie (lineare Anbindung)" if theory == "linear" else "Theorie (Zufallsanbindung)"
        fig.add_trace(go.Scatter(x=grid, y=[pmf(k, m) for k in grid], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name=label))
    fig.update_xaxes(title_text="Grad", type="log")
    fig.update_yaxes(title_text="Anteil der Knoten", type="log")
    return _base(fig)


def build_hub_growth(trace, m: int):
    """Größter Grad nach der Ankunft jedes Knotens (beide Achsen logarithmisch) neben der Wurzel-Kurve m·√t, an die der Wert bei der linearen Anbindung anschließt."""
    n = len(trace)
    ts = np.arange(1, n + 1)
    step = max(1, n // 400)
    sel = ts[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sel, y=np.asarray(trace)[sel - 1], mode="lines", line=dict(color=C.COLOR_MEASURED, width=3), name="größter Grad",
                             hovertemplate="%{x} Knoten: %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sel, y=[m * math.sqrt(t) for t in sel], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="m · √t"))
    fig.update_xaxes(title_text="Zahl der Knoten", type="log")
    fig.update_yaxes(title_text="größter Grad", type="log")
    return _base(fig)


def build_exponent(rows):
    """Geschätzter Exponent über der Untergrenze kmin (Log-Achse) mit der Theorie 3 als Linie."""
    ks = [r["kmin"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[min(ks), max(ks)], y=[3.0, 3.0], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="Theorie (n → ∞)"))
    fig.add_trace(go.Scatter(x=ks, y=[r["alpha"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=3), name="geschätzt (Median)",
                             hovertemplate="kmin %{x}: α̂ = %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Untergrenze kmin des Schätzers", type="log")
    fig.update_yaxes(title_text="geschätzter Exponent α̂")
    return _base(fig)


def build_hub_scaling(rows):
    """Größter Grad über n (beide Achsen logarithmisch) mit Band Minimum bis Maximum und der Wurzel-Referenz m·√n."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns + ns[::-1], y=[r["high"] for r in rows] + [r["low"] for r in rows][::-1], fill="toself", fillcolor="rgba(47,107,101,0.15)", line=dict(width=0),
                             hoverinfo="skip", name="Minimum bis Maximum"))
    fig.add_trace(go.Scatter(x=ns, y=[C.HUB_M * math.sqrt(n) for n in ns], mode="lines", line=dict(color=C.COLOR_THEORY, width=2, dash="dash"), name="m · √n"))
    fig.add_trace(go.Scatter(x=ns, y=[r["kmax"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=3), name="größter Grad (Median)",
                             hovertemplate="n = %{x}: %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title_text="Knotenzahl n", type="log")
    fig.update_yaxes(title_text="größter Grad", type="log")
    return _base(fig)


def build_models(rows):
    """Größter Grad je Modell (Log-Achse), der Anteil des größten Hubs an allen Kantenenden steht im Hover."""
    names = [r["name"] for r in rows]
    colors = [C.COLOR_ALERT if r["share"] >= 0.1 else C.COLOR_MEASURED for r in rows]
    fig = go.Figure(go.Bar(x=names, y=[r["kmax"] for r in rows], marker_color=colors, customdata=[[r["share"] * 100] for r in rows],
                           hovertemplate="%{x}: größter Grad %{y:.0f}, Hub hält %{customdata[0]:.1f} % der Kantenenden<extra></extra>"))
    fig.update_xaxes(title_text="Modell", tickangle=-25)
    fig.update_yaxes(title_text="größter Grad (Median)", type="log")
    return _base(fig, height=420)


def build_kernel(rows):
    """Anteil des größten Hubs an allen Kantenenden über dem Exponenten der Anbindung, mit Band Minimum bis Maximum."""
    xs = [r["a"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r["high"] * 100 for r in rows] + [r["low"] * 100 for r in rows][::-1], fill="toself", fillcolor="rgba(47,107,101,0.15)", line=dict(width=0),
                             hoverinfo="skip", name="Minimum bis Maximum"))
    fig.add_trace(go.Scatter(x=xs, y=[r["share"] * 100 for r in rows], mode="lines+markers", line=dict(color=C.COLOR_MEASURED, width=3), name="größter Hub (Median)",
                             hovertemplate="a = %{x}: %{y:.1f} %<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=C.COLOR_ALERT, width=1.5, dash="dash"))
    fig.update_xaxes(title_text="Exponent a der Anbindung (Gewicht k hoch a)")
    fig.update_yaxes(title_text="Anteil des größten Hubs an allen Kantenenden (%)")
    return _base(fig)
