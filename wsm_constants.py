"""Konstanten: Regler-Grenzen, Rechengrenzen, feste Pläne der Messreihen, Modelle, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 500, 10000, 3000      # Knoten; allgemeines Wachstum quadratisch in n (10 000 Knoten etwa eine Sekunde), lineare Anbindung fast linear
M_MIN, M_MAX, DEFAULT_M = 1, 5, 2               # Kanten je neuem Knoten
A_MIN, A_MAX, DEFAULT_A = 0.0, 2.0, 1.0         # Exponent der Anbindung: Gewicht (k + A0)^a; 0 = Zufall, 1 = linear
A0_MIN, A0_MAX, DEFAULT_A0 = 0.0, 10.0, 0.0     # anfängliche Attraktivität A0
NU_MIN, NU_MAX, DEFAULT_NU = 0.0, 2.0, 0.0      # Alterung: Gewicht mal (Alter + 1)^(−nu)
SEED_MAX, DEFAULT_SEED = 999999, 35

FITNESS = {"none": "keine", "uniform": "gleichverteilt (0 bis 1)"}
DEFAULT_FITNESS = "none"
MLE_KMIN_FACTOR = 3                             # kleinster Grad des Exponent-Schätzers in der Hauptansicht: 3 · m

SWEEP_SEEDS = tuple(range(100000, 100005))      # feste Pläne für alle Messreihen: nie ein einzelner Seed als Befund

EXP_N, EXP_M = 16000, 3                         # Reihe „Exponent und Untergrenze“
EXP_KMINS = (3, 4, 6, 10, 20, 40)
HUB_NS, HUB_M = (1000, 4000, 16000, 64000), 3   # Reihe „größter Grad gegen n“
MODEL_N, MODEL_M = 4000, 2                      # Reihe „Modelle im Vergleich“
KERNEL_N, KERNEL_M = 3000, 2                    # Reihe „überlineare Anbindung“
KERNEL_EXPONENTS = (0.0, 0.5, 1.0, 1.25, 1.5, 2.0)

# Modelle der Vergleichsreihe: Name, Exponent a, anfängliche Attraktivität A0, Alterung nu, Fitness
MODELS = (
    ("Zufallsanbindung (a = 0)", 0.0, 0.0, 0.0, False),
    ("unterlinear (a = 0,5)", 0.5, 0.0, 0.0, False),
    ("linear (Barabási–Albert)", 1.0, 0.0, 0.0, False),
    ("überlinear (a = 1,5)", 1.5, 0.0, 0.0, False),
    ("linear mit Alterung (nu = 1)", 1.0, 0.0, 1.0, False),
    ("linear mit Fitness", 1.0, 0.0, 0.0, True),
    ("linear mit Anfangsattraktivität 3", 1.0, 3.0, 0.0, False),
)

COLOR_TARGET = "rgba(120,120,120,0.55)"
COLOR_MEASURED = "#2F6B65"
COLOR_THEORY = "#4c78a8"
COLOR_THIRD = "#e8a33d"
COLOR_ALERT = "#d62728"

PRESETS = {
    "Standardfall: lineare Anbindung": {"n": DEFAULT_N, "m": DEFAULT_M, "a": DEFAULT_A, "a0": DEFAULT_A0, "nu": DEFAULT_NU, "fitness": DEFAULT_FITNESS, "seed": DEFAULT_SEED},
    "Zufallsanbindung (a = 0)": {"n": 2500, "m": DEFAULT_M, "a": 0.0, "a0": DEFAULT_A0, "nu": DEFAULT_NU, "fitness": DEFAULT_FITNESS, "seed": DEFAULT_SEED},
    "Überlinear: ein Hub gewinnt (a = 1,5)": {"n": 2500, "m": DEFAULT_M, "a": 1.5, "a0": DEFAULT_A0, "nu": DEFAULT_NU, "fitness": DEFAULT_FITNESS, "seed": DEFAULT_SEED},
    "Alterung (nu = 1)": {"n": 2500, "m": DEFAULT_M, "a": DEFAULT_A, "a0": DEFAULT_A0, "nu": 1.0, "fitness": DEFAULT_FITNESS, "seed": DEFAULT_SEED},
    "Fitness: Neulinge können aufholen": {"n": 2500, "m": DEFAULT_M, "a": DEFAULT_A, "a0": DEFAULT_A0, "nu": DEFAULT_NU, "fitness": "uniform", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall: lineare Anbindung": "3 000 Knoten, jeder neue Knoten wählt zwei Ziele mit Wahrscheinlichkeit proportional zum Grad: das Barabási–Albert-Modell. Die Gradverteilung folgt einem Potenzgesetz, der größte Grad wächst etwa mit der Wurzel aus n.",
    "Zufallsanbindung (a = 0)": "Jedes Ziel gleich wahrscheinlich: ohne Bevorzugung gibt es keine Hubs, die Gradverteilung fällt exponentiell und der größte Grad bleibt klein.",
    "Überlinear: ein Hub gewinnt (a = 1,5)": "Gewicht k hoch 1,5: wer schon viele hat, bekommt überproportional mehr, und ein einziger Hub zieht einen großen Teil aller Kanten an sich.",
    "Alterung (nu = 1)": "Das Gewicht sinkt mit dem Alter der Knoten: alte Knoten verlieren ihren Vorsprung, es entstehen keine großen Hubs und die Verteilung fällt steil ab.",
    "Fitness: Neulinge können aufholen": "Jeder Knoten hat eine zufällige Fitness als Faktor im Gewicht: Spätankömmlinge mit hoher Fitness holen auf, und der größte Hub wird größer als ohne Fitness.",
}
