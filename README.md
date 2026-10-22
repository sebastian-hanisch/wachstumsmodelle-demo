# Wachstumsmodelle – was erzeugt Hubs? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-wachstumsmodelle-demo.streamlit.app/)** -->

Drittes Stück der **Netzmodelle-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vorgänger sind
[zufallsgraph-demo](https://github.com/sebastian-hanisch/zufallsgraph-demo) (das einfachste Zufallsnetz, ohne Hubs) und [konfigurationsmodell-demo](https://github.com/sebastian-hanisch/konfigurationsmodell-demo)
(Netze zu vorgegebenen Graden): das Konfigurationsmodell würfelt Netze zu einer gegebenen Gradfolge, **woher** eine Gradfolge mit Hubs kommen kann, zeigt dieses Stück. Die Linie erzeugt Netze aus Zufallsmodellen und prüft sie gegen die Theorie, während
die Linie „Netzwerkanalyse“ ein gegebenes Netz vermisst.

Die Demo lässt ein Netz Knoten für Knoten wachsen: jeder neue Knoten wählt m verschiedene Ziele mit einem Gewicht, das vom Grad des Ziels abhängt (**bevorzugte Anbindung**, Barabási und Albert), und verändert die Regel:
Zufallsanbindung ohne Bevorzugung, unter- und **überlineare** Bevorzugung, anfängliche Attraktivität, **Alterung** und **Fitness**. Sie zeigt Gradverteilung, größten Grad und den geschätzten Exponenten gegen die Theorie.
`networkx` und `scipy` kommen nur als Gegenproben in die Tests.

## Warum dieses Problem

Reale Netze haben Hubs. Die einfachste Erklärung ist eine **Wachstumsregel**: wer schon viele Kanten hat, bekommt mit größerer Wahrscheinlichkeit weitere. Die Frage der Demo:
**Welche Regel erzeugt Hubs und ein Potenzgesetz, wie groß wird der größte Hub, und was passiert, wenn man die Regel verändert?** Dazu gehört eine Warnung: der Exponent des Potenzgesetzes
lässt sich aus der Gradverteilung eines endlichen Netzes nicht einfach ablesen, die Schätzung hängt von der gewählten Untergrenze ab.

## Modell

Start mit m + 1 vollständig verbundenen Knoten. Der neue Knoten t wählt m **verschiedene** Ziele nacheinander ohne Zurücklegen, jedes bestehende Knoten i mit Wahrscheinlichkeit proportional zum Gewicht
(k_i + A₀)^a · (t − t_i + 1)^(−ν) · η_i. Dabei ist k_i der Grad, a der Exponent der Anbindung (0: Zufall, 1: linear), A₀ die anfängliche Attraktivität, ν die Alterung (t_i Ankunftszeit) und η_i die Fitness (gleichverteilt, oder überall 1).
Die Theorie (Grenzwert n → ∞, linear, A₀ = 0): P(k) = 2m(m+1) / (k(k+1)(k+2)) für k ≥ m, also ein Potenzgesetz mit Exponent 3; bei Zufallsanbindung P(k) = (1/(m+1))·(m/(m+1))^(k−m), also exponentiell;
der größte Grad wächst mit der Größenordnung m·√n; mit Anfangsattraktivität A₀ ist der Exponent 3 + A₀/m.
Formeln und Lesehilfe stehen im 📐-Expander der App. Die Literaturangaben dort stammen aus dem Gedächtnis und sind nicht im Original geprüft.

## Methodik

- **Kern** (`wsm_algorithm.py`): Gewichte nach der Formel, Ziehen von m verschiedenen Zielen ohne Zurücklegen, das allgemeine Wachstum (quadratische Laufzeit) und ein schneller Sampler für die rein lineare Regel
  (Liste der Kantenenden, fast linear), Gradzählung, Exponent-Schätzer für diskrete Daten (1 + N / Σ ln(k / (kmin − 0,5))), Anteil des größten Hubs an allen Kantenenden, die Theorie.
- **Auswertung** (`wsm_evaluation.py`): Kennzahlen eines Netzes, vier Messreihen mit festem Programm (Median über fünf feste Zufalls-Seeds), alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): fünf Schnellstart-Szenarien, Regler für n, m, a, A₀, ν und Fitness, Gradverteilung und Wachstum des größten Grades, Kernfrage, Messreihen auf Klick, Annahmen und Formeln.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen mit festen Zufalls-Seeds (`SWEEP_SEEDS` 100000 bis 100004), nie ein einzelner Seed; Messzahlen der Reihen sind Mediane. Die Schnellstart-Szenarien sind **einzelne Läufe** mit dem Zufalls-Seed 35 (die Zahlen schwanken
von Zufalls-Seed zu Zufalls-Seed). **Die Zufallszahlen eines `numpy`-Generators sind nicht über Versionen stabil**: die Tests prüfen deshalb Bänder um diese Zahlen.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Der Endgrad des ältesten Knotens stimmt für fünf Regeln (linear, Zufall, Alterung, überlinear, mit Anfangsattraktivität und Alterung) mit der **exakt durch Aufzählung aller Zustände** berechneten Verteilung überein (m = 1, 7 Knoten, je 12 000 Läufe, innerhalb von fünf Standardfehlern); der schnelle Sampler hat dieselbe Verteilung wie das allgemeine Wachstum (KS-Test, 2 500 Läufe); `networkx.barabasi_albert_graph` liefert dieselbe Hubgröße (innerhalb 20 %) und denselben Exponenten (innerhalb 0,08). | `tests/test_oracle_wachstumsmodelle.py` |
| Welche Werte hat die Theorie? | Für m = 2: P(2) = **0,5**, P(3) = **0,2** bei linearer Anbindung, P(2) = **1/3** bei Zufallsanbindung; Exponent 3 bei A₀ = 0 und **4,5** bei m = 2, A₀ = 3; Kantenzahl bei n = 3 000, m = 2: **5 997**. Ein Netz mit 64 000 Knoten liegt bei den Graden 3 bis 10 innerhalb von 8 % der Theorie, bei Zufallsanbindung (8 000 Knoten) die Grade 2 bis 8 innerhalb von 15 %. | `test_claims.py::test_theory_values`, Orakel |
| Was zeigen die Schnellstart-Szenarien? | Standardfall (n = 3 000, m = 2, linear): größter Grad **103**, der Hub hält **0,9 %** aller Kantenenden, geschätzter Exponent **2,79**. Zufallsanbindung (n = 2 500): größter Grad **22**, α̂ = 3,98 (bedeutungslos, die Verteilung ist exponentiell). Überlinear (a = 1,5): größter Grad **2 037**, ein Hub hält **20,4 %** aller Kantenenden. Alterung (ν = 1): größter Grad **18**. Fitness: größter Grad **233**, mehr als im Standardfall bei etwas kleinerem n. | `test_claims.py::test_presets_hold_what_their_help_texts_say` |
| Hängt der Exponent von der Untergrenze ab? | Bei n = 16 000, m = 3 und linearer Anbindung steigt der geschätzte Exponent mit der Untergrenze kmin: **2,58** (kmin 3, alle 16 000 Knoten), **2,78** (kmin 6, rund 4 570 Knoten), **2,95** (kmin 40, rund 120 Knoten); die Theorie für n → ∞ ist 3. Die Schätzung liegt in der ganzen Reihe darunter. | `test_claims.py::test_the_estimated_exponent_depends_on_the_lower_bound` |
| Wie wächst der größte Grad? | Mit m = 3: **117** (n = 1 000), **254** (4 000), **488** (16 000), **972** (64 000); geteilt durch √n zwischen **3,7 und 4,0**, also das 1,2- bis 1,3-fache von m·√n. Er wächst wie √n; die Konstante der Näherung m·√n trifft nicht. | `test_claims.py::test_the_largest_degree_grows_like_the_root_of_n` |
| Wie unterscheiden sich die Regeln? | Größter Grad (Median bei n = 4 000, m = 2): Zufallsanbindung **20**, unterlinear (a = 0,5) **36**, Anfangsattraktivität 3 **54**, linear **145**, linear mit Fitness **746**, überlinear (a = 1,5) **3 327**; mit Alterung **21**. Bei überlinearer Anbindung hält der größte Hub **20,8 %** aller Kantenenden, bei linearer **0,9 %**. Geschätzter Exponent: linear **2,78**, mit Alterung **4,52**, mit Anfangsattraktivität 3 **3,20** (Theorie für n → ∞: 4,5). | `test_claims.py::test_the_model_comparison` |
| Wo kippt die Anbindung? | Anteil des größten Hubs an allen Kantenenden bei n = 3 000, m = 2: a = 0: **0,2 %**, a = 0,5: **0,3 %**, a = 1: **1,0 %**, a = 1,25: **6,8 %**, a = 1,5: **20,4 %**, a = 2: **24,7 %**. Der Anteil wächst mit a steil ab 1 und sättigt bei etwa 25 % (= 1/(2m)): die Erklärung, dass der Hub bei a = 2 praktisch eine von zwei Kanten jedes neuen Knotens abbekommt, ist nicht gesondert geprüft. | `test_claims.py::test_the_kernel_series` |

## Ehrliche Grenzen

- **Die Schnellstart-Szenarien sind einzelne Läufe.** Der Standardfall hat dort größten Grad 103, der Median der Reihe bei n = 4 000 liegt bei 145; die Streuung zwischen Zufalls-Seeds ist groß (bei n = 4 000, m = 3 liegt der kleinste größte Grad über fünf Pläne bei 169, der größte bei 294).
- **Der Exponent-Schätzer ist keine Anpassungsprüfung.** Er rechnet auch für Verteilungen, die kein Potenzgesetz sind, eine Zahl aus (bei Zufallsanbindung 3,98, bei Alterung 4,34); die Demo prüft nicht, ob ein Potenzgesetz zu den Daten passt, und nutzt eine feste Untergrenze 3·m. Wie stark die Zahl von kmin abhängt, zeigt die Messreihe.
- **Für Alterung, Fitness und überlineare Anbindung gibt die Demo keine Theorie der Gradverteilung an.** Ihre Zahlen sind Messungen; ob sie zu bekannten Ergebnissen der Literatur passen, ist nicht geprüft.
- **m·√n ist nur die Größenordnung.** Gemessen liegt der größte Grad beim 1,2- bis 1,3-fachen; die Demo zeigt die Wurzel-Skalierung (das Verhältnis zu √n schwankt zwischen 3,7 und 4,0), nicht die genaue Konstante.
- **Kleine Netze.** n liegt zwischen 500 und 10 000 (das allgemeine Wachstum ist quadratisch in n); die Messreihen laufen bis 64 000 Knoten nur mit der schnellen linearen Regel. Aussagen über n → ∞ sind Extrapolation.
- **Keine realen Netze.** Die Demo vergleicht Modelle mit Theorie, nicht mit Daten (dafür steht die Linie „Netzwerkanalyse“); keine gerichteten, gewichteten oder mehrfachen Kanten.
- **Literatur nicht im Original geprüft** (Barabási und Albert 1999, Krapivsky, Redner und Leyvraz 2000, Bianconi und Barabási 2001, Clauset, Shalizi und Newman 2009; aus dem Gedächtnis).

## Befunde und Korrekturen gegenüber dem Plan

- **Port:** 9159 statt 9158 (9158 hält `zufallsverfahren-demo`).
- **Größter Grad gegen √n:** die Vormessung im Plan fand für m = 3 Verhältnisse von 2,9 bis 3,4; die Demo misst mit den festen Plänen **3,7 bis 4,0**. Die Richtung (Wurzelwachstum) ist dieselbe, die Konstante hängt von den Zufalls-Seeds ab (andere Pläne, fünf statt weniger Läufe).
- **Fitness:** die Vormessung fand für den größten Grad bei Fitness 435 bis 580, die Demo im Vergleich bei n = 4 000 **746** (Median); in beiden Fällen deutlich über linear, die Zahlen sind nicht vergleichbar (andere Pläne).

## Verifikation

- **Kern:** Gewichte und Ziehen an einer Mini-Instanz von Hand (`tests/conftest.py`), Invarianten des Wachstums (Gradsumme, Kantenzahl, keine Doppelkanten, Startklique, Determinismus), Schätzer und Theorie von Hand.
- **Orakel:** exakte Verteilung des Endgrades durch Aufzählung aller Zustände (fünf Regeln), KS-Test schneller gegen allgemeinen Sampler, Theorie gegen große Netze, `networkx.barabasi_albert_graph`, Schätzer an bekannten diskreten Potenzgesetzen (Exponent 2,5, 3, 3,5).
- **Verdikte:** reine Funktionen, jede Zahl im Text an handgesetzten Werten, einschließlich der Schwelle von 10 % für „ein einziger Hub“ (`tests/test_verdicts.py`).
- **Presets, Layout, Oberfläche:** wie in allen Demos (Presets vollständig und im Regelbereich, Schnellstart oben, aktives Preset hervorgehoben, Auswahlfeld in der Layoutprüfung, AppTest jeder Button und jeder Regler an den Rändern).
- **Mutationsprüfung** (`tools/mutate_demo.py`): 107 Stellen im Kern `wsm_algorithm.py`, **99 gefangen** (4 davon über wortgleiche Zeilen in Nachbar-Demos), 8 als äquivalent festgehalten. Der erste Lauf ließ 13 Mutanten überleben und deckte drei echte Lücken auf (Ziehen am oberen Rand der Rundung, Wechsel des Zufallszahlen-Puffers der schnellen Regel, Schätzer bei kmin = 1), für die es jetzt Tests gibt (`tests/test_edge_cases.py`). Die äquivalenten Stellen: Größe des Zufallszahlen-Puffers in `grow_linear` (die Konstanten 64 und 4 ändern nur, wann der Puffer erneuert wird, nicht die Verteilung; vier Mutanten), `>` gegen `>=` beim Mitführen des größten Grades (derselbe Wert), und im Schätzer `kmin < 0` statt `kmin < 1` (kmin = 0 ergibt über nan dasselbe Ergebnis) sowie die zwei Mutanten der Abfrage `s <= 0`, die nie wahr wird (jeder Summand ist positiv, weil jedes k mindestens kmin ist).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und networkx (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `wsm_algorithm.py` | Kern: Gewichte, Ziehen, Wachstum (allgemein und schnell linear), Schätzer, Theorie |
| `wsm_constants.py` | Regler-Grenzen, feste Pläne der Messreihen, Modelle, Farben, Presets |
| `wsm_evaluation.py` | Kennzahlen, Messreihen, Verdikte |
| `wsm_layout.py` | Fester Kopf (Stylesheet) |
| `wsm_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `wsm_scenario.py` | Modell der Einstellungen, Wahl des Samplers, Art der Theorie |
| `wsm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `messen.py` | Einmalige Messung der Zahlen in dieser README (nicht Teil der App) |
| `tests/` | Mini-Instanz, Kern, Orakel, Verdikte, Randfälle, Presets, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

- **Anpassungsprüfung des Potenzgesetzes** (Kolmogorov–Smirnov-Abstand mit Wahl von kmin nach Clauset, Shalizi und Newman): würde die Abhängigkeit von kmin ersetzen, ist ein eigenes Stück.
- **Kopieren und Triadenschluss** (Wachstum mit hohem Clustering): eigene Folgemodelle der Linie, hier nur die Gradregel.
- **Gerichtete Netze und Zitationsnetze** (alte Knoten zitieren nicht neue): andere Gradverteilung (Eingrad), nicht Teil dieser Demo.
- **Anpassung an reale Gradfolgen:** gehört zur Netzwerkanalyse.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
