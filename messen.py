"""Einmalige Messung der README-Zahlen (nicht Teil der App): feste Pläne, Ausgabe auf stdout."""

import sys
import time

sys.path.insert(0, ".")
import wsm_constants as C
import wsm_evaluation as ev

t0 = time.time()
print("PRESETS")
for name, p in C.PRESETS.items():
    a = ev.analyse(ev.Settings(**p))
    print(name, "kmax", a.kmax, "share", round(a.share, 4), "alpha", round(a.alpha, 3), "kmin", a.kmin, "theory", a.theory)
print("EXPONENT")
for r in ev.series_exponent():
    print(r)
print("HUB")
for r in ev.series_hub_growth():
    print(r)
print("MODELS")
for r in ev.series_models():
    print(r)
print("KERNEL")
for r in ev.series_kernel():
    print(r)
print("sec", round(time.time() - t0, 1))
