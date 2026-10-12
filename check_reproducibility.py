"""Verifica que dos ejecuciones con los mismos argumentos producen exactamente las mismas métricas.

Uso:  python check_reproducibility.py [argumentos de train.py]
Las corridas se registran en un backend temporal para no ensuciar el experimento principal.
"""
import sys
import tempfile
from pathlib import Path

import train

with tempfile.TemporaryDirectory() as tmp:
    uri = f"sqlite:///{Path(tmp) / 'repro.db'}"
    extra = sys.argv[1:] + ["--tracking-uri", uri, "--experiment-name", "reproducibility-check"]
    first = train.main(extra)
    second = train.main(extra)

assert first == second, f"Las métricas difieren:\n{first}\n{second}"
print("\nOK: ambas ejecuciones produjeron métricas idénticas (comparación exacta, sin redondeo).")
