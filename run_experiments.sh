#!/usr/bin/env bash
# Ejecuta la batería de corridas del experimento "bank-marketing-logreg-tuning".
# Cada línea varía uno o dos hiperparámetros respecto a la configuración base.
set -euo pipefail
PY=${PYTHON:-python}

$PY train.py                                                      # 1. base: C=1, l2, lbfgs, balanced, umbral 0.5
$PY train.py --C 0.01                                             # 2. regularización fuerte
$PY train.py --C 0.1                                              # 3. regularización media
$PY train.py --C 10                                               # 4. regularización débil
$PY train.py --C 0.1  --penalty l1 --solver liblinear             # 5. L1 (selección de variables)
$PY train.py --C 0.01 --penalty l1 --solver liblinear             # 6. L1 muy fuerte (modelo disperso)
$PY train.py --class-weight none                                  # 7. sin compensar el desbalance
$PY train.py --class-weight none --threshold 0.2                  # 8. sin pesos, umbral ajustado
$PY train.py --seed 7                                             # 9. base con otra semilla (variación por partición)
