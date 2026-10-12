# Bank Marketing: experimentación reproducible con MLflow

Este repositorio contiene un modelo base, una regresión logística que predice si un cliente contratará un depósito a plazo (dataset *Bank Marketing* de UCI). El modelo se convirtió en un script de línea de comandos que registra cada corrida en **MLflow Tracking**.

## Estructura

```
.
├── train.py                  # carga → limpieza → split → entrenamiento → evaluación → MLflow
├── run_experiments.sh        # las 9 corridas comparadas en el análisis
├── check_reproducibility.py  # verifica que dos ejecuciones iguales dan métricas idénticas
├── requirements.txt          # dependencias con versiones exactas
├── data/                     # bank-full.csv (UCI, CC BY 4.0) y su descripción
├── notebooks/eda.ipynb       # solo exploración de datos y decisiones de limpieza
└── docs/capturas/            # evidencias de la interfaz de MLflow
```

## Instalación

Requisitos: Python 3.9 o superior. Los resultados se generaron con Python 3.9.6.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
# Con valores por defecto (C=1.0, l2, lbfgs, class_weight=balanced, umbral=0.5, seed=42)
python train.py

# Cambiando hiperparámetros sin editar el código
python train.py --C 0.1 --penalty l1 --solver liblinear --class-weight balanced --threshold 0.5 --seed 42

# Todas las corridas del análisis
bash run_experiments.sh

# Lista completa de argumentos
python train.py --help
```

| Argumento | Defecto | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la fuerza de regularización |
| `--penalty` | l2 | `l1`, `l2` o `none` |
| `--solver` | lbfgs | `lbfgs`, `liblinear` o `saga` (`l1` requiere `liblinear` o `saga`) |
| `--class-weight` | balanced | `balanced` o `none` |
| `--max-iter` | 1000 | Iteraciones máximas |
| `--threshold` | 0.5 | Umbral de probabilidad para predecir "sí" |
| `--test-size` | 0.2 | Proporción del conjunto de prueba |
| `--winsor-pct` | 0.01 | Recorte de outliers en p y 1−p (0 = desactivado) |
| `--seed` | 42 | Semilla global |
| `--experiment-name` | bank-marketing-logreg-tuning | Experimento de MLflow |
| `--tracking-uri` | `sqlite:///mlflow.db` | Backend de MLflow (también se toma de `MLFLOW_TRACKING_URI`) |

## Ver y comparar las corridas

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://127.0.0.1:5000 y entra al experimento **bank-marketing-logreg-tuning**. Marca las corridas y pulsa **Compare**.

Cada corrida registra lo siguiente:

- **Parámetros** (`log_param`): C, penalty, solver, class_weight, max_iter, threshold, test_size, winsor_pct, seed, n_train y n_test.
- **Métricas** (`log_metric`):
  - de prueba: ROC-AUC, PR-AUC, F1, recall, precision, accuracy y lift en el top 20 %;
  - de entrenamiento: ROC-AUC y F1;
  - número de iteraciones.
- **Artefactos** (`log_artifact`, carpeta `evaluacion/`): matriz de confusión, curva ROC, `metrics.json`, reporte de clasificación y coeficientes.
- **Modelo** (`mlflow.sklearn.log_model`): el pipeline completo (preprocesamiento y modelo) con firma y ejemplo de entrada.
- **Tags**: el hash SHA-256 del CSV, para identificar la versión de los datos.

## Reproducibilidad

- `set_seed()` fija `random`, `numpy` y `PYTHONHASHSEED`. La misma semilla se pasa a `train_test_split` y a `LogisticRegression` como `random_state`.
- Los límites de winsorización se calculan solo con el conjunto de entrenamiento.
- Las dependencias tienen versiones exactas en `requirements.txt`.
- Para comprobar la reproducibilidad, ejecuta:

  ```bash
  python check_reproducibility.py            # admite los mismos argumentos que train.py
  ```

  El script ejecuta `train.py` dos veces con los mismos argumentos en un backend temporal y comprueba que todas las métricas coincidan exactamente.

## Datos

Moro, S., Cortez, P. & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. UCI Machine Learning Repository. Licencia CC BY 4.0.
