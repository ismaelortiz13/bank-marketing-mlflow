"""Entrenamiento reproducible del modelo base (regresión logística) para Bank Marketing.

Migrado desde el notebook de la semana anterior. Cada ejecución:
  1. carga y limpia los datos,
  2. divide en entrenamiento/prueba de forma estratificada,
  3. entrena un Pipeline (preprocesamiento + LogisticRegression),
  4. evalúa en prueba,
  5. registra parámetros, métricas, artefactos y el modelo en MLflow.

Uso:
    python train.py                         # valores por defecto
    python train.py --C 0.1 --penalty l1 --solver liblinear --seed 7
    python train.py --help                  # lista completa de argumentos
"""

import argparse
import hashlib
import json
import logging
import os
import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silencia avisos no críticos (p. ej. urllib3/LibreSSL en macOS)

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz gráfica: funciona en terminal y servidores
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

# --- Definición de columnas (decididas en el EDA del notebook) ---
NUM_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
BIN_FEATURES = ["default", "housing", "loan", "contactado_antes"]
CAT_FEATURES = ["job", "marital", "education", "contact", "month", "poutcome"]
WINSOR_COLS = ["balance", "campaign", "previous"]
TARGET = "y"


# ---------------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Define y lee los argumentos. Todos tienen valor por defecto."""
    p = argparse.ArgumentParser(
        description="Entrena una regresión logística sobre Bank Marketing y la registra en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparámetros del modelo
    p.add_argument("--C", type=float, default=1.0, help="Inverso de la fuerza de regularización")
    p.add_argument("--penalty", choices=["l1", "l2", "none"], default="l2", help="Tipo de regularización")
    p.add_argument("--solver", choices=["lbfgs", "liblinear", "saga"], default="lbfgs", help="Algoritmo de optimización")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                   help="Ponderación de clases para compensar el desbalance")
    p.add_argument("--max-iter", type=int, default=1000, help="Iteraciones máximas del optimizador")
    p.add_argument("--threshold", type=float, default=0.5, help="Umbral de probabilidad para predecir la clase positiva")
    # Datos y reproducibilidad
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba")
    p.add_argument("--winsor-pct", type=float, default=0.01,
                   help="Percentil de recorte de outliers (se recorta en p y 1-p; 0 desactiva)")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global")
    p.add_argument("--data-path", default="data/bank-full.csv", help="Ruta al CSV original")
    # MLflow
    p.add_argument("--experiment-name", default="bank-marketing-logreg-tuning", help="Experimento de MLflow")
    p.add_argument("--run-name", default=None, help="Nombre de la corrida (por defecto se genera a partir de los hiperparámetros)")
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                   help="URI del servidor/backend de MLflow")
    args = p.parse_args(argv)

    # Validaciones de combinaciones que scikit-learn no admite
    if args.penalty == "l1" and args.solver == "lbfgs":
        p.error("--penalty l1 requiere --solver liblinear o saga")
    if args.penalty == "none" and args.solver == "liblinear":
        p.error("--penalty none no es compatible con --solver liblinear")
    if not 0 < args.threshold < 1:
        p.error("--threshold debe estar entre 0 y 1")
    return args


def set_seed(seed):
    """Fija todas las fuentes de aleatoriedad usadas por el script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def file_sha256(path):
    """Huella del archivo de datos: permite saber con qué versión exacta se entrenó."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    """Lee el CSV original (separado por ';')."""
    return pd.read_csv(path, sep=";")


def clean_data(df):
    """Aplica las decisiones de limpieza documentadas en el notebook de EDA."""
    df = df.drop_duplicates().reset_index(drop=True)
    # yes/no -> 1/0
    for col in ["default", "housing", "loan", TARGET]:
        df[col] = (df[col] == "yes").astype(int)
    # pdays = -1 significa "nunca contactado": se vuelve indicador explícito
    df["contactado_antes"] = (df["pdays"] != -1).astype(int)
    df.loc[df["pdays"] == -1, "pdays"] = 0
    # 'unknown' se mantiene como categoría propia
    df[CAT_FEATURES] = df[CAT_FEATURES].astype("category")
    # 'duration' solo se conoce después de la llamada -> fuga de información
    return df.drop(columns=["duration"])


def split_data(df, test_size, seed):
    """División estratificada entrenamiento/prueba."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def winsorize(X_train, X_test, pct):
    """Recorta outliers con límites aprendidos SOLO del entrenamiento (sin fuga hacia prueba)."""
    X_train, X_test = X_train.copy(), X_test.copy()
    if pct <= 0:
        return X_train, X_test
    for col in WINSOR_COLS:
        lo, hi = X_train[col].quantile([pct, 1 - pct])
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col] = X_test[col].clip(lo, hi)
    return X_train, X_test


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def build_pipeline(args):
    """Preprocesamiento (escalado + one-hot) y regresión logística en un solo Pipeline."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_FEATURES),
        ("bin", "passthrough", BIN_FEATURES),
    ])
    clf = LogisticRegression(
        C=args.C,
        penalty=None if args.penalty == "none" else args.penalty,
        solver=args.solver,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    return Pipeline([("preprocesamiento", preprocessor), ("clasificador", clf)])


def evaluate(model, X, y, threshold):
    """Calcula métricas de clasificación binaria con el umbral indicado."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    # Lift en el 20 % de clientes con mayor probabilidad (utilidad para la campaña)
    top = np.argsort(-proba)[: int(0.2 * len(proba))]
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "lift_top20": y.iloc[top].mean() / y.mean(),
    }, pred


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------
def save_artifacts(model, X_test, y_test, pred, metrics, out_dir):
    """Genera figuras y reportes de la corrida en out_dir y devuelve sus rutas."""
    out_dir = Path(out_dir)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["no", "sí"],
                                            cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Matriz de confusión (prueba)")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=110); plt.close(fig)
    paths.append(out_dir / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="LogReg")
    ax.plot([0, 1], [0, 1], "k--", label="Azar")
    ax.legend(); ax.set_title("Curva ROC (prueba)")
    fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=110); plt.close(fig)
    paths.append(out_dir / "roc_curve.png")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    paths.append(out_dir / "metrics.json")

    (out_dir / "classification_report.txt").write_text(classification_report(y_test, pred, digits=3))
    paths.append(out_dir / "classification_report.txt")

    names = model.named_steps["preprocesamiento"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["clasificador"].coef_[0], index=names, name="coef")
    coefs.sort_values().to_csv(out_dir / "coefficients.csv")
    paths.append(out_dir / "coefficients.csv")
    return paths


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    df = clean_data(load_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)
    X_train, X_test = winsorize(X_train, X_test, args.winsor_pct)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    run_name = args.run_name or (
        f"C={args.C}_{args.penalty}_{args.solver}_cw={args.class_weight}_thr={args.threshold}_seed={args.seed}")

    with mlflow.start_run(run_name=run_name) as run:
        # 1) Hiperparámetros y configuración
        params = {
            "C": args.C, "penalty": args.penalty, "solver": args.solver,
            "class_weight": args.class_weight, "max_iter": args.max_iter,
            "threshold": args.threshold, "test_size": args.test_size,
            "winsor_pct": args.winsor_pct, "seed": args.seed,
            "n_train": len(X_train), "n_test": len(X_test),
        }
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.set_tags({"modelo": "LogisticRegression", "dataset": "UCI bank-full",
                         "data_sha256": file_sha256(args.data_path)[:16]})

        # 2) Entrenamiento
        model = build_pipeline(args)
        model.fit(X_train, y_train)
        n_iter = int(np.max(model.named_steps["clasificador"].n_iter_))

        # 3) Evaluación
        test_metrics, pred = evaluate(model, X_test, y_test, args.threshold)
        train_metrics, _ = evaluate(model, X_train, y_train, args.threshold)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        for name in ["roc_auc", "f1"]:  # para vigilar sobreajuste
            mlflow.log_metric(f"train_{name}", train_metrics[name])
        mlflow.log_metric("n_iter", n_iter)

        # 4) Artefactos (figuras y reportes)
        with tempfile.TemporaryDirectory() as tmp:
            for path in save_artifacts(model, X_test, y_test, pred, test_metrics, tmp):
                mlflow.log_artifact(str(path), artifact_path="evaluacion")

        # 5) Modelo entrenado con firma y ejemplo de entrada
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(sk_model=model, name="model",
                                 signature=signature, input_example=X_train.head(3))

        print(f"Corrida: {run_name}  (run_id={run.info.run_id})")
        print(json.dumps({k: round(v, 4) for k, v in test_metrics.items()}, indent=2))
    return test_metrics


if __name__ == "__main__":
    main()
