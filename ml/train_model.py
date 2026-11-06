"""
Exercise 4 training script: Group-aware ML Pipeline.
Compares Logistic Regression and Random Forest using GroupShuffleSplit on robot_id,
evaluates macro F1 score, and serializes the winning pipeline.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA = Path("data/training.csv")

FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]
TARGET = "status"
GROUP_COL = "robot_id"

def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            "Training dataset not found. Run: python simulator/generate_dataset.py"
        )

    df = pd.read_csv(DATA)
    print("=== 1. Exploración Inicial de Datos ===")
    print(f"Total registros: {len(df)}")
    print("\nDistribución de clases:")
    print(df[TARGET].value_counts(normalize=True).round(4) * 100)

    print("\n=== 2. División Entrenamiento / Prueba por Grupos (robot_id) ===")
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP_COL]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = sorted(list(groups.iloc[train_idx].unique()))
    test_groups = sorted(list(groups.iloc[test_idx].unique()))
    print(f"Robots en entrenamiento ({len(train_groups)}): {train_groups}")
    print(f"Robots en prueba ({len(test_groups)}): {test_groups}")
    assert set(train_groups).isdisjoint(set(test_groups)), "¡Error: Fuga de datos entre grupos!"

    print("\n=== 3. Comparación de Modelos ===")

    # Modelo 1: Regresión Logística
    pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    f1_lr = f1_score(y_test, y_pred_lr, average="macro")
    print(f"F1 Macro - Regresión Logística: {f1_lr:.4f}")

    # Modelo 2: Random Forest
    pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average="macro")
    print(f"F1 Macro - Random Forest: {f1_rf:.4f}")

    # Selección del mejor modelo según F1 Macro
    if f1_rf >= f1_lr:
        best_model = pipe_rf
        best_name = "Random Forest"
        best_f1 = f1_rf
        best_pred = y_pred_rf
    else:
        best_model = pipe_lr
        best_name = "Regresión Logística"
        best_f1 = f1_lr
        best_pred = y_pred_lr

    print(f"\nModelo ganador: {best_name} (F1 Macro: {best_f1:.4f})")
    print("\nReporte de clasificación detallado:")
    print(classification_report(y_test, best_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, best_pred))

    print("\n=== 4. Guardar Artefacto Serializado ===")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Modelo guardado exitosamente en: {model_path}")

if __name__ == "__main__":
    main()