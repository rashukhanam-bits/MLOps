"""
End-to-end training pipeline for the Heart Disease classifier.
Reproduces preprocessing, model training, hyperparameter tuning,
and MLflow experiment tracking as a script (so it can run in CI/CD).
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "heart_clean.csv"
MODELS_DIR = BASE_DIR / "models"
TRACKING_DIR = BASE_DIR / "mlruns"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_DIR.resolve().as_uri())
mlflow.set_experiment("Heart Disease Prediction")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

preprocessor.fit(X_train)
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
joblib.dump(X_test, MODELS_DIR / "X_test.pkl")
joblib.dump(y_test, MODELS_DIR / "y_test.pkl")


def log_eval(y_true, y_pred, y_prob, cmap, title, artifact_prefix):
    """Compute metrics, log them, and save+log a confusion matrix + ROC curve."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"Accuracy : {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1 Score : {f1}")
    print(f"ROC AUC  : {roc_auc}")

    mlflow.log_metrics({
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1, "roc_auc": roc_auc
    })

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap)
    plt.title(f"{title} Confusion Matrix")
    cm_path = f"{artifact_prefix}_confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{title} ROC Curve")
    roc_path = f"{artifact_prefix}_roc_curve.png"
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)
    plt.close()

    return roc_auc


# ---------------------------------------------------------------------------
# 1) Logistic Regression
# ---------------------------------------------------------------------------
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

mlflow.start_run(run_name="Logistic_Regression")
logistic_pipeline.fit(X_train, y_train)
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_params(logistic_pipeline.named_steps["classifier"].get_params())

y_pred_lr = logistic_pipeline.predict(X_test)
y_prob_lr = logistic_pipeline.predict_proba(X_test)[:, 1]

roc_auc_lr = log_eval(y_test, y_pred_lr, y_prob_lr, "Blues", "Logistic Regression", "lr")

cv_scores = cross_val_score(logistic_pipeline, X, y, cv=5, scoring="accuracy")
mlflow.log_metrics({"cv_accuracy_mean": cv_scores.mean(), "cv_accuracy_std": cv_scores.std()})

joblib.dump(logistic_pipeline, MODELS_DIR / "heart_disease_model_lr.pkl")
mlflow.sklearn.log_model(logistic_pipeline, artifact_path="model")
mlflow.end_run()

# ---------------------------------------------------------------------------
# 2) Random Forest
# ---------------------------------------------------------------------------
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

mlflow.start_run(run_name="Random_Forest")
rf_pipeline.fit(X_train, y_train)
mlflow.log_param("model_type", "RandomForestClassifier")
mlflow.log_params(rf_pipeline.named_steps["classifier"].get_params())

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

roc_auc_rf = log_eval(y_test, y_pred_rf, y_prob_rf, "Greens", "Random Forest", "rf")

cv_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring="accuracy")
mlflow.log_metrics({"cv_accuracy_mean": cv_scores.mean(), "cv_accuracy_std": cv_scores.std()})

joblib.dump(rf_pipeline, MODELS_DIR / "heart_disease_model_rf.pkl")
mlflow.sklearn.log_model(rf_pipeline, artifact_path="model")
mlflow.end_run()

# ---------------------------------------------------------------------------
# 3) GridSearchCV (Random Forest)
# ---------------------------------------------------------------------------
mlflow.start_run(run_name="RF_GridSearchCV")

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline, param_grid=param_grid,
    cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

mlflow.log_params(grid_search.best_params_)
mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

best_rf = grid_search.best_estimator_
joblib.dump(best_rf, MODELS_DIR / "heart_disease_model_rf_gridsearch.pkl")

y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]

roc_auc_grid = log_eval(y_test, y_pred_best, y_prob_best, "Purples", "Tuned Random Forest", "gridsearch")

mlflow.sklearn.log_model(best_rf, artifact_path="model")
mlflow.end_run()

# ---------------------------------------------------------------------------
# 4) RandomizedSearchCV (Random Forest)
# ---------------------------------------------------------------------------
mlflow.start_run(run_name="RF_RandomizedSearchCV")

random_grid = {
    "classifier__n_estimators": np.arange(100, 501, 50),
    "classifier__max_depth": [None, 5, 10, 15, 20, 25],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"]
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline, param_distributions=random_grid,
    n_iter=20, cv=5, scoring="accuracy",
    random_state=42, n_jobs=-1, verbose=1
)
random_search.fit(X_train, y_train)

mlflow.log_params(random_search.best_params_)
mlflow.log_metric("best_cv_accuracy", random_search.best_score_)

best_random_rf = random_search.best_estimator_

y_pred_random = best_random_rf.predict(X_test)
y_prob_random = best_random_rf.predict_proba(X_test)[:, 1]

roc_auc_random = log_eval(y_test, y_pred_random, y_prob_random, "Oranges", "RandomizedSearch RF", "randomsearch")

joblib.dump(best_random_rf, MODELS_DIR / "heart_disease_model_random_search.pkl")
mlflow.sklearn.log_model(best_random_rf, artifact_path="model")
mlflow.end_run()

# ---------------------------------------------------------------------------
# 5) Pick the best model overall and save it as final_model.pkl
# ---------------------------------------------------------------------------
models_dict = {
    "Logistic Regression": (logistic_pipeline, roc_auc_lr),
    "Random Forest": (rf_pipeline, roc_auc_rf),
    "GridSearch RF": (best_rf, roc_auc_grid),
    "RandomSearch RF": (best_random_rf, roc_auc_random),
}

best_model_name = max(models_dict, key=lambda k: models_dict[k][1])
best_model = models_dict[best_model_name][0]

print(f"Best model: {best_model_name} (ROC-AUC={models_dict[best_model_name][1]:.4f})")

joblib.dump(best_model, MODELS_DIR / "final_model.pkl")
print(f"Saved as {MODELS_DIR / 'final_model.pkl'}")

print(mlflow.get_tracking_uri())