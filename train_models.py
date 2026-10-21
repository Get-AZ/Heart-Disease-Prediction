from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_models(X_train, y_train, model_dir=None):
    project_dir = Path(__file__).resolve().parent

    # Step 1: Prepare local output folders.
    model_dir = (
        project_dir / "models"
        if model_dir is None
        else Path(model_dir)
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = project_dir / "mlflow_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Configure local MLflow tracking and registry.
    database_path = (project_dir / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{database_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    experiment_name = "Heart_Disease_Classification"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_dir.as_uri(),
        )

    mlflow.set_experiment(experiment_name)

    print(f"\nMLflow tracking URI: {tracking_uri}")

    # Step 3: Define models and tuning parameters.
    # SMOTE is omitted for this baseline.
    models = {
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=1,
            ),
            "params": {
                "n_estimators": [100],
                "max_depth": [3, 5],
                "learning_rate": [0.05],
                "subsample": [0.8],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=42,
                n_jobs=1,
            ),
            "params": {
                "n_estimators": [100],
                "max_depth": [5, None],
                "min_samples_split": [2, 5],
            },
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5],
                "min_samples_split": [2, 4],
            },
        },
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    trained_models = {}

    # Step 4: Tune, log, register, and save each model.
    for name, config in models.items():
        print(f"\nTraining {name}...", flush=True)

        with mlflow.start_run(run_name=f"{name}_training") as run:
            mlflow.set_tags({
                "model_name": name,
                "input_type": "preprocessed_selected_features",
                "cv_limitation": (
                    "Preprocessing and selection were fitted before CV."
                ),
            })

            grid = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                error_score="raise",
                return_train_score=False,
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            trained_models[name] = best_model

            # Record parameters and cross-validation results.
            mlflow.log_params(grid.best_params_)
            mlflow.log_params({
                "cv_folds": cv.n_splits,
                "scoring": "accuracy",
                "training_samples": X_train.shape[0],
                "input_features": X_train.shape[1],
                "random_state": 42,
                "smote_used": False,
            })

            best_index = grid.best_index_

            mlflow.log_metrics({
                "mean_cv_accuracy": float(grid.best_score_),
                "std_cv_accuracy": float(
                    grid.cv_results_["std_test_score"][best_index]
                ),
            })

            # Save the full tuning table as an MLflow artifact.
            cv_results = pd.DataFrame(grid.cv_results_)
            mlflow.log_text(
                cv_results.to_csv(index=False),
                artifact_file="cv_results.csv",
            )

            # Preserve pickle filenames used by evaluate_models.py.
            model_path = model_dir / f"{name.lower()}_model.pkl"

            with open(model_path, "wb") as file:
                pickle.dump(best_model, file)

            # Input example must match the processed training features.
            input_example = (
                X_train.iloc[:5].copy()
                if hasattr(X_train, "iloc")
                else X_train[:5].copy()
            )

            signature = infer_signature(
                input_example,
                best_model.predict(input_example),
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=best_model,
                name=f"{name.lower()}_model",
                serialization_format="cloudpickle",
                input_example=input_example,
                signature=signature,
                registered_model_name=f"HeartDisease_{name}",
            )

            print(f"Best parameters: {grid.best_params_}")
            print(f"Mean CV accuracy: {grid.best_score_:.4f}")
            print(f"Saved model: {model_path}")
            print(f"MLflow run ID: {run.info.run_id}")
            print(f"MLflow model URI: {model_info.model_uri}")

    return trained_models
