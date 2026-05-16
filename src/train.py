"""
Glass Classification - Academic ML Project

This script trains and evaluates several supervised classification models
on a glass classification dataset.

The project demonstrates a basic machine learning workflow:
- Load dataset
- Split features and target
- Train/test split
- Preprocess numerical features when needed
- Train multiple classifiers
- Evaluate accuracy, classification report, and confusion matrix
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "glass.csv"
RESULTS_DIR = ROOT_DIR / "results"
TARGET_COLUMN = "Type of glass"
ID_COLUMN = "Id"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the glass dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into input features X and target labels y."""
    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_models() -> dict[str, Pipeline | RandomForestClassifier]:
    """Return classification models for comparison."""
    return {
        "Gaussian Naive Bayes": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "SVM (RBF Kernel)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", random_state=42)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Train and evaluate models, then save summary results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    results: list[dict[str, float | str]] = []
    best_model_name = None
    best_accuracy = -1.0
    best_predictions = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results.append(
            {
                "model": model_name,
                "accuracy": round(accuracy, 4),
                "wrong_predictions": int((predictions != y_test).sum()),
                "test_samples": len(y_test),
            }
        )

        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification report:")
        print(classification_report(y_test, predictions, zero_division=0))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_predictions = predictions

    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    if best_predictions is not None:
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, best_predictions, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"actual_{label}" for label in labels],
            columns=[f"predicted_{label}" for label in labels],
        )
        cm_df.to_csv(RESULTS_DIR / "confusion_matrix_random_forest.csv")

    print("=" * 70)
    print("Model comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name} ({best_accuracy:.2%})")

    return results_df


def main() -> None:
    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    X, y = prepare_features(df)
    evaluate_models(X, y)


if __name__ == "__main__":
    main()
