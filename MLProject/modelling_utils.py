import os
import sys
import io
import urllib.request
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.utils import estimator_html_repr
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)
import numpy as np
import tempfile
from dotenv import load_dotenv

# Force UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load environment variables
load_dotenv()

def setup_mlflow(server, experiment_name):
    """Configure MLflow and set the experiment."""
    print(f"Setup MLflow for {server} server")
    if server == "dagshub":
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")    
        repo_name = os.getenv("DAGSHUB_REPO_NAME")
        token = os.getenv("DAGSHUB_TOKEN")
        dagshub.auth.add_app_token(token)
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        print("Successfully connected to DagsHub!")
    if server == "local":
        # Check if the local MLflow server is running on port 5000
        try:
            with urllib.request.urlopen("http://localhost:5000", timeout=1):
                mlflow.set_tracking_uri("http://localhost:5000")
                print("MLflow tracking to local server at: http://localhost:5000")
        except Exception:
            mlflow.set_tracking_uri("file:///./mlruns")
            print("MLflow tracking locally to directory: ./mlruns")

    # If run in 'mlflow run' context, verify if the run exists in the newly set tracking URI.
    if "MLFLOW_RUN_ID" in os.environ:
        run_id = os.environ["MLFLOW_RUN_ID"]
        try:
            mlflow.tracking.MlflowClient().get_run(run_id)
        except Exception:
            print(f"Warning: Run ID {run_id} from MLFLOW_RUN_ID was not found in the current tracking URI.")
            os.environ.pop("MLFLOW_RUN_ID", None)

    mlflow.set_experiment(experiment_name)

def load_data(dataset_dir):
    """Load preprocessed datasets."""
    print(f"Loading datasets from {dataset_dir}...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, dataset_dir)

    X_train = pd.read_csv(os.path.join(dataset_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(dataset_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(dataset_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(dataset_dir, 'y_test.csv')).values.ravel()
    X_train_smote = pd.read_csv(os.path.join(dataset_dir, 'X_train_smote.csv'))
    y_train_smote = pd.read_csv(os.path.join(dataset_dir, 'y_train_smote.csv')).values.ravel()
    
    print("Datasets loaded successfully!")
    return X_train, y_train, X_train_smote, y_train_smote, X_test, y_test

def save_and_log_artifacts(best_model, X_train, X_test, y_test, metrics_dict, output_dir=None, path=None):
    """Generate and log evaluation plots and reports as artifacts."""

    if path is not None:
        output_dir = path

    y_test_pred = best_model.predict(X_test)

    def _write_artifacts(target_dir):
        # 1. Confusion Matrix
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            best_model, X_test, y_test, 
            display_labels=['No Churn', 'Churn'], 
            cmap='Blues', ax=ax
        )
        plt.title('Confusion Matrix')
        plt.tight_layout()
        fig_cm.savefig(os.path.join(target_dir, 'confusion_matrix.png'))
        plt.close(fig_cm)

        # 2. Estimator HTML representation
        with open(os.path.join(target_dir, 'estimator.html'), 'w', encoding='utf-8') as f:
            f.write(estimator_html_repr(best_model))

        # 3. Metric Info JSON
        with open(os.path.join(target_dir, 'metric_info.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4)

        # 4. ROC Curve
        fig_roc, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
        plt.title('ROC Curve')
        plt.tight_layout()
        fig_roc.savefig(os.path.join(target_dir, 'roc_curve.png'))
        plt.close(fig_roc)

        # 5. Classification Report
        report_text = classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn'])
        with open(os.path.join(target_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)

        # 6. Precision-Recall Curve
        fig_pr, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
        plt.title('Precision-Recall Curve')
        plt.tight_layout()
        fig_pr.savefig(os.path.join(target_dir, 'precision_recall_curve.png'))
        plt.close(fig_pr)

        # 7. Feature Importance Plot (Coefficients)
        if hasattr(best_model, "coef_"):
            coefficients = best_model.coef_[0]
            feature_names = X_train.columns
            indices = np.argsort(np.abs(coefficients))
            
            fig_fi = plt.figure(figsize=(8, 6))
            plt.barh(range(len(indices)), coefficients[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Coefficient Value')
            plt.title('Feature Importance (Logistic Regression Coefficients)')
            plt.tight_layout()
            fig_fi.savefig(os.path.join(target_dir, 'feature_importance.png'))
            plt.close(fig_fi)

        # 8. Save model locally
        local_model_path = os.path.join(target_dir, 'model')
        if os.path.exists(local_model_path):
            import shutil
            shutil.rmtree(local_model_path)
        mlflow.sklearn.save_model(sk_model=best_model, path=local_model_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _write_artifacts(output_dir)
        mlflow.log_artifacts(output_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_artifacts(tmp_dir)
            mlflow.log_artifacts(tmp_dir)

def eval_metric(model, data, target):
    """
    Evaluate the model performance.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        log_loss,
        confusion_matrix
    )
    y_pred = model.predict(data)
    y_prob = model.predict_proba(data)[:, 1]

    accuracy = accuracy_score(target, y_pred)
    precision = precision_score(target, y_pred, average='binary')
    recall = recall_score(target, y_pred, average='binary')
    f1 = f1_score(target, y_pred, average='binary')
    roc_auc = roc_auc_score(target, y_prob)
    log_loss_val = log_loss(target, y_prob)
    score = model.score(data, target)
    
    tn, fp, fn, tp = confusion_matrix(target, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return accuracy, precision, recall, f1, roc_auc, log_loss_val, score, specificity, fpr
