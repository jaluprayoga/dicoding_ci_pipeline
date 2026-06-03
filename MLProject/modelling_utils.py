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
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
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

def save_and_log_artifacts(best_model, X_train, X_test, y_test, metrics_dict, path):
    """Generate and log evaluation plots and reports as artifacts."""
    os.makedirs(path, exist_ok=True)
    
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    def save_and_log_fig(filename):
        full_path = os.path.join(path, filename)
        plt.tight_layout()
        plt.savefig(full_path)
        plt.close()
        mlflow.log_artifact(full_path)

    def write_and_log_file(filename, content, is_json=False):
        full_path = os.path.join(path, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            if is_json:
                json.dump(content, f, indent=4)
            else:
                f.write(content)
        mlflow.log_artifact(full_path)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    save_and_log_fig('confusion_matrix.png')

    # Estimator HTML representation
    write_and_log_file('estimator.html', estimator_html_repr(best_model))

    # Metric Info JSON
    write_and_log_file('metric_info.json', metrics_dict, is_json=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    save_and_log_fig('roc_curve.png')

    # Classification Report
    report_text = classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn'])
    write_and_log_file('classification_report.txt', report_text)

    # Save model locally
    local_model_path = os.path.join(path, 'model')
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
    mlflow.sklearn.save_model(sk_model=best_model, path=local_model_path)
