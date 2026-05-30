import argparse
import mlflow
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
from modelling_utils import (
    setup_mlflow,
    load_data,
    save_and_log_artifacts
)

# Load Data
X_train, y_train, X_train_smote, y_train_smote, X_test, y_test = load_data(dataset_dir='customer_churn_preprocessed')

def run_tuning(n_trials):

    def objective(trial):
        C = trial.suggest_float("C", 1e-5, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        max_iter = trial.suggest_int("max_iter", 100, 500)

        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
        scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="f1")
        f1_score = scores.mean()

        return f1_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

def eval_metric(model, data, target):
    """
    Evaluate the model performance.
    """
    y_pred = model.predict(data)
    y_prob = model.predict_proba(data)[:, 1]

    accuracy = accuracy_score(target, y_pred)
    precision = precision_score(target, y_pred, average='binary')
    recall = recall_score(target, y_pred, average='binary')
    f1 = f1_score(target, y_pred, average='binary')
    roc_auc = roc_auc_score(target, y_prob)
    log_loss_val = log_loss(target, y_prob)
    score = model.score(data, target)
    
    return accuracy, precision, recall, f1, roc_auc, log_loss_val, score


def main():
    parser = argparse.ArgumentParser(description="Optuna + MLflow Tuning")
    parser.add_argument("--trials", type=int, default=20, help="number of trials")
    args = parser.parse_args()
    best_params = run_tuning(n_trials=args.trials)
    setup_mlflow('dagshub', 'customer_churn')
    model_name = "Logistic Regression-Optuna"
    input_example=X_train_smote[0:5]

    with mlflow.start_run(run_name= model_name):
        
        # Log Best Hyperparameters
        mlflow.log_params({
            'model_name': 'LR-Optuna', 
            **best_params, 
            'input_example': input_example})

        # Train final model
        best_model = LogisticRegression(**best_params)
        best_model.fit(X_train_smote, y_train_smote)

        # Calculate train metrics
        train_accuracy, train_precision, train_recall, train_f1, train_roc_auc, train_log_loss, train_score = eval_metric(best_model, X_train_smote, y_train_smote)
        metrics_train_dict = {
            'train_accuracy_score': train_accuracy,
            'train_precision_score': train_precision,
            'train_recall_score': train_recall,
            'train_f1_score': train_f1,
            'train_roc_auc': train_roc_auc,
            'train_log_loss': train_log_loss,
            'train_score': train_score
        }

        # Calculate evaluation metrics
        accuracy, precision, recall, f1, roc_auc, log_loss, score = eval_metric(best_model, X_test, y_test)
        metrics_dict = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1,
            'test_roc_auc': roc_auc,
            'test_log_loss': log_loss,
            'test_score': score
        }

        # Log Metrics
        mlflow.log_metrics(metrics_train_dict)
        mlflow.log_metrics(metrics_dict)

        # Log Model and Register it
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name=model_name,
            registered_model_name=model_name
        )

        # Save and log model
        save_and_log_artifacts(best_model, X_train_smote, X_test, y_test, metrics_dict)

        print(metrics_dict)
    print("\nModelling and logging complete!")

if __name__ == '__main__':
    main()
