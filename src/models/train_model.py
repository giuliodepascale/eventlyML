import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
import pickle
import joblib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri("https://dagshub.com/giuliodepascale/eventlyML.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv('USERNAME')
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv('PASSWORD')

# Carica il dataset
df = pd.read_csv('data/processed/user_event_similarity.csv')

# Seleziona le feature e il target
X = df[['region_match', 'user_likes_for_category', 'event_popularity']]
y = df['score']

# Suddividi in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="LinearRegression-user-event"):
    # Inizializza e addestra il modello
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predici sul test set
    y_pred = model.predict(X_test)

    # Valuta il modello
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Log parametri, metriche e modello su MLflow
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Prepara input_example e signature per MLflow
    input_example = X_test.iloc[:2]
    signature = infer_signature(X_test, y_pred)

    # Log del modello su MLflow
    mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)
    
    # Salva il modello localmente
    # Crea la directory 'models' se non esiste
    models_dir = Path('model-linear-regression')
    models_dir.mkdir(exist_ok=True)
    
    # Salva il modello usando pickle
    pickle_path = models_dir / 'linear_regression_model.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Salva il modello anche usando joblib (più efficiente per oggetti grandi)
    joblib_path = models_dir / 'linear_regression_model.joblib'
    joblib.dump(model, joblib_path)
    
    print(f"\nModello salvato localmente in:\n- {pickle_path}\n- {joblib_path}")
    
    # Salva anche i metadati del modello
    metadata = {
        'features': list(X.columns),
        'metrics': {
            'mse': mse,
            'r2': r2
        }
    }
    
    # Salva i metadati come CSV
    metrics_path = Path('reports') / 'training_metrics.csv'
    pd.DataFrame([metadata['metrics']]).to_csv(metrics_path, index=False)

