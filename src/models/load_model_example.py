import pickle
import joblib
from pathlib import Path
import pandas as pd
import numpy as np


def load_model_pickle():
    """Carica il modello salvato con pickle"""
    model_path = Path('models') / 'linear_regression_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_model_joblib():
    """Carica il modello salvato con joblib"""
    model_path = Path('models') / 'linear_regression_model.joblib'
    model = joblib.load(model_path)
    return model


def predict_example():
    # Carica il modello (puoi scegliere tra pickle o joblib)
    model = load_model_pickle()  # oppure load_model_joblib()
    
    # Esempio di dati di input (assicurati che abbiano le stesse feature del modello)
    # Le feature sono: 'region_match', 'user_likes_for_category', 'event_popularity'
    example_data = pd.DataFrame({
        'region_match': [1, 0, 1],
        'user_likes_for_category': [2, 0, 5],
        'event_popularity': [10, 3, 20]
    })
    
    # Effettua la predizione
    predictions = model.predict(example_data)
    
    print("Dati di input:")
    print(example_data)
    print("\nPredizioni (score):")
    for i, pred in enumerate(predictions):
        print(f"Esempio {i+1}: {pred:.4f}")
    
    return predictions


if __name__ == "__main__":
    # Carica il modello e fai una predizione di esempio
    predictions = predict_example()
    
    # Carica anche i metadati del modello
    metrics_path = Path('reports') / 'training_metrics.csv'
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        print("\nMetriche del modello:")
        print(f"MSE: {metrics['mse'].values[0]:.6f}")
        print(f"R²: {metrics['r2'].values[0]:.6f}")